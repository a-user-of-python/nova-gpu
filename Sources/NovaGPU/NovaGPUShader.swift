// NovaGPUShader.swift — NovaGPU shader instruction set + SIMT warp executor.
//
// The NovaGPU runs small shader programs (vertex, fragment and compute
// kernels all share ONE instruction set) on warps of 32 lanes. A warp is
// the unit of issue: one instruction is decoded once and applied to every
// active lane, which is what "SIMT" means here. Control flow is
// warp-uniform (a branch tests lane 0), the classic early-GPU model;
// per-lane divergence is expressed with CMPLT + SEL predication.
//
// Instruction encoding (32-bit words, little-endian fields):
//   bits  0.. 7  opcode
//   bits  8..15  dst register (0..15) — for OUT: the output slot
//   bits 16..23  source A register (0..15) — for OUT: the source register
//   bits 24..31  source B register (0..15) — for LDTEX: the texture unit
// MOVI and the branch instructions consume a second word / pack an imm16.
//
// Each lane owns 16 vector registers r0..r15 of 4 x FP32. Uniforms are
// 16 shared vec4 constant registers c0..c15, set per draw / launch.
//
// Stage conventions:
//   vertex   : LDATTR n reads vertex attribute n (NVF1 layout, see
//              NovaGPU.swift). OUT 0 = clip-space position (x,y,z,w);
//              OUT 1..7 = varyings handed to the fragment stage.
//   fragment : LDATTR n reads interpolated varying n. OUT 0 = colour.
//   compute  : LDATTR 0 returns (globalLaneId, 0, 0, 0). LDBUF/STBUF
//              address the launch's target buffer by byte offset.

import Foundation

// MARK: - Opcodes

enum GPOpcode: UInt32 {
    case nop    = 0x00
    case mov    = 0x01   // rd = ra
    case movi   = 0x02   // rd = splat(Float(bitPattern: imm32)); imm in next word
    case ldunif = 0x03   // rd = c[a]
    case ldattr = 0x04   // rd = attribute/varying a (stage-dependent)
    case out    = 0x05   // output slot dst = ra
    // FP32 ALU (component-wise unless noted)
    case add    = 0x10   // rd = ra + rb
    case sub    = 0x11
    case mul    = 0x12
    case div    = 0x13
    case min    = 0x14
    case max    = 0x15
    case mad    = 0x16   // rd = ra * rb + rd   (accumulating multiply-add)
    case rcp    = 0x17   // rd = 1 / ra
    case sqrt   = 0x18
    case rsq    = 0x19   // rd = 1 / sqrt(ra)
    case sin    = 0x1A
    case cos    = 0x1B
    case dot3   = 0x1C   // rd = splat(dot(ra.xyz, rb.xyz))
    case dot4   = 0x1D   // rd = splat(dot(ra.xyzw, rb.xyzw))
    case cmplt  = 0x20   // rd = (ra < rb) ? 1 : 0 per component
    case sel    = 0x21   // rd = (ra != 0) ? rb : rd per component (predication)
    case f2i    = 0x22   // rd bits = Int32(ra) per component (truncation)
    case i2f    = 0x23   // rd = Float(Int32(bits of ra)) per component
    // INT32 ALU on the FP32 bit patterns, per component
    case iadd   = 0x30
    case isub   = 0x31
    case imul   = 0x32
    case iand   = 0x33
    case ior    = 0x34
    case ixor   = 0x35
    case ishl   = 0x36
    case ishr   = 0x37   // arithmetic shift
    // Memory / sampling
    case ldtex  = 0x40   // rd = texture(unit b).sample(ra.xy), bilinear
    case ldbuf  = 0x41   // rd.x bits = u32 at byte addr Int32(bits of ra.x); rd.yzw = 0
    case stbuf  = 0x42   // u32 at byte addr Int32(bits of ra.x) = bits of rb.x
    // Control (warp-uniform; condition read from lane 0)
    case bra    = 0x50   // pc += imm16 (signed, packed in a|b)
    case brz    = 0x51   // if ra.x == 0 (lane 0) then pc += imm16
    case end    = 0x52   // end of program for this warp
}

// MARK: - Program builder (used by the test app / tests to author shaders)

struct ShaderBuilder {
    private(set) var words: [UInt32] = []
    private var labels: [Int] = []            // word index per label id
    private var fixups: [(at: Int, label: Int)] = []

    private mutating func emit(_ op: GPOpcode, _ dst: Int = 0, _ a: Int = 0, _ b: Int = 0) {
        words.append(op.rawValue | (UInt32(dst & 0xFF) << 8) | (UInt32(a & 0xFF) << 16) | (UInt32(b & 0xFF) << 24))
    }
    mutating func label() -> Int { labels.append(words.count); return labels.count - 1 }
    private mutating func emitBranch(_ op: GPOpcode, _ a: Int, to label: Int) {
        let at = words.count
        emit(op, 0, a, 0)
        fixups.append((at: at, label: label))
    }
    mutating func build() -> [UInt32] {
        var w = words
        for f in fixups {
            let target = labels[f.label]
            let rel = Int16(target - (f.at + 1))
            let bits = UInt32(UInt16(bitPattern: rel))
            w[f.at] |= ((bits & 0xFF) << 16) | (((bits >> 8) & 0xFF) << 24)
        }
        return w
    }

    mutating func nop()                 { emit(.nop) }
    mutating func mov(_ d: Int, _ a: Int)            { emit(.mov, d, a) }
    mutating func movi(_ d: Int, _ imm: UInt32)      { emit(.movi, d); words.append(imm) }
    mutating func movf(_ d: Int, _ v: Float)         { emit(.movi, d); words.append(v.bitPattern) }
    mutating func ldunif(_ d: Int, _ idx: Int)       { emit(.ldunif, d, idx) }
    mutating func ldattr(_ d: Int, _ slot: Int)      { emit(.ldattr, d, slot) }
    mutating func out(_ slot: Int, _ a: Int)         { emit(.out, slot, a) }
    mutating func add(_ d: Int, _ a: Int, _ b: Int)  { emit(.add, d, a, b) }
    mutating func sub(_ d: Int, _ a: Int, _ b: Int)  { emit(.sub, d, a, b) }
    mutating func mul(_ d: Int, _ a: Int, _ b: Int)  { emit(.mul, d, a, b) }
    mutating func div(_ d: Int, _ a: Int, _ b: Int)  { emit(.div, d, a, b) }
    mutating func min(_ d: Int, _ a: Int, _ b: Int)  { emit(.min, d, a, b) }
    mutating func max(_ d: Int, _ a: Int, _ b: Int)  { emit(.max, d, a, b) }
    mutating func mad(_ d: Int, _ a: Int, _ b: Int)  { emit(.mad, d, a, b) }
    mutating func rcp(_ d: Int, _ a: Int)            { emit(.rcp, d, a) }
    mutating func sqrt(_ d: Int, _ a: Int)           { emit(.sqrt, d, a) }
    mutating func rsq(_ d: Int, _ a: Int)            { emit(.rsq, d, a) }
    mutating func sin(_ d: Int, _ a: Int)            { emit(.sin, d, a) }
    mutating func cos(_ d: Int, _ a: Int)            { emit(.cos, d, a) }
    mutating func dot3(_ d: Int, _ a: Int, _ b: Int) { emit(.dot3, d, a, b) }
    mutating func dot4(_ d: Int, _ a: Int, _ b: Int) { emit(.dot4, d, a, b) }
    mutating func cmplt(_ d: Int, _ a: Int, _ b: Int){ emit(.cmplt, d, a, b) }
    mutating func sel(_ d: Int, _ a: Int, _ b: Int)  { emit(.sel, d, a, b) }
    mutating func f2i(_ d: Int, _ a: Int)            { emit(.f2i, d, a) }
    mutating func i2f(_ d: Int, _ a: Int)            { emit(.i2f, d, a) }
    mutating func iadd(_ d: Int, _ a: Int, _ b: Int) { emit(.iadd, d, a, b) }
    mutating func isub(_ d: Int, _ a: Int, _ b: Int) { emit(.isub, d, a, b) }
    mutating func imul(_ d: Int, _ a: Int, _ b: Int) { emit(.imul, d, a, b) }
    mutating func iand(_ d: Int, _ a: Int, _ b: Int) { emit(.iand, d, a, b) }
    mutating func ior(_ d: Int, _ a: Int, _ b: Int)  { emit(.ior, d, a, b) }
    mutating func ixor(_ d: Int, _ a: Int, _ b: Int) { emit(.ixor, d, a, b) }
    mutating func ishl(_ d: Int, _ a: Int, _ b: Int) { emit(.ishl, d, a, b) }
    mutating func ishr(_ d: Int, _ a: Int, _ b: Int) { emit(.ishr, d, a, b) }
    mutating func ldtex(_ d: Int, _ a: Int, unit: Int) { emit(.ldtex, d, a, unit) }
    mutating func ldbuf(_ d: Int, _ a: Int)          { emit(.ldbuf, d, a) }
    mutating func stbuf(_ a: Int, _ b: Int)          { emit(.stbuf, 0, a, b) }
    mutating func bra(to label: Int)                 { emitBranch(.bra, 0, to: label) }
    /// BRZ encoding: condition register goes in the DST byte; the imm16
    /// branch offset occupies the a|b bytes (patched by the fixup).
    mutating func brz(_ condReg: Int, to label: Int) {
        let at = words.count
        emit(.brz, condReg, 0, 0)
        fixups.append((at: at, label: label))
    }
    mutating func end()                              { emit(.end) }
}

// MARK: - Warp execution context (stage-provided I/O)

struct WarpContext {
    /// 16 uniform vec4s, flattened [idx*4 + comp].
    var uniforms: [Float]
    /// (lane, slot) -> vec4. Vertex: attributes. Fragment: varyings.
    /// Compute: slot 0 = (gid, 0, 0, 0).
    var attrProvider: (Int, Int) -> (Float, Float, Float, Float)
    /// (unit, u, v) -> RGBA in 0..1.
    var texSampler: (Int, Float, Float) -> (Float, Float, Float, Float)
    /// Compute-only raw buffer access by byte address (target buffer).
    var bufLoad: (Int) -> UInt32
    var bufStore: (Int, UInt32) -> Void
    /// (lane, slot, vec4) — vertex/fragment stage output sink.
    var outWriter: (Int, Int, Float, Float, Float, Float) -> Void

    static func dummy() -> WarpContext {
        WarpContext(uniforms: [Float](repeating: 0, count: 64),
                    attrProvider: { _, _ in (0, 0, 0, 0) },
                    texSampler: { _, _, _ in (0, 0, 0, 0) },
                    bufLoad: { _ in 0 },
                    bufStore: { _, _ in },
                    outWriter: { _, _, _, _, _, _ in })
    }
}

// MARK: - Decoded program (speed-round fast path)
//
// A shader program decoded ONCE into flat operand arrays, so a warp's
// inner loop never re-decodes instruction words. The arrays are indexed
// by WORD index, not instruction index: a MOVI's immediate word gets its
// own slot, decoded from its own bits exactly as the v1 interpreter
// would decode it if a branch landed there, and branch offsets stay
// word-relative. Control flow is therefore bit-identical to v1 — this
// is a decode cache, not a new program representation.

struct DecodedShader {
    var op: [UInt32] = []     // raw opcode byte per word
    var d: [UInt32] = []      // dst byte per word
    var a: [UInt32] = []      // source A byte per word
    var b: [UInt32] = []      // source B byte per word
    var imm: [UInt32] = []    // MOVI only: the following word (0 past end)
    /// Bitmask of LDATTR slots the program can read (bit = slot).
    /// The rasteriser uses the fragment program's mask to interpolate
    /// only the varyings that can actually be observed.
    var attrSlotMask: UInt32 = 0
    var count: Int { op.count }

    init(_ program: [UInt32]) {
        let n = program.count
        op = [UInt32](repeating: 0, count: n)
        d = [UInt32](repeating: 0, count: n)
        a = [UInt32](repeating: 0, count: n)
        b = [UInt32](repeating: 0, count: n)
        imm = [UInt32](repeating: 0, count: n)
        for i in 0..<n {
            let w = program[i]
            let o = w & 0xFF
            op[i] = o
            d[i] = (w >> 8) & 0xFF
            a[i] = (w >> 16) & 0xFF
            b[i] = (w >> 24) & 0xFF
            if o == GPOpcode.movi.rawValue {
                imm[i] = (i + 1 < n) ? program[i + 1] : 0
            } else if o == GPOpcode.ldattr.rawValue, a[i] < 32 {
                attrSlotMask |= UInt32(1) << a[i]
            }
        }
    }
}

// MARK: - Executor
//
// Speed round (v2): the v1 executor re-decoded every instruction from
// its packed word on every warp, indexed the register file through a
// bounds-checked Swift array with a computed index per component, and
// routed every operation through nested helper closures. The semantics
// were never the problem — the per-element overhead was. This executor
// runs the SAME semantics out of a DecodedShader:
//   * register file in a stack-allocated SIMD4<Float> buffer (the
//     [reg][lane][comp] layout becomes 512 vec4s; a register's lane
//     vectors are contiguous), so component-wise FP ops are single
//     vector ops per lane with no bounds checks anywhere;
//   * operations whose results depend on Swift's exact scalar semantics
//     (MIN/MAX NaN behaviour, DOT accumulation order, F2I/I2F
//     conversions, the integer ops on bit patterns) run scalar loops
//     over raw pointers with the v1 expressions copied verbatim;
//   * per-issue bookkeeping (pc, steps guard, lane-op counting) mirrors
//     v1 exactly, including END not counting itself and unknown
//     opcodes executing as NOP.

enum ShaderExecutor {
    static let warpSize = 32
    static let regCount = 16

    /// Runs one warp of `program` with `lanes` active lanes (1...32).
    /// Returns the number of lane-instructions executed (issue count x
    /// active lanes at each issue) — the number the device reports as
    /// compute-unit work. (v1-compatible entry point: decodes per call;
    /// the pipeline decodes once per draw/launch and uses the
    /// DecodedShader overload.)
    static func runWarp(program: [UInt32], lanes: Int, ctx: WarpContext) -> Int {
        let n = max(0, min(warpSize, lanes))
        if n == 0 || program.isEmpty { return 0 }
        return runWarp(decoded: DecodedShader(program), lanes: n, ctx: ctx)
    }

    static func runWarp(decoded: DecodedShader, lanes: Int, ctx: WarpContext) -> Int {
        let n = max(0, min(warpSize, lanes))
        if n == 0 || decoded.count == 0 { return 0 }
        return ctx.uniforms.withUnsafeBufferPointer { ubuf in
            withUnsafeTemporaryAllocation(of: SIMD4<Float>.self,
                                          capacity: regCount * warpSize) { vbuf in
                vbuf.initialize(repeating: SIMD4<Float>())
                let vb = vbuf.baseAddress!
                let rp = UnsafeMutableRawPointer(vb).assumingMemoryBound(to: Float.self)
                let up = UnsafeMutableRawPointer(vb).assumingMemoryBound(to: UInt32.self)
                return runLoop(decoded: decoded, n: n, ctx: ctx,
                               ubuf: ubuf, vb: vb, rp: rp, up: up)
            }
        }
    }

    private static func runLoop(decoded: DecodedShader, n: Int, ctx: WarpContext,
                                ubuf: UnsafeBufferPointer<Float>,
                                vb: UnsafeMutablePointer<SIMD4<Float>>,
                                rp: UnsafeMutablePointer<Float>,
                                up: UnsafeMutablePointer<UInt32>) -> Int {
        let opsA = decoded.op, dA = decoded.d, aA = decoded.a, bA = decoded.b, immA = decoded.imm
        let wc = decoded.count
        var laneOps = 0
        var pc = 0
        var steps = 0
        let maxSteps = 100_000   // runaway-shader guard; a fault, not a hang
        while pc >= 0 && pc < wc && steps < maxSteps {
            steps += 1
            let opc = opsA[pc]
            let d = Int(dA[pc])
            let a = Int(aA[pc])
            let b = Int(bA[pc])
            var nextPc = pc + 1
            switch opc {
            case 0x52: // END — returns without counting itself (as v1)
                return laneOps
            case 0x50: // BRA — imm16 packed in the a|b bytes
                laneOps += n
                let rel = Int(Int16(bitPattern: UInt16(a | (b << 8))))
                nextPc = pc + 1 + rel
            case 0x51: // BRZ — condition register rides in the dst byte
                laneOps += n
                if rp[d * 128] == 0 {
                    let rel = Int(Int16(bitPattern: UInt16(a | (b << 8))))
                    nextPc = pc + 1 + rel
                }
            case 0x02: // MOVI — splat of the following word's FP32 bits
                laneOps += n
                let v = SIMD4<Float>(repeating: Float(bitPattern: immA[pc]))
                let base = d * warpSize
                for l in 0..<n { vb[base + l] = v }
                nextPc = pc + 2
            default:
                laneOps += n
                execute(opc, d: d, a: a, b: b, n: n, ctx: ctx, ubuf: ubuf, vb: vb, rp: rp, up: up)
            }
            pc = nextPc
        }
        return laneOps
    }

    private static func execute(_ opc: UInt32, d: Int, a: Int, b: Int, n: Int,
                                ctx: WarpContext, ubuf: UnsafeBufferPointer<Float>,
                                vb: UnsafeMutablePointer<SIMD4<Float>>,
                                rp: UnsafeMutablePointer<Float>,
                                up: UnsafeMutablePointer<UInt32>) {
        // Register bases: SIMD4 units for vector ops, Float units for
        // scalar ops. (reg * warpSize lanes; reg * 128 floats.)
        let db = d * warpSize, ab = a * warpSize, bb = b * warpSize
        let df = d * 128, af = a * 128, bf = b * 128
        let span = n * 4
        switch opc {
        case 0x01: // MOV
            for l in 0..<n { vb[db + l] = vb[ab + l] }
        case 0x03: // LDUNIF — warp-uniform value; guarded read done once
            let base = a * 4
            let v = SIMD4<Float>(base + 0 < ubuf.count ? ubuf[base + 0] : 0,
                                 base + 1 < ubuf.count ? ubuf[base + 1] : 0,
                                 base + 2 < ubuf.count ? ubuf[base + 2] : 0,
                                 base + 3 < ubuf.count ? ubuf[base + 3] : 0)
            for l in 0..<n { vb[db + l] = v }
        case 0x04: // LDATTR
            for l in 0..<n {
                let v = ctx.attrProvider(l, a)
                vb[db + l] = SIMD4<Float>(v.0, v.1, v.2, v.3)
            }
        case 0x05: // OUT
            for l in 0..<n {
                let v = vb[ab + l]
                ctx.outWriter(l, d, v.x, v.y, v.z, v.w)
            }
        case 0x10: // ADD
            for l in 0..<n { vb[db + l] = vb[ab + l] + vb[bb + l] }
        case 0x11: // SUB
            for l in 0..<n { vb[db + l] = vb[ab + l] - vb[bb + l] }
        case 0x12: // MUL
            for l in 0..<n { vb[db + l] = vb[ab + l] * vb[bb + l] }
        case 0x13: // DIV
            for l in 0..<n { vb[db + l] = vb[ab + l] / vb[bb + l] }
        case 0x14: // MIN — Swift.min per component (v1 expression)
            for i in 0..<span { rp[df + i] = Swift.min(rp[af + i], rp[bf + i]) }
        case 0x15: // MAX — Swift.max per component (v1 expression)
            for i in 0..<span { rp[df + i] = Swift.max(rp[af + i], rp[bf + i]) }
        case 0x16: // MAD — rd = ra * rb + rd (two ops, no fusion, as v1)
            for l in 0..<n { vb[db + l] = vb[ab + l] * vb[bb + l] + vb[db + l] }
        case 0x17: // RCP
            let one = SIMD4<Float>(repeating: 1)
            for l in 0..<n { vb[db + l] = one / vb[ab + l] }
        case 0x18: // SQRT — v1: x <= 0 ? 0 : sqrtf(x)
            for i in 0..<span { let x = rp[af + i]; rp[df + i] = x <= 0 ? 0 : sqrtf(x) }
        case 0x19: // RSQ — v1: x <= 0 ? 0 : 1 / sqrtf(x)
            for i in 0..<span { let x = rp[af + i]; rp[df + i] = x <= 0 ? 0 : 1 / sqrtf(x) }
        case 0x1A: // SIN
            for i in 0..<span { rp[df + i] = sinf(rp[af + i]) }
        case 0x1B: // COS
            for i in 0..<span { rp[df + i] = cosf(rp[af + i]) }
        case 0x1C, 0x1D: // DOT3 / DOT4 — v1 accumulation order (s += per comp)
            let lim = opc == 0x1C ? 3 : 4
            for l in 0..<n {
                let ao = af + l * 4, bo = bf + l * 4
                var s: Float = 0
                for c in 0..<lim { s += rp[ao + c] * rp[bo + c] }
                vb[db + l] = SIMD4<Float>(repeating: s)
            }
        case 0x20: // CMPLT — (ra < rb) ? 1 : 0 per component
            for i in 0..<span { rp[df + i] = rp[af + i] < rp[bf + i] ? 1 : 0 }
        case 0x21: // SEL — rd = (ra != 0) ? rb : rd per component
            for i in 0..<span { if rp[af + i] != 0 { rp[df + i] = rp[bf + i] } }
        case 0x22: // F2I — v1 expression verbatim
            for i in 0..<span {
                let v = rp[af + i]
                rp[df + i] = Float(bitPattern: UInt32(bitPattern: Int32(v.isFinite ? v : 0)))
            }
        case 0x23: // I2F — v1 expression verbatim
            for i in 0..<span { rp[df + i] = Float(Int32(bitPattern: rp[af + i].bitPattern)) }
        case 0x30: // IADD
            for i in 0..<span { up[df + i] = up[af + i] &+ up[bf + i] }
        case 0x31: // ISUB
            for i in 0..<span { up[df + i] = up[af + i] &- up[bf + i] }
        case 0x32: // IMUL
            for i in 0..<span { up[df + i] = up[af + i] &* up[bf + i] }
        case 0x33: // IAND
            for i in 0..<span { up[df + i] = up[af + i] & up[bf + i] }
        case 0x34: // IOR
            for i in 0..<span { up[df + i] = up[af + i] | up[bf + i] }
        case 0x35: // IXOR
            for i in 0..<span { up[df + i] = up[af + i] ^ up[bf + i] }
        case 0x36: // ISHL — shift counts masked to 31, as v1
            for i in 0..<span {
                let x = Int32(bitPattern: up[af + i]), y = Int32(bitPattern: up[bf + i])
                up[df + i] = UInt32(bitPattern: x << (y & 31))
            }
        case 0x37: // ISHR — arithmetic shift, counts masked to 31, as v1
            for i in 0..<span {
                let x = Int32(bitPattern: up[af + i]), y = Int32(bitPattern: up[bf + i])
                up[df + i] = UInt32(bitPattern: x >> (y & 31))
            }
        case 0x40: // LDTEX — texture(unit b).sample(ra.xy)
            for l in 0..<n {
                let v = ctx.texSampler(b, rp[af + l * 4], rp[af + l * 4 + 1])
                vb[db + l] = SIMD4<Float>(v.0, v.1, v.2, v.3)
            }
        case 0x41: // LDBUF — rd.x bits = u32 at byte addr from ra.x bits
            for l in 0..<n {
                let addr = Int(Int32(bitPattern: rp[af + l * 4].bitPattern))
                vb[db + l] = SIMD4<Float>(Float(bitPattern: ctx.bufLoad(addr)), 0, 0, 0)
            }
        case 0x42: // STBUF — u32 at byte addr from ra.x bits = rb.x bits
            for l in 0..<n {
                let addr = Int(Int32(bitPattern: rp[af + l * 4].bitPattern))
                ctx.bufStore(addr, rp[bf + l * 4].bitPattern)
            }
        default:
            break // NOP (0x00) and every unknown opcode, exactly as v1
        }
    }
}
