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
    // v4 vector ops (appended; existing opcodes never renumbered).
    // These are the shader-side face of the vector unit: world-space
    // FP32 math, executed as wide SIMD ops in the C core with the
    // same accumulation order as DOT3/DOT4.
    case vcross = 0x1E   // rd.xyz = cross(ra.xyz, rb.xyz), rd.w = 0
    case vnorm  = 0x1F   // rd.xyz = normalize(ra.xyz) (RSQ semantics), rd.w = ra.w
    case cmplt  = 0x20   // rd = (ra < rb) ? 1 : 0 per component
    case sel    = 0x21   // rd = (ra != 0) ? rb : rd per component (predication)
    case f2i    = 0x22   // rd bits = Int32(ra) per component (truncation)
    case i2f    = 0x23   // rd = Float(Int32(bits of ra)) per component
    case vmat4  = 0x24   // rd = M * ra, M = uniform rows c[b..b+3] (row-major);
                         // component j = dot4(ra, c[b+j]); out-of-range rows read 0
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
    mutating func vcross(_ d: Int, _ a: Int, _ b: Int) { emit(.vcross, d, a, b) }
    mutating func vnorm(_ d: Int, _ a: Int)          { emit(.vnorm, d, a) }
    mutating func vmat4(_ d: Int, _ a: Int, rows: Int) { emit(.vmat4, d, a, rows) }
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
    /// Stage fast path (speed round 3, generalized in v4): when set,
    /// LDATTR reads straight from this buffer instead of calling
    /// attrProvider per lane — layout [lane][attrStride] floats,
    /// slot s component c at lane*attrStride + s*4 + c. Fragment:
    /// attrStride 32 (varyings, slots 0..7). Vertex (v4 grab unit):
    /// attrStride 16 (attributes, slots 0..3). Bit-identical either
    /// way: the buffer holds exactly what attrProvider returned.
    var attrPtr: UnsafePointer<Float>? = nil
    var attrStride: Int = 32
    /// Stage fast path: when set, OUT writes straight to this buffer
    /// instead of calling outWriter per lane. Fragment layout
    /// [lane][4]: only slot 0 is recorded, exactly as the fragment
    /// outWriter closure recorded it. Vertex (v4): outFull = true
    /// switches to layout [lane][32] and records every slot 0..7,
    /// exactly as the vertex outWriter closure recorded them.
    var outPtr: UnsafeMutablePointer<Float>? = nil
    var outFull: Bool = false
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

// MARK: - C hot core bindings (speed round 3)
//
// The component-wise FP32 vector ops live in NovaGPUFast.c. The C
// functions perform exactly the IEEE-754 operations the Swift vector
// code performed (one rounding per op, no FMA fusion — see
// NovaGPUFast.h), so results are bit-identical; the framebuffer
// checksums police this on every CI run. Every build that compiles
// these Swift sources also compiles and links NovaGPUFast.c.

@_silgen_name("ngp_add_f32")
private func ngp_add_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_sub_f32")
private func ngp_sub_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_mul_f32")
private func ngp_mul_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_div_f32")
private func ngp_div_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_mad_f32")
private func ngp_mad_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_rcp_f32")
private func ngp_rcp_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_mov_f32")
private func ngp_mov_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_min_f32")
func ngp_min_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_max_f32")
func ngp_max_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_sqrt_f32")
func ngp_sqrt_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_rsq_f32")
func ngp_rsq_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_cmplt_f32")
func ngp_cmplt_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_sel_f32")
func ngp_sel_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_dot3_f32")
func ngp_dot3_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ lanes: Int32)
@_silgen_name("ngp_dot4_f32")
func ngp_dot4_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ lanes: Int32)
@_silgen_name("ngp_cross_f32")
func ngp_cross_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ lanes: Int32)
@_silgen_name("ngp_vnorm_f32")
func ngp_vnorm_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ lanes: Int32)
@_silgen_name("ngp_mat4_f32")
func ngp_mat4_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<Float>, _ u: UnsafePointer<Float>, _ lanes: Int32)
@_silgen_name("ngp_f2i_u32")
func ngp_f2i_u32(_ d: UnsafeMutablePointer<UInt32>, _ a: UnsafePointer<Float>, _ n: Int32)
@_silgen_name("ngp_i2f_f32")
func ngp_i2f_f32(_ d: UnsafeMutablePointer<Float>, _ a: UnsafePointer<UInt32>, _ n: Int32)
@_silgen_name("ngp_iadd_u32")
func ngp_iadd_u32(_ d: UnsafeMutablePointer<UInt32>, _ a: UnsafePointer<UInt32>, _ b: UnsafePointer<UInt32>, _ n: Int32)
@_silgen_name("ngp_isub_u32")
func ngp_isub_u32(_ d: UnsafeMutablePointer<UInt32>, _ a: UnsafePointer<UInt32>, _ b: UnsafePointer<UInt32>, _ n: Int32)
@_silgen_name("ngp_imul_u32")
func ngp_imul_u32(_ d: UnsafeMutablePointer<UInt32>, _ a: UnsafePointer<UInt32>, _ b: UnsafePointer<UInt32>, _ n: Int32)
@_silgen_name("ngp_iand_u32")
func ngp_iand_u32(_ d: UnsafeMutablePointer<UInt32>, _ a: UnsafePointer<UInt32>, _ b: UnsafePointer<UInt32>, _ n: Int32)
@_silgen_name("ngp_ior_u32")
func ngp_ior_u32(_ d: UnsafeMutablePointer<UInt32>, _ a: UnsafePointer<UInt32>, _ b: UnsafePointer<UInt32>, _ n: Int32)
@_silgen_name("ngp_ixor_u32")
func ngp_ixor_u32(_ d: UnsafeMutablePointer<UInt32>, _ a: UnsafePointer<UInt32>, _ b: UnsafePointer<UInt32>, _ n: Int32)
@_silgen_name("ngp_ishl_u32")
func ngp_ishl_u32(_ d: UnsafeMutablePointer<UInt32>, _ a: UnsafePointer<UInt32>, _ b: UnsafePointer<UInt32>, _ n: Int32)
@_silgen_name("ngp_ishr_u32")
private func ngp_ishr_u32(_ d: UnsafeMutablePointer<UInt32>, _ a: UnsafePointer<UInt32>, _ b: UnsafePointer<UInt32>, _ n: Int32)

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
        // Speed round 3: the decoded arrays are read through raw
        // pointers for the whole warp — no bounds-checked array
        // subscripts in the issue loop. Same values, same order.
        return decoded.op.withUnsafeBufferPointer { opB in
        decoded.d.withUnsafeBufferPointer { dB in
        decoded.a.withUnsafeBufferPointer { aB in
        decoded.b.withUnsafeBufferPointer { bB in
        decoded.imm.withUnsafeBufferPointer { immB in
        ctx.uniforms.withUnsafeBufferPointer { ubuf in
            withUnsafeTemporaryAllocation(of: SIMD4<Float>.self,
                                          capacity: regCount * warpSize) { vbuf in
                vbuf.initialize(repeating: SIMD4<Float>())
                let vb = vbuf.baseAddress!
                let rp = UnsafeMutableRawPointer(vb).assumingMemoryBound(to: Float.self)
                let up = UnsafeMutableRawPointer(vb).assumingMemoryBound(to: UInt32.self)
                return runLoop(opP: opB.baseAddress!, dP: dB.baseAddress!,
                               aP: aB.baseAddress!, bP: bB.baseAddress!,
                               immP: immB.baseAddress!, wc: decoded.count,
                               n: n, ctx: ctx, ubuf: ubuf, vb: vb, rp: rp, up: up)
            }
        }}}}}}
    }

    private static func runLoop(opP: UnsafePointer<UInt32>, dP: UnsafePointer<UInt32>,
                                aP: UnsafePointer<UInt32>, bP: UnsafePointer<UInt32>,
                                immP: UnsafePointer<UInt32>, wc: Int,
                                n: Int, ctx: WarpContext,
                                ubuf: UnsafeBufferPointer<Float>,
                                vb: UnsafeMutablePointer<SIMD4<Float>>,
                                rp: UnsafeMutablePointer<Float>,
                                up: UnsafeMutablePointer<UInt32>) -> Int {
        var laneOps = 0
        var pc = 0
        var steps = 0
        let maxSteps = 100_000   // runaway-shader guard; a fault, not a hang
        while pc >= 0 && pc < wc && steps < maxSteps {
            steps += 1
            let opc = opP[pc]
            let d = Int(dP[pc])
            let a = Int(aP[pc])
            let b = Int(bP[pc])
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
                let v = SIMD4<Float>(repeating: Float(bitPattern: immP[pc]))
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
        let db = d * warpSize, ab = a * warpSize
        let df = d * 128, af = a * 128, bf = b * 128
        let span = n * 4
        switch opc {
        case 0x01: // MOV — C core (component-wise copy, bit-exact)
            ngp_mov_f32(rp + df, rp + af, Int32(span))
        case 0x03: // LDUNIF — warp-uniform value; guarded read done once
            let base = a * 4
            let v = SIMD4<Float>(base + 0 < ubuf.count ? ubuf[base + 0] : 0,
                                 base + 1 < ubuf.count ? ubuf[base + 1] : 0,
                                 base + 2 < ubuf.count ? ubuf[base + 2] : 0,
                                 base + 3 < ubuf.count ? ubuf[base + 3] : 0)
            for l in 0..<n { vb[db + l] = v }
        case 0x04: // LDATTR — fast path reads the stage buffer
            // directly; same values the closure returned. Slots past
            // the buffer's slot count read as zero, as the vertex
            // closure returned zero for them and the fragment closure
            // was never asked (no OUT slot >= 8 feeds it).
            if let ap = ctx.attrPtr {
                let stride = ctx.attrStride
                if a < stride / 4 {
                    for l in 0..<n {
                        let o = l * stride + a * 4
                        vb[db + l] = SIMD4<Float>(ap[o], ap[o + 1], ap[o + 2], ap[o + 3])
                    }
                } else {
                    for l in 0..<n { vb[db + l] = SIMD4<Float>() }
                }
            } else {
                for l in 0..<n {
                    let v = ctx.attrProvider(l, a)
                    vb[db + l] = SIMD4<Float>(v.0, v.1, v.2, v.3)
                }
            }
        case 0x05: // OUT — fast path writes the stage buffer directly.
            if let op = ctx.outPtr {
                if ctx.outFull {
                    // Vertex (v4): [lane][32], every slot recorded.
                    if d < 8 {
                        for l in 0..<n {
                            let v = vb[ab + l]
                            let o = l * 32 + d * 4
                            op[o] = v.x; op[o + 1] = v.y; op[o + 2] = v.z; op[o + 3] = v.w
                        }
                    }
                } else if d == 0 {
                    // Fragment: [lane][4], only slot 0, as before.
                    for l in 0..<n {
                        let v = vb[ab + l]
                        let o = l * 4
                        op[o] = v.x; op[o + 1] = v.y; op[o + 2] = v.z; op[o + 3] = v.w
                    }
                }
            } else {
                for l in 0..<n {
                    let v = vb[ab + l]
                    ctx.outWriter(l, d, v.x, v.y, v.z, v.w)
                }
            }
        case 0x10: // ADD — C core (IEEE per component, bit-exact)
            ngp_add_f32(rp + df, rp + af, rp + bf, Int32(span))
        case 0x11: // SUB — C core
            ngp_sub_f32(rp + df, rp + af, rp + bf, Int32(span))
        case 0x12: // MUL — C core
            ngp_mul_f32(rp + df, rp + af, rp + bf, Int32(span))
        case 0x13: // DIV — C core
            ngp_div_f32(rp + df, rp + af, rp + bf, Int32(span))
        case 0x14: // MIN — C core (Comparable comparison form, as v1)
            ngp_min_f32(rp + df, rp + af, rp + bf, Int32(span))
        case 0x15: // MAX — C core (Comparable comparison form, as v1)
            ngp_max_f32(rp + df, rp + af, rp + bf, Int32(span))
        case 0x16: // MAD — rd = ra * rb + rd (two roundings, no fusion, as v1) — C core
            ngp_mad_f32(rp + df, rp + af, rp + bf, Int32(span))
        case 0x17: // RCP — C core (1 / x per component)
            ngp_rcp_f32(rp + df, rp + af, Int32(span))
        case 0x18: // SQRT — C core (v1: x <= 0 ? 0 : sqrtf(x))
            ngp_sqrt_f32(rp + df, rp + af, Int32(span))
        case 0x19: // RSQ — C core (v1: x <= 0 ? 0 : 1 / sqrtf(x))
            ngp_rsq_f32(rp + df, rp + af, Int32(span))
        case 0x1A: // SIN — libm per component (stays scalar: no vector libm without fast-math)
            for i in 0..<span { rp[df + i] = sinf(rp[af + i]) }
        case 0x1B: // COS — libm per component
            for i in 0..<span { rp[df + i] = cosf(rp[af + i]) }
        case 0x1C: // DOT3 — C core (v1 accumulation order, NEON planes)
            ngp_dot3_f32(rp + df, rp + af, rp + bf, Int32(n))
        case 0x1D: // DOT4 — C core (v1 accumulation order, NEON planes)
            ngp_dot4_f32(rp + df, rp + af, rp + bf, Int32(n))
        case 0x1E: // VCROSS (v4) — C core; rd.xyz = cross, rd.w = 0
            ngp_cross_f32(rp + df, rp + af, rp + bf, Int32(n))
        case 0x1F: // VNORM (v4) — C core; RSQ semantics, w pass-through
            ngp_vnorm_f32(rp + df, rp + af, Int32(n))
        case 0x20: // CMPLT — C core ((ra < rb) ? 1 : 0 per component)
            ngp_cmplt_f32(rp + df, rp + af, rp + bf, Int32(span))
        case 0x21: // SEL — C core (rd = (ra != 0) ? rb : rd per component)
            ngp_sel_f32(rp + df, rp + af, rp + bf, Int32(span))
        case 0x22: // F2I — C core (v1 expression: Int32(v.isFinite ? v : 0) bits)
            ngp_f2i_u32(up + df, rp + af, Int32(span))
        case 0x23: // I2F — C core (Float(Int32(bits of ra)) per component)
            ngp_i2f_f32(rp + df, up + af, Int32(span))
        case 0x24: // VMAT4 (v4) — rd = M * ra, M = uniform rows c[b..b+3];
            // rows past the uniform file read as zero (LDUNIF's rule).
            withUnsafeTemporaryAllocation(of: Float.self, capacity: 16) { rows in
                rows.initialize(repeating: 0)
                let base = b * 4
                let avail = max(0, min(16, ubuf.count - base))
                for i in 0..<avail { rows[i] = ubuf[base + i] }
                ngp_mat4_f32(rp + df, rp + af, rows.baseAddress!, Int32(n))
            }
        case 0x30: // IADD — C core
            ngp_iadd_u32(up + df, up + af, up + bf, Int32(span))
        case 0x31: // ISUB — C core
            ngp_isub_u32(up + df, up + af, up + bf, Int32(span))
        case 0x32: // IMUL — C core
            ngp_imul_u32(up + df, up + af, up + bf, Int32(span))
        case 0x33: // IAND — C core
            ngp_iand_u32(up + df, up + af, up + bf, Int32(span))
        case 0x34: // IOR — C core
            ngp_ior_u32(up + df, up + af, up + bf, Int32(span))
        case 0x35: // IXOR — C core
            ngp_ixor_u32(up + df, up + af, up + bf, Int32(span))
        case 0x36: // ISHL — C core (counts masked to 31, as v1)
            ngp_ishl_u32(up + df, up + af, up + bf, Int32(span))
        case 0x37: // ISHR — C core (arithmetic, counts masked to 31, as v1)
            ngp_ishr_u32(up + df, up + af, up + bf, Int32(span))
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
