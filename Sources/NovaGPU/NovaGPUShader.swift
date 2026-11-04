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

// MARK: - Executor

enum ShaderExecutor {
    static let warpSize = 32
    static let regCount = 16

    /// Runs one warp of `program` with `lanes` active lanes (1...32).
    /// Returns the number of lane-instructions executed (issue count x
    /// active lanes at each issue) — the number the device reports as
    /// compute-unit work.
    static func runWarp(program: [UInt32], lanes: Int, ctx: WarpContext) -> Int {
        let n = max(0, min(warpSize, lanes))
        if n == 0 || program.isEmpty { return 0 }
        // Register file: [reg][lane][comp] flattened as (reg*32 + lane)*4 + comp.
        var regs = [Float](repeating: 0, count: regCount * warpSize * 4)
        @inline(__always) func idx(_ r: Int, _ lane: Int, _ c: Int) -> Int { (r * warpSize + lane) * 4 + c }
        var laneOps = 0
        var pc = 0
        var steps = 0
        let maxSteps = 100_000   // runaway-shader guard; a fault, not a hang
        while pc >= 0 && pc < program.count && steps < maxSteps {
            steps += 1
            let w = program[pc]
            let op = GPOpcode(rawValue: w & 0xFF) ?? .nop
            let d = Int((w >> 8) & 0xFF)
            let a = Int((w >> 16) & 0xFF)
            let b = Int((w >> 24) & 0xFF)
            var nextPc = pc + 1
            switch op {
            case .end:
                return laneOps
            case .bra:
                let rel = Int(Int16(bitPattern: UInt16(a | (b << 8))))
                nextPc = pc + 1 + rel
                laneOps += n
            case .brz:
                // Condition register is in the dst byte (builder encoding).
                laneOps += n
                if regs[idx(d, 0, 0)] == 0 {
                    let rel = Int(Int16(bitPattern: UInt16(a | (b << 8))))
                    nextPc = pc + 1 + rel
                }
            case .movi:
                laneOps += n
                let imm: UInt32 = (pc + 1 < program.count) ? program[pc + 1] : 0
                let v = Float(bitPattern: imm)
                for lane in 0..<n {
                    regs[idx(d, lane, 0)] = v
                    regs[idx(d, lane, 1)] = v
                    regs[idx(d, lane, 2)] = v
                    regs[idx(d, lane, 3)] = v
                }
                nextPc = pc + 2
            default:
                laneOps += n
                execute(op, d: d, a: a, b: b, regs: &regs, lanes: n, ctx: ctx, idx: idx)
            }
            pc = nextPc
        }
        return laneOps
    }

    private static func execute(_ op: GPOpcode, d: Int, a: Int, b: Int,
                                regs: inout [Float], lanes n: Int,
                                ctx: WarpContext,
                                idx: (Int, Int, Int) -> Int) {
        func comp(_ op: GPOpcode, _ f: (Float, Float) -> Float) {
            for lane in 0..<n { for c in 0..<4 {
                regs[idx(d, lane, c)] = f(regs[idx(a, lane, c)], regs[idx(b, lane, c)])
            } }
        }
        func comp1(_ op: GPOpcode, _ f: (Float) -> Float) {
            for lane in 0..<n { for c in 0..<4 {
                regs[idx(d, lane, c)] = f(regs[idx(a, lane, c)])
            } }
        }
        switch op {
        case .nop, .end, .bra, .brz:
            break
        case .mov:
            for lane in 0..<n { for c in 0..<4 { regs[idx(d, lane, c)] = regs[idx(a, lane, c)] } }
        case .movi:
            break // handled in runWarp (needs the following word)
        case .ldunif:
            let base = a * 4
            for lane in 0..<n { for c in 0..<4 {
                regs[idx(d, lane, c)] = base + c < ctx.uniforms.count ? ctx.uniforms[base + c] : 0
            } }
        case .ldattr:
            for lane in 0..<n {
                let v = ctx.attrProvider(lane, a)
                regs[idx(d, lane, 0)] = v.0; regs[idx(d, lane, 1)] = v.1
                regs[idx(d, lane, 2)] = v.2; regs[idx(d, lane, 3)] = v.3
            }
        case .out:
            for lane in 0..<n {
                ctx.outWriter(lane, d, regs[idx(a, lane, 0)], regs[idx(a, lane, 1)],
                              regs[idx(a, lane, 2)], regs[idx(a, lane, 3)])
            }
        case .add: comp(op) { $0 + $1 }
        case .sub: comp(op) { $0 - $1 }
        case .mul: comp(op) { $0 * $1 }
        case .div: comp(op) { $0 / $1 }
        case .min: comp(op) { Swift.min($0, $1) }
        case .max: comp(op) { Swift.max($0, $1) }
        case .mad:
            for lane in 0..<n { for c in 0..<4 {
                regs[idx(d, lane, c)] = regs[idx(a, lane, c)] * regs[idx(b, lane, c)] + regs[idx(d, lane, c)]
            } }
        case .rcp:  comp1(op) { 1 / $0 }
        case .sqrt: comp1(op) { $0 <= 0 ? 0 : sqrtf($0) }
        case .rsq:  comp1(op) { $0 <= 0 ? 0 : 1 / sqrtf($0) }
        case .sin:  comp1(op) { sinf($0) }
        case .cos:  comp1(op) { cosf($0) }
        case .dot3, .dot4:
            let lim = op == .dot3 ? 3 : 4
            for lane in 0..<n {
                var s: Float = 0
                for c in 0..<lim { s += regs[idx(a, lane, c)] * regs[idx(b, lane, c)] }
                for c in 0..<4 { regs[idx(d, lane, c)] = s }
            }
        case .cmplt: comp(op) { $0 < $1 ? 1 : 0 }
        case .sel:
            for lane in 0..<n { for c in 0..<4 {
                if regs[idx(a, lane, c)] != 0 { regs[idx(d, lane, c)] = regs[idx(b, lane, c)] }
            } }
        case .f2i:
            comp1(op) { v in Float(bitPattern: UInt32(bitPattern: Int32(v.isFinite ? v : 0))) }
        case .i2f:
            comp1(op) { v in Float(Int32(bitPattern: v.bitPattern)) }
        case .iadd: icomp(regs: &regs, d: d, a: a, b: b, n: n, idx: idx) { $0 &+ $1 }
        case .isub: icomp(regs: &regs, d: d, a: a, b: b, n: n, idx: idx) { $0 &- $1 }
        case .imul: icomp(regs: &regs, d: d, a: a, b: b, n: n, idx: idx) { $0 &* $1 }
        case .iand: icomp(regs: &regs, d: d, a: a, b: b, n: n, idx: idx) { $0 & $1 }
        case .ior:  icomp(regs: &regs, d: d, a: a, b: b, n: n, idx: idx) { $0 | $1 }
        case .ixor: icomp(regs: &regs, d: d, a: a, b: b, n: n, idx: idx) { $0 ^ $1 }
        case .ishl: icomp(regs: &regs, d: d, a: a, b: b, n: n, idx: idx) { $0 << ($1 & 31) }
        case .ishr: icomp(regs: &regs, d: d, a: a, b: b, n: n, idx: idx) { $0 >> ($1 & 31) }
        case .ldtex:
            for lane in 0..<n {
                let v = ctx.texSampler(b, regs[idx(a, lane, 0)], regs[idx(a, lane, 1)])
                regs[idx(d, lane, 0)] = v.0; regs[idx(d, lane, 1)] = v.1
                regs[idx(d, lane, 2)] = v.2; regs[idx(d, lane, 3)] = v.3
            }
        case .ldbuf:
            for lane in 0..<n {
                let addr = Int(Int32(bitPattern: regs[idx(a, lane, 0)].bitPattern))
                regs[idx(d, lane, 0)] = Float(bitPattern: ctx.bufLoad(addr))
                regs[idx(d, lane, 1)] = 0; regs[idx(d, lane, 2)] = 0; regs[idx(d, lane, 3)] = 0
            }
        case .stbuf:
            for lane in 0..<n {
                let addr = Int(Int32(bitPattern: regs[idx(a, lane, 0)].bitPattern))
                ctx.bufStore(addr, regs[idx(b, lane, 0)].bitPattern)
            }
        }
    }

    private static func icomp(regs: inout [Float], d: Int, a: Int, b: Int, n: Int,
                              idx: (Int, Int, Int) -> Int, _ f: (Int32, Int32) -> Int32) {
        for lane in 0..<n { for c in 0..<4 {
            let x = Int32(bitPattern: regs[idx(a, lane, c)].bitPattern)
            let y = Int32(bitPattern: regs[idx(b, lane, c)].bitPattern)
            regs[idx(d, lane, c)] = Float(bitPattern: UInt32(bitPattern: f(x, y)))
        } }
    }
}
