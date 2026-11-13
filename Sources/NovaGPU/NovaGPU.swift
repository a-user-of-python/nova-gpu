// NovaGPU.swift — the NovaGPU device core.
//
// A from-scratch software model of a GPU, in the same spirit as NovaCPU:
// a documented virtual architecture, executed on the host. The mapping
// to the host is honest and stated plainly:
//
//   * The GPU has 1536 compute units (CUs): 48 clusters x 32 lanes.
//     A "lane" is one SIMT thread of a warp. Warps are issued to
//     clusters round-robin and clusters run concurrently on host
//     threads (DispatchQueue.concurrentPerform). No Metal, no host GPU:
//     every shader instruction is executed by this code.
//   * VRAM is a 2 GiB virtual address space, sparsely committed in
//     64 KiB blocks from the app's host memory: an allocation reserves
//     address space; host pages are only committed when written. The
//     device can therefore address the full 2 GiB specified while
//     only paying host RAM for what a scene actually touches.
//   * The framebuffer is RGBA8888: 16,777,216 colours, computed from
//     FP32 per-channel shading. (Requirement was >= 1024 colours; this
//     exceeds it by four orders of magnitude.)
//
// Standard vertex format "NVF1" (48 bytes, all FP32):
//   attr0 position xyz (w read as 1), attr1 normal xyz,
//   attr2 uv, attr3 colour rgba.
//
// Pipeline: vertex fetch -> vertex shader (warps over CUs) -> clip
// (Sutherland-Hodgman, 6 planes) -> viewport -> rasterise (row chunks
// over clusters, fragments in warps of 32) -> fragment shader -> depth
// test (early-z; shaders cannot discard) -> RGBA8888 framebuffer in
// VRAM.
//
// Speed round (v2) — execution only, architecture and results
// unchanged. Measured profile of v1 (share of frame CPU time): the
// rasteriser rescanned every triangle against every one of 48 row
// bands (~1.4M calls per frame); every accepted fragment allocated a
// fresh 8x4 nested varying array and interpolated all 8 slots whether
// or not the fragment program could read them; every pixel went back
// through the locked VRAM write path; vertex/index data was fetched
// one float at a time through that same locked path. v2: triangles
// are binned once into the row chunks they actually touch and chunks
// run over the clusters; warp varyings live in one flat scratch
// buffer per chunk and only the slots the fragment program reads
// (DecodedShader.attrSlotMask) are interpolated; framebuffer pixels
// are written through a block-pointer table resolved once per draw
// (uncommitted blocks fall back to the locked write, so semantics are
// unchanged); vertex and index buffers are snapshotted with one
// locked read each. Every FP expression in the geometry, coverage,
// interpolation and shading paths is the v1 expression verbatim, and
// per-pixel processing order (triangle order, row order) is
// preserved — the output is byte-identical, proven by checksum.

import Foundation

enum GPUError: Error, CustomStringConvertible {
    case outOfMemory
    case badHandle(UInt64)
    case outOfBounds
    case badProgram
    case badState(String)
    var description: String {
        switch self {
        case .outOfMemory: return "VRAM out of memory"
        case .badHandle(let h): return "bad VRAM handle 0x\(String(h, radix: 16))"
        case .outOfBounds: return "VRAM access out of bounds"
        case .badProgram: return "bad shader program"
        case .badState(let s): return "bad device state: \(s)"
        }
    }
}

/// Per-draw stage timings (v4 profiling). Wall time of each pipeline
/// section of the most recent draw, in milliseconds. The sections run
/// their internal parallelism to completion before the next begins,
/// so the section walls sum to (approximately) the draw's wall time
/// and show exactly where a frame goes. Filled by draw() always —
/// the cost is a handful of clock reads per draw.
struct DrawProfile {
    var snapshotMs: Double = 0   // VRAM -> host snapshot of vertex/index data
    var vertexMs: Double = 0     // vertex stage (warps over clusters)
    var clipMs: Double = 0       // clip + fan + screen transform (parallel ranges)
    var binMs: Double = 0        // triangle binning into row chunks
    var rasterMs: Double = 0     // rasterise + fragment shading + writeback
    var fragExecMs: Double = 0   // of rasterMs: batcher flush (executor + writeback);
                                 // only accumulated when NovaGPU.profileFlushTiming is set
    var totalMs: Double = 0
}

/// Per-call pass timings for plotVectors (v5 profiling). Pass 1 is
/// the vector work (grab + VPU positioning + DDA + depth merge);
/// pass 2 shades the accepted pixels through the fragment path.
/// Filled by plotVectors always — two extra clock reads per call.
struct PlotProfile {
    var pass1Ms: Double = 0
    var pass2Ms: Double = 0
    var totalMs: Double = 0
}

/// Accumulators for the v5 in-draw VPU test (see NovaGPU.armVPUTest
/// and the test pass inside draw()). A mismatch is a triangle
/// corner whose VPU clip position differs in ANY component's bit
/// pattern from the shader stage's clip position for the same
/// vertex. maxAbsError is the largest finite |VPU - shader|
/// component difference seen — informational only; the verdict is
/// the mismatch count, so even a +0/-0 bit difference (abs error
/// exactly 0) still counts as a mismatch and is never rounded
/// into a pass.
struct VPUTestResult {
    var trianglesTested = 0
    var cornersCompared = 0
    var mismatches = 0
    var maxAbsError: Float = 0
    var passMs: Double = 0
    var clustersEngaged = 0
    /// One VPU workload is issued per triangle tested.
    var vpusEngaged: Int { trianglesTested }
    /// VPU positioning throughput over the accumulated window.
    var trianglesPerSecond: Double {
        passMs > 0 ? Double(trianglesTested) / (passMs / 1000) : 0
    }
    mutating func record(triangles: Int, mismatches: Int, maxErr: Float,
                         passMs: Double, clusters: Int) {
        trianglesTested += triangles
        cornersCompared += triangles * 3
        self.mismatches += mismatches
        if maxErr > maxAbsError { maxAbsError = maxErr }
        self.passMs += passMs
        clustersEngaged = clusters
    }
}

struct GPUStats {
    var laneOps: UInt64 = 0                 // total lane-instructions executed
    var warpsIssued: UInt64 = 0
    var vectorsPlotted: UInt64 = 0          // v4: plot-unit vectors drawn
    var lanesPerCluster: [UInt64] = [UInt64](repeating: 0, count: NovaGPU.clusterCount)
    var trianglesIn: UInt64 = 0
    var trianglesOut: UInt64 = 0           // after clipping
    var fragmentsShaded: UInt64 = 0
    var draws: UInt64 = 0
    var launches: UInt64 = 0
    var lastFrameMs: Double = 0
    mutating func reset() { self = GPUStats() }
    var activeComputeUnits: Int { lanesPerCluster.filter { $0 > 0 }.count * ShaderExecutor.warpSize }
}

/// Speed round 3 fragment batcher: accepted pixels accumulate
/// until 32 fill a warp, then shade. v2 shaded each row segment's
/// partial warp immediately — warps break at every row and triangle
/// end, so the average fill was ≈19/32 and roughly 40% of fragment
/// executor issues ran on lanes that produce no pixel — and it
/// rebuilt the warp context (six closures) for every warp. The
/// buffer is FIFO: pixels shade in exactly the order they were
/// accepted, depth is still resolved at acceptance time in scan
/// order, and every pixel's shading inputs are bit-identical to
/// v2's. Only the issue granularity moved, so stats.laneOps counts
/// fewer padded issues than v2 while fragmentsShaded is unchanged.
/// The context is built once per chunk, not once per warp.
private final class FragBatcher {
    let px: UnsafeMutablePointer<Int>
    let vars: UnsafeMutablePointer<Float>
    let col: UnsafeMutablePointer<Float>
    let decoded: DecodedShader
    var ctx: WarpContext
    let fbTable: UnsafeBufferPointer<UnsafeMutablePointer<UInt8>?>
    let fbFirstBlock: Int
    let fb: UInt64
    let fbFallback: (UInt64, [UInt8]) -> Void
    var count = 0
    var ops: UInt64 = 0
    var frags: UInt64 = 0
    var warps: UInt64 = 0
    var execMs: Double = 0   // flush wall time (profileFlushTiming only)

    init(px: UnsafeMutablePointer<Int>, vars: UnsafeMutablePointer<Float>,
         col: UnsafeMutablePointer<Float>, decoded: DecodedShader, unis: [Float],
         texSampler: @escaping (Int, Float, Float) -> (Float, Float, Float, Float),
         fbTable: UnsafeBufferPointer<UnsafeMutablePointer<UInt8>?>,
         fbFirstBlock: Int, fb: UInt64,
         fbFallback: @escaping (UInt64, [UInt8]) -> Void) {
        self.px = px; self.vars = vars; self.col = col
        self.decoded = decoded
        self.fbTable = fbTable; self.fbFirstBlock = fbFirstBlock; self.fb = fb
        self.fbFallback = fbFallback
        var c = WarpContext.dummy()
        c.uniforms = unis
        // The executor reads varyings and writes colours through
        // these buffers directly (identical contents to what the
        // closures below serve); the closures are the general path.
        c.attrPtr = UnsafePointer(vars)
        c.outPtr = col
        c.attrProvider = { lane, slot in
            // Varyings exist for slots 0..7. (v1 indexed a
            // per-fragment array and trapped on slot >= 8; no valid
            // program reads those — there is no OUT slot >= 8 into
            // the fragment stage — so zero is returned, not a trap.)
            guard slot >= 0 && slot < 8 else { return (0, 0, 0, 0) }
            let o = lane * 32 + slot * 4
            return (vars[o], vars[o + 1], vars[o + 2], vars[o + 3])
        }
        c.texSampler = texSampler
        c.outWriter = { lane, slot, r, g, b, a in
            if slot == 0 {
                let o = lane * 4
                col[o] = r; col[o + 1] = g
                col[o + 2] = b; col[o + 3] = a
            }
        }
        self.ctx = c
    }

    /// Shades the accumulated lanes and writes their pixels out, in
    /// acceptance order. Called when the warp fills and once at the
    /// end of each chunk.
    func flush() {
        if count == 0 { return }
        let fT0 = NovaGPU.profileFlushTiming ? NovaGPU.nowMs() : 0
        defer { if NovaGPU.profileFlushTiming { execMs += NovaGPU.nowMs() - fT0 } }
        // v1 default lane colour: (0, 0, 0, 1) when the program
        // never writes OUT 0.
        for i in 0..<count {
            col[i * 4] = 0; col[i * 4 + 1] = 0
            col[i * 4 + 2] = 0; col[i * 4 + 3] = 1
        }
        ops += UInt64(ShaderExecutor.runWarp(decoded: decoded, lanes: count, ctx: ctx))
        frags += UInt64(count)
        warps += 1
        // Lanes hold only the pixels that passed coverage + depth,
        // so they are not guaranteed consecutive: write each pixel at
        // its own framebuffer offset, through the block table
        // resolved for this draw.
        for i in 0..<count {
            let co = i * 4
            let rb = UInt8(min(max(col[co], 0), 1) * 255)
            let gb = UInt8(min(max(col[co + 1], 0), 1) * 255)
            let bb = UInt8(min(max(col[co + 2], 0), 1) * 255)
            let ab = UInt8(min(max(col[co + 3], 0), 1) * 255)
            let byteAddr = fb + UInt64(px[i] * 4)
            let tableIdx = Int(byteAddr >> 16) - fbFirstBlock
            if tableIdx >= 0 && tableIdx < fbTable.count, let p = fbTable[tableIdx] {
                let o = Int(byteAddr & 0xFFFF)
                p[o] = rb; p[o + 1] = gb; p[o + 2] = bb; p[o + 3] = ab
            } else {
                fbFallback(byteAddr, [rb, gb, bb, ab])
            }
        }
        count = 0
    }
}

final class NovaGPU {
    /// Architectural cluster count (v4: configurable). Set before
    /// creating a device; the default is the architecture's defining
    /// 48 clusters x 32 lanes = 1536 CUs. The count sizes the stats
    /// buckets warps are accounted to; host execution parallelism
    /// comes from the pipeline's work decomposition (warps, clip
    /// ranges, row chunks), so the bench — not intuition — decides
    /// whether a different count earns its place (see BENCH-v4.md).
    static var clusterCount = 48
    static let lanesPerCluster = 32
    static var computeUnits: Int { clusterCount * lanesPerCluster }   // 1536 at default
    /// v4 vector hardware sizing (Sean, 2026-11-13). MAX_SCREEN_
    /// VECTORS is the plot unit's per-frame on-screen vector
    /// budget: 65,536. The VPUs are grouped into MGVPU clusters —
    /// MGVPU_COUNT = 50 clusters of VPUS_PER_MGVPU = 131,072 VPUs
    /// each, an addressable pool of 6,553,600 VPUs. A VPU address
    /// is (mgvpuIndex, vpuIndex); the global id is
    /// mgvpuIndex * 131,072 + vpuIndex.
    ///
    /// The pool is VIRTUAL, and at 6.55M units it has to be: a VPU
    /// is a stateless 48-byte work slot (one triangle's 9 axis
    /// values + w), not a persistent object — slots materialize on
    /// demand on the stack of whichever host thread runs the
    /// workload, and scheduling distributes whole MGVPU clusters
    /// (contiguous sub-ranges of a command) across host threads.
    /// Idle cost is zero: nothing allocates per-VPU state. At
    /// absolute full occupancy the pool's position data would be
    /// 6,553,600 × 3 positions × 16 B ≈ 315 MB — but positions
    /// live in the caller's vertex buffers inside the 2 GiB
    /// sparse VRAM, committed block-by-block on touch, never
    /// eagerly. The triangle pipeline never touches the pool.
    static let maxScreenVectors = 65536      // MAX_SCREEN_VECTORS
    static let mgvpuCount = 50               // MGVPU_COUNT
    static let vpusPerMgvpu = 131072         // VPUS_PER_MGVPU
    static var vpuCount: Int { mgvpuCount * vpusPerMgvpu }   // 6,553,600
    /// Plot pass-1 switches from serial to MGVPU-cluster-parallel
    /// at this many vectors (internal; host tests lower it to
    /// force the parallel path on small scenes).
    static var plotParallelThreshold = 2048
    static let vramSize: UInt64 = 2 * 1024 * 1024 * 1024       // 2 GiB
    static let blockSize = 65536                               // sparse commit granularity

    // MARK: - Sparse VRAM

    private var blocks: [Int: UnsafeMutablePointer<UInt8>] = [:]
    private let vramLock = NSLock()
    /// Bytes of host memory actually committed (touched blocks).
    var vramCommitted: Int { blocks.count * Self.blockSize }

    private func blockIndex(_ addr: UInt64) -> Int { Int(addr / UInt64(Self.blockSize)) }

    /// Read bytes from VRAM; untouched blocks read as zero.
    func vramRead(at addr: UInt64, count: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: count)
        if count == 0 { return out }
        vramLock.lock(); defer { vramLock.unlock() }
        out.withUnsafeMutableBufferPointer { dst in
            guard let dp = dst.baseAddress else { return }
            var done = 0
            while done < count {
                let a = addr + UInt64(done)
                let bi = blockIndex(a)
                let off = Int(a % UInt64(Self.blockSize))
                let n = min(Self.blockSize - off, count - done)
                if let p = blocks[bi] { memcpy(dp + done, p + off, n) }
                done += n
            }
        }
        return out
    }

    /// Write bytes into VRAM, committing host blocks on first touch.
    func vramWrite(at addr: UInt64, bytes: [UInt8]) {
        if bytes.isEmpty { return }
        vramLock.lock(); defer { vramLock.unlock() }
        bytes.withUnsafeBufferPointer { src in
            guard let sp = src.baseAddress else { return }
            var done = 0
            while done < bytes.count {
                let a = addr + UInt64(done)
                let bi = blockIndex(a)
                let off = Int(a % UInt64(Self.blockSize))
                let n = min(Self.blockSize - off, bytes.count - done)
                let p: UnsafeMutablePointer<UInt8>
                if let existing = blocks[bi] { p = existing }
                else {
                    p = UnsafeMutablePointer<UInt8>.allocate(capacity: Self.blockSize)
                    p.initialize(repeating: 0, count: Self.blockSize)
                    blocks[bi] = p
                }
                memcpy(p + off, sp + done, n)
                done += n
            }
        }
    }

    /// Fills a VRAM range with a repeated 32-bit pattern (block-level
    /// memset; used by the bus FILL command and the memory self-test).
    func vramFill(at addr: UInt64, count: Int, pattern: UInt32) {
        if count <= 0 { return }
        var chunk = [UInt8](repeating: 0, count: min(count, Self.blockSize))
        for i in stride(from: 0, to: chunk.count - 3, by: 4) {
            chunk[i] = UInt8(pattern & 0xFF); chunk[i + 1] = UInt8((pattern >> 8) & 0xFF)
            chunk[i + 2] = UInt8((pattern >> 16) & 0xFF); chunk[i + 3] = UInt8((pattern >> 24) & 0xFF)
        }
        var done = 0
        while done < count {
            let n = min(chunk.count, count - done)
            vramWrite(at: addr + UInt64(done), bytes: Array(chunk[0..<n]))
            done += n
        }
    }

    func vramReadU32(at addr: UInt64) -> UInt32 {
        let b = vramRead(at: addr, count: 4)
        return UInt32(b[0]) | (UInt32(b[1]) << 8) | (UInt32(b[2]) << 16) | (UInt32(b[3]) << 24)
    }
    func vramWriteU32(at addr: UInt64, _ v: UInt32) {
        vramWrite(at: addr, bytes: [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF),
                                    UInt8((v >> 16) & 0xFF), UInt8((v >> 24) & 0xFF)])
    }
    func vramReadF32(at addr: UInt64) -> Float { Float(bitPattern: vramReadU32(at: addr)) }
    func vramWriteF32(at addr: UInt64, _ v: Float) { vramWriteU32(at: addr, v.bitPattern) }

    // MARK: - Block-pointer tables (speed round)
    //
    // The hot paths (framebuffer writes during rasterisation, buffer
    // access during compute launches) resolve the range's 64 KiB block
    // pointers ONCE under the lock, then run lock-free against raw
    // pointers. Blocks are never freed while a draw/launch that
    // resolved them is in flight (bus commands execute serially), so
    // the pointers stay valid for the operation's duration.

    /// Snapshot of the block pointers covering a VRAM range; nil
    /// entries are uncommitted blocks (reads as zero; writes must go
    /// through the locked vramWrite, which commits — callers do this).
    func blockPointers(addr: UInt64, size: UInt64) -> [UnsafeMutablePointer<UInt8>?] {
        let first = blockIndex(addr)
        let last = blockIndex(addr + max(size, 1) - 1)
        vramLock.lock(); defer { vramLock.unlock() }
        var out: [UnsafeMutablePointer<UInt8>?] = []
        out.reserveCapacity(last - first + 1)
        for bi in first...last { out.append(blocks[bi]) }
        return out
    }

    /// Like blockPointers, but commits every block in the range first.
    /// Used for compute-launch buffers (≤ 256 MiB), where the whole
    /// buffer is the kernel's working set; ranges above that keep the
    /// sparse behaviour via the snapshot + locked-fallback path.
    func committedBlockPointers(addr: UInt64, size: UInt64) -> [UnsafeMutablePointer<UInt8>] {
        let first = blockIndex(addr)
        let last = blockIndex(addr + max(size, 1) - 1)
        vramLock.lock(); defer { vramLock.unlock() }
        var out: [UnsafeMutablePointer<UInt8>] = []
        out.reserveCapacity(last - first + 1)
        for bi in first...last {
            if let p = blocks[bi] { out.append(p) }
            else {
                let p = UnsafeMutablePointer<UInt8>.allocate(capacity: Self.blockSize)
                p.initialize(repeating: 0, count: Self.blockSize)
                blocks[bi] = p
                out.append(p)
            }
        }
        return out
    }

    // MARK: - Allocator (first-fit free list over the 2 GiB space)

    private var freeList: [(start: UInt64, size: UInt64)] = [(0, NovaGPU.vramSize)]
    private var allocations: [UInt64: UInt64] = [:]   // handle -> size
    private(set) var vramReserved: UInt64 = 0         // sum of live allocations

    @discardableResult
    func alloc(size: UInt64, alignment: UInt64 = 256) throws -> UInt64 {
        let align = max(UInt64(1), alignment)
        let want = max(size, 1)
        for i in freeList.indices {
            let seg = freeList[i]
            let start = (seg.start + align - 1) & ~(align - 1)
            let pad = start - seg.start
            if seg.size >= pad + want {
                var rest = seg
                freeList.remove(at: i)
                if pad > 0 { freeList.append((seg.start, pad)) }
                let tailSize = seg.size - pad - want
                if tailSize > 0 { rest = (start + want, tailSize); freeList.append(rest) }
                freeList.sort { $0.start < $1.start }
                allocations[start] = want
                vramReserved += want
                return start
            }
        }
        throw GPUError.outOfMemory
    }

    func free(_ handle: UInt64) throws {
        guard let size = allocations.removeValue(forKey: handle) else { throw GPUError.badHandle(handle) }
        vramReserved -= size
        // Release host blocks that lie fully inside the freed range
        // and are not shared with any remaining live allocation (edge
        // blocks can be shared: allocations are 256-byte aligned,
        // blocks are 64 KiB).
        let end = handle + size
        let firstBlock = blockIndex(handle)
        let lastBlock = blockIndex(end - 1)
        if lastBlock > firstBlock {
            vramLock.lock()
            for bi in (firstBlock + 1)..<lastBlock {
                let bStart = UInt64(bi) * UInt64(Self.blockSize)
                let bEnd = bStart + UInt64(Self.blockSize)
                let shared = allocations.contains { kv in
                    kv.key < bEnd && kv.key + kv.value > bStart
                }
                if !shared, let p = blocks.removeValue(forKey: bi) {
                    p.deinitialize(count: Self.blockSize)
                    p.deallocate()
                }
            }
            vramLock.unlock()
        }
        freeList.append((handle, size))
        freeList.sort { $0.start < $1.start }
        // Coalesce neighbours.
        var merged: [(start: UInt64, size: UInt64)] = []
        for seg in freeList {
            if let last = merged.last, last.start + last.size == seg.start {
                merged[merged.count - 1].size += seg.size
            } else { merged.append(seg) }
        }
        freeList = merged
    }

    func allocationSize(_ handle: UInt64) -> UInt64? { allocations[handle] }

    func reset() {
        vramLock.lock()
        for (_, p) in blocks { p.deinitialize(count: Self.blockSize); p.deallocate() }
        blocks.removeAll()
        vramLock.unlock()
        freeList = [(0, Self.vramSize)]
        allocations.removeAll()
        vramReserved = 0
        stats.reset()
        vpuTestEnabled = false
        vpuTest = VPUTestResult()
        vpuTestLifetime = VPUTestResult()
        textures = [:]
    }

    // MARK: - Device state set over the bus

    var vertexProgram: [UInt32] = []
    var fragmentProgram: [UInt32] = []
    var computeProgram: [UInt32] = []
    var uniforms: [Float] = [Float](repeating: 0, count: 64)
    var vertexBuffer: UInt64 = 0
    var indexBuffer: UInt64 = 0
    var framebuffer: UInt64 = 0
    var fbWidth: Int = 0
    var fbHeight: Int = 0
    var textures: [Int: (handle: UInt64, w: Int, h: Int)] = [:]
    private(set) var stats = GPUStats()
    /// Stage timings of the most recent draw (v4 profiling).
    private(set) var lastDrawProfile = DrawProfile()
    /// Pass timings of the most recent plotVectors call (v5 profiling).
    private(set) var lastPlotProfile = PlotProfile()
    /// v5 in-scene VPU test state (armed over the bus, command 20).
    /// `vpuTest` accumulates over the current window — arming
    /// resets it, so the app arms at the top of each frame and
    /// reads per-frame numbers; `vpuTestLifetime` accumulates
    /// since the last device reset.
    private(set) var vpuTest = VPUTestResult()
    private(set) var vpuTestLifetime = VPUTestResult()
    private(set) var vpuTestEnabled = false
    /// Host-test fault injection for the VPU test pass: when
    /// nonzero, this value is added to corner 0's clip x inside
    /// the pass (before the compare AND the gather), proving the
    /// test detects a wrong VPU and that armed frames really are
    /// rendered from VPU positions. Never set by the app; the
    /// default 0 is the honest path.
    static var vpuTestFault: Float = 0

    /// Arms/disarms the in-draw VPU test. Arming resets the
    /// window accumulator.
    func armVPUTest(_ on: Bool) {
        vpuTestEnabled = on
        if on { vpuTest = VPUTestResult() }
    }
    static func nowMs() -> Double { ProcessInfo.processInfo.systemUptime * 1000 }
    /// When true, FragBatcher.flush() accumulates its wall time into
    /// DrawProfile.fragExecMs (two clock reads per warp flush — a
    /// profiling cost the bench pays deliberately; off in production).
    static var profileFlushTiming = false

    func setUniform(index: Int, _ v: (Float, Float, Float, Float)) {
        guard index >= 0 && index < 16 else { return }
        uniforms[index * 4 + 0] = v.0; uniforms[index * 4 + 1] = v.1
        uniforms[index * 4 + 2] = v.2; uniforms[index * 4 + 3] = v.3
    }

    // MARK: - Texture sampling (bilinear, clamp-to-edge, RGBA8888 source)

    func sampleTexture(unit: Int, u: Float, v: Float) -> (Float, Float, Float, Float) {
        guard let t = textures[unit], t.w > 0, t.h > 0 else { return (0, 0, 0, 0) }
        let x = (min(max(u, 0), 1) * Float(t.w) - 0.5)
        let y = (min(max(v, 0), 1) * Float(t.h) - 0.5)
        let x0 = Int(floorf(x)), y0 = Int(floorf(y))
        let fx = x - Float(x0), fy = y - Float(y0)
        func px(_ ix: Int, _ iy: Int) -> (Float, Float, Float, Float) {
            let cx = min(max(ix, 0), t.w - 1), cy = min(max(iy, 0), t.h - 1)
            let o = t.handle + UInt64((cy * t.w + cx) * 4)
            let b = vramRead(at: o, count: 4)
            return (Float(b[0]) / 255, Float(b[1]) / 255, Float(b[2]) / 255, Float(b[3]) / 255)
        }
        let c00 = px(x0, y0), c10 = px(x0 + 1, y0), c01 = px(x0, y0 + 1), c11 = px(x0 + 1, y0 + 1)
        func mix(_ a: Float, _ b: Float, _ f: Float) -> Float { a + (b - a) * f }
        return (mix(mix(c00.0, c10.0, fx), mix(c01.0, c11.0, fx), fy),
                mix(mix(c00.1, c10.1, fx), mix(c01.1, c11.1, fx), fy),
                mix(mix(c00.2, c10.2, fx), mix(c01.2, c11.2, fx), fy),
                mix(mix(c00.3, c10.3, fx), mix(c01.3, c11.3, fx), fy))
    }

    // MARK: - Compute launch

    /// Runs the compute program over `lanes` lanes (a 1D grid), addressing
    /// `buffer` by byte offset through LDBUF/STBUF.
    func launchCompute(lanes: Int, buffer: UInt64) throws {
        guard !computeProgram.isEmpty else { throw GPUError.badProgram }
        guard let bufSize = allocationSize(buffer) else { throw GPUError.badHandle(buffer) }
        let warpCount = (lanes + ShaderExecutor.warpSize - 1) / ShaderExecutor.warpSize
        let decoded = DecodedShader(computeProgram)
        let unis = uniforms
        let firstBlock = blockIndex(buffer)
        // Buffer access path: buffers up to 256 MiB are committed up
        // front and accessed through raw block pointers; larger (still
        // sparse) buffers use a pointer snapshot whose uncommitted
        // blocks fall back to the locked v1 accessors, so results are
        // identical either way.
        let fastTable: [UnsafeMutablePointer<UInt8>] =
            bufSize <= 256 * 1024 * 1024 ? committedBlockPointers(addr: buffer, size: bufSize) : []
        let snapTable: [UnsafeMutablePointer<UInt8>?] =
            fastTable.isEmpty ? blockPointers(addr: buffer, size: bufSize) : []
        let perWarpOps = UnsafeMutablePointer<UInt64>.allocate(capacity: max(warpCount, 1))
        perWarpOps.initialize(repeating: 0, count: max(warpCount, 1))
        defer { perWarpOps.deinitialize(count: max(warpCount, 1)); perWarpOps.deallocate() }

        func runWarps(_ body: @escaping (Int, Int) -> WarpContext) {
            DispatchQueue.concurrentPerform(iterations: warpCount) { warp in
                let base = warp * ShaderExecutor.warpSize
                let count = min(ShaderExecutor.warpSize, lanes - base)
                if count <= 0 { return }
                let ops = ShaderExecutor.runWarp(decoded: decoded, lanes: count,
                                                 ctx: body(warp, base))
                perWarpOps[warp] = UInt64(ops)
            }
        }

        if !fastTable.isEmpty {
            fastTable.withUnsafeBufferPointer { tb in
                runWarps { _, base in
                    var ctx = WarpContext.dummy()
                    ctx.uniforms = unis
                    ctx.attrProvider = { lane, slot in
                        slot == 0 ? (Float(base + lane), 0, 0, 0) : (0, 0, 0, 0)
                    }
                    ctx.bufLoad = { addr in
                        guard addr >= 0, UInt64(addr) + 4 <= bufSize else { return 0 }
                        var v: UInt32 = 0
                        var byteAddr = buffer + UInt64(addr)
                        for k in 0..<4 {
                            v |= UInt32(tb[Int(byteAddr >> 16) - firstBlock][Int(byteAddr & 0xFFFF)]) << UInt32(k * 8)
                            byteAddr += 1
                        }
                        return v
                    }
                    ctx.bufStore = { addr, value in
                        guard addr >= 0, UInt64(addr) + 4 <= bufSize else { return }
                        var byteAddr = buffer + UInt64(addr)
                        for k in 0..<4 {
                            tb[Int(byteAddr >> 16) - firstBlock][Int(byteAddr & 0xFFFF)] =
                                UInt8((value >> UInt32(k * 8)) & 0xFF)
                            byteAddr += 1
                        }
                    }
                    return ctx
                }
            }
        } else {
            snapTable.withUnsafeBufferPointer { tb in
                runWarps { _, base in
                    var ctx = WarpContext.dummy()
                    ctx.uniforms = unis
                    ctx.attrProvider = { lane, slot in
                        slot == 0 ? (Float(base + lane), 0, 0, 0) : (0, 0, 0, 0)
                    }
                    ctx.bufLoad = { [self] addr in
                        guard addr >= 0, UInt64(addr) + 4 <= bufSize else { return 0 }
                        let a = buffer + UInt64(addr)
                        if tb[Int(a >> 16) - firstBlock] != nil,
                           tb[Int((a + 3) >> 16) - firstBlock] != nil {
                            var v: UInt32 = 0
                            var byteAddr = a
                            for k in 0..<4 {
                                v |= UInt32(tb[Int(byteAddr >> 16) - firstBlock]![Int(byteAddr & 0xFFFF)]) << UInt32(k * 8)
                                byteAddr += 1
                            }
                            return v
                        }
                        return vramReadU32(at: a)
                    }
                    ctx.bufStore = { [self] addr, value in
                        guard addr >= 0, UInt64(addr) + 4 <= bufSize else { return }
                        let a = buffer + UInt64(addr)
                        if tb[Int(a >> 16) - firstBlock] != nil,
                           tb[Int((a + 3) >> 16) - firstBlock] != nil {
                            var byteAddr = a
                            for k in 0..<4 {
                                tb[Int(byteAddr >> 16) - firstBlock]![Int(byteAddr & 0xFFFF)] =
                                    UInt8((value >> UInt32(k * 8)) & 0xFF)
                                byteAddr += 1
                            }
                            return
                        }
                        vramWriteU32(at: a, value)
                    }
                    return ctx
                }
            }
        }
        stats.launches += 1
        for warp in 0..<warpCount {
            let v = perWarpOps[warp]
            stats.lanesPerCluster[warp % Self.clusterCount] += v
            stats.laneOps += v
        }
        stats.warpsIssued += UInt64(warpCount)
    }

    // MARK: - 3D pipeline

    /// Varyings storage (v4): 8 vec4 slots held INLINE in the
    /// vertex. v3 heap-allocated a 32-float Swift array per vertex
    /// (~15k allocations per draw on the app scene) and paid array
    /// refcount/COW traffic on every clip-stage copy; the inline
    /// block copies as plain bytes. Layout contract: slot s
    /// component c is float s*4 + c of the block, slots 0..7; the
    /// eight SIMD4 properties are contiguous (each 16 bytes,
    /// 16-aligned — no padding), guarded by the framebuffer
    /// checksum tests, which render through these pointers.
    /// (Slot 0 is never written — OUT 0 is the clip position.)
    struct VarBlock {
        var s0 = SIMD4<Float>()
        var s1 = SIMD4<Float>()
        var s2 = SIMD4<Float>()
        var s3 = SIMD4<Float>()
        var s4 = SIMD4<Float>()
        var s5 = SIMD4<Float>()
        var s6 = SIMD4<Float>()
        var s7 = SIMD4<Float>()
        func withFloats<R>(_ body: (UnsafePointer<Float>) -> R) -> R {
            withUnsafePointer(to: self) { p in
                body(UnsafeRawPointer(p).assumingMemoryBound(to: Float.self))
            }
        }
        mutating func withMutableFloats<R>(_ body: (UnsafeMutablePointer<Float>) -> R) -> R {
            withUnsafeMutablePointer(to: &self) { p in
                body(UnsafeMutableRawPointer(p).assumingMemoryBound(to: Float.self))
            }
        }
    }

    private struct ShadedVertex {
        var clip: (Float, Float, Float, Float) = (0, 0, 0, 1)
        var varyings = VarBlock()
    }

    /// A clipped, fan-triangulated triangle with its vertices' screen
    /// coordinates precomputed once (v1 recomputed them per band).
    private struct WorkTri {
        var v0: ShadedVertex, v1: ShadedVertex, v2: ShadedVertex
        var s0: (Float, Float, Float, Float)   // (x, y, z, invW)
        var s1: (Float, Float, Float, Float)
        var s2: (Float, Float, Float, Float)
    }

    /// Clears the framebuffer to an RGBA8888 colour + depth far.
    func clearFramebuffer(rgba: UInt32) throws {
        guard allocationSize(framebuffer) != nil, fbWidth > 0, fbHeight > 0 else { throw GPUError.badState("no framebuffer") }
        // Speed round 3 fast path: when every framebuffer block is
        // already committed (true from the second clear on), fill the
        // words straight through the block table — v2 issued one
        // locked vramWrite per row (480 lock round-trips per frame).
        // Identical bytes either way; the first clear still takes the
        // row path below, which is what commits the blocks.
        let total = fbWidth * fbHeight * 4
        let table = blockPointers(addr: framebuffer, size: UInt64(total))
        var allCommitted = !table.isEmpty
        for p in table where p == nil { allCommitted = false }
        if allCommitted {
            var done = 0
            var bi = 0
            while done < total {
                let addr = framebuffer + UInt64(done)
                let off = Int(addr % UInt64(Self.blockSize))
                let n = min(Self.blockSize - off, total - done)
                let dst = table[bi]! + off
                if addr % 4 == 0 {
                    let words = n / 4
                    dst.withMemoryRebound(to: UInt32.self, capacity: words) { wp in
                        for w in 0..<words { wp[w] = rgba }
                    }
                    let tail = n % 4
                    if tail > 0 {
                        let bytes: [UInt8] = [UInt8(rgba & 0xFF), UInt8((rgba >> 8) & 0xFF),
                                              UInt8((rgba >> 16) & 0xFF), UInt8((rgba >> 24) & 0xFF)]
                        for k in 0..<tail { dst[words * 4 + k] = bytes[k] }
                    }
                } else {
                    let bytes: [UInt8] = [UInt8(rgba & 0xFF), UInt8((rgba >> 8) & 0xFF),
                                          UInt8((rgba >> 16) & 0xFF), UInt8((rgba >> 24) & 0xFF)]
                    let phase = Int(addr % 4)
                    for k in 0..<n { dst[k] = bytes[(phase + k) % 4] }
                }
                done += n
                bi += 1
            }
            return
        }
        var rowBytes = [UInt8](repeating: 0, count: fbWidth * 4)
        for x in 0..<fbWidth {
            rowBytes[x * 4 + 0] = UInt8(rgba & 0xFF); rowBytes[x * 4 + 1] = UInt8((rgba >> 8) & 0xFF)
            rowBytes[x * 4 + 2] = UInt8((rgba >> 16) & 0xFF); rowBytes[x * 4 + 3] = UInt8((rgba >> 24) & 0xFF)
        }
        for y in 0..<fbHeight {
            vramWrite(at: framebuffer + UInt64(y * fbWidth * 4), bytes: rowBytes)
        }
    }

    /// Executes one indexed draw. Returns frame time in milliseconds.
    @discardableResult
    func draw(indexCount: Int) throws -> Double {
        let t0 = Date()
        guard !vertexProgram.isEmpty, !fragmentProgram.isEmpty else { throw GPUError.badProgram }
        guard let vbSize = allocationSize(vertexBuffer) else { throw GPUError.badHandle(vertexBuffer) }
        guard let ibSize = allocationSize(indexBuffer) else { throw GPUError.badHandle(indexBuffer) }
        guard let fbAllocSize = allocationSize(framebuffer), fbWidth > 0, fbHeight > 0 else { throw GPUError.badState("no framebuffer") }
        let vertexCount = Int(vbSize / 48)
        let idxCount = min(indexCount, Int(ibSize / 4))
        guard idxCount >= 3 else { throw GPUError.badState("draw with < 3 indices") }

        let vDecoded = DecodedShader(vertexProgram)
        let fDecoded = DecodedShader(fragmentProgram)
        // Varying slots the fragment program can read — the only ones
        // clip interpolation and raster interpolation need to compute.
        // (Values in unread slots are unobservable: the fragment stage
        // can only read varyings through LDATTR.)
        var usedSlots: [Int] = []
        for s in 0..<8 where fDecoded.attrSlotMask & (UInt32(1) << UInt32(s)) != 0 {
            usedSlots.append(s)
        }
        let unis = uniforms
        let width = fbWidth, height = fbHeight

        var prof = DrawProfile()
        let pT0 = Self.nowMs()
        // ---- Buffer snapshots: one locked read each (v1 fetched the
        // vertex buffer one float and the index buffer one word at a
        // time through the lock).
        let vbFloats = UnsafeMutablePointer<Float>.allocate(capacity: max(vertexCount * 12, 1))
        defer { vbFloats.deallocate() }
        vramRead(at: vertexBuffer, count: vertexCount * 48).withUnsafeBufferPointer { src in
            if let sp = src.baseAddress, vertexCount > 0 {
                memcpy(vbFloats, sp, vertexCount * 48)
            }
        }
        let ibWords = UnsafeMutablePointer<UInt32>.allocate(capacity: max(idxCount, 1))
        defer { ibWords.deallocate() }
        vramRead(at: indexBuffer, count: idxCount * 4).withUnsafeBufferPointer { src in
            if let sp = src.baseAddress, idxCount > 0 {
                memcpy(ibWords, sp, idxCount * 4)
            }
        }

        // ---- Vertex stage: one warp per 32 vertices, warps over clusters.
        let shadedPtr = UnsafeMutablePointer<ShadedVertex>.allocate(capacity: max(vertexCount, 1))
        shadedPtr.initialize(repeating: ShadedVertex(), count: max(vertexCount, 1))
        defer { shadedPtr.deinitialize(count: max(vertexCount, 1)); shadedPtr.deallocate() }
        let warpCount = (vertexCount + 31) / 32
        let perWarpOps = UnsafeMutablePointer<UInt64>.allocate(capacity: max(warpCount, 1))
        perWarpOps.initialize(repeating: 0, count: max(warpCount, 1))
        defer { perWarpOps.deinitialize(count: max(warpCount, 1)); perWarpOps.deallocate() }
        prof.snapshotMs = Self.nowMs() - pT0
        let pV0 = Self.nowMs()
        let vbF = vbFloats
        DispatchQueue.concurrentPerform(iterations: warpCount) { warp in
            let base = warp * 32
            let count = min(32, vertexCount - base)
            // v4 grab unit: fetch this warp's attributes with wide
            // sequential loads into [lane][16] records (vec3
            // attributes expanded to vec4 exactly as the v3 closure
            // presented them), and let the executor read/write
            // through raw pointers — v3 paid ~128 closure calls per
            // warp for LDATTR and ~224 for OUT. The closures below
            // remain as the general path and return identical
            // values; the executor prefers the pointers.
            let grab = UnsafeMutablePointer<Float>.allocate(capacity: 32 * GrabUnit.recordStride)
            defer { grab.deallocate() }
            GrabUnit.grabWarp(snapshot: vbF, base: base, count: count, into: grab)
            let outBuf = UnsafeMutablePointer<Float>.allocate(capacity: 32 * 32)
            outBuf.initialize(repeating: 0, count: 32 * 32)
            defer { outBuf.deallocate() }
            var ctx = WarpContext.dummy()
            ctx.uniforms = unis
            ctx.attrPtr = UnsafePointer(grab)
            ctx.attrStride = GrabUnit.recordStride
            ctx.outPtr = outBuf
            ctx.outFull = true
            ctx.attrProvider = { lane, slot in
                let o = (base + lane) * 12
                switch slot {
                case 0: return (vbF[o], vbF[o + 1], vbF[o + 2], 1)
                case 1: return (vbF[o + 3], vbF[o + 4], vbF[o + 5], 0)
                case 2: return (vbF[o + 6], vbF[o + 7], 0, 0)
                case 3: return (vbF[o + 8], vbF[o + 9], vbF[o + 10], vbF[o + 11])
                default: return (0, 0, 0, 0)
                }
            }
            ctx.outWriter = { lane, slot, x, y, z, w in
                // General path (unused while outPtr is set): fill
                // the same outBuf the scatter below consumes, so
                // both paths land identically.
                if slot >= 0 && slot < 8 {
                    let o = lane * 32 + slot * 4
                    outBuf[o] = x; outBuf[o + 1] = y; outBuf[o + 2] = z; outBuf[o + 3] = w
                }
            }
            let ops = ShaderExecutor.runWarp(decoded: vDecoded, lanes: count, ctx: ctx)
            perWarpOps[warp] = UInt64(ops)
            // Scatter the warp's outputs into the shaded-vertex
            // buffer (disjoint vertices per warp): slot 0 → clip;
            // slots 1..7 → varyings. Varyings slot 0 stays zero,
            // exactly as in every earlier version (OUT 0 is the
            // clip position, never a varying).
            for lane in 0..<count {
                let o = lane * 32
                shadedPtr[base + lane].clip = (outBuf[o], outBuf[o + 1], outBuf[o + 2], outBuf[o + 3])
                shadedPtr[base + lane].varyings.withMutableFloats { dst in
                    for i in 4..<32 { dst[i] = outBuf[o + i] }
                }
            }
        }
        for warp in 0..<warpCount {
            let v = perWarpOps[warp]
            stats.lanesPerCluster[warp % Self.clusterCount] += v
            stats.laneOps += v
        }
        prof.vertexMs = Self.nowMs() - pV0
        stats.trianglesIn += UInt64(idxCount / 3)
        let triCount = (idxCount - 3) / 3 + 1   // idxCount >= 3 (guarded above)

        // ---- v5 VPU test pass (only when armed over the bus):
        // every triangle is re-positioned through the actual VPU
        // path and bit-compared against the clip positions the
        // vertex stage just produced; the shaded clips are then
        // overwritten with the VPU clips, so an armed frame is
        // RENDERED FROM the VPU positions — a wrong VPU visibly
        // breaks the picture. With correct VPUs the pass is
        // observational only (the host tests prove armed and
        // unarmed frames byte-identical). Its wall time lands in
        // neither stage timer; it is reported in vpuTest.passMs.
        if vpuTestEnabled {
            runVPUTestPass(triCount: triCount, vertexCount: vertexCount,
                           snapshot: vbF, indices: ibWords,
                           uniforms: unis, shaded: shadedPtr)
        }

        // ---- Clip + fan-triangulate once, into a work list with
        // precomputed screen coordinates, then bin each triangle into
        // the row chunks its screen Y range touches. (v1 re-ran the
        // full clip/screen/rasterise prologue for every triangle in
        // every one of the 48 bands.)
        func screenOf(_ v: ShadedVertex) -> (Float, Float, Float, Float) {
            let iw = 1 / v.clip.3
            return ((v.clip.0 * iw * 0.5 + 0.5) * Float(width),
                    (0.5 - v.clip.1 * iw * 0.5) * Float(height),
                    v.clip.2 * iw, iw)
        }
        // Speed round 3: the clip runs over contiguous triangle
        // ranges on all cores — it was the draw's serial tail (the
        // vertex and raster stages already fan out). Each range
        // builds its own work list; the lists are concatenated in
        // range order, so the merged list is element-for-element the
        // serial list: triangle order, and therefore depth-tie
        // behaviour, is unchanged.
        let pC0 = Self.nowMs()
        final class WorkBox { var tris: [WorkTri] = [] }
        let rangeCount = min(32, max(triCount, 1))
        let boxes = (0..<rangeCount).map { _ in WorkBox() }
        DispatchQueue.concurrentPerform(iterations: rangeCount) { range in
            let lo = triCount * range / rangeCount
            let hi = triCount * (range + 1) / rangeCount
            var local: [WorkTri] = []
            local.reserveCapacity(max(hi - lo, 0))
            for k in lo..<hi {
                let t = k * 3
                let tri = [shadedPtr[Int(ibWords[t])],
                           shadedPtr[Int(ibWords[t + 1])],
                           shadedPtr[Int(ibWords[t + 2])]]
                let clipped = Self.clipTriangle(tri, usedSlots: usedSlots)
                if clipped.count < 3 { continue }
                for fan in 1..<(clipped.count - 1) {
                    let a = clipped[0], b = clipped[fan], c = clipped[fan + 1]
                    local.append(WorkTri(v0: a, v1: b, v2: c,
                                         s0: screenOf(a), s1: screenOf(b), s2: screenOf(c)))
                }
            }
            boxes[range].tris = local
        }
        var work: [WorkTri] = []
        work.reserveCapacity(idxCount / 3)
        var workStarts: [Int] = [0]
        for box in boxes {
            work.append(contentsOf: box.tris)
            workStarts.append(work.count)
        }
        stats.trianglesOut += UInt64(work.count)
        prof.clipMs = Self.nowMs() - pC0
        let pB0 = Self.nowMs()

        // ---- Rasterise in horizontal row chunks over the clusters.
        // Chunks own disjoint framebuffer rows, so the shared depth
        // buffer (raw pointer) and framebuffer writes never overlap
        // between concurrent chunks — the same invariant v1's bands
        // used, at finer granularity for load balance. Within a chunk,
        // triangles are processed in work-list order and rows in
        // increasing order, exactly as v1 processed them within a band.
        // Depth convention: this pipeline's projection does not flip
        // z, so a LARGER NDC z is NEARER the viewer. The buffer clears
        // to -1 (below the [0, 1] clip range) and the test keeps the
        // greater z (see rasterise).
        let depthPtr = UnsafeMutablePointer<Float>.allocate(capacity: width * height)
        depthPtr.initialize(repeating: -1.0, count: width * height)
        defer { depthPtr.deinitialize(count: width * height); depthPtr.deallocate() }
        let fbBlocks = blockPointers(addr: framebuffer, size: fbAllocSize)
        let fbFirstBlock = blockIndex(framebuffer)
        let fb = framebuffer
        let chunkRows = 4
        let chunkCount = (height + chunkRows - 1) / chunkRows
        // Binning fans out over the same ranges (per-range bin
        // lists merged in range order keep every chunk's triangle
        // order identical to the serial build).
        final class BinBox { var bins: [[Int]] = [] }
        let binBoxes = (0..<rangeCount).map { _ in BinBox() }
        DispatchQueue.concurrentPerform(iterations: rangeCount) { range in
            var local: [[Int]] = [[Int]](repeating: [], count: chunkCount)
            for wi in workStarts[range]..<workStarts[range + 1] {
                let wt = work[wi]
                var ry0 = Int(floorf(min(wt.s0.1, min(wt.s1.1, wt.s2.1))))
                var ry1 = Int(ceilf(max(wt.s0.1, max(wt.s1.1, wt.s2.1))))
                if ry0 < 0 { ry0 = 0 }
                if ry1 > height - 1 { ry1 = height - 1 }
                if ry0 > ry1 { continue }
                for c in (ry0 / chunkRows)...(ry1 / chunkRows) { local[c].append(wi) }
            }
            binBoxes[range].bins = local
        }
        var bins: [[Int]] = [[Int]](repeating: [], count: chunkCount)
        for box in binBoxes {
            for c in 0..<chunkCount { bins[c].append(contentsOf: box.bins[c]) }
        }
        prof.binMs = Self.nowMs() - pB0
        let pR0 = Self.nowMs()
        let chunkOps = UnsafeMutablePointer<UInt64>.allocate(capacity: chunkCount)
        let chunkFrags = UnsafeMutablePointer<UInt64>.allocate(capacity: chunkCount)
        let chunkWarps = UnsafeMutablePointer<UInt64>.allocate(capacity: chunkCount)
        let chunkExec = UnsafeMutablePointer<Double>.allocate(capacity: chunkCount)
        for p in [chunkOps, chunkFrags, chunkWarps] { p.initialize(repeating: 0, count: chunkCount) }
        chunkExec.initialize(repeating: 0, count: chunkCount)
        defer {
            for p in [chunkOps, chunkFrags, chunkWarps] {
                p.deinitialize(count: chunkCount); p.deallocate()
            }
            chunkExec.deinitialize(count: chunkCount); chunkExec.deallocate()
        }

        fbBlocks.withUnsafeBufferPointer { fbTable in
            DispatchQueue.concurrentPerform(iterations: chunkCount) { chunk in
                let y0 = chunk * chunkRows
                let y1 = min(height, y0 + chunkRows)
                // Per-chunk warp scratch, allocated once (v1 allocated
                // a fresh nested varying array per accepted fragment).
                let warpPx = UnsafeMutablePointer<Int>.allocate(capacity: 32)
                let warpVar = UnsafeMutablePointer<Float>.allocate(capacity: 32 * 32)
                let warpCol = UnsafeMutablePointer<Float>.allocate(capacity: 32 * 4)
                defer { warpPx.deallocate(); warpVar.deallocate(); warpCol.deallocate() }
                // One batcher (and one warp context) per chunk;
                // fragments accumulate across rows and triangles
                // until a full warp of 32 is ready (round 3).
                let batcher = FragBatcher(
                    px: warpPx, vars: warpVar, col: warpCol,
                    decoded: fDecoded, unis: unis,
                    texSampler: { [self] unit, u, v in self.sampleTexture(unit: unit, u: u, v: v) },
                    fbTable: fbTable, fbFirstBlock: fbFirstBlock, fb: fb,
                    fbFallback: { [self] addr, bytes in self.vramWrite(at: addr, bytes: bytes) })
                for wi in bins[chunk] {
                    self.rasterise(work[wi], width: width,
                                   y0: y0, y1: y1, depth: depthPtr,
                                   usedSlots: usedSlots, batcher: batcher)
                }
                batcher.flush()
                chunkOps[chunk] = batcher.ops
                chunkFrags[chunk] = batcher.frags
                chunkWarps[chunk] = batcher.warps
                chunkExec[chunk] = batcher.execMs
            }
        }

        var fragWarps: UInt64 = 0
        for c in 0..<chunkCount {
            stats.lanesPerCluster[c % Self.clusterCount] += chunkOps[c]
            stats.laneOps += chunkOps[c]
            stats.fragmentsShaded += chunkFrags[c]
            fragWarps += chunkWarps[c]
            prof.fragExecMs += chunkExec[c]
        }
        prof.rasterMs = Self.nowMs() - pR0
        stats.warpsIssued += UInt64(warpCount) + fragWarps
        stats.draws += 1
        let ms = Date().timeIntervalSince(t0) * 1000
        stats.lastFrameMs = ms
        prof.totalMs = ms
        lastDrawProfile = prof
        return ms
    }

    /// The v5 VPU test pass, run inside draw() between the vertex
    /// and clip stages while armed. For every triangle: the grab
    /// unit fetches its 3 vertex positions from the draw's
    /// snapshot, ONE VPU runs the 3-axis calculator (ngp_mat4 over
    /// uniforms c0..c3 — the same matrix the vertex stage's VMAT4
    /// consumed) over the triangle's 9 axis values, and the
    /// resulting clip position of each corner is bit-compared
    /// against the clip the shader stage produced for the same
    /// vertex (the independently computed reference — snapshotted
    /// before the pass, so the gather below can never contaminate
    /// a later comparison: every corner is checked against the
    /// pristine shader value). Workloads are split into
    /// contiguous sub-ranges over the MGVPU clusters and run
    /// concurrently. After the compare, each shaded vertex's clip
    /// is OVERWRITTEN with the VPU's clip (the gather), so the
    /// frame that renders is positioned by the VPUs. Results
    /// accumulate into `vpuTest` (window) and `vpuTestLifetime`.
    private func runVPUTestPass(triCount: Int, vertexCount: Int,
                                snapshot vbF: UnsafePointer<Float>,
                                indices ibWords: UnsafePointer<UInt32>,
                                uniforms unis: [Float],
                                shaded shadedPtr: UnsafeMutablePointer<ShadedVertex>) {
        let t0 = Self.nowMs()
        // Pristine reference: the shader stage's clip positions,
        // copied before any gather write can touch them.
        let refClip = UnsafeMutablePointer<Float>.allocate(capacity: max(vertexCount, 1) * 4)
        defer { refClip.deallocate() }
        for v in 0..<vertexCount {
            let c = shadedPtr[v].clip
            refClip[v * 4] = c.0; refClip[v * 4 + 1] = c.1
            refClip[v * 4 + 2] = c.2; refClip[v * 4 + 3] = c.3
        }
        let out = UnsafeMutablePointer<Float>.allocate(capacity: triCount * 12)
        defer { out.deallocate() }
        let ranges = MGVPU.subRanges(count: triCount)
        var clusters = 0
        for r in ranges where r.1 > r.0 { clusters += 1 }
        unis.withUnsafeBufferPointer { ub in
            let mPtr = ub.baseAddress!
            DispatchQueue.concurrentPerform(iterations: ranges.count) { r in
                let lo = ranges[r].0, hi = ranges[r].1
                if lo >= hi { return }
                var vpu = VPU()
                withUnsafeTemporaryAllocation(of: Float.self, capacity: 12) { s in
                    s.initialize(repeating: 0)
                    let sPtr = s.baseAddress!
                    for k in lo..<hi {
                        let t = k * 3
                        GrabUnit.grabTrianglePositions(
                            snapshot: vbF,
                            Int(ibWords[t]), Int(ibWords[t + 1]), Int(ibWords[t + 2]),
                            into: sPtr)
                        vpu.load(positions: sPtr)
                        vpu.position(matrix: mPtr, into: out + k * 12)
                    }
                }
            }
        }
        // Compare + gather (serial, in triangle order): the
        // reference is the pristine snapshot of the shader stage's
        // clip for the corner's vertex; the gather writes the VPU
        // clip over the shaded vertex.
        let fault = Self.vpuTestFault
        var mismatches = 0
        var maxErr: Float = 0
        for k in 0..<triCount {
            let t = k * 3
            for c in 0..<3 {
                let vi = Int(ibWords[t + c])
                let o = k * 12 + c * 4
                var g0 = out[o]
                let g1 = out[o + 1], g2 = out[o + 2], g3 = out[o + 3]
                if fault != 0 && c == 0 { g0 += fault }   // host-test fault hook
                let r0 = refClip[vi * 4], r1 = refClip[vi * 4 + 1]
                let r2 = refClip[vi * 4 + 2], r3 = refClip[vi * 4 + 3]
                if g0.bitPattern != r0.bitPattern || g1.bitPattern != r1.bitPattern
                    || g2.bitPattern != r2.bitPattern || g3.bitPattern != r3.bitPattern {
                    mismatches += 1
                }
                let d0 = abs(g0 - r0), d1 = abs(g1 - r1)
                let d2 = abs(g2 - r2), d3 = abs(g3 - r3)
                if d0.isFinite && d0 > maxErr { maxErr = d0 }
                if d1.isFinite && d1 > maxErr { maxErr = d1 }
                if d2.isFinite && d2 > maxErr { maxErr = d2 }
                if d3.isFinite && d3 > maxErr { maxErr = d3 }
                shadedPtr[vi].clip = (g0, g1, g2, g3)
            }
        }
        let dt = Self.nowMs() - t0
        vpuTest.record(triangles: triCount, mismatches: mismatches,
                       maxErr: maxErr, passMs: dt, clusters: clusters)
        vpuTestLifetime.record(triangles: triCount, mismatches: mismatches,
                               maxErr: maxErr, passMs: dt, clusters: clusters)
    }

    /// v4 plot unit: plots 3D vectors from the current vertex and
    /// index buffers, bypassing the triangle pipeline. Mode 0 =
    /// lines (index pairs), mode 1 = points (one per index). The
    /// transform is the uniform matrix c0..c3 (row-major, the same
    /// uniforms the standard vertex shader consumes) and the plot
    /// colour is uniform c10, delivered to the fragment program as
    /// varying slot 1 — the slot the standard lit fragment program
    /// reads as colour (it modulates by its lighting, like any
    /// fragment); a flat fragment program (LDATTR 1 -> OUT 0)
    /// shows the c10 colour verbatim.
    ///
    /// Path per vector: the grab unit fetches the endpoint
    /// positions, a VPU runs the 3-axis calculator (ngp_mat4 —
    /// bit-identical to the shader stage's VMAT4), endpoints with
    /// w <= 0 or non-finite clip are skipped (the same rule the
    /// clip stage applies), survivors convert to the 16-bit-per-
    /// axis screen grid, and the DDA rasteriser emits samples that
    /// depth-test exactly like triangle fragments (larger NDC z is
    /// nearer; buffer clears to -1). Depth interpolates linearly
    /// in screen space along the vector. Indices outside the
    /// vertex buffer are skipped. Returns frame time in ms.
    @discardableResult
    func plotVectors(indexCount: Int, mode: Int) throws -> Double {
        let t0 = Date()
        let pP0 = Self.nowMs()
        guard !fragmentProgram.isEmpty else { throw GPUError.badProgram }
        guard let vbSize = allocationSize(vertexBuffer) else { throw GPUError.badHandle(vertexBuffer) }
        guard let ibSize = allocationSize(indexBuffer) else { throw GPUError.badHandle(indexBuffer) }
        guard let fbAllocSize = allocationSize(framebuffer), fbWidth > 0, fbHeight > 0 else { throw GPUError.badState("no framebuffer") }
        let vertexCount = Int(vbSize / 48)
        var idxCount = min(indexCount, Int(ibSize / 4))
        if mode == 0 { idxCount -= idxCount % 2 }
        guard idxCount > 0 else { return 0 }
        let width = fbWidth, height = fbHeight
        let fDecoded = DecodedShader(fragmentProgram)
        let unis = uniforms
        func u(_ i: Int) -> Float { i < unis.count ? unis[i] : 0 }

        let vbFloats = UnsafeMutablePointer<Float>.allocate(capacity: max(vertexCount * 12, 1))
        defer { vbFloats.deallocate() }
        vramRead(at: vertexBuffer, count: vertexCount * 48).withUnsafeBufferPointer { src in
            if let sp = src.baseAddress, vertexCount > 0 {
                memcpy(vbFloats, sp, vertexCount * 48)
            }
        }
        let ibWords = UnsafeMutablePointer<UInt32>.allocate(capacity: max(idxCount, 1))
        defer { ibWords.deallocate() }
        vramRead(at: indexBuffer, count: idxCount * 4).withUnsafeBufferPointer { src in
            if let sp = src.baseAddress, idxCount > 0 {
                memcpy(ibWords, sp, idxCount * 4)
            }
        }

        let depthPtr = UnsafeMutablePointer<Float>.allocate(capacity: width * height)
        depthPtr.initialize(repeating: -1.0, count: width * height)
        defer { depthPtr.deinitialize(count: width * height); depthPtr.deallocate() }
        let chunkRows = 4
        let chunkCount = (height + chunkRows - 1) / chunkRows
        var chunkPx: [[Int]] = [[Int]](repeating: [], count: chunkCount)

        // Per-frame budget: at most MAX_SCREEN_VECTORS vectors are
        // plotted (the plot unit's rated on-screen capacity);
        // excess vectors in the command are dropped, as a display
        // list that overruns its hardware budget would be.
        let vectorCount = min(mode == 0 ? idxCount / 2 : idxCount,
                              Self.maxScreenVectors)

        // Pass 1 worker: positions vectors [lo, hi) (grab + VPU),
        // projects them to the screen grid, DDA-rasterises, and
        // hands each sample to `emit` as (pixelIndex, z). Consecu-
        // tive samples landing on the same pixel (rounding can
        // repeat a pixel on shallow slopes) are coalesced to their
        // max z first: the serial walk's outcome for such a run is
        // exactly "the max-z sample tests once" (z is monotonic
        // along a vector) — the framebuffer and final depth are
        // bit-identical, only redundant fragment shading is
        // removed.
        func processRange(_ lo: Int, _ hi: Int, _ emit: (Int, Float) -> Void) -> UInt64 {
            var count: UInt64 = 0
            var vpu = VPU()
            withUnsafeTemporaryAllocation(of: Float.self, capacity: 16) { mbuf in
                mbuf.initialize(repeating: 0)
                for i in 0..<16 { mbuf[i] = u(i) }
                let mPtr = mbuf.baseAddress!
                withUnsafeTemporaryAllocation(of: Float.self, capacity: 12) { scratch in
                    scratch.initialize(repeating: 0)
                    let sPtr = scratch.baseAddress!
                    withUnsafeTemporaryAllocation(of: Float.self, capacity: 12) { clipBuf in
                        clipBuf.initialize(repeating: 0)
                        let cPtr = clipBuf.baseAddress!
                        for v in lo..<hi {
                            let i0 = Int(ibWords[mode == 0 ? v * 2 : v])
                            let i1 = mode == 0 ? Int(ibWords[v * 2 + 1]) : i0
                            guard i0 < vertexCount, i1 < vertexCount else { continue }
                            GrabUnit.grabTrianglePositions(snapshot: vbFloats, i0, i1, i0, into: sPtr)
                            vpu.load(positions: sPtr)
                            vpu.position(matrix: mPtr, into: cPtr)
                            // Endpoints: skip on w <= 0 / non-
                            // finite, as the clip stage does.
                            let w0 = cPtr[3]
                            guard w0 > 0, w0.isFinite else { continue }
                            let iw0 = 1 / w0
                            let nx0 = cPtr[0] * iw0, ny0 = cPtr[1] * iw0, nz0 = cPtr[2] * iw0
                            guard nx0.isFinite, ny0.isFinite, nz0.isFinite else { continue }
                            let gx0 = ScreenSpace.gridX(nx0, width: width)
                            let gy0 = ScreenSpace.gridY(ny0, height: height)
                            if mode == 1 {
                                let px = ScreenSpace.pixel(gx0), py = ScreenSpace.pixel(gy0)
                                if px >= 0 && px < width && py >= 0 && py < height {
                                    emit(py * width + px, nz0)
                                }
                                count += 1
                                continue
                            }
                            let w1 = cPtr[7]
                            guard w1 > 0, w1.isFinite else { continue }
                            let iw1 = 1 / w1
                            let nx1 = cPtr[4] * iw1, ny1 = cPtr[5] * iw1, nz1 = cPtr[6] * iw1
                            guard nx1.isFinite, ny1.isFinite, nz1.isFinite else { continue }
                            let gx1 = ScreenSpace.gridX(nx1, width: width)
                            let gy1 = ScreenSpace.gridY(ny1, height: height)
                            count += 1
                            var runDi = -1
                            var runZ: Float = 0
                            PlotUnit.rasterizeLine(x0: gx0, y0: gy0, x1: gx1, y1: gy1) { gx, gy, s, steps in
                                let px = ScreenSpace.pixel(gx), py = ScreenSpace.pixel(gy)
                                guard px >= 0 && px < width && py >= 0 && py < height else {
                                    if runDi >= 0 { emit(runDi, runZ); runDi = -1 }
                                    return
                                }
                                let z = steps == 0 ? nz0 : nz0 + (nz1 - nz0) * (Float(s) / Float(steps))
                                let di = py * width + px
                                if di == runDi {
                                    if z > runZ { runZ = z }
                                } else {
                                    if runDi >= 0 { emit(runDi, runZ) }
                                    runDi = di; runZ = z
                                }
                            }
                            if runDi >= 0 { emit(runDi, runZ) }
                        }
                    }
                }
            }
            return count
        }

        // Depth-testing emit, shared by the serial path and the
        // parallel path's merge: the triangle rule (z >= 0 and
        // z >= depth wins), samples in vector order.
        let depthEmit: (Int, Float) -> Void = { di, z in
            if z >= 0 && z >= depthPtr[di] {
                depthPtr[di] = z
                chunkPx[(di / width) / chunkRows].append(di)
            }
        }

        // Pass 1: below the threshold, run serially in index order.
        // At scale, distribute contiguous vector sub-ranges across
        // the MGVPU clusters (the scheduling layer: one sub-range
        // per cluster, run concurrently, each cluster collecting
        // its own sample list), then merge the lists in cluster
        // order through the same depth test — the merged order is
        // exactly the serial order, so the frame is bit-identical
        // either way (host-tested).
        var plotted: UInt64 = 0
        if vectorCount >= Self.plotParallelThreshold {
            let ranges = MGVPU.subRanges(count: vectorCount)
            final class SampleBox { var samples: [(Int, Float)] = [] }
            let boxes = ranges.map { _ in SampleBox() }
            let counts = UnsafeMutablePointer<UInt64>.allocate(capacity: ranges.count)
            counts.initialize(repeating: 0, count: ranges.count)
            defer { counts.deinitialize(count: ranges.count); counts.deallocate() }
            DispatchQueue.concurrentPerform(iterations: ranges.count) { k in
                let box = boxes[k]
                counts[k] = processRange(ranges[k].0, ranges[k].1) { di, z in
                    box.samples.append((di, z))
                }
            }
            for k in 0..<ranges.count {
                plotted += counts[k]
                for (di, z) in boxes[k].samples { depthEmit(di, z) }
            }
        } else {
            plotted = processRange(0, vectorCount, depthEmit)
        }
        stats.vectorsPlotted += plotted
        let pP1 = Self.nowMs()

        // Pass 2: shade the accepted pixels per row chunk through
        // the ordinary fragment path (one FragBatcher per chunk,
        // pixels in acceptance order — the draw() protocol).
        let fbBlocks = blockPointers(addr: framebuffer, size: fbAllocSize)
        let fbFirstBlock = blockIndex(framebuffer)
        let fb = framebuffer
        let colour = (u(40), u(41), u(42), u(43))   // uniform c10
        let chunkOps = UnsafeMutablePointer<UInt64>.allocate(capacity: chunkCount)
        let chunkFrags = UnsafeMutablePointer<UInt64>.allocate(capacity: chunkCount)
        let chunkWarps = UnsafeMutablePointer<UInt64>.allocate(capacity: chunkCount)
        for p in [chunkOps, chunkFrags, chunkWarps] { p.initialize(repeating: 0, count: chunkCount) }
        defer {
            for p in [chunkOps, chunkFrags, chunkWarps] {
                p.deinitialize(count: chunkCount); p.deallocate()
            }
        }
        fbBlocks.withUnsafeBufferPointer { fbTable in
            DispatchQueue.concurrentPerform(iterations: chunkCount) { chunk in
                let list = chunkPx[chunk]
                if list.isEmpty { return }
                let warpPx = UnsafeMutablePointer<Int>.allocate(capacity: 32)
                let warpVar = UnsafeMutablePointer<Float>.allocate(capacity: 32 * 32)
                let warpCol = UnsafeMutablePointer<Float>.allocate(capacity: 32 * 4)
                defer { warpPx.deallocate(); warpVar.deallocate(); warpCol.deallocate() }
                let batcher = FragBatcher(
                    px: warpPx, vars: warpVar, col: warpCol,
                    decoded: fDecoded, unis: unis,
                    texSampler: { [self] unit, uu, vv in self.sampleTexture(unit: unit, u: uu, v: vv) },
                    fbTable: fbTable, fbFirstBlock: fbFirstBlock, fb: fb,
                    fbFallback: { [self] addr, bytes in self.vramWrite(at: addr, bytes: bytes) })
                for di in list {
                    let base = batcher.count * 32
                    for i in 0..<32 { warpVar[base + i] = 0 }
                    warpVar[base + 4] = colour.0
                    warpVar[base + 5] = colour.1
                    warpVar[base + 6] = colour.2
                    warpVar[base + 7] = colour.3
                    warpPx[batcher.count] = di
                    batcher.count += 1
                    if batcher.count == 32 { batcher.flush() }
                }
                batcher.flush()
                chunkOps[chunk] = batcher.ops
                chunkFrags[chunk] = batcher.frags
                chunkWarps[chunk] = batcher.warps
            }
        }
        var fragWarps: UInt64 = 0
        for c in 0..<chunkCount {
            stats.lanesPerCluster[c % Self.clusterCount] += chunkOps[c]
            stats.laneOps += chunkOps[c]
            stats.fragmentsShaded += chunkFrags[c]
            fragWarps += chunkWarps[c]
        }
        stats.warpsIssued += fragWarps
        let ms = Date().timeIntervalSince(t0) * 1000
        stats.lastFrameMs = ms
        lastPlotProfile = PlotProfile(pass1Ms: pP1 - pP0,
                                      pass2Ms: Self.nowMs() - pP1,
                                      totalMs: ms)
        return ms
    }

    // Sutherland–Hodgman clipping against the 6 clip planes, in clip
    // space (before the perspective divide), interpolating varyings.
    // Only the fragment-readable slots are interpolated at clip points
    // (see draw()); all other slots of a generated vertex stay zero,
    // which no fragment program can observe.
    private static func clipTriangle(_ tri: [ShadedVertex], usedSlots: [Int]) -> [ShadedVertex] {
        // Plane signed distances, keep >= 0 — the v2 plane expressions
        // verbatim, evaluated directly (speed round 3 removed the
        // per-call array of six closures; same values, same order).
        func dist(_ plane: Int, _ v: ShadedVertex) -> Float {
            switch plane {
            case 0: return v.clip.3 + v.clip.0    // x >= -w
            case 1: return v.clip.3 - v.clip.0    // x <=  w
            case 2: return v.clip.3 + v.clip.1    // y >= -w
            case 3: return v.clip.3 - v.clip.1    // y <=  w
            case 4: return v.clip.2               // z >= 0
            default: return v.clip.3 - v.clip.2   // z <= w
            }
        }
        // Fast path: fully inside every plane — the Sutherland–Hodgman
        // loop below returns the input unchanged in that case.
        var allIn = true
        outer: for v in tri {
            for p in 0..<6 where dist(p, v) < 0 { allIn = false; break outer }
        }
        if allIn { return tri }
        var poly = tri
        for p in 0..<6 {
            if poly.isEmpty { break }
            var out: [ShadedVertex] = []
            for i in 0..<poly.count {
                let cur = poly[i]
                let prev = poly[(i + poly.count - 1) % poly.count]
                let dc = dist(p, cur), dp = dist(p, prev)
                let curIn = dc >= 0, prevIn = dp >= 0
                if curIn != prevIn {
                    let f = dp / (dp - dc)
                    out.append(lerpVertex(prev, cur, f, usedSlots: usedSlots))
                }
                if curIn { out.append(cur) }
            }
            poly = out
        }
        return poly
    }

    private static func lerpVertex(_ a: ShadedVertex, _ b: ShadedVertex, _ f: Float,
                                   usedSlots: [Int]) -> ShadedVertex {
        var v = ShadedVertex()
        v.clip = (a.clip.0 + (b.clip.0 - a.clip.0) * f,
                  a.clip.1 + (b.clip.1 - a.clip.1) * f,
                  a.clip.2 + (b.clip.2 - a.clip.2) * f,
                  a.clip.3 + (b.clip.3 - a.clip.3) * f)
        // v4: varyings are inline VarBlocks — lerp through raw
        // pointers with the identical expression per component.
        a.varyings.withFloats { ap in
            b.varyings.withFloats { bp in
                v.varyings.withMutableFloats { vp in
                    for s in usedSlots {
                        let o = s * 4
                        for c in 0..<4 { vp[o + c] = ap[o + c] + (bp[o + c] - ap[o + c]) * f }
                    }
                }
            }
        }
        return v
    }

    /// Rasterises one screen-space triangle into rows [y0, y1),
    /// feeding accepted pixels to the chunk's fragment batcher.
    /// Coverage, barycentric weights, depth and interpolation use
    /// the v1 expressions verbatim, evaluated per pixel in the same
    /// order.
    private func rasterise(_ tri: WorkTri, width: Int,
                           y0: Int, y1: Int, depth: UnsafeMutablePointer<Float>,
                           usedSlots: [Int], batcher: FragBatcher) {
        // Speed round 3: interpolation reads the three vertices'
        // varyings through raw pointers — v2 subscripted the Swift
        // arrays per component (bounds check + COW check each, ~24
        // per pixel). v4: the varyings are inline VarBlocks; the
        // pointers address the copies inside the WorkTri, which
        // outlives the call and is never mutated during
        // rasterisation.
        tri.v0.varyings.withFloats { p0 in
            tri.v1.varyings.withFloats { p1 in
                tri.v2.varyings.withFloats { p2 in
                    rasteriseBody(tri, width: width, y0: y0, y1: y1, depth: depth,
                                  usedSlots: usedSlots, batcher: batcher,
                                  v0p: p0, v1p: p1, v2p: p2)
                }
            }
        }
    }

    private func rasteriseBody(_ tri: WorkTri, width: Int,
                               y0: Int, y1: Int, depth: UnsafeMutablePointer<Float>,
                               usedSlots: [Int], batcher: FragBatcher,
                               v0p: UnsafePointer<Float>,
                               v1p: UnsafePointer<Float>,
                               v2p: UnsafePointer<Float>) {
        let s0 = tri.s0, s1 = tri.s1, s2 = tri.s2
        let area = (s1.0 - s0.0) * (s2.1 - s0.1) - (s1.1 - s0.1) * (s2.0 - s0.0)
        if area == 0 { return }
        let minY = max(y0, Int(floorf(min(s0.1, min(s1.1, s2.1)))))
        let maxY = min(y1 - 1, Int(ceilf(max(s0.1, max(s1.1, s2.1)))))
        if minY > maxY { return }
        let minX = max(0, Int(floorf(min(s0.0, min(s1.0, s2.0)))))
        let maxX = min(width - 1, Int(ceilf(max(s0.0, max(s1.0, s2.0)))))
        if minX > maxX { return }

        let invArea = 1 / area
        let bvars = batcher.vars
        let bpx = batcher.px

        for y in minY...maxY {
            let py = Float(y) + 0.5
            // Row constants: each edge's (py - sY) product is the
            // same for every pixel in the row — the identical
            // rounded value, computed once (v2 recomputed the whole
            // edge expression per pixel).
            let e0row = (s2.0 - s1.0) * (py - s1.1)
            let e1row = (s0.0 - s2.0) * (py - s2.1)
            let e2row = (s1.0 - s0.0) * (py - s0.1)
            let b0 = s2.1 - s1.1, b1 = s0.1 - s2.1, b2 = s1.1 - s0.1
            let rowBase = y * width
            // v4 span narrowing: in a row, each edge function is
            // affine in px, so the pixels satisfying each sign case
            // (all e >= 0 / all e <= 0) form a single interval.
            // Compute both cases' intervals from the row constants,
            // take their hull, widen by 2 px (the interval math's
            // rounding error is < 0.01 px at these magnitudes), and
            // clamp to the bounding box. The per-pixel `inside`
            // test below still decides every visited pixel with the
            // identical expression, so the drawn set is bit-
            // identical — only certainly-outside pixels are skipped
            // (about half the bounding box of a typical triangle).
            var spanLo = 0, spanHi = -1   // empty until a case lands
            for sign: Float in [1, -1] {
                var loF = -Float.greatestFiniteMagnitude
                var hiF = Float.greatestFiniteMagnitude
                var ok = true
                // Edge constraints for this sign case: require
                // sign * (erow - b * (px - sx)) >= 0 per edge —
                // e0 pairs with s1.x, e1 with s2.x, e2 with s0.x,
                // as in the per-pixel expressions below. (Unrolled:
                // no per-row array allocation in the hot loop.)
                if ok {
                    let eb = sign * b0, sr = sign * e0row
                    if eb > 0 { hiF = min(hiF, s1.0 + sr / eb) }
                    else if eb < 0 { loF = max(loF, s1.0 + sr / eb) }
                    else if sr < 0 { ok = false }
                }
                if ok {
                    let eb = sign * b1, sr = sign * e1row
                    if eb > 0 { hiF = min(hiF, s2.0 + sr / eb) }
                    else if eb < 0 { loF = max(loF, s2.0 + sr / eb) }
                    else if sr < 0 { ok = false }
                }
                if ok {
                    let eb = sign * b2, sr = sign * e2row
                    if eb > 0 { hiF = min(hiF, s0.0 + sr / eb) }
                    else if eb < 0 { loF = max(loF, s0.0 + sr / eb) }
                    else if sr < 0 { ok = false }
                }
                if !ok { continue }
                var xLo = minX, xHi = maxX
                if loF > -1e6 { xLo = max(xLo, Int(ceilf(loF - 0.5)) - 2) }
                if hiF < 1e6 { xHi = min(xHi, Int(floorf(hiF - 0.5)) + 2) }
                if xLo <= xHi {
                    if spanHi < spanLo { spanLo = xLo; spanHi = xHi }
                    else { spanLo = min(spanLo, xLo); spanHi = max(spanHi, xHi) }
                }
            }
            if spanHi < spanLo { continue }
            for x in spanLo...spanHi {
                let px = Float(x) + 0.5
                let e0 = e0row - b0 * (px - s1.0)
                let e1 = e1row - b1 * (px - s2.0)
                let e2 = e2row - b2 * (px - s0.0)
                let inside = (e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)
                if inside {
                    let w0 = e0 * invArea, w1 = e1 * invArea, w2 = e2 * invArea
                    let z = w0 * s0.2 + w1 * s1.2 + w2 * s2.2
                    let di = rowBase + x
                    if z >= 0 && z >= depth[di] {
                        // Perspective-correct interpolation of the
                        // slots the fragment program can read.
                        let iw = w0 * s0.3 + w1 * s1.3 + w2 * s2.3
                        let varBase = batcher.count * 32
                        if iw != 0 {
                            // The three weight products are the
                            // same for every slot and component of
                            // this pixel, and the varyings are read
                            // through raw pointers — the expression
                            // tree per component is unchanged.
                            let a0 = w0 * s0.3, a1 = w1 * s1.3, a2 = w2 * s2.3
                            for s in usedSlots {
                                let so = s * 4
                                for c in 0..<4 {
                                    bvars[varBase + so + c] =
                                        (a0 * v0p[so + c]
                                       + a1 * v1p[so + c]
                                       + a2 * v2p[so + c]) / iw
                                }
                            }
                        } else {
                            for s in usedSlots {
                                let so = s * 4
                                for c in 0..<4 { bvars[varBase + so + c] = 0 }
                            }
                        }
                        depth[di] = z
                        bpx[batcher.count] = di
                        batcher.count += 1
                        if batcher.count == 32 { batcher.flush() }
                    }
                }
            }
        }
    }
}
