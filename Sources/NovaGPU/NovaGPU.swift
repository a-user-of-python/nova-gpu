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

struct GPUStats {
    var laneOps: UInt64 = 0                 // total lane-instructions executed
    var warpsIssued: UInt64 = 0
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
    static let clusterCount = 48
    static let lanesPerCluster = 32
    static let computeUnits = clusterCount * lanesPerCluster   // 1536
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

    private struct ShadedVertex {
        var clip: (Float, Float, Float, Float) = (0, 0, 0, 1)
        /// Varyings, flat: slot s occupies [s*4 + 0 ... s*4 + 3], slots 0..7.
        /// (Slot 0 is never written — OUT 0 is the clip position.)
        var varyings: [Float] = [Float](repeating: 0, count: 32)
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
        let vbF = vbFloats
        DispatchQueue.concurrentPerform(iterations: warpCount) { warp in
            let base = warp * 32
            let count = min(32, vertexCount - base)
            var ctx = WarpContext.dummy()
            ctx.uniforms = unis
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
            // Speed round 3: write straight into the shaded-vertex
            // buffer (disjoint vertices per warp) — v2 staged through
            // a per-warp array of 32 fresh ShadedVertex values and
            // copied them out, ~11k array allocations per draw.
            ctx.outWriter = { lane, slot, x, y, z, w in
                if slot == 0 { shadedPtr[base + lane].clip = (x, y, z, w) }
                else if slot >= 1 && slot < 8 {
                    let o = slot * 4
                    shadedPtr[base + lane].varyings[o] = x
                    shadedPtr[base + lane].varyings[o + 1] = y
                    shadedPtr[base + lane].varyings[o + 2] = z
                    shadedPtr[base + lane].varyings[o + 3] = w
                }
            }
            let ops = ShaderExecutor.runWarp(decoded: vDecoded, lanes: count, ctx: ctx)
            perWarpOps[warp] = UInt64(ops)
        }
        for warp in 0..<warpCount {
            let v = perWarpOps[warp]
            stats.lanesPerCluster[warp % Self.clusterCount] += v
            stats.laneOps += v
        }
        stats.trianglesIn += UInt64(idxCount / 3)

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
        var work: [WorkTri] = []
        work.reserveCapacity(idxCount / 3)
        for t in stride(from: 0, to: idxCount - 2, by: 3) {
            let tri = [shadedPtr[Int(ibWords[t])],
                       shadedPtr[Int(ibWords[t + 1])],
                       shadedPtr[Int(ibWords[t + 2])]]
            let clipped = Self.clipTriangle(tri, usedSlots: usedSlots)
            if clipped.count < 3 { continue }
            for fan in 1..<(clipped.count - 1) {
                let a = clipped[0], b = clipped[fan], c = clipped[fan + 1]
                work.append(WorkTri(v0: a, v1: b, v2: c,
                                    s0: screenOf(a), s1: screenOf(b), s2: screenOf(c)))
            }
        }
        stats.trianglesOut += UInt64(work.count)

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
        var bins: [[Int]] = [[Int]](repeating: [], count: chunkCount)
        for (wi, wt) in work.enumerated() {
            var ry0 = Int(floorf(min(wt.s0.1, min(wt.s1.1, wt.s2.1))))
            var ry1 = Int(ceilf(max(wt.s0.1, max(wt.s1.1, wt.s2.1))))
            if ry0 < 0 { ry0 = 0 }
            if ry1 > height - 1 { ry1 = height - 1 }
            if ry0 > ry1 { continue }
            for c in (ry0 / chunkRows)...(ry1 / chunkRows) { bins[c].append(wi) }
        }
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
            }
        }

        var fragWarps: UInt64 = 0
        for c in 0..<chunkCount {
            stats.lanesPerCluster[c % Self.clusterCount] += chunkOps[c]
            stats.laneOps += chunkOps[c]
            stats.fragmentsShaded += chunkFrags[c]
            fragWarps += chunkWarps[c]
        }
        stats.warpsIssued += UInt64(warpCount) + fragWarps
        stats.draws += 1
        let ms = Date().timeIntervalSince(t0) * 1000
        stats.lastFrameMs = ms
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
        for s in usedSlots {
            let o = s * 4
            for c in 0..<4 { v.varyings[o + c] = a.varyings[o + c] + (b.varyings[o + c] - a.varyings[o + c]) * f }
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
        // per pixel). The buffers outlive the call (the WorkTri holds
        // them) and are never mutated during rasterisation.
        tri.v0.varyings.withUnsafeBufferPointer { p0 in
            tri.v1.varyings.withUnsafeBufferPointer { p1 in
                tri.v2.varyings.withUnsafeBufferPointer { p2 in
                    rasteriseBody(tri, width: width, y0: y0, y1: y1, depth: depth,
                                  usedSlots: usedSlots, batcher: batcher,
                                  v0p: p0.baseAddress!, v1p: p1.baseAddress!, v2p: p2.baseAddress!)
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
            for x in minX...maxX {
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
