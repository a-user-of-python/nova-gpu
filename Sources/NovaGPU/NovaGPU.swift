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
// (Sutherland-Hodgman, 6 planes) -> viewport -> rasterise (bands over
// clusters, fragments in warps of 32) -> fragment shader -> depth test
// (early-z; shaders cannot discard) -> RGBA8888 framebuffer in VRAM.

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
        let perClusterPtr = UnsafeMutablePointer<UInt64>.allocate(capacity: Self.clusterCount)
        perClusterPtr.initialize(repeating: 0, count: Self.clusterCount)
        defer { perClusterPtr.deinitialize(count: Self.clusterCount); perClusterPtr.deallocate() }
        let prog = computeProgram
        let unis = uniforms
        DispatchQueue.concurrentPerform(iterations: warpCount) { warp in
            let base = warp * ShaderExecutor.warpSize
            let count = min(ShaderExecutor.warpSize, lanes - base)
            if count <= 0 { return }
            var ctx = WarpContext.dummy()
            ctx.uniforms = unis
            ctx.attrProvider = { lane, slot in
                slot == 0 ? (Float(base + lane), 0, 0, 0) : (0, 0, 0, 0)
            }
            ctx.bufLoad = { [self] addr in
                guard addr >= 0, UInt64(addr) + 4 <= bufSize else { return 0 }
                return vramReadU32(at: buffer + UInt64(addr))
            }
            ctx.bufStore = { [self] addr, value in
                guard addr >= 0, UInt64(addr) + 4 <= bufSize else { return }
                vramWriteU32(at: buffer + UInt64(addr), value)
            }
            let ops = ShaderExecutor.runWarp(program: prog, lanes: count, ctx: ctx)
            perClusterPtr[warp % Self.clusterCount] += UInt64(ops)
        }
        stats.launches += 1
        for i in 0..<Self.clusterCount {
            let v = perClusterPtr[i]
            stats.lanesPerCluster[i] += v; stats.laneOps += v
        }
        stats.warpsIssued += UInt64(warpCount)
    }

    // MARK: - 3D pipeline

    private struct ShadedVertex {
        var clip: (Float, Float, Float, Float) = (0, 0, 0, 1)
        var varyings: [[Float]] = [[Float]](repeating: [0, 0, 0, 0], count: 8)  // slots 1..7 used
    }

    /// Clears the framebuffer to an RGBA8888 colour + depth far.
    func clearFramebuffer(rgba: UInt32) throws {
        guard allocationSize(framebuffer) != nil, fbWidth > 0, fbHeight > 0 else { throw GPUError.badState("no framebuffer") }
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
        guard allocationSize(framebuffer) != nil, fbWidth > 0, fbHeight > 0 else { throw GPUError.badState("no framebuffer") }
        let vertexCount = Int(vbSize / 48)
        let idxCount = min(indexCount, Int(ibSize / 4))
        guard idxCount >= 3 else { throw GPUError.badState("draw with < 3 indices") }

        // ---- Vertex stage: one warp per 32 vertices, warps over clusters.
        let shadedPtr = UnsafeMutablePointer<ShadedVertex>.allocate(capacity: max(vertexCount, 1))
        shadedPtr.initialize(repeating: ShadedVertex(), count: max(vertexCount, 1))
        defer { shadedPtr.deinitialize(count: max(vertexCount, 1)); shadedPtr.deallocate() }
        let vb = vertexBuffer
        let vprog = vertexProgram
        let unis = uniforms
        let warpCount = (vertexCount + 31) / 32
        let perClusterVPtr = UnsafeMutablePointer<UInt64>.allocate(capacity: Self.clusterCount)
        perClusterVPtr.initialize(repeating: 0, count: Self.clusterCount)
        defer { perClusterVPtr.deinitialize(count: Self.clusterCount); perClusterVPtr.deallocate() }
        DispatchQueue.concurrentPerform(iterations: warpCount) { warp in
            let base = warp * 32
            let count = min(32, vertexCount - base)
            var outs = [ShadedVertex](repeating: ShadedVertex(), count: 32)
            var ctx = WarpContext.dummy()
            ctx.uniforms = unis
            ctx.attrProvider = { [self] lane, slot in
                let vi = base + lane
                let o = vb + UInt64(vi * 48)
                switch slot {
                case 0: return (vramReadF32(at: o), vramReadF32(at: o + 4), vramReadF32(at: o + 8), 1)
                case 1: return (vramReadF32(at: o + 12), vramReadF32(at: o + 16), vramReadF32(at: o + 20), 0)
                case 2: return (vramReadF32(at: o + 24), vramReadF32(at: o + 28), 0, 0)
                case 3: return (vramReadF32(at: o + 32), vramReadF32(at: o + 36), vramReadF32(at: o + 40), vramReadF32(at: o + 44))
                default: return (0, 0, 0, 0)
                }
            }
            ctx.outWriter = { lane, slot, x, y, z, w in
                if slot == 0 { outs[lane].clip = (x, y, z, w) }
                else if slot >= 1 && slot < 8 { outs[lane].varyings[slot] = [x, y, z, w] }
            }
            let ops = ShaderExecutor.runWarp(program: vprog, lanes: count, ctx: ctx)
            perClusterVPtr[warp % Self.clusterCount] += UInt64(ops)
            for i in 0..<count { shadedPtr[base + i] = outs[i] }
        }
        var shaded = [ShadedVertex](repeating: ShadedVertex(), count: vertexCount)
        for i in 0..<vertexCount { shaded[i] = shadedPtr[i] }
        var perClusterV = [UInt64](repeating: 0, count: Self.clusterCount)
        for i in 0..<Self.clusterCount { perClusterV[i] = perClusterVPtr[i] }

        // ---- Index fetch + primitive assembly.
        var indices = [Int](repeating: 0, count: idxCount)
        for i in 0..<idxCount { indices[i] = Int(vramReadU32(at: indexBuffer + UInt64(i * 4))) }
        stats.trianglesIn += UInt64(idxCount / 3)

        // ---- Clip + viewport + rasterise, in horizontal bands.
        // Bands own disjoint framebuffer rows, so the shared depth
        // buffer (raw pointer) is written without overlap per band.
        let width = fbWidth, height = fbHeight
        // Depth convention: this pipeline's projection does not flip
        // z, so a LARGER NDC z is NEARER the viewer. The buffer clears
        // to -1 (below the [0, 1] clip range) and the test keeps the
        // greater z (see rasterise).
        let depthPtr = UnsafeMutablePointer<Float>.allocate(capacity: width * height)
        depthPtr.initialize(repeating: -1.0, count: width * height)
        defer { depthPtr.deinitialize(count: width * height); depthPtr.deallocate() }
        let bandCount = min(Self.clusterCount, height)
        let perClusterFPtr = UnsafeMutablePointer<UInt64>.allocate(capacity: Self.clusterCount)
        let fragCountsPtr = UnsafeMutablePointer<UInt64>.allocate(capacity: Self.clusterCount)
        let fragWarpsPtr = UnsafeMutablePointer<UInt64>.allocate(capacity: Self.clusterCount)
        for p in [perClusterFPtr, fragCountsPtr, fragWarpsPtr] {
            p.initialize(repeating: 0, count: Self.clusterCount)
        }
        defer {
            for p in [perClusterFPtr, fragCountsPtr, fragWarpsPtr] {
                p.deinitialize(count: Self.clusterCount); p.deallocate()
            }
        }
        let fprog = fragmentProgram
        let fb = framebuffer

        // Clip + fan-triangulate once, into a read-only work list the
        // bands share.
        var work: [[ShadedVertex]] = []
        work.reserveCapacity(idxCount / 3)
        for t in stride(from: 0, to: idxCount - 2, by: 3) {
            let tri = [shaded[indices[t]], shaded[indices[t + 1]], shaded[indices[t + 2]]]
            let clipped = Self.clipTriangle(tri)
            if clipped.count < 3 { continue }
            for fan in 1..<(clipped.count - 1) {
                work.append([clipped[0], clipped[fan], clipped[fan + 1]])
            }
        }
        stats.trianglesOut += UInt64(work.count)

        DispatchQueue.concurrentPerform(iterations: bandCount) { band in
            let y0 = band * height / bandCount
            let y1 = (band + 1) * height / bandCount
            var localOps: UInt64 = 0
            var localFrags: UInt64 = 0
            var localWarps: UInt64 = 0

            for poly in work {
                let r = self.rasterise(poly, width: width, height: height,
                                       y0: y0, y1: y1, depth: depthPtr,
                                       fb: fb, program: fprog, unis: unis)
                localOps += r.ops; localFrags += r.frags; localWarps += r.warps
            }
            perClusterFPtr[band] = localOps
            fragCountsPtr[band] = localFrags
            fragWarpsPtr[band] = localWarps
        }

        var fragWarps: UInt64 = 0
        for i in 0..<Self.clusterCount {
            stats.lanesPerCluster[i] += perClusterV[i] + perClusterFPtr[i]
            stats.laneOps += perClusterV[i] + perClusterFPtr[i]
            stats.fragmentsShaded += fragCountsPtr[i]
            fragWarps += fragWarpsPtr[i]
        }
        stats.warpsIssued += UInt64(warpCount) + fragWarps
        stats.draws += 1
        let ms = Date().timeIntervalSince(t0) * 1000
        stats.lastFrameMs = ms
        return ms
    }

    // Sutherland–Hodgman clipping against the 6 clip planes, in clip
    // space (before the perspective divide), interpolating varyings.
    private static func clipTriangle(_ tri: [ShadedVertex]) -> [ShadedVertex] {
        // Plane test functions: signed distance, keep >= 0.
        let planes: [(ShadedVertex) -> Float] = [
            { $0.clip.3 + $0.clip.0 },   // x >= -w
            { $0.clip.3 - $0.clip.0 },   // x <=  w
            { $0.clip.3 + $0.clip.1 },   // y >= -w
            { $0.clip.3 - $0.clip.1 },   // y <=  w
            { $0.clip.2 },               // z >= 0
            { $0.clip.3 - $0.clip.2 },   // z <= w
        ]
        var poly = tri
        for plane in planes {
            if poly.isEmpty { break }
            var out: [ShadedVertex] = []
            for i in 0..<poly.count {
                let cur = poly[i]
                let prev = poly[(i + poly.count - 1) % poly.count]
                let dc = plane(cur), dp = plane(prev)
                let curIn = dc >= 0, prevIn = dp >= 0
                if curIn != prevIn {
                    let f = dp / (dp - dc)
                    out.append(lerpVertex(prev, cur, f))
                }
                if curIn { out.append(cur) }
            }
            poly = out
        }
        return poly
    }

    private static func lerpVertex(_ a: ShadedVertex, _ b: ShadedVertex, _ f: Float) -> ShadedVertex {
        var v = ShadedVertex()
        v.clip = (a.clip.0 + (b.clip.0 - a.clip.0) * f,
                  a.clip.1 + (b.clip.1 - a.clip.1) * f,
                  a.clip.2 + (b.clip.2 - a.clip.2) * f,
                  a.clip.3 + (b.clip.3 - a.clip.3) * f)
        for s in 0..<8 {
            for c in 0..<4 { v.varyings[s][c] = a.varyings[s][c] + (b.varyings[s][c] - a.varyings[s][c]) * f }
        }
        return v
    }

    /// Rasterises one screen-space triangle into rows [y0, y1).
    /// Fragments are shaded in warps of 32 along each row.
    private func rasterise(_ tri: [ShadedVertex], width: Int, height: Int,
                           y0: Int, y1: Int, depth: UnsafeMutablePointer<Float>,
                           fb: UInt64, program: [UInt32], unis: [Float])
        -> (ops: UInt64, frags: UInt64, warps: UInt64, touched: Bool) {
        // Perspective divide + viewport.
        func screen(_ v: ShadedVertex) -> (x: Float, y: Float, z: Float, invW: Float) {
            let iw = 1 / v.clip.3
            return ((v.clip.0 * iw * 0.5 + 0.5) * Float(width),
                    (0.5 - v.clip.1 * iw * 0.5) * Float(height),
                    v.clip.2 * iw, iw)
        }
        let s0 = screen(tri[0]), s1 = screen(tri[1]), s2 = screen(tri[2])
        let area = (s1.x - s0.x) * (s2.y - s0.y) - (s1.y - s0.y) * (s2.x - s0.x)
        if area == 0 { return (0, 0, 0, false) }
        let minY = max(y0, Int(floorf(min(s0.y, min(s1.y, s2.y)))))
        let maxY = min(y1 - 1, Int(ceilf(max(s0.y, max(s1.y, s2.y)))))
        if minY > maxY { return (0, 0, 0, false) }
        let minX = max(0, Int(floorf(min(s0.x, min(s1.x, s2.x)))))
        let maxX = min(width - 1, Int(ceilf(max(s0.x, max(s1.x, s2.x)))))
        if minX > maxX { return (0, 0, 0, false) }

        var totalOps: UInt64 = 0
        var totalFrags: UInt64 = 0
        var totalWarps: UInt64 = 0
        let invArea = 1 / area

        for y in minY...maxY {
            let py = Float(y) + 0.5
            var x = minX
            while x <= maxX {
                // Build one warp: up to 32 consecutive pixels in this row.
                var lanesPx: [Int] = []
                var lanesVar: [[[Float]]] = []   // per lane: varyings[8][4]
                var cursor = x
                while cursor <= maxX && lanesPx.count < 32 {
                    let px = Float(cursor) + 0.5
                    let e0 = (s2.x - s1.x) * (py - s1.y) - (s2.y - s1.y) * (px - s1.x)
                    let e1 = (s0.x - s2.x) * (py - s2.y) - (s0.y - s2.y) * (px - s2.x)
                    let e2 = (s1.x - s0.x) * (py - s0.y) - (s1.y - s0.y) * (px - s0.x)
                    let inside = (e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)
                    if inside {
                        let w0 = e0 * invArea, w1 = e1 * invArea, w2 = e2 * invArea
                        let z = w0 * s0.z + w1 * s1.z + w2 * s2.z
                        let di = y * width + cursor
                        if z >= 0 && z >= depth[di] {
                            // Perspective-correct varying interpolation.
                            let iw = w0 * s0.invW + w1 * s1.invW + w2 * s2.invW
                            var vary = [[Float]](repeating: [0, 0, 0, 0], count: 8)
                            if iw != 0 {
                                for s in 0..<8 { for c in 0..<4 {
                                    vary[s][c] = (w0 * s0.invW * tri[0].varyings[s][c]
                                                + w1 * s1.invW * tri[1].varyings[s][c]
                                                + w2 * s2.invW * tri[2].varyings[s][c]) / iw
                                } }
                            }
                            depth[di] = z
                            lanesPx.append(di)
                            lanesVar.append(vary)
                        }
                    }
                    cursor += 1
                }
                if !lanesPx.isEmpty {
                    var colours = [(Float, Float, Float, Float)](repeating: (0, 0, 0, 1), count: lanesPx.count)
                    var ctx = WarpContext.dummy()
                    ctx.uniforms = unis
                    ctx.attrProvider = { lane, slot in
                        let v = lanesVar[lane][slot]
                        return (v[0], v[1], v[2], v[3])
                    }
                    ctx.texSampler = { [self] unit, u, v in self.sampleTexture(unit: unit, u: u, v: v) }
                    ctx.outWriter = { lane, slot, r, g, b, a in
                        if slot == 0 { colours[lane] = (r, g, b, a) }
                    }
                    totalOps += UInt64(ShaderExecutor.runWarp(program: program, lanes: lanesPx.count, ctx: ctx))
                    totalFrags += UInt64(lanesPx.count)
                    totalWarps += 1
                    // Lanes hold only the pixels that passed coverage +
                    // depth, so they are not guaranteed consecutive:
                    // write each pixel at its own framebuffer offset.
                    for i in 0..<lanesPx.count {
                        let c = colours[i]
                        let px: [UInt8] = [UInt8(min(max(c.0, 0), 1) * 255),
                                           UInt8(min(max(c.1, 0), 1) * 255),
                                           UInt8(min(max(c.2, 0), 1) * 255),
                                           UInt8(min(max(c.3, 0), 1) * 255)]
                        vramWrite(at: fb + UInt64(lanesPx[i] * 4), bytes: px)
                    }
                }
                x = cursor
            }
        }
        return (totalOps, totalFrags, totalWarps, true)
    }
}
