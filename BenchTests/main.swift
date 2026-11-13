// BenchTests/main.swift — NovaGPU v4 benchmark harness (CI host).
//
// Drives the GPU device directly (no bus) so the per-stage draw
// profile is visible: snapshot / vertex / clip / bin / raster walls
// for the app scene, plus isolation scenes that separate vertex cost
// from fragment cost, a SAXPY compute rate, and a cluster-count
// sweep. Output lines are prefixed BENCH for log scraping. These are
// HOST numbers (macos-26 CI runner, Apple Silicon) — the device
// verdict (iPad Air M3, v3 baseline ~27 FPS) is Sean's to measure.

import Foundation

func bench(_ line: String) { print("BENCH \(line)") }
func ms(_ v: Double) -> String { String(format: "%.3f", v) }

func median(_ xs: [Double]) -> Double {
    if xs.isEmpty { return 0 }
    let s = xs.sorted()
    return s[s.count / 2]
}

func mean(_ xs: [Double]) -> Double {
    xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count)
}

func stdev(_ xs: [Double]) -> Double {
    if xs.count < 2 { return 0 }
    let m = mean(xs)
    return (xs.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(xs.count - 1)).squareRoot()
}

/// Percentile by nearest-rank on the sorted series.
func pct(_ xs: [Double], _ p: Double) -> Double {
    if xs.isEmpty { return 0 }
    let s = xs.sorted()
    return s[min(s.count - 1, Int(p * Double(s.count)))]
}

// Sets the lit-shader uniforms directly on the device (same values
// NovaGPUStandard.setLitUniforms sends over the bus).
func setLit(_ gpu: NovaGPU, mvp: [Float]) {
    for r in 0..<4 {
        gpu.setUniform(index: r, (mvp[r * 4], mvp[r * 4 + 1], mvp[r * 4 + 2], mvp[r * 4 + 3]))
    }
    gpu.setUniform(index: 4, (1, 0, 0, 0))
    gpu.setUniform(index: 5, (0, 1, 0, 0))
    gpu.setUniform(index: 6, (0, 0, 1, 0))
    gpu.setUniform(index: 7, (0, 0, 0, 1))
    gpu.setUniform(index: 8, (0.577, 0.577, 0.577, 0))
    gpu.setUniform(index: 9, (1, 1, 1, 1))
}

func uploadMesh(_ gpu: NovaGPU, _ m: NovaGPUStandard.Mesh) throws -> (UInt64, UInt64) {
    let v = try gpu.alloc(size: UInt64(m.vertices.count * 4))
    let i = try gpu.alloc(size: UInt64(m.indices.count * 4))
    gpu.vramWrite(at: v, bytes: NovaGPUStandard.vertexBytes(m))
    gpu.vramWrite(at: i, bytes: NovaGPUStandard.indexBytes(m))
    return (v, i)
}

/// Runs the app scene (icosphere 5 + torus, 640x480, lit programs)
/// for `frames` frames on a fresh device built with the current
/// NovaGPU.clusterCount; returns per-frame wall + profile medians.
func appScene(frames: Int, vProg: [UInt32], fProg: [UInt32], vpuTest: Bool = false,
              configure: (NovaGPU) -> Void = { _ in }) throws
    -> (frame: Double, prof: DrawProfile, frags: Double) {
    let gpu = NovaGPU()
    configure(gpu)
    let fbW = 640, fbH = 480
    let fb = try gpu.alloc(size: UInt64(fbW * fbH * 4))
    gpu.framebuffer = fb; gpu.fbWidth = fbW; gpu.fbHeight = fbH
    gpu.vertexProgram = vProg
    gpu.fragmentProgram = fProg
    if vpuTest { gpu.armVPUTest(true) }
    let sphere = NovaGPUStandard.icosphere(subdivisions: 5)
    let torus = NovaGPUStandard.torus(rings: 96, slices: 48)
    let (sv, si) = try uploadMesh(gpu, sphere)
    let (tv, ti) = try uploadMesh(gpu, torus)
    let proj = NovaGPUStandard.matPerspective(k: 0.45)
    var frameMs: [Double] = []
    var profs: [DrawProfile] = []
    var fragCounts: [Double] = []
    for f in 0..<frames {
        let angle: Float = 0.8 + Float(f) * 0.02
        let t0 = NovaGPU.nowMs()
        try gpu.clearFramebuffer(rgba: 0xFF200810)
        let mvpS = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
            NovaGPUStandard.matRotX(0.5),
            NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle), NovaGPUStandard.matScale(0.72))))
        setLit(gpu, mvp: mvpS)
        gpu.vertexBuffer = sv; gpu.indexBuffer = si
        let fragsBefore = gpu.stats.fragmentsShaded
        _ = try gpu.draw(indexCount: sphere.indices.count)
        let pS = gpu.lastDrawProfile
        let mvpT = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
            NovaGPUStandard.matTranslate(-0.05, -0.1, 0.35),
            NovaGPUStandard.matMul(NovaGPUStandard.matRotX(1.1),
                                   NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle * 0.6),
                                                          NovaGPUStandard.matScale(0.62)))))
        setLit(gpu, mvp: mvpT)
        gpu.vertexBuffer = tv; gpu.indexBuffer = ti
        _ = try gpu.draw(indexCount: torus.indices.count)
        let pT = gpu.lastDrawProfile
        frameMs.append(NovaGPU.nowMs() - t0)
        var sum = DrawProfile()
        sum.snapshotMs = pS.snapshotMs + pT.snapshotMs
        sum.vertexMs = pS.vertexMs + pT.vertexMs
        sum.clipMs = pS.clipMs + pT.clipMs
        sum.binMs = pS.binMs + pT.binMs
        sum.rasterMs = pS.rasterMs + pT.rasterMs
        sum.fragExecMs = pS.fragExecMs + pT.fragExecMs
        sum.totalMs = pS.totalMs + pT.totalMs
        profs.append(sum)
        fragCounts.append(Double(gpu.stats.fragmentsShaded - fragsBefore))
    }
    if vpuTest {
        let t = gpu.vpuTest
        bench("vputest-result triangles=\(t.trianglesTested) corners=\(t.cornersCompared) " +
              "mismatches=\(t.mismatches) maxErr=\(t.maxAbsError) " +
              "passMs_total=\(ms(t.passMs)) clusters=\(t.clustersEngaged) " +
              "tri_s=\(ms(t.trianglesPerSecond))")
    }
    // Drop the first frame (warm-up: block commits, allocator, caches).
    let p = Array(profs.dropFirst())
    var med = DrawProfile()
    med.snapshotMs = median(p.map { $0.snapshotMs })
    med.vertexMs = median(p.map { $0.vertexMs })
    med.clipMs = median(p.map { $0.clipMs })
    med.binMs = median(p.map { $0.binMs })
    med.rasterMs = median(p.map { $0.rasterMs })
    med.fragExecMs = median(p.map { $0.fragExecMs })
    med.totalMs = median(p.map { $0.totalMs })
    return (median(Array(frameMs.dropFirst())), med, median(Array(fragCounts.dropFirst())))
}

func reportScene(_ name: String, _ r: (frame: Double, prof: DrawProfile, frags: Double)) {
    bench("\(name) frame_ms=\(ms(r.frame)) draws_ms=\(ms(r.prof.totalMs)) " +
          "snapshot=\(ms(r.prof.snapshotMs)) vertex=\(ms(r.prof.vertexMs)) " +
          "clip=\(ms(r.prof.clipMs)) bin=\(ms(r.prof.binMs)) raster=\(ms(r.prof.rasterMs)) " +
          "fragExecCpu=\(ms(r.prof.fragExecMs)) " +
          "fragments=\(Int(r.frags)) host_fps=\(ms(1000 / max(r.frame, 0.001)))")
}

let vsLit = NovaGPUStandard.vertexLitProgram()
let vsLitV4 = NovaGPUStandard.vertexLitProgramV4()
let fsLit = NovaGPUStandard.fragmentLitProgram()

do {
    NovaGPU.profileFlushTiming = true
    // 1. The app scene, as shipped.
    let app = try appScene(frames: 8, vProg: vsLit, fProg: fsLit)
    reportScene("app-scene-lit", app)

    // 1a. Same scene, v4 vertex program (VMAT4): same-run A/B for
    //     the vertex stage against the v1 program above.
    let appV4 = try appScene(frames: 8, vProg: vsLitV4, fProg: fsLit)
    reportScene("app-scene-lit-v4vert", appV4)

    // 1e. v5: the app scene with the VPU test armed — draws render
    //     FROM the VPU positions; this is the frame cost of the
    //     in-scene test (same-run A/B vs app-scene-lit-v4vert).
    let appT = try appScene(frames: 8, vProg: vsLitV4, fProg: fsLit, vpuTest: true)
    reportScene("app-scene-vputest", appT)

    // 1b. Executor floor: fragment program is a bare END (default
    //     colour out) — raster wall then holds coverage +
    //     interpolation + executor fixed overhead only.
    var eb = ShaderBuilder()
    eb.end()
    let nofrag = try appScene(frames: 8, vProg: vsLit, fProg: eb.build())
    reportScene("app-scene-nofrag", nofrag)

    // 1c. Pure executor micro-bench: 20,000 warps of the lit
    //     fragment program against static varying/colour buffers.
    do {
        let decoded = DecodedShader(fsLit)
        let varBuf = UnsafeMutablePointer<Float>.allocate(capacity: 32 * 32)
        let colBuf = UnsafeMutablePointer<Float>.allocate(capacity: 32 * 4)
        for i in 0..<(32 * 32) { varBuf[i] = Float(i % 17) * 0.05 }
        defer { varBuf.deallocate(); colBuf.deallocate() }
        var ctx = WarpContext.dummy()
        ctx.uniforms = [Float](repeating: 0.5, count: 64)
        ctx.attrPtr = UnsafePointer(varBuf)
        ctx.outPtr = colBuf
        let warps = 20000
        let t0 = NovaGPU.nowMs()
        var ops = 0
        for _ in 0..<warps {
            ops += ShaderExecutor.runWarp(decoded: decoded, lanes: 32, ctx: ctx)
        }
        let dt = NovaGPU.nowMs() - t0
        bench("executor-fraglit warps=\(warps) ms=\(ms(dt)) " +
              "us_per_warp=\(ms(dt * 1000 / Double(warps))) laneOps=\(ops)")
    }
    // 1d. Pure executor micro-bench: lit VERTEX program through the
    //     closure attribute path (the path draw() uses today).
    do {
        let decoded = DecodedShader(vsLit)
        let raw = UnsafeMutablePointer<Float>.allocate(capacity: 32 * 12)
        for i in 0..<(32 * 12) { raw[i] = Float(i % 13) * 0.1 - 0.6 }
        defer { raw.deallocate() }
        var ctx = WarpContext.dummy()
        ctx.uniforms = NovaGPUStandard.matIdentity()
            + [Float](repeating: 0, count: 48)
        // c0..c3 = identity rows; c4..c7 masks and c8/c9 set below.
        for r in 0..<4 { for c in 0..<4 { ctx.uniforms[r * 4 + c] = r == c ? 1 : 0 } }
        ctx.uniforms[16] = 1; ctx.uniforms[21] = 1; ctx.uniforms[26] = 1; ctx.uniforms[31] = 1
        ctx.uniforms[32] = 0.577; ctx.uniforms[33] = 0.577; ctx.uniforms[34] = 0.577
        ctx.uniforms[36] = 1; ctx.uniforms[37] = 1; ctx.uniforms[38] = 1; ctx.uniforms[39] = 1
        ctx.attrProvider = { lane, slot in
            let o = lane * 12
            switch slot {
            case 0: return (raw[o], raw[o + 1], raw[o + 2], 1)
            case 1: return (raw[o + 3], raw[o + 4], raw[o + 5], 0)
            case 2: return (raw[o + 6], raw[o + 7], 0, 0)
            case 3: return (raw[o + 8], raw[o + 9], raw[o + 10], raw[o + 11])
            default: return (0, 0, 0, 0)
            }
        }
        var sink: Float = 0
        ctx.outWriter = { _, _, x, _, _, _ in sink += x }
        let warps = 20000
        let t0 = NovaGPU.nowMs()
        var ops = 0
        for _ in 0..<warps {
            ops += ShaderExecutor.runWarp(decoded: decoded, lanes: 32, ctx: ctx)
        }
        let dt = NovaGPU.nowMs() - t0
        bench("executor-vertlit warps=\(warps) ms=\(ms(dt)) " +
              "us_per_warp=\(ms(dt * 1000 / Double(warps))) laneOps=\(ops) sink=\(sink)")
    }

    // 2. Fragment-cost isolation: identical geometry and coverage,
    //    fragment program reduced to pass-through colour.
    var fb = ShaderBuilder()
    fb.ldattr(0, 1)
    fb.out(0, 0)
    fb.end()
    let flat = try appScene(frames: 8, vProg: vsLit, fProg: fb.build())
    reportScene("app-scene-flatfrag", flat)

    // 3. Vertex-cost isolation: the lit vertex program's instruction
    //    count is visible in the vertex wall above; a minimal vertex
    //    program (position passthrough) shows the floor.
    var vb = ShaderBuilder()
    vb.ldattr(0, 0)
    vb.out(0, 0)
    vb.end()
    let vmin = try appScene(frames: 8, vProg: vb.build(), fProg: fsLit)
    reportScene("app-scene-minvert", vmin)

    // 4. SAXPY compute rate (same shape as the host test).
    do {
        let gpu = NovaGPU()
        let n = 1_000_003
        let bufBytes = UInt64(n * 4 * 2)
        let h = try gpu.alloc(size: bufBytes)
        var data = [UInt8](repeating: 0, count: Int(bufBytes))
        for i in 0..<n {
            let x = (Float(i % 97) * 0.25 - 12).bitPattern
            let y = (Float(i % 13) * 0.5).bitPattern
            for k in 0..<4 {
                data[i * 4 + k] = UInt8((x >> UInt32(k * 8)) & 0xFF)
                data[(n + i) * 4 + k] = UInt8((y >> UInt32(k * 8)) & 0xFF)
            }
        }
        gpu.vramWrite(at: h, bytes: data)
        gpu.computeProgram = NovaGPUStandard.saxpyProgram()
        gpu.setUniform(index: 0, (Float(bitPattern: UInt32(n * 4)), 0, 0, 0))
        gpu.setUniform(index: 1, (1.75, 1.75, 1.75, 1.75))
        var times: [Double] = []
        for _ in 0..<5 {
            let t0 = NovaGPU.nowMs()
            try gpu.launchCompute(lanes: n, buffer: h)
            times.append(NovaGPU.nowMs() - t0)
        }
        let med = median(times)
        bench("saxpy lanes=\(n) ms=\(ms(med)) Mlanes_s=\(ms(Double(n) / med / 1000))")
    }

    // 5. Cluster-count sweep on the app scene.
    for clusters in [24, 48, 96] {
        NovaGPU.clusterCount = clusters
        let r = try appScene(frames: 6, vProg: vsLit, fProg: fsLit)
        bench("clusters=\(clusters) cus=\(NovaGPU.computeUnits) frame_ms=\(ms(r.frame)) " +
              "vertex=\(ms(r.prof.vertexMs)) raster=\(ms(r.prof.rasterMs))")
    }
    NovaGPU.clusterCount = 48

    // 6. Plot-unit saturation (v4): random line vectors at rising
    //    counts up to the full MAX_SCREEN_VECTORS budget, showing
    //    where the serial positioning + DDA pass saturates. The
    //    vectors are short random segments in NDC [-0.9, 0.9]^2.
    do {
        let gpu = NovaGPU()
        let fbW = 640, fbH = 480
        let fb = try gpu.alloc(size: UInt64(fbW * fbH * 4))
        gpu.framebuffer = fb; gpu.fbWidth = fbW; gpu.fbHeight = fbH
        var flatB = ShaderBuilder()
        flatB.ldattr(0, 1); flatB.out(0, 0); flatB.end()
        gpu.fragmentProgram = flatB.build()
        let maxV = NovaGPU.maxScreenVectors
        let nv = maxV * 2
        var rng: UInt64 = 0xABCDEF
        func fr() -> Float {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return Float((rng >> 40) & 0xFFFFFF) / Float(0x1000000) * 1.8 - 0.9
        }
        let vbh = try gpu.alloc(size: UInt64(nv * 48))
        var vbytes = [UInt8]()
        vbytes.reserveCapacity(nv * 48)
        for _ in 0..<nv {
            var rec = [Float](repeating: 0, count: 12)
            rec[0] = fr(); rec[1] = fr(); rec[2] = 0.5
            for f in rec { withUnsafeBytes(of: f.bitPattern) { vbytes.append(contentsOf: $0) } }
        }
        gpu.vramWrite(at: vbh, bytes: vbytes)
        let ibh = try gpu.alloc(size: UInt64(maxV * 2 * 4))
        var ibytes = [UInt8]()
        ibytes.reserveCapacity(maxV * 2 * 4)
        for i in 0..<(maxV * 2) { withUnsafeBytes(of: UInt32(i)) { ibytes.append(contentsOf: $0) } }
        gpu.vramWrite(at: ibh, bytes: ibytes)
        gpu.vertexBuffer = vbh; gpu.indexBuffer = ibh
        let ident: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        for r in 0..<4 { gpu.setUniform(index: r, (ident[r * 4], ident[r * 4 + 1], ident[r * 4 + 2], ident[r * 4 + 3])) }
        gpu.setUniform(index: 10, (0.25, 1.0, 0.35, 1.0))
        for count in [1024, 8192, 32768, 65536] {
            var times: [Double] = []
            var p1s: [Double] = []
            var p2s: [Double] = []
            var frags: UInt64 = 0
            for rep in 0..<3 {
                try gpu.clearFramebuffer(rgba: 0xFF0A0A12)
                let fBefore = gpu.stats.fragmentsShaded
                let t = try gpu.plotVectors(indexCount: count * 2, mode: 0)
                times.append(t)
                p1s.append(gpu.lastPlotProfile.pass1Ms)
                p2s.append(gpu.lastPlotProfile.pass2Ms)
                if rep == 2 { frags = gpu.stats.fragmentsShaded - fBefore }
            }
            let med = median(times)
            bench("plot-lines vectors=\(count) ms=\(ms(med)) " +
                  "pass1=\(ms(median(p1s))) pass2=\(ms(median(p2s))) " +
                  "us_per_vector=\(ms(med * 1000 / Double(count))) fragments=\(frags)")
        }

        // 6a. Plot parallel-threshold sweep (v5): the threshold only
        //     gates serial vs MGVPU-cluster-parallel pass 1; find the
        //     crossover by measurement instead of keeping v4's 2,048
        //     by default.
        for count in [1024, 8192, 65536] {
            for th in [256, 512, 1024, 2048, 4096, 8192] {
                NovaGPU.plotParallelThreshold = th
                var times: [Double] = []
                for _ in 0..<3 {
                    try gpu.clearFramebuffer(rgba: 0xFF0A0A12)
                    times.append(try gpu.plotVectors(indexCount: count * 2, mode: 0))
                }
                bench("plot-threshold vectors=\(count) threshold=\(th) ms=\(ms(median(times)))")
            }
        }
        NovaGPU.plotParallelThreshold = 2048
    }

    // 7. VPU micro-bench (v5 baseline): position the app scene's
    //    triangle count through the per-triangle VPU path (grab
    //    unit -> VPU load -> 3-axis calculator), serially and split
    //    over MGVPU sub-ranges. This is the throughput the v5
    //    wide-issue VPU path has to beat.
    do {
        let tris = 29696
        var rng: UInt64 = 0x5EED
        func fr() -> Float {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return Float((rng >> 40) & 0xFFFFFF) / Float(0x1000000) * 2 - 1
        }
        let snap = UnsafeMutablePointer<Float>.allocate(capacity: tris * 36)
        defer { snap.deallocate() }
        for i in 0..<(tris * 36) { snap[i] = fr() }
        let mat: [Float] = [0.8, 0.1, -0.3, 0.5, -0.2, 1.1, 0.4, -0.6,
                            0.9, -0.7, 0.3, 0.25, -0.1, 0.2, 0.45, 1.0]
        let out = UnsafeMutablePointer<Float>.allocate(capacity: tris * 12)
        defer { out.deallocate() }
        func runRange(_ lo: Int, _ hi: Int, _ mPtr: UnsafePointer<Float>) {
            var vpu = VPU()
            withUnsafeTemporaryAllocation(of: Float.self, capacity: 12) { s in
                s.initialize(repeating: 0)
                let sPtr = s.baseAddress!
                for k in lo..<hi {
                    GrabUnit.grabTrianglePositions(snapshot: snap, k * 3, k * 3 + 1, k * 3 + 2, into: sPtr)
                    vpu.load(positions: sPtr)
                    vpu.position(matrix: mPtr, into: out + k * 12)
                }
            }
        }
        var serialMs: [Double] = []
        var parMs: [Double] = []
        for _ in 0..<3 {
            let t0 = NovaGPU.nowMs()
            mat.withUnsafeBufferPointer { mp in runRange(0, tris, mp.baseAddress!) }
            serialMs.append(NovaGPU.nowMs() - t0)
            let ranges = MGVPU.subRanges(count: tris)
            let t1 = NovaGPU.nowMs()
            mat.withUnsafeBufferPointer { mp in
                let mPtr = mp.baseAddress!
                DispatchQueue.concurrentPerform(iterations: ranges.count) { r in
                    runRange(ranges[r].0, ranges[r].1, mPtr)
                }
            }
            parMs.append(NovaGPU.nowMs() - t1)
        }
        bench("vpu-micro tris=\(tris) serial_ms=\(ms(median(serialMs))) " +
              "serial_tri_s=\(ms(Double(tris) / median(serialMs) * 1000)) " +
              "mgvpu_parallel_ms=\(ms(median(parMs))) " +
              "parallel_tri_s=\(ms(Double(tris) / median(parMs) * 1000)) " +
              "first_clip=\(out[0])")

        // Wide issue (v5): the SAME grab + position work, but the
        // calculator is issued in 256-triangle groups (one call
        // per group) instead of one 3-lane call per triangle.
        func wideRange(_ lo: Int, _ hi: Int, _ mPtr: UnsafePointer<Float>) {
            withUnsafeTemporaryAllocation(of: Float.self, capacity: 256 * 12) { gbuf in
                gbuf.initialize(repeating: 0)
                let gPtr = gbuf.baseAddress!
                var g = lo
                while g < hi {
                    let gn = min(256, hi - g)
                    for k in 0..<gn {
                        let kk = g + k
                        GrabUnit.grabTrianglePositions(snapshot: snap, kk * 3, kk * 3 + 1, kk * 3 + 2, into: gPtr + k * 12)
                    }
                    VPU.positionWide(positions: gPtr, vertices: gn * 3, matrix: mPtr, into: out + g * 12)
                    g += gn
                }
            }
        }
        var wideMs: [Double] = []
        var wideParMs: [Double] = []
        for _ in 0..<3 {
            let t0 = NovaGPU.nowMs()
            mat.withUnsafeBufferPointer { mp in wideRange(0, tris, mp.baseAddress!) }
            wideMs.append(NovaGPU.nowMs() - t0)
            let ranges = MGVPU.subRanges(count: tris)
            let t1 = NovaGPU.nowMs()
            mat.withUnsafeBufferPointer { mp in
                let mPtr = mp.baseAddress!
                DispatchQueue.concurrentPerform(iterations: ranges.count) { r in
                    wideRange(ranges[r].0, ranges[r].1, mPtr)
                }
            }
            wideParMs.append(NovaGPU.nowMs() - t1)
        }
        bench("vpu-micro-wide tris=\(tris) serial_ms=\(ms(median(wideMs))) " +
              "serial_tri_s=\(ms(Double(tris) / median(wideMs) * 1000)) " +
              "mgvpu_parallel_ms=\(ms(median(wideParMs))) " +
              "parallel_tri_s=\(ms(Double(tris) / median(wideParMs) * 1000))")
    }
    // 8. v6 knob sweeps on the app scene: raster grain, stage
    //    width cap, clip fan-out. The knobs change partitioning
    //    only — the sweeps exist to put numbers on the governor's
    //    levers before the governor is allowed to touch them.
    for rows in [1, 2, 4, 8] {
        let r = try appScene(frames: 6, vProg: vsLitV4, fProg: fsLit) { g in
            g.rasterChunkRows = rows
        }
        bench("sweep-grain rows=\(rows) frame_ms=\(ms(r.frame)) raster=\(ms(r.prof.rasterMs))")
    }
    for w in [0, 3, 6, 12] {
        let r = try appScene(frames: 6, vProg: vsLitV4, fProg: fsLit) { g in
            g.stageWidth = w
        }
        bench("sweep-width width=\(w) frame_ms=\(ms(r.frame)) raster=\(ms(r.prof.rasterMs))")
    }
    for rc in [8, 32] {
        let r = try appScene(frames: 6, vProg: vsLitV4, fProg: fsLit) { g in
            g.clipRangeCount = rc
        }
        bench("sweep-clipranges ranges=\(rc) frame_ms=\(ms(r.frame)) clip=\(ms(r.prof.clipMs)) bin=\(ms(r.prof.binMs))")
    }
    // v6 attempt 3 A/B: retained draw scratch on (v6) vs off
    // (v5 allocate-per-draw), same scene, same run.
    for retain in [true, false] {
        let r = try appScene(frames: 6, vProg: vsLitV4, fProg: fsLit) { g in
            g.scratchRetentionEnabled = retain
        }
        bench("sweep-scratch retain=\(retain) frame_ms=\(ms(r.frame)) snapshot=\(ms(r.prof.snapshotMs)) raster=\(ms(r.prof.rasterMs))")
    }

    // 9. v6 jitter profile: the Live Scene's exact bus shape —
    //    one submitBatch (clear + 2 draws + present) and one
    //    1.2 MB download per frame, over the bus client on a
    //    128 MB guest, 90 frames. This is the series the device
    //    FPS instability has to be explained against: mean AND
    //    spread, not just the median the scene reports print.
    do {
        let mem = NovaMemory(totalSize: 128 * 1024 * 1024)
        let device = NovaGPUDevice()
        let client = NovaBusClient(memory: mem, device: device)
        let fbW = 640, fbH = 480
        let STAGE: UInt64 = 0x40000, DSTAGE: UInt64 = 0x80000
        let sphere = NovaGPUStandard.icosphere(subdivisions: 5)
        let torus = NovaGPUStandard.torus(rings: 96, slices: 48)
        let fb = try client.alloc(UInt64(fbW * fbH * 4))
        let sh = try client.alloc(8192)
        try client.uploadData(sh, data: NovaGPUStandard.programBytes(vsLitV4), guestStaging: STAGE)
        try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fsLit), guestStaging: STAGE)
        try client.setShader(.setVShader, programHandle: sh, off: 0, len: vsLitV4.count * 4)
        try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fsLit.count * 4)
        try client.setFBuf(fb, fbW, fbH)
        try client.setUniform(4, (1, 0, 0, 0))
        try client.setUniform(5, (0, 1, 0, 0))
        try client.setUniform(6, (0, 0, 1, 0))
        try client.setUniform(7, (0, 0, 0, 1))
        try client.setUniform(8, (0.577, 0.577, 0.577, 0))
        try client.setUniform(9, (1, 1, 1, 1))
        let sphereV = try client.alloc(UInt64(sphere.vertices.count * 4))
        let sphereI = try client.alloc(UInt64(sphere.indices.count * 4))
        try client.uploadData(sphereV, data: NovaGPUStandard.vertexBytes(sphere), guestStaging: STAGE)
        try client.uploadData(sphereI, data: NovaGPUStandard.indexBytes(sphere), guestStaging: STAGE)
        let torusV = try client.alloc(UInt64(torus.vertices.count * 4))
        let torusI = try client.alloc(UInt64(torus.indices.count * 4))
        try client.uploadData(torusV, data: NovaGPUStandard.vertexBytes(torus), guestStaging: STAGE)
        try client.uploadData(torusI, data: NovaGPUStandard.indexBytes(torus), guestStaging: STAGE)
        let proj = NovaGPUStandard.matPerspective(k: 0.45)
        var series: [Double] = []
        var angle: Float = 0
        for _ in 0..<90 {
            angle += 0.02
            let mvpS = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
                NovaGPUStandard.matRotX(0.5),
                NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle), NovaGPUStandard.matScale(0.72))))
            let mvpT = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
                NovaGPUStandard.matTranslate(-0.05, -0.1, 0.35),
                NovaGPUStandard.matMul(NovaGPUStandard.matRotX(1.1),
                                       NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle * 0.6),
                                                              NovaGPUStandard.matScale(0.62)))))
            var batch: [(opcode: GPUCommand, fence: UInt64, args: [UInt64])] = []
            batch.append((opcode: .clear, fence: 0, args: [UInt64(0xFF200810), 0, 0, 0, 0]))
            for (mvp, vb, ib, idx) in [(mvpS, sphereV, sphereI, sphere.indices.count),
                                       (mvpT, torusV, torusI, torus.indices.count)] {
                for r in 0..<4 {
                    batch.append((opcode: .setUniform, fence: 0,
                                  args: [UInt64(r), UInt64(mvp[r * 4].bitPattern),
                                         UInt64(mvp[r * 4 + 1].bitPattern), UInt64(mvp[r * 4 + 2].bitPattern),
                                         UInt64(mvp[r * 4 + 3].bitPattern)]))
                }
                batch.append((opcode: .setVBuf, fence: 0, args: [vb, 0, 0, 0, 0]))
                batch.append((opcode: .setIBuf, fence: 0, args: [ib, 0, 0, 0, 0]))
                batch.append((opcode: .draw, fence: 0, args: [UInt64(idx), 0, 0, 0, 0]))
            }
            batch.append((opcode: .present, fence: 0, args: [fb, 0, 0, 0, 0]))
            let t0 = NovaGPU.nowMs()
            try client.submitBatch(batch)
            _ = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
            series.append(NovaGPU.nowMs() - t0)
        }
        let s = Array(series.dropFirst(5))
        bench("engine-series frames=\(s.count) mean_ms=\(ms(mean(s))) stdev_ms=\(ms(stdev(s))) " +
              "p50_ms=\(ms(pct(s, 0.50))) p95_ms=\(ms(pct(s, 0.95))) max_ms=\(ms(s.max() ?? 0)) " +
              "min_ms=\(ms(s.min() ?? 0)) fps_at_mean=\(ms(1000 / max(mean(s), 0.001)))")
    }

    // 10. v6 governor A/B: the section-9 loop with the supervisor
    //     chip attached, enabled over the ring, and the app's
    //     deadline pacing applied (frame starts held to the chip's
    //     pace tier, exactly GPUEngine's algorithm). The series
    //     recorded is the frame INTERVAL (start to start — what a
    //     viewer experiences as FPS stability); section 9's
    //     back-to-back loop is the unpaced control, where the
    //     interval equals the work time.
    do {
        let mem = NovaMemory(totalSize: 128 * 1024 * 1024)
        let device = NovaGPUDevice()
        let client = NovaBusClient(memory: mem, device: device)
        let gov = NovaGovernorDevice()
        gov.attach(to: mem)
        gov.supervise(device)
        device.governor = gov
        let fbW = 640, fbH = 480
        let STAGE: UInt64 = 0x40000, DSTAGE: UInt64 = 0x80000
        let sphere = NovaGPUStandard.icosphere(subdivisions: 5)
        let torus = NovaGPUStandard.torus(rings: 96, slices: 48)
        let fb = try client.alloc(UInt64(fbW * fbH * 4))
        let sh = try client.alloc(8192)
        try client.uploadData(sh, data: NovaGPUStandard.programBytes(vsLitV4), guestStaging: STAGE)
        try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fsLit), guestStaging: STAGE)
        try client.setShader(.setVShader, programHandle: sh, off: 0, len: vsLitV4.count * 4)
        try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fsLit.count * 4)
        try client.setFBuf(fb, fbW, fbH)
        try client.setUniform(4, (1, 0, 0, 0))
        try client.setUniform(5, (0, 1, 0, 0))
        try client.setUniform(6, (0, 0, 1, 0))
        try client.setUniform(7, (0, 0, 0, 1))
        try client.setUniform(8, (0.577, 0.577, 0.577, 0))
        try client.setUniform(9, (1, 1, 1, 1))
        let sphereV = try client.alloc(UInt64(sphere.vertices.count * 4))
        let sphereI = try client.alloc(UInt64(sphere.indices.count * 4))
        try client.uploadData(sphereV, data: NovaGPUStandard.vertexBytes(sphere), guestStaging: STAGE)
        try client.uploadData(sphereI, data: NovaGPUStandard.indexBytes(sphere), guestStaging: STAGE)
        let torusV = try client.alloc(UInt64(torus.vertices.count * 4))
        let torusI = try client.alloc(UInt64(torus.indices.count * 4))
        try client.uploadData(torusV, data: NovaGPUStandard.vertexBytes(torus), guestStaging: STAGE)
        try client.uploadData(torusI, data: NovaGPUStandard.indexBytes(torus), guestStaging: STAGE)
        try client.governorControl(1)   // enable over the ring, as the app does
        let proj = NovaGPUStandard.matPerspective(k: 0.45)
        var intervals: [Double] = []
        var works: [Double] = []
        var angle: Float = 0
        var paceNextStart: Double? = nil   // ms on the NovaGPU.nowMs() clock
        var lastStart: Double? = nil
        for _ in 0..<90 {
            // GPUEngine's pacing: hold the frame start to the
            // chip's current pace tier; late frames start fresh.
            let interval = gov.chip.paceTargetMs
            let now0 = NovaGPU.nowMs()
            if let next = paceNextStart, now0 < next {
                governorWait(untilMs: next)
            }
            let start = NovaGPU.nowMs()
            paceNextStart = max(start, paceNextStart ?? start) + interval
            if let ls = lastStart { intervals.append(start - ls) }
            lastStart = start
            angle += 0.02
            let mvpS = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
                NovaGPUStandard.matRotX(0.5),
                NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle), NovaGPUStandard.matScale(0.72))))
            let mvpT = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
                NovaGPUStandard.matTranslate(-0.05, -0.1, 0.35),
                NovaGPUStandard.matMul(NovaGPUStandard.matRotX(1.1),
                                       NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle * 0.6),
                                                              NovaGPUStandard.matScale(0.62)))))
            var batch: [(opcode: GPUCommand, fence: UInt64, args: [UInt64])] = []
            batch.append((opcode: .clear, fence: 0, args: [UInt64(0xFF200810), 0, 0, 0, 0]))
            for (mvp, vb, ib, idx) in [(mvpS, sphereV, sphereI, sphere.indices.count),
                                       (mvpT, torusV, torusI, torus.indices.count)] {
                for r in 0..<4 {
                    batch.append((opcode: .setUniform, fence: 0,
                                  args: [UInt64(r), UInt64(mvp[r * 4].bitPattern),
                                         UInt64(mvp[r * 4 + 1].bitPattern), UInt64(mvp[r * 4 + 2].bitPattern),
                                         UInt64(mvp[r * 4 + 3].bitPattern)]))
                }
                batch.append((opcode: .setVBuf, fence: 0, args: [vb, 0, 0, 0, 0]))
                batch.append((opcode: .setIBuf, fence: 0, args: [ib, 0, 0, 0, 0]))
                batch.append((opcode: .draw, fence: 0, args: [UInt64(idx), 0, 0, 0, 0]))
            }
            batch.append((opcode: .present, fence: 0, args: [fb, 0, 0, 0, 0]))
            let t0 = NovaGPU.nowMs()
            try client.submitBatch(batch)
            _ = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
            works.append(NovaGPU.nowMs() - t0)
        }
        let iv = Array(intervals.dropFirst(5))
        let wk = Array(works.dropFirst(5))
        bench("governor-series frames=\(iv.count) interval_mean_ms=\(ms(mean(iv))) interval_stdev_ms=\(ms(stdev(iv))) " +
              "p50_ms=\(ms(pct(iv, 0.50))) p95_ms=\(ms(pct(iv, 0.95))) max_ms=\(ms(iv.max() ?? 0)) min_ms=\(ms(iv.min() ?? 0)) " +
              "work_mean_ms=\(ms(mean(wk))) work_stdev_ms=\(ms(stdev(wk)))")
        // The chip needs its 24-frame window + cooldown to settle;
        // the tail of the series is the governed steady state.
        let settled = Array(intervals.dropFirst(45))
        bench("governor-settled frames=\(settled.count) interval_mean_ms=\(ms(mean(settled))) interval_stdev_ms=\(ms(stdev(settled))) " +
              "p50_ms=\(ms(pct(settled, 0.50))) p95_ms=\(ms(pct(settled, 0.95))) max_ms=\(ms(settled.max() ?? 0)) min_ms=\(ms(settled.min() ?? 0))")
        bench("governor-state pace_ms=\(ms(gov.chip.paceTargetMs)) width=\(gov.chip.gpuStageWidth) " +
              "grain=\(gov.chip.rasterChunkRows) actions=\(gov.chip.actionsIssued) ticks=\(gov.chip.tickCount)")
        for a in gov.chip.actionLog { bench("governor-action \(a.text)") }
    }
} catch {
    print("BENCH FAILED: \(error)")
    exit(1)
}
print("BENCH DONE")
