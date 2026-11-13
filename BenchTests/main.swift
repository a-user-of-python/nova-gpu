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
func appScene(frames: Int, vProg: [UInt32], fProg: [UInt32]) throws
    -> (frame: Double, prof: DrawProfile, frags: Double) {
    let gpu = NovaGPU()
    let fbW = 640, fbH = 480
    let fb = try gpu.alloc(size: UInt64(fbW * fbH * 4))
    gpu.framebuffer = fb; gpu.fbWidth = fbW; gpu.fbHeight = fbH
    gpu.vertexProgram = vProg
    gpu.fragmentProgram = fProg
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
            var frags: UInt64 = 0
            for rep in 0..<3 {
                try gpu.clearFramebuffer(rgba: 0xFF0A0A12)
                let fBefore = gpu.stats.fragmentsShaded
                let t = try gpu.plotVectors(indexCount: count * 2, mode: 0)
                times.append(t)
                if rep == 2 { frags = gpu.stats.fragmentsShaded - fBefore }
            }
            let med = median(times)
            bench("plot-lines vectors=\(count) ms=\(ms(med)) " +
                  "us_per_vector=\(ms(med * 1000 / Double(count))) fragments=\(frags)")
        }
    }
} catch {
    print("BENCH FAILED: \(error)")
    exit(1)
}
print("BENCH DONE")
