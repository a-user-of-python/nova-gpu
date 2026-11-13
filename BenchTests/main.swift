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
    med.totalMs = median(p.map { $0.totalMs })
    return (median(Array(frameMs.dropFirst())), med, median(Array(fragCounts.dropFirst())))
}

func reportScene(_ name: String, _ r: (frame: Double, prof: DrawProfile, frags: Double)) {
    bench("\(name) frame_ms=\(ms(r.frame)) draws_ms=\(ms(r.prof.totalMs)) " +
          "snapshot=\(ms(r.prof.snapshotMs)) vertex=\(ms(r.prof.vertexMs)) " +
          "clip=\(ms(r.prof.clipMs)) bin=\(ms(r.prof.binMs)) raster=\(ms(r.prof.rasterMs)) " +
          "fragments=\(Int(r.frags)) host_fps=\(ms(1000 / max(r.frame, 0.001)))")
}

let vsLit = NovaGPUStandard.vertexLitProgram()
let fsLit = NovaGPUStandard.fragmentLitProgram()

do {
    // 1. The app scene, as shipped.
    let app = try appScene(frames: 8, vProg: vsLit, fProg: fsLit)
    reportScene("app-scene-lit", app)

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
} catch {
    print("BENCH FAILED: \(error)")
    exit(1)
}
print("BENCH DONE")
