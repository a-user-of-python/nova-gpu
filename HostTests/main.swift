// main.swift — NovaGPU host self-tests, run on CI (macos-26).
// Everything talks to the GPU over NovaBus (descriptors in guest RAM +
// MMIO doorbell) exactly as a NovaCPU driver would. Statistics are
// read from the device object by the harness only (that is test
// instrumentation, not the app path).

import Foundation

var failures = 0
func check(_ cond: Bool, _ name: String, _ detail: String = "") {
    print("\(cond ? "PASS" : "FAIL")  \(name)\(detail.isEmpty ? "" : " — " + detail)")
    if !cond { failures += 1 }
}
func note(_ s: String) { print("  .. \(s)") }

let mem = NovaMemory(totalSize: 64 * 1024 * 1024)
let device = NovaGPUDevice()
let client = NovaBusClient(memory: mem, device: device)
let STAGE: UInt64 = 0x40000          // guest staging for uploads
let DSTAGE: UInt64 = 0x80000         // guest staging for downloads
let base = NovaMemory.gpuBaseAddr

// MARK: 1. Device identity + registers
let id = mem.mmioRead(base + GPURegister.id.rawValue)
check(id == 0x4E47_5055_0001_0000, "device ID register", String(format: "0x%016X", id))
check(NovaGPU.computeUnits == 1536, "1536 compute units",
      "\(NovaGPU.clusterCount) clusters x \(NovaGPU.lanesPerCluster) lanes")
check(NovaGPU.vramSize == 2 * 1024 * 1024 * 1024, "2 GiB VRAM space")

// MARK: 2. Compute correctness: SAXPY y = a*x + y over 1,000,003 lanes
do {
    let n = 1_000_003
    let alpha: Float = 1.75
    let bufBytes = n * 4 * 2
    let h = try client.alloc(UInt64(bufBytes))
    var data = [UInt8](repeating: 0, count: bufBytes)
    func setF(_ idx: Int, _ v: Float) {
        let b = v.bitPattern
        data[idx * 4] = UInt8(b & 0xFF); data[idx * 4 + 1] = UInt8((b >> 8) & 0xFF)
        data[idx * 4 + 2] = UInt8((b >> 16) & 0xFF); data[idx * 4 + 3] = UInt8((b >> 24) & 0xFF)
    }
    func getF(_ idx: Int) -> Float {
        let b = UInt32(data[idx * 4]) | (UInt32(data[idx * 4 + 1]) << 8)
            | (UInt32(data[idx * 4 + 2]) << 16) | (UInt32(data[idx * 4 + 3]) << 24)
        return Float(bitPattern: b)
    }
    for i in 0..<n {
        setF(i, Float(i % 97) * 0.25 - 12)
        setF(n + i, Float(i % 13) * 0.5)
    }
    try client.uploadData(h, data: data, guestStaging: STAGE)
    let prog = NovaGPUStandard.saxpyProgram()
    let ph = try client.alloc(UInt64(prog.count * 4))
    try client.uploadData(ph, data: NovaGPUStandard.programBytes(prog), guestStaging: STAGE)
    try client.setShader(.setCompute, programHandle: ph, off: 0, len: prog.count * 4)
    try client.setUniform(0, (Float(bitPattern: UInt32(n * 4)), 0, 0, 0))
    try client.setUniform(1, (alpha, alpha, alpha, alpha))
    let t0 = Date()
    try client.launch(lanes: n, buffer: h)
    let dt = Date().timeIntervalSince(t0)
    let back = try client.downloadData(h, len: bufBytes, guestStaging: DSTAGE)
    data = back
    var maxErr: Float = 0
    for i in 0..<n {
        let expect = alpha * (Float(i % 97) * 0.25 - 12) + Float(i % 13) * 0.5
        maxErr = max(maxErr, abs(getF(n + i) - expect))
    }
    check(maxErr < 0.01, "SAXPY 1,000,003 lanes matches CPU reference",
          "max abs err \(maxErr)")
    note(String(format: "SAXPY wall time %.3f s -> %.1f M lanes/s (CI host)",
                dt, Double(n) / dt / 1e6))
    check(device.gpu.stats.laneOps > UInt64(n), "lane-op counter advanced",
          "\(device.gpu.stats.laneOps) lane-ops")
    try client.free(h); try client.free(ph)
} catch {
    check(false, "SAXPY suite", "\(error)")
}

// MARK: 3. Memory: large allocation, fill, spot verify, sparse addressing
do {
    let big = try client.alloc(512 * 1024 * 1024)
    let t0 = Date()
    try client.fill(big, off: 0, len: 512 * 1024 * 1024, pattern: 0xA5A5A5A5)
    let dt = Date().timeIntervalSince(t0)
    var fillOK = true
    for off in [0, 256 * 1024 * 1024, 512 * 1024 * 1024 - 4096] {
        let w = try client.downloadData(big, off: UInt64(off), len: 4096, guestStaging: DSTAGE)
        if w.count != 4096 || !w.allSatisfy({ $0 == 0xA5 }) { fillOK = false }
    }
    check(fillOK, "512 MiB alloc + FILL + spot verify",
          String(format: "fill took %.2f s (CI host)", dt))
    try client.free(big)

    let sparse = try client.alloc(1536 * 1024 * 1024)   // 1.5 GiB reservation
    let before = device.gpu.vramCommitted
    let tailBytes: [UInt8] = [0xDE, 0xAD, 0xBE, 0xEF]
    try client.uploadData(sparse, off: 1536 * 1024 * 1024 - 4, data: tailBytes, guestStaging: STAGE)
    let rb = try client.downloadData(sparse, off: 1536 * 1024 * 1024 - 4, len: 4, guestStaging: DSTAGE)
    check(rb == tailBytes, "write/read at 1.5 GiB offset (sparse addressing)")
    note("reserved \(device.gpu.vramReserved / 1048576) MiB, committed after end-write \(device.gpu.vramCommitted / 1048576) MiB (was \(before / 1048576))")
    check(device.gpu.vramCommitted < 64 * 1024 * 1024,
          "sparse commit stays small", "\(device.gpu.vramCommitted) bytes committed")
    try client.free(sparse)
    let oom = try? client.alloc(3 * 1024 * 1024 * 1024)
    check(oom == nil, "3 GiB request fails cleanly (over the 2 GiB space)")
} catch {
    check(false, "memory suite", "\(error)")
}

// MARK: 4. Error behaviour: unknown opcode sets ERROR + IRQ, CTRL clears
do {
    // Fresh memory + device so the ring head starts at index 0.
    let mem2 = NovaMemory(totalSize: 64 * 1024 * 1024)
    let dev2 = NovaGPUDevice()
    dev2.attach(to: mem2)
    mem2.mmioWrite(base + GPURegister.ringBase.rawValue, 0x20000)
    mem2.mmioWrite(base + GPURegister.ringSlots.rawValue, 64)
    var desc = GPUDescriptor(opcode: 222, fence: 0, args: [0, 0, 0, 0, 0], status: 0)
    mem2.storeHostBytes(desc.packed(), at: 0x20000)
    mem2.mmioWrite(base + GPURegister.doorbell.rawValue, 1)
    dev2.servicePendingWork()   // hosts drain the device at service points
    let bytes = mem2.loadHostBytes(from: 0x20000, count: GPUDescriptor.byteSize)
    desc = GPUDescriptor.unpack(bytes)
    check(desc.status == GPUDescriptorStatus.error.rawValue, "bad opcode -> descriptor error status")
    check(mem2.mmioRead(base + GPURegister.error.rawValue) == GPUErrorCode.unknownCommand.rawValue,
          "ERROR register reports unknownCommand")
    check(mem2.mmioRead(base + GPURegister.irqStatus.rawValue) & 2 != 0, "error IRQ bit raised")
    mem2.mmioWrite(base + GPURegister.ctrl.rawValue, 2)
    check(mem2.mmioRead(base + GPURegister.error.rawValue) == 0, "CTRL clear-error works")
}

// MARK: 5. Bus round-trip timing (NOP descriptors)
do {
    let t0 = Date()
    let rounds = 500
    for _ in 0..<rounds { _ = try client.submit(opcode: .nop) }
    let dt = Date().timeIntervalSince(t0)
    note(String(format: "bus round trip %.1f us avg over %d NOP submits (CI host)",
                dt / Double(rounds) * 1e6, rounds))
    check(dt / Double(rounds) < 0.01, "round trips complete promptly")
} catch {
    check(false, "round-trip suite", "\(error)")
}

// MARK: 6. Two-way service channel from the Swift side (ASSET request)
do {
    let cube = NovaGPUStandard.proofCube()
    let fbW = 256, fbH = 192
    let fb = try client.alloc(UInt64(fbW * fbH * 4))
    let vb = try client.alloc(UInt64(cube.vertices.count * 4))
    let ib = try client.alloc(UInt64(cube.indices.count * 4))
    let vs = NovaGPUStandard.vertexLitProgram()
    let fs = NovaGPUStandard.fragmentLitProgram()
    let sh = try client.alloc(8192)
    try client.uploadData(sh, data: NovaGPUStandard.programBytes(vs), guestStaging: STAGE)
    try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fs), guestStaging: STAGE)
    try client.setShader(.setVShader, programHandle: sh, off: 0, len: vs.count * 4)
    try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fs.count * 4)
    try client.uploadData(ib, data: NovaGPUStandard.indexBytes(cube), guestStaging: STAGE)
    try client.setFBuf(fb, fbW, fbH)
    try client.setVBuf(vb)
    try client.setIBuf(ib)
    let mvp: [Float] = [0.8, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 1]
    try NovaGPUStandard.setLitUniforms(client, mvp: mvp, light: (0.5, 1.0, 0.25))
    try client.clear(0xFF200810)
    var serviced = false
    client.serviceHandler = { words in
        // words: [seq, type, staging addr, length, target handle, fence, 0, 0]
        mem.storeHostBytes(NovaGPUStandard.vertexBytes(cube), at: words[2])
        serviced = true
    }
    try client.draw(indexCount: cube.indices.count, flags: 1,
                    assetStaging: 0x30000, assetLen: UInt64(cube.vertices.count * 4),
                    assetTarget: vb)
    check(serviced, "GPU raised ASSET service request and CPU serviced it")
    try client.present(fb)
    // DEBUG: verify the service DMA actually landed the cube bytes.
    let vbBack = try client.downloadData(vb, len: cube.vertices.count * 4, guestStaging: DSTAGE)
    let vbExpect = NovaGPUStandard.vertexBytes(cube)
    note("vertex buffer after service DMA matches cube data: \(vbBack == vbExpect)")
    if vbBack != vbExpect {
        note("first 16 bytes got \(Array(vbBack[0..<16])) want \(Array(vbExpect[0..<16]))")
    }
    let px = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
    let nonClear = px.filter { $0 != 0 }.count
    note("framebuffer non-zero bytes after serviced draw: \(nonClear)")
    check(nonClear > 10000, "serviced draw actually rendered the cube")
    var counts: [UInt32: Int] = [:]
    for i in stride(from: 0, to: px.count, by: 4) {
        let v = UInt32(px[i]) | (UInt32(px[i + 1]) << 8) | (UInt32(px[i + 2]) << 16) | (UInt32(px[i + 3]) << 24)
        counts[v, default: 0] += 1
    }
    note("distinct RGBA8888 values in cube frame: \(counts.count)")
    for (v, c) in counts.sorted(by: { $0.value > $1.value }).prefix(6) {
        note(String(format: "  colour 0x%08X x %d", v, c))
    }
    // ASCII map: 64x24 cells, centre-sampled.
    for gy in 0..<24 {
        var rowStr = ""
        for gx in 0..<64 {
            let sx = gx * fbW / 64 + fbW / 128, sy = gy * fbH / 24 + fbH / 48
            let o = (sy * fbW + sx) * 4
            let v = UInt32(px[o]) | (UInt32(px[o + 1]) << 8) | (UInt32(px[o + 2]) << 16) | (UInt32(px[o + 3]) << 24)
            rowStr += v == 0xFF200810 ? "." : (v == 0 ? " " : "#")
        }
        note(rowStr)
    }
    // A flat-shaded cube from a fixed angle shows exactly one face.
    // The strong check is the exact pixel value: front face colour
    // (1, 0.25, 0.2), normal rotated to (0, 0, 0.5) by the proof MVP,
    // light (0.5, 1, 0.25): shade = 0.25 + 0.75 * 0.125 = 0.34375.
    let shade: Float = 0.25 + 0.75 * 0.125
    func byte(_ v: Float) -> UInt32 { UInt32(min(max(v, 0), 1) * 255) }
    let expectFace = byte(1.0 * shade) | (byte(0.25 * shade) << 8) | (byte(0.2 * shade) << 16) | (0xFF << 24)
    let faceCount = counts[expectFace] ?? 0
    check(counts.count == 2 && faceCount > 10000,
          "cube face renders at the exact expected shaded colour",
          String(format: "expected 0x%08X x %d px", expectFace, faceCount))
} catch {
    check(false, "service-channel suite", "\(error)")
}

// MARK: 7. Full scene: icosphere + torus, colour depth, CU activity, timing
do {
    let sphere = NovaGPUStandard.icosphere(subdivisions: 5)   // 20,480 tris
    let torus = NovaGPUStandard.torus(rings: 96, slices: 48)  // 9,216 tris
    let fbW = 640, fbH = 480
    let fb = try client.alloc(UInt64(fbW * fbH * 4))
    let vs = NovaGPUStandard.vertexLitProgram()
    let fs = NovaGPUStandard.fragmentLitProgram()
    let sh = try client.alloc(8192)
    try client.uploadData(sh, data: NovaGPUStandard.programBytes(vs), guestStaging: STAGE)
    try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fs), guestStaging: STAGE)
    try client.setShader(.setVShader, programHandle: sh, off: 0, len: vs.count * 4)
    try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fs.count * 4)
    try client.setFBuf(fb, fbW, fbH)

    func uploadMesh(_ m: NovaGPUStandard.Mesh) throws -> (UInt64, UInt64) {
        let v = try client.alloc(UInt64(m.vertices.count * 4))
        let i = try client.alloc(UInt64(m.indices.count * 4))
        try client.uploadData(v, data: NovaGPUStandard.vertexBytes(m), guestStaging: STAGE)
        try client.uploadData(i, data: NovaGPUStandard.indexBytes(m), guestStaging: STAGE)
        return (v, i)
    }
    let (sv, si) = try uploadMesh(sphere)
    let (tv, ti) = try uploadMesh(torus)

    let proj = NovaGPUStandard.matPerspective(k: 0.45)
    let t0 = Date()
    try client.clear(0xFF200810)
    var mvp = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
        NovaGPUStandard.matRotX(0.5),
        NovaGPUStandard.matMul(NovaGPUStandard.matRotY(0.8), NovaGPUStandard.matScale(0.72))))
    try NovaGPUStandard.setLitUniforms(client, mvp: mvp)
    try client.setVBuf(sv); try client.setIBuf(si)
    try client.draw(indexCount: sphere.indices.count)
    mvp = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
        NovaGPUStandard.matTranslate(-0.05, -0.1, 0.35),
        NovaGPUStandard.matMul(NovaGPUStandard.matRotX(1.1),
                               NovaGPUStandard.matMul(NovaGPUStandard.matRotY(0.4),
                                                      NovaGPUStandard.matScale(0.62)))))
    try NovaGPUStandard.setLitUniforms(client, mvp: mvp)
    try client.setVBuf(tv); try client.setIBuf(ti)
    try client.draw(indexCount: torus.indices.count)
    try client.present(fb)
    let frameMs = Date().timeIntervalSince(t0) * 1000

    let px = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
    var distinct = Set<UInt32>()
    for i in stride(from: 0, to: px.count, by: 4) {
        distinct.insert(UInt32(px[i]) | (UInt32(px[i + 1]) << 8) | (UInt32(px[i + 2]) << 16) | (UInt32(px[i + 3]) << 24))
    }
    let totalTris = sphere.triangleCount + torus.triangleCount
    let st = device.gpu.stats
    note("triangles: \(totalTris) (in \(st.trianglesIn), out \(st.trianglesOut)), fragments shaded \(st.fragmentsShaded)")
    note(String(format: "frame time %.1f ms (CI host), lane-ops this scene counted in stats", frameMs))
    note("VRAM reserved \(device.gpu.vramReserved / 1024) KiB, committed \(device.gpu.vramCommitted / 1024) KiB")
    check(totalTris > 20000, "complex scene: tens of thousands of triangles", "\(totalTris)")
    check(distinct.count > 1024, "COLOUR DEPTH: distinct RGBA8888 colours exceed 1024",
          "\(distinct.count) distinct colours")
    check(st.activeComputeUnits == 1536, "all 1536 CUs saw work in the scene",
          "\(st.activeComputeUnits) active")
    check(st.fragmentsShaded > 100000, "fragment shading at scale", "\(st.fragmentsShaded) fragments")
} catch {
    check(false, "scene suite", "\(error)")
}

// MARK: 8. v4 vector ops (VCROSS / VNORM / VMAT4) through the executor
do {
    // Runs a program over `lanes` lanes; attrs[lane][slot] feeds
    // LDATTR, outputs are captured per (lane, slot).
    func runProg(_ prog: [UInt32], lanes: Int,
                 attrs: [[(Float, Float, Float, Float)]],
                 uniforms: [Float])
        -> [[(Float, Float, Float, Float)]] {
        var outs = [[(Float, Float, Float, Float)]](repeating: [], count: lanes)
        var ctx = WarpContext.dummy()
        ctx.uniforms = uniforms
        ctx.attrProvider = { lane, slot in
            (lane < attrs.count && slot < attrs[lane].count) ? attrs[lane][slot] : (0, 0, 0, 0)
        }
        ctx.outWriter = { lane, slot, x, y, z, w in
            while outs[lane].count <= slot { outs[lane].append((0, 0, 0, 0)) }
            outs[lane][slot] = (x, y, z, w)
        }
        _ = ShaderExecutor.runWarp(program: prog, lanes: lanes, ctx: ctx)
        return outs
    }
    func bitsEq(_ a: (Float, Float, Float, Float), _ b: (Float, Float, Float, Float)) -> Bool {
        a.0.bitPattern == b.0.bitPattern && a.1.bitPattern == b.1.bitPattern
            && a.2.bitPattern == b.2.bitPattern && a.3.bitPattern == b.3.bitPattern
    }
    // 5 lanes: not a multiple of 4, so the NEON group AND the scalar
    // tail path both execute.
    let lanes = 5
    let avec: [(Float, Float, Float, Float)] = [
        (1, 0, 0, 9), (0, 1, 0, 9), (2, -3, 5, 9), (-1.5, 0.25, 4, 9), (0, 0, 0, 9),
    ]
    let bvec: [(Float, Float, Float, Float)] = [
        (0, 1, 0, 9), (0, 0, 1, 9), (-4, 2, 0.5, 9), (3, -2, 1, 9), (1, 2, 3, 9),
    ]
    let attrs: [[(Float, Float, Float, Float)]] = (0..<lanes).map { [avec[$0], bvec[$0]] }
    let zeroU = [Float](repeating: 0, count: 64)

    // VCROSS vs the scalar reference (same mul/sub expressions).
    var pb = ShaderBuilder()
    pb.ldattr(0, 0); pb.ldattr(1, 1); pb.vcross(2, 0, 1); pb.out(0, 2); pb.end()
    let crossOut = runProg(pb.build(), lanes: lanes, attrs: attrs, uniforms: zeroU)
    var crossOK = true
    for l in 0..<lanes {
        let a = avec[l], b = bvec[l]
        let want = (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0, Float(0))
        if !bitsEq(crossOut[l][0], want) { crossOK = false }
    }
    check(crossOK, "VCROSS matches scalar reference bit-exactly (5 lanes, NEON + tail)")

    // VCROSS in place (rd == ra) must not corrupt its inputs.
    var pbi = ShaderBuilder()
    pbi.ldattr(0, 0); pbi.ldattr(1, 1); pbi.vcross(0, 0, 1); pbi.out(0, 0); pbi.end()
    let crossIn = runProg(pbi.build(), lanes: lanes, attrs: attrs, uniforms: zeroU)
    var crossInOK = true
    for l in 0..<lanes { if !bitsEq(crossIn[l][0], crossOut[l][0]) { crossInOK = false } }
    check(crossInOK, "VCROSS in-place (rd == ra) is correct")

    // VNORM vs the documented formula (DOT3-order len2, RSQ rule).
    var nb = ShaderBuilder()
    nb.ldattr(0, 0); nb.vnorm(1, 0); nb.out(0, 1); nb.end()
    let nvec: [(Float, Float, Float, Float)] = [
        (3, 0, 4, 7), (0, 0, 0, -2), (1, 2, 2, 0), (-6, -8, 0, 1), (0.25, -0.5, 0.75, 3),
    ]
    let nattrs: [[(Float, Float, Float, Float)]] = (0..<lanes).map { [nvec[$0]] }
    let normOut = runProg(nb.build(), lanes: lanes, attrs: nattrs, uniforms: zeroU)
    var normOK = true
    for l in 0..<lanes {
        let v = nvec[l]
        var len2: Float = 0
        len2 += v.0 * v.0; len2 += v.1 * v.1; len2 += v.2 * v.2
        let inv: Float = len2 <= 0 ? 0 : 1 / sqrtf(len2)
        let want = (v.0 * inv, v.1 * inv, v.2 * inv, v.3)
        if !bitsEq(normOut[l][0], want) { normOK = false }
    }
    check(normOK, "VNORM matches the documented formula bit-exactly (incl. zero vector, w pass-through)")

    // VMAT4 vs a manual DOT4 chain: bit-identical by construction —
    // component j of the VMAT4 result must equal the DOT4 splat of
    // row j, for every lane.
    let m: [Float] = [
        0.8, 0.1, -0.3, 0.5,
        -0.2, 1.1, 0.4, -0.6,
        0.9, -0.7, 0.3, 0.25,
        -0.1, 0.2, 0.45, 1.0,
    ]
    var unis = zeroU
    for i in 0..<16 { unis[i] = m[i] }
    var mb = ShaderBuilder()
    mb.ldattr(0, 0); mb.vmat4(1, 0, rows: 0); mb.out(0, 1); mb.end()
    let matOut = runProg(mb.build(), lanes: lanes, attrs: nattrs, uniforms: unis)
    var db = ShaderBuilder()
    db.ldattr(0, 0)
    for j in 0..<4 {
        db.ldunif(2, j); db.dot4(3, 0, 2); db.out(j, 3)
    }
    db.end()
    let dotOut = runProg(db.build(), lanes: lanes, attrs: nattrs, uniforms: unis)
    var matOK = true
    for l in 0..<lanes {
        let mv = matOut[l][0]
        let comps = [dotOut[l][0].0, dotOut[l][1].1, dotOut[l][2].2, dotOut[l][3].3]
        if mv.0.bitPattern != comps[0].bitPattern || mv.1.bitPattern != comps[1].bitPattern
            || mv.2.bitPattern != comps[2].bitPattern || mv.3.bitPattern != comps[3].bitPattern {
            matOK = false
        }
    }
    check(matOK, "VMAT4 is bit-identical to the manual DOT4-row chain")

    // VMAT4 in place + out-of-range rows read as zero.
    var mi = ShaderBuilder()
    mi.ldattr(0, 0); mi.vmat4(0, 0, rows: 0); mi.out(0, 0); mi.end()
    let matIn = runProg(mi.build(), lanes: lanes, attrs: nattrs, uniforms: unis)
    var matInOK = true
    for l in 0..<lanes { if !bitsEq(matIn[l][0], matOut[l][0]) { matInOK = false } }
    check(matInOK, "VMAT4 in-place (rd == ra) is correct")
    var mz = ShaderBuilder()
    mz.ldattr(0, 0); mz.vmat4(1, 0, rows: 14); mz.out(0, 1); mz.end()
    let matZ = runProg(mz.build(), lanes: lanes, attrs: nattrs, uniforms: unis)
    var matZOK = true
    for l in 0..<lanes {
        // Rows 14, 15 exist (zeros); rows 16, 17 are out of range (zero).
        if !bitsEq(matZ[l][0], (0, 0, 0, 0)) { matZOK = false }
    }
    check(matZOK, "VMAT4 rows past the uniform file read as zero")
}

// MARK: 9. v4 vector hardware: VPU, plot unit, screen grid
do {
    check(NovaGPU.maxScreenVectors == 65536 && NovaGPU.mgvpuCount == 50
          && NovaGPU.vpusPerMgvpu == 131072 && NovaGPU.vpuCount == 6553600
          && MGVPU.totalVPUs == 6553600 && MGVPU.address(ofGlobalID: MGVPU.globalID(mgvpu: 49, vpu: 131071)) == (49, 131071),
          "vector hardware sizing: 50 MGVPU x 131,072 VPUs = 6,553,600 addressable")
    // 9a. VPU.position is bit-identical to the shader stage's
    // VMAT4 on the same triangle workload (both are ngp_mat4, but
    // prove the VPU's load/axes plumbing preserves it).
    let triPos: [Float] = [0.5, -1.25, 2.0, 1, -0.75, 0.33, 1.5, 1, 3.25, 2.5, -0.125, 1]
    let mat: [Float] = [0.8, 0.1, -0.3, 0.5, -0.2, 1.1, 0.4, -0.6,
                        0.9, -0.7, 0.3, 0.25, -0.1, 0.2, 0.45, 1.0]
    var vpu = VPU()
    let vpuOut = UnsafeMutablePointer<Float>.allocate(capacity: 12)
    defer { vpuOut.deallocate() }
    triPos.withUnsafeBufferPointer { tp in
        mat.withUnsafeBufferPointer { mp in
            vpu.load(positions: tp.baseAddress!)
            vpu.position(matrix: mp.baseAddress!, into: vpuOut)
        }
    }
    var vb = ShaderBuilder()
    vb.ldattr(0, 0); vb.vmat4(1, 0, rows: 0); vb.out(0, 1); vb.end()
    var vctx = WarpContext.dummy()
    var unis = [Float](repeating: 0, count: 64)
    for i in 0..<16 { unis[i] = mat[i] }
    vctx.uniforms = unis
    var vmatOut: [(Float, Float, Float, Float)] = []
    vctx.attrProvider = { lane, _ in
        (triPos[lane * 4], triPos[lane * 4 + 1], triPos[lane * 4 + 2], triPos[lane * 4 + 3])
    }
    vctx.outWriter = { _, slot, x, y, z, w in if slot == 0 { vmatOut.append((x, y, z, w)) } }
    _ = ShaderExecutor.runWarp(program: vb.build(), lanes: 3, ctx: vctx)
    var vpuOK = vmatOut.count == 3
    for v in 0..<3 {
        let o = vmatOut[v]
        if o.0.bitPattern != vpuOut[v * 4].bitPattern || o.1.bitPattern != vpuOut[v * 4 + 1].bitPattern
            || o.2.bitPattern != vpuOut[v * 4 + 2].bitPattern || o.3.bitPattern != vpuOut[v * 4 + 3].bitPattern {
            vpuOK = false
        }
    }
    check(vpuOK, "VPU 3-axis calculator is bit-identical to shader VMAT4")

    // 9b. Screen grid: FP32 projection lands within 1 grid unit
    // (1/64 px) of the Double-precision reference on random NDC
    // endpoints, and exactly on it for dyadic coordinates.
    var rng: UInt64 = 0x12345678
    func frand() -> Float {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return Float((rng >> 40) & 0xFFFFFF) / Float(0x1000000) * 2 - 1
    }
    var gridOK = true
    for _ in 0..<2000 {
        let nx = frand(), ny = frand()
        let gx = ScreenSpace.gridX(nx, width: 2360), gy = ScreenSpace.gridY(ny, height: 1640)
        let dgx = Int((Double(nx) + 1) * 0.5 * 2360.0 * 64.0)
        let dgy = Int((1 - Double(ny)) * 0.5 * 1640.0 * 64.0)
        if abs(gx - dgx) > 1 || abs(gy - dgy) > 1 { gridOK = false }
    }
    check(gridOK, "screen grid (16 bits/axis, Q10.6) within 1 grid unit of Double reference")
    check(ScreenSpace.gridX(-0.75, width: 128) == 1024 && ScreenSpace.gridY(0.5, height: 96) == 1536,
          "screen grid exact on dyadic coordinates")

    // 9c. Full-frame plot: three crafted vectors whose samples are
    // exact in both Float and Double (integer grid endpoints,
    // dyadic step counts) must produce EXACTLY the Double-DDA
    // reference pixel set, in the c10 colour, via the bus.
    let fbW = 128, fbH = 96
    let fb = try client.alloc(UInt64(fbW * fbH * 4))
    // positions only matter; normals/uv/colour are unused by plot
    let verts: [[Float]] = [
        [-0.75, 0.5, 0.5],   // 0 -> pixel (16, 24)
        [0.75, 0.5, 0.5],    // 1 -> pixel (112, 24)
        [-0.5, 0.75, 0.5],   // 2 -> pixel (32, 12)
        [-0.5, -0.75, 0.5],  // 3 -> pixel (32, 84)
        [-0.75, 0.75, 0.5],  // 4 -> pixel (16, 12)
        [-0.25, 0.25, 0.5],  // 5 -> pixel (48, 36)
    ]
    var vdata = [UInt8]()
    for v in verts {
        var rec = [Float](repeating: 0, count: 12)
        rec[0] = v[0]; rec[1] = v[1]; rec[2] = v[2]
        for f in rec { withUnsafeBytes(of: f.bitPattern) { vdata.append(contentsOf: $0) } }
    }
    let vbh = try client.alloc(UInt64(vdata.count))
    try client.uploadData(vbh, data: vdata, guestStaging: STAGE)
    let idx: [UInt32] = [0, 1, 2, 3, 4, 5]
    var idata = [UInt8]()
    for i in idx { withUnsafeBytes(of: i) { idata.append(contentsOf: $0) } }
    let ibh = try client.alloc(UInt64(idata.count))
    try client.uploadData(ibh, data: idata, guestStaging: STAGE)
    var flat = ShaderBuilder()
    flat.ldattr(0, 1); flat.out(0, 0); flat.end()
    let flatProg = NovaGPUStandard.programBytes(flat.build())
    let sh = try client.alloc(4096)
    try client.uploadData(sh, data: flatProg, guestStaging: STAGE)
    try client.setShader(.setFShader, programHandle: sh, off: 0, len: flatProg.count)
    try client.setFBuf(fb, fbW, fbH)
    try client.setVBuf(vbh)
    try client.setIBuf(ibh)
    let ident: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    for r in 0..<4 { try client.setUniform(r, (ident[r * 4], ident[r * 4 + 1], ident[r * 4 + 2], ident[r * 4 + 3])) }
    try client.setUniform(10, (1.0, 0.5, 0.25, 1.0))
    let plottedBefore = device.gpu.stats.vectorsPlotted
    try client.clear(0xFF000000)
    try client.plot(indexCount: idx.count, mode: 0)
    try client.present(fb)
    let px = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
    func wordAt(_ x: Int, _ y: Int) -> UInt32 {
        let o = (y * fbW + x) * 4
        return UInt32(px[o]) | (UInt32(px[o + 1]) << 8) | (UInt32(px[o + 2]) << 16) | (UInt32(px[o + 3]) << 24)
    }
    let colourWord: UInt32 = 0xFF3F7FFF   // (1.0, 0.5, 0.25, 1.0) as RGBA8888 bytes
    var actual = Set<Int>()
    for y in 0..<fbH { for x in 0..<fbW where wordAt(x, y) == colourWord { actual.insert(y * fbW + x) } }
    func expectedLine(_ gx0: Int, _ gy0: Int, _ gx1: Int, _ gy1: Int) -> Set<Int> {
        // The plot unit's DDA spec, in Double: pixel-resolution
        // steps over the grid, same rounding, same gap insurance.
        var set = Set<Int>()
        let dx = gx1 - gx0, dy = gy1 - gy0
        let gridSteps = max(abs(dx), abs(dy))
        if gridSteps == 0 { set.insert((gy0 >> 6) * fbW + (gx0 >> 6)); return set }
        let steps = (gridSteps + 63) / 64
        var lastPx = Int.min, lastPy = Int.min
        for s in 0...steps {
            let t = Double(s) / Double(steps)
            let gx = Double(gx0) + Double(dx) * t
            let gy = Double(gy0) + Double(dy) * t
            let ex = Int(gx + 0.5) >> 6, ey = Int(gy + 0.5) >> 6
            if lastPx != Int.min && (abs(ex - lastPx) > 1 || abs(ey - lastPy) > 1) {
                let mx = (lastPx + ex) / 2, my = (lastPy + ey) / 2
                if mx >= 0 && mx < fbW && my >= 0 && my < fbH { set.insert(my * fbW + mx) }
            }
            if ex >= 0 && ex < fbW && ey >= 0 && ey < fbH { set.insert(ey * fbW + ex) }
            lastPx = ex; lastPy = ey
        }
        return set
    }
    var expected = Set<Int>()
    expected.formUnion(expectedLine(1024, 1536, 7168, 1536))   // horizontal
    expected.formUnion(expectedLine(2048, 768, 2048, 5376))    // vertical
    expected.formUnion(expectedLine(1024, 768, 3072, 2304))    // diagonal
    check(actual == expected, "plot lines: pixel set exactly matches the Double DDA reference",
          "actual \(actual.count) px, expected \(expected.count) px, symdiff \(actual.symmetricDifference(expected).count)")
    let endpoints: [(Int, Int)] = [(16, 24), (112, 24), (32, 12), (32, 84), (16, 12), (48, 36)]
    check(endpoints.allSatisfy { wordAt($0.0, $0.1) == colourWord },
          "plot endpoints land on the full-pipeline pixels")
    check(device.gpu.stats.vectorsPlotted == plottedBefore + 3, "vectorsPlotted accounted (3)")

    // 9d. Points mode: exactly the two endpoint pixels.
    try client.clear(0xFF000000)
    try client.plot(indexCount: 2, mode: 1)   // indices 0, 1 as points
    let px2 = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
    var pts = Set<Int>()
    for y in 0..<fbH {
        for x in 0..<fbW {
            let o = (y * fbW + x) * 4
            let w = UInt32(px2[o]) | (UInt32(px2[o + 1]) << 8) | (UInt32(px2[o + 2]) << 16) | (UInt32(px2[o + 3]) << 24)
            if w == colourWord { pts.insert(y * fbW + x) }
        }
    }
    check(pts == [24 * fbW + 16, 24 * fbW + 112], "plot points mode draws exactly the indexed points",
          "\(pts.count) px")

    // 9e. MGVPU-parallel pass 1 must be frame-identical to serial:
    // force the parallel path on this small scene and compare the
    // whole framebuffer byte for byte.
    try client.clear(0xFF000000)
    try client.plot(indexCount: idx.count, mode: 0)
    let serialFrame = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
    let savedThreshold = NovaGPU.plotParallelThreshold
    NovaGPU.plotParallelThreshold = 1
    try client.clear(0xFF000000)
    try client.plot(indexCount: idx.count, mode: 0)
    NovaGPU.plotParallelThreshold = savedThreshold
    let parallelFrame = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
    check(serialFrame == parallelFrame,
          "MGVPU-cluster-parallel plot is byte-identical to serial plot")

    // 9f. v5 wide issue: the plotted frame must not depend on the
    // pass-1 batch size (endpoints are positioned in batches
    // through the shared calculator). serialFrame above was
    // produced at the default batch size; re-plot at 1, 7 and
    // 1024 and compare byte for byte.
    let savedBatch = NovaGPU.plotBatchSize
    for bs in [1, 7, 1024] {
        NovaGPU.plotBatchSize = bs
        try client.clear(0xFF000000)
        try client.plot(indexCount: idx.count, mode: 0)
        let f = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
        check(f == serialFrame, "plot frame is batch-size invariant (batch = \(bs))")
    }
    NovaGPU.plotBatchSize = savedBatch
} catch {
    check(false, "vector hardware suite", "\(error)")
}

// MARK: 10. v4 standard vertex program renders bit-identically to v1
do {
    let cube = NovaGPUStandard.proofCube()
    let fbW = 256, fbH = 192
    let fb = try client.alloc(UInt64(fbW * fbH * 4))
    let vb = try client.alloc(UInt64(cube.vertices.count * 4))
    let ib = try client.alloc(UInt64(cube.indices.count * 4))
    try client.uploadData(vb, data: NovaGPUStandard.vertexBytes(cube), guestStaging: STAGE)
    try client.uploadData(ib, data: NovaGPUStandard.indexBytes(cube), guestStaging: STAGE)
    let fs = NovaGPUStandard.fragmentLitProgram()
    let sh = try client.alloc(8192)
    try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fs), guestStaging: STAGE)
    try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fs.count * 4)
    try client.setFBuf(fb, fbW, fbH)
    try client.setVBuf(vb)
    try client.setIBuf(ib)
    let mvp: [Float] = [0.8, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 1]
    try NovaGPUStandard.setLitUniforms(client, mvp: mvp, light: (0.5, 1.0, 0.25))

    func renderWith(_ vs: [UInt32]) throws -> [UInt8] {
        try client.uploadData(sh, data: NovaGPUStandard.programBytes(vs), guestStaging: STAGE)
        try client.setShader(.setVShader, programHandle: sh, off: 0, len: vs.count * 4)
        try client.clear(0xFF200810)
        try client.draw(indexCount: cube.indices.count)
        try client.present(fb)
        return try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
    }
    let frameV1 = try renderWith(NovaGPUStandard.vertexLitProgram())
    let frameV4 = try renderWith(NovaGPUStandard.vertexLitProgramV4())
    check(frameV1 == frameV4, "vertexLitProgramV4 (VMAT4) frame is byte-identical to v1 (DOT4+SEL)",
          "\(frameV1.count) bytes compared")
} catch {
    check(false, "v4 vertex program suite", "\(error)")
}

// MARK: 11. v5 in-draw VPU test: armed rendering, exact counts, fault detection
do {
    let cube = NovaGPUStandard.proofCube()                    // 12 triangles
    let sphere = NovaGPUStandard.icosphere(subdivisions: 5)   // 20,480 triangles
    let fbW = 256, fbH = 192
    let fb = try client.alloc(UInt64(fbW * fbH * 4))
    let sh = try client.alloc(8192)
    let fs = NovaGPUStandard.fragmentLitProgram()
    try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fs), guestStaging: STAGE)
    try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fs.count * 4)
    try client.setFBuf(fb, fbW, fbH)
    let mvp: [Float] = [0.8, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 1]
    try NovaGPUStandard.setLitUniforms(client, mvp: mvp, light: (0.5, 1.0, 0.25))

    func uploadMesh(_ m: NovaGPUStandard.Mesh) throws -> (UInt64, UInt64) {
        let v = try client.alloc(UInt64(m.vertices.count * 4))
        let i = try client.alloc(UInt64(m.indices.count * 4))
        try client.uploadData(v, data: NovaGPUStandard.vertexBytes(m), guestStaging: STAGE)
        try client.uploadData(i, data: NovaGPUStandard.indexBytes(m), guestStaging: STAGE)
        return (v, i)
    }
    let (cv, ci) = try uploadMesh(cube)
    let (sv, si) = try uploadMesh(sphere)

    func setVShader(_ vs: [UInt32]) throws {
        try client.uploadData(sh, data: NovaGPUStandard.programBytes(vs), guestStaging: STAGE)
        try client.setShader(.setVShader, programHandle: sh, off: 0, len: vs.count * 4)
    }
    func render(_ m: NovaGPUStandard.Mesh, _ v: UInt64, _ i: UInt64, arm: Bool) throws -> [UInt8] {
        try client.setVBuf(v)
        try client.setIBuf(i)
        if arm { try client.vpuTest(arm: true) }
        try client.clear(0xFF200810)
        try client.draw(indexCount: m.indices.count)
        if arm { try client.vpuTest(arm: false) }
        try client.present(fb)
        return try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
    }

    // 11a. Armed rendering is byte-identical to unarmed (the VPU
    // positions ARE the shader positions), with exact counts.
    try setVShader(NovaGPUStandard.vertexLitProgramV4())
    let cubeOff = try render(cube, cv, ci, arm: false)
    let cubeOn = try render(cube, cv, ci, arm: true)
    check(cubeOn == cubeOff, "VPU test armed: cube frame byte-identical to unarmed")
    var t = device.gpu.vpuTest
    check(t.trianglesTested == 12 && t.cornersCompared == 36 && t.mismatches == 0
          && t.maxAbsError == 0 && t.clustersEngaged == 12,
          "VPU test cube counts: 12 triangles, 36 corners, 0 mismatches, 0 error, 12 clusters",
          "got tris=\(t.trianglesTested) corners=\(t.cornersCompared) mism=\(t.mismatches) err=\(t.maxAbsError) clusters=\(t.clustersEngaged)")
    let sphereOff = try render(sphere, sv, si, arm: false)
    let sphereOn = try render(sphere, sv, si, arm: true)
    check(sphereOn == sphereOff, "VPU test armed: sphere frame byte-identical to unarmed")
    t = device.gpu.vpuTest
    check(t.trianglesTested == 20480 && t.mismatches == 0 && t.maxAbsError == 0
          && t.clustersEngaged == 50,
          "VPU test sphere counts: 20,480 triangles, 0 mismatches, all 50 MGVPU clusters",
          "got tris=\(t.trianglesTested) mism=\(t.mismatches) err=\(t.maxAbsError) clusters=\(t.clustersEngaged)")
    let life = device.gpu.vpuTestLifetime
    check(life.trianglesTested == 12 + 20480 && life.mismatches == 0,
          "VPU test lifetime accumulates across arms",
          "lifetime tris=\(life.trianglesTested) mism=\(life.mismatches)")

    // 11b. The reference is the pipeline, not one program: the v1
    // (DOT4+SEL) vertex program's clips must also match the VPUs.
    try setVShader(NovaGPUStandard.vertexLitProgram())
    _ = try render(cube, cv, ci, arm: true)
    t = device.gpu.vpuTest
    check(t.trianglesTested == 12 && t.mismatches == 0 && t.maxAbsError == 0,
          "VPU test also passes against the v1 vertex program's clips",
          "got mism=\(t.mismatches) err=\(t.maxAbsError)")

    // 11c. Fault injection: a perturbed VPU output must be
    // DETECTED (one mismatched corner per triangle — corner 0 —
    // with a real error magnitude) and must CHANGE the rendered
    // frame, proving armed frames render from VPU positions.
    try setVShader(NovaGPUStandard.vertexLitProgramV4())
    NovaGPU.vpuTestFault = 0.5
    let cubeFault = try render(cube, cv, ci, arm: true)
    NovaGPU.vpuTestFault = 0
    t = device.gpu.vpuTest
    check(t.mismatches == 12 && t.maxAbsError > 0.4,
          "VPU test detects the injected fault (12 corners, error ~0.5)",
          "got mism=\(t.mismatches) err=\(t.maxAbsError)")
    check(cubeFault != cubeOff, "faulty VPU positions visibly change the armed frame")

    // 11d. Disarmed draws do not accumulate into the window.
    let before = device.gpu.vpuTest.trianglesTested
    _ = try render(cube, cv, ci, arm: false)
    check(device.gpu.vpuTest.trianglesTested == before,
          "disarmed draws leave the VPU test window untouched")
} catch {
    check(false, "VPU test suite", "\(error)")
}

// MARK: 12. v5 wide VPU issue: grouped positioning == per-triangle VPU, bitwise
do {
    var rng: UInt64 = 0xB17C4
    func frand() -> Float {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return Float((rng >> 40) & 0xFFFFFF) / Float(0x1000000) * 4 - 2
    }
    let mat: [Float] = [0.8, 0.1, -0.3, 0.5, -0.2, 1.1, 0.4, -0.6,
                        0.9, -0.7, 0.3, 0.25, -0.1, 0.2, 0.45, 1.0]
    var wideOK = true
    for group in [1, 2, 3, 4, 5, 17, 64, 250] {
        let n = group * 12
        let pos = UnsafeMutablePointer<Float>.allocate(capacity: n)
        let outA = UnsafeMutablePointer<Float>.allocate(capacity: n)
        let outB = UnsafeMutablePointer<Float>.allocate(capacity: n)
        defer { pos.deallocate(); outA.deallocate(); outB.deallocate() }
        for i in 0..<n { pos[i] = frand() }
        var vpu = VPU()
        mat.withUnsafeBufferPointer { mp in
            for k in 0..<group {
                vpu.load(positions: pos + k * 12)
                vpu.position(matrix: mp.baseAddress!, into: outA + k * 12)
            }
            VPU.positionWide(positions: pos, vertices: group * 3,
                             matrix: mp.baseAddress!, into: outB)
        }
        for i in 0..<n where outA[i].bitPattern != outB[i].bitPattern { wideOK = false }
    }
    check(wideOK, "VPU.positionWide is bit-identical to per-triangle VPU.position (groups 1..250 triangles)")
}

// MARK: 13. Governor chip: registers, ring, lever policy, pixel invariance
do {
    let govBase = NovaMemory.govBaseAddr
    check(NovaGovernor.chipName == "NovaGovernor", "governor chip name lives in one place")

    // MMIO register block on the shared memory (no GPU supervised).
    let govA = NovaGovernorDevice()
    govA.attach(to: mem)
    check(mem.govDevice === govA, "governor device attaches to NovaMemory")
    let gid = mem.mmioRead(govBase + GovRegister.id.rawValue)
    check(gid == NovaGovernorDevice.deviceID, "governor device ID register",
          String(format: "0x%016X", gid))
    mem.mmioWrite(govBase + GovRegister.enable.rawValue, 1)
    check(govA.chip.enabled, "governor enables via MMIO")
    check(mem.mmioRead(govBase + GovRegister.status.rawValue) & 1 == 1,
          "governor status register reports enabled")
    for _ in 0..<40 { mem.mmioWrite(govBase + GovRegister.tick.rawValue, 30000) }
    check(govA.chip.lastFrameMs == 30, "governor MMIO tick carries frame microseconds")
    check(mem.mmioRead(govBase + GovRegister.medianUs.rawValue) == 30000,
          "governor median register reads back the scripted window")
    check(govA.chip.paceCeilingMs == 30 && !govA.chip.pacingEngaged,
          "governor ceiling tracks the work p25; pacing stays off on work-only ticks",
          "ceiling \(govA.chip.paceCeilingMs) ms, state \(govA.chip.pacingStateText)")
    check(mem.mmioRead(govBase + GovRegister.paceCeilingUs.rawValue) == 30000,
          "governor ceiling register reads back the work p25")
    mem.mmioWrite(govBase + GovRegister.enable.rawValue, 0)
    check(!govA.chip.enabled && !govA.chip.pacingEngaged && govA.chip.gpuStageWidth == 0,
          "governor disable via MMIO restores the v5 operating point")

    // Ring opcode 21 on the shared device.
    let govRing = NovaGovernorDevice()
    govRing.supervise(device)
    device.governor = govRing
    try client.governorControl(1)
    check(govRing.chip.enabled, "governor enables via ring opcode 21")
    try client.governorControl(2)
    check(!govRing.chip.enabled, "governor disables via ring opcode 21")
    device.governor = nil

    // Pacing policy traces live in section 14 (the v7
    // never-slower law replaced the v6 tier ladder — the ladder
    // was the bug: see V7-LEDGER.md).

    // Policy: jitter refines the grain, then caps the width;
    // stability restores both.
    let jit = NovaGovernor()
    jit.setEnabled(true)
    for i in 0..<120 { jit.tick(GovernorTelemetry(frameWorkMs: i % 6 == 5 ? 40 : 14)) }
    check(jit.rasterChunkRows == 2 && jit.gpuStageWidth == 6,
          "jitter refines grain to 2 then caps GPU width at 6",
          "grain \(jit.rasterChunkRows), width \(jit.gpuStageWidth)")
    for _ in 0..<200 { jit.tick(GovernorTelemetry(frameWorkMs: 14)) }
    check(jit.gpuStageWidth == 0 && jit.rasterChunkRows == 4,
          "stability restores width and grain",
          "grain \(jit.rasterChunkRows), width \(jit.gpuStageWidth)")

    // Policy: disabling mid-adaptation restores everything.
    let dis = NovaGovernor()
    dis.setEnabled(true)
    for i in 0..<40 { dis.tick(GovernorTelemetry(frameWorkMs: i % 6 == 5 ? 40 : 14)) }
    dis.setEnabled(false)
    check(dis.rasterChunkRows == 4 && dis.gpuStageWidth == 0 && !dis.enabled,
          "disable restores grain and width")
    check(dis.actionLog.last?.command == .disable, "disable is the last logged action")

    // Policy: CPU slice steering (halve over target, double far
    // under, deadband between).
    let cpu = NovaGovernor()
    cpu.setEnabled(true)
    cpu.tickCPU(sliceMs: 9.0, steps: cpu.cpuSliceSteps)
    cpu.tickCPU(sliceMs: 9.0, steps: cpu.cpuSliceSteps)
    cpu.tickCPU(sliceMs: 9.0, steps: cpu.cpuSliceSteps)
    check(cpu.cpuSliceSteps == 12_500, "CPU slice halves toward the 2 ms target",
          "\(cpu.cpuSliceSteps) steps")
    cpu.tickCPU(sliceMs: 0.4, steps: cpu.cpuSliceSteps)
    check(cpu.cpuSliceSteps == 25_000, "CPU slice doubles when far under target")
    cpu.tickCPU(sliceMs: 2.0, steps: cpu.cpuSliceSteps)
    check(cpu.cpuSliceSteps == 25_000, "CPU slice holds inside the deadband")

    // Policy: VRAM pressure sheds scratch retention; relief
    // restores it.
    let memChip = NovaGovernor()
    memChip.setEnabled(true)
    for _ in 0..<40 {
        memChip.tick(GovernorTelemetry(frameWorkMs: 16,
                                       vramCommitted: 300 * 1024 * 1024))
    }
    check(!memChip.scratchRetention, "scratch shed under VRAM commit pressure")
    for _ in 0..<40 {
        memChip.tick(GovernorTelemetry(frameWorkMs: 16,
                                       vramCommitted: 10 * 1024 * 1024))
    }
    check(memChip.scratchRetention, "scratch restored when VRAM pressure clears")

    // Pixel invariance: the same scene renders byte-identically
    // with raw knobs twisted and with the chip attached + live.
    func renderGoverned(_ configure: (NovaMemory, NovaGPUDevice) -> Void) throws -> [UInt8] {
        let m = NovaMemory(totalSize: 64 * 1024 * 1024)
        let d = NovaGPUDevice()
        let c = NovaBusClient(memory: m, device: d)
        configure(m, d)
        let cube = NovaGPUStandard.proofCube()
        let fbW = 256, fbH = 192
        let fb = try c.alloc(UInt64(fbW * fbH * 4))
        let vb = try c.alloc(UInt64(cube.vertices.count * 4))
        let ib = try c.alloc(UInt64(cube.indices.count * 4))
        try c.uploadData(vb, data: NovaGPUStandard.vertexBytes(cube), guestStaging: STAGE)
        try c.uploadData(ib, data: NovaGPUStandard.indexBytes(cube), guestStaging: STAGE)
        let vs = NovaGPUStandard.vertexLitProgram()
        let fs = NovaGPUStandard.fragmentLitProgram()
        let sh = try c.alloc(8192)
        try c.uploadData(sh, data: NovaGPUStandard.programBytes(vs), guestStaging: STAGE)
        try c.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fs), guestStaging: STAGE)
        try c.setShader(.setVShader, programHandle: sh, off: 0, len: vs.count * 4)
        try c.setShader(.setFShader, programHandle: sh, off: 4096, len: fs.count * 4)
        try c.setFBuf(fb, fbW, fbH)
        try c.setVBuf(vb)
        try c.setIBuf(ib)
        let mvp: [Float] = [0.8, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 1]
        try NovaGPUStandard.setLitUniforms(c, mvp: mvp, light: (0.5, 1.0, 0.25))
        try c.clear(0xFF200810)
        try c.draw(indexCount: cube.indices.count)
        try c.present(fb)
        return try c.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
    }
    let plain = try renderGoverned { _, _ in }
    let twisted = try renderGoverned { _, d in
        d.gpu.rasterChunkRows = 1
        d.gpu.stageWidth = 1
        d.gpu.clipRangeCount = 4
    }
    check(plain == twisted, "governor knobs (grain 1, width 1, clip 4) are pixel-invariant")
    var govTicks: UInt64 = 0
    let governed = try renderGoverned { m, d in
        let g = NovaGovernorDevice()
        g.attach(to: m)
        g.supervise(d)
        d.governor = g
        g.chip.setEnabled(true)
        for _ in 0..<40 { g.chip.tick(GovernorTelemetry(frameWorkMs: 30)) }
        govTicks = g.chip.tickCount
    }
    check(plain == governed, "frame renders byte-identically with the governor attached and enabled")
    note("governor ticks during governed render: \(govTicks)")
} catch {
    check(false, "governor suite", "\(error)")
}

// MARK: 14. v7 pacing policy: the never-slower law (scripted traces)
do {
    // Work-bound loop (interval == work): pacing must never
    // engage, and the ceiling must equal the demonstrated work.
    let wb = NovaGovernor()
    wb.setEnabled(true)
    for _ in 0..<120 { wb.tick(GovernorTelemetry(frameWorkMs: 35, frameIntervalMs: 35)) }
    check(!wb.pacingEngaged && wb.paceCeilingMs == 35,
          "work-bound loop: pacing stays disengaged, ceiling = work p25",
          "state \(wb.pacingStateText), ceiling \(wb.paceCeilingMs)")
    check(wb.pacingStateText.contains("work-bound"),
          "pacing state text names the work-bound reason",
          wb.pacingStateText)

    // The device trace: the numbers v6's sensor saw on the iPad
    // (work 35 ms, intervals inflated to ~52 by its own padding).
    // v7 may engage only at a ceiling of the work p25 — 35 ms,
    // never a 50 ms tier — and when the work drops, pacing
    // disengages and the ceiling re-derives with the window.
    let dev = NovaGovernor()
    dev.setEnabled(true)
    for i in 0..<120 {
        dev.tick(GovernorTelemetry(frameWorkMs: 35,
                                   frameIntervalMs: [46.0, 52.0, 58.0][i % 3]))
    }
    check(dev.pacingEngaged && dev.paceCeilingMs == 35,
          "device trace: pacing engages at the work-p25 ceiling (35), not a 50 tier",
          "ceiling \(dev.paceCeilingMs)")
    for _ in 0..<60 { dev.tick(GovernorTelemetry(frameWorkMs: 12, frameIntervalMs: 12)) }
    check(!dev.pacingEngaged && dev.paceCeilingMs == 12,
          "no lock-in: fast work disengages pacing and the ceiling re-derives to 12",
          "ceiling \(dev.paceCeilingMs), state \(dev.pacingStateText)")

    // Headroom + jitter: pacing engages. Then the governed
    // intervals inflate past the ungoverned estimate — governing
    // is costing frames — and the A/B guard must trip, disengage
    // pacing, and say so in plain words. Afterwards a lockout +
    // fresh probe keep it off even when headroom returns.
    let gd = NovaGovernor()
    gd.setEnabled(true)
    for i in 0..<120 {
        gd.tick(GovernorTelemetry(frameWorkMs: 14,
                                  frameIntervalMs: [18.0, 20.0, 22.0][i % 3]))
    }
    check(gd.pacingEngaged && gd.paceCeilingMs == 14,
          "headroom + jitter: pacing engages at the work p25",
          "ceiling \(gd.paceCeilingMs)")
    for _ in 0..<80 { gd.tick(GovernorTelemetry(frameWorkMs: 14, frameIntervalMs: 30)) }
    check(!gd.pacingEngaged &&
          gd.actionLog.contains { $0.text.contains("pacing disengaged: it was costing frames") },
          "A/B guard trips and logs 'pacing disengaged: it was costing frames'")
    for i in 0..<60 {
        gd.tick(GovernorTelemetry(frameWorkMs: 14,
                                  frameIntervalMs: [18.0, 20.0, 22.0][i % 3]))
    }
    check(!gd.pacingEngaged, "guard lockout + fresh probe keep pacing off after a trip")

    // The guard covers the other levers too: jitter with slop
    // caps the GPU width; when intervals then inflate, the guard
    // reverts the width cap with the same plain-words log.
    let wd = NovaGovernor()
    wd.setEnabled(true)
    for i in 0..<140 {
        let w: Double = i % 6 == 5 ? 40 : 14
        wd.tick(GovernorTelemetry(frameWorkMs: w, frameIntervalMs: w + 6))
    }
    check(wd.gpuStageWidth == 6, "jitter with slop caps GPU width at 6",
          "width \(wd.gpuStageWidth)")
    for i in 0..<80 {
        let w: Double = i % 6 == 5 ? 40 : 14
        wd.tick(GovernorTelemetry(frameWorkMs: w, frameIntervalMs: w + 26))
    }
    check(wd.gpuStageWidth == 0 &&
          wd.actionLog.contains { $0.text.contains("costing frames") },
          "A/B guard reverts the width cap when it costs frames",
          "width \(wd.gpuStageWidth)")
}

print(failures == 0 ? "\nALL HOST TESTS PASSED" : "\n\(failures) HOST TEST(S) FAILED")
exit(failures == 0 ? 0 : 1)
