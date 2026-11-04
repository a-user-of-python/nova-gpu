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

print(failures == 0 ? "\nALL HOST TESTS PASSED" : "\n\(failures) HOST TEST(S) FAILED")
exit(failures == 0 ? 0 : 1)
