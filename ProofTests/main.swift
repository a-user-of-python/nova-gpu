// main.swift — NovaCPU <-> NovaGPU compatibility proof, run on CI.
//
// 1) Boots the REAL NovaCPU core (byte-identical copy of nova-cpu's
//    NovaCPU.swift) on the guest program gpu_driver.bin (assembled from
//    Driver/gpu_driver.nova with the project's novas.py). The driver
//    talks to the GPU only through NovaBus MMIO + the guest-RAM
//    command ring, and services the GPU's ASSET request itself.
// 2) Renders the identical scene through the Swift bus client and
//    compares the framebuffer byte-sum the driver printed with the
//    sum of the client-rendered framebuffer.
//
// Usage: proof <path-to-gpu_driver.bin>

import Foundation

var failures = 0
func check(_ cond: Bool, _ name: String, _ detail: String = "") {
    print("\(cond ? "PASS" : "FAIL")  \(name)\(detail.isEmpty ? "" : " — " + detail)")
    if !cond { failures += 1 }
}

guard CommandLine.arguments.count >= 2 else {
    print("usage: proof <gpu_driver.bin>")
    exit(2)
}
let driverPath = CommandLine.arguments[1]
guard let driverBin = try? Data(contentsOf: URL(fileURLWithPath: driverPath)) else {
    print("cannot read \(driverPath)")
    exit(2)
}

// Shared scene constants — MUST match Driver/gpu_driver.nova.
let fbW = 256, fbH = 192
let clearColour: UInt32 = 0xFF200810
let proofMVP: [Float] = [0.8, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 1]
let proofLight: (Float, Float, Float) = (0.5, 1.0, 0.25)
let cube = NovaGPUStandard.proofCube()
let vsProg = NovaGPUStandard.vertexLitProgram()
let fsProg = NovaGPUStandard.fragmentLitProgram()

// MARK: - Expected framebuffer via the Swift bus client

var expectedSum: UInt64 = 0
do {
    let mem = NovaMemory(totalSize: 64 * 1024 * 1024)
    let device = NovaGPUDevice()
    let client = NovaBusClient(memory: mem, device: device)
    let STAGE: UInt64 = 0x40000, DSTAGE: UInt64 = 0x80000
    let fb = try client.alloc(UInt64(fbW * fbH * 4))
    let vb = try client.alloc(UInt64(cube.vertices.count * 4))
    let ib = try client.alloc(UInt64(cube.indices.count * 4))
    let sh = try client.alloc(8192)
    try client.uploadData(sh, data: NovaGPUStandard.programBytes(vsProg), guestStaging: STAGE)
    try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fsProg), guestStaging: STAGE)
    try client.setShader(.setVShader, programHandle: sh, off: 0, len: vsProg.count * 4)
    try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fsProg.count * 4)
    try client.uploadData(ib, data: NovaGPUStandard.indexBytes(cube), guestStaging: STAGE)
    try client.setFBuf(fb, fbW, fbH)
    try client.setVBuf(vb)
    try client.setIBuf(ib)
    try NovaGPUStandard.setLitUniforms(client, mvp: proofMVP, light: proofLight)
    try client.clear(clearColour)
    client.serviceHandler = { words in
        mem.storeHostBytes(NovaGPUStandard.vertexBytes(cube), at: words[2])
    }
    try client.draw(indexCount: cube.indices.count, flags: 1,
                    assetStaging: 0x30000, assetLen: UInt64(cube.vertices.count * 4),
                    assetTarget: vb)
    try client.present(fb)
    let px = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
    expectedSum = px.reduce(UInt64(0)) { $0 + UInt64($1) }
    print("  .. client-rendered framebuffer byte-sum: \(expectedSum)")
} catch {
    check(false, "client reference render", "\(error)")
}

// MARK: - Boot the real NovaCPU on the assembly driver (twice in v6)
//
// v6 runs the driver twice: once with the v5 fixed 100k-step
// slices, once with the supervisor chip steering the slice size
// (SET_CPU_SLICE, steered by each slice's measured wall time).
// The full check battery applies to BOTH runs — the governor
// changes scheduling only, never the result.

struct DriverRun {
    var console: String
    var wallSeconds: Double
    var descriptors: UInt64
    var slices: Int
    var trap: String?
    var governorActions: [String]
}

func runDriver(governed: Bool) -> DriverRun {
    let memory = NovaMemory(totalSize: 64 * 1024 * 1024)
    let cpu = NovaCPU()
    cpu.memory = memory
    let gpuDevice = NovaGPUDevice()
    gpuDevice.attach(to: memory)
    var gov: NovaGovernorDevice? = nil
    if governed {
        let g = NovaGovernorDevice()
        g.attach(to: memory)
        g.supervise(gpuDevice)
        gpuDevice.governor = g
        g.chip.setEnabled(true)
        gov = g
    }

    var consoleText = ""
    memory.consoleOutput = { s in consoleText += s }
    memory.netHandler = { _, _, _, _ in -1 }
    var trapDesc: String? = nil
    cpu.trapHandler = { trap, _ in
        trapDesc = "\(trap)"
        cpu.running = false
    }

    // Loader role (like the harness loading a program): a small
    // header (u64 vs length, u64 fs length) at 0x22000, the shader
    // binaries at 0x22010 / 0x22800, and the cube indices at
    // 0x23000. The vertex data is deliberately NOT provided — the
    // driver must produce it when the GPU asks (the two-way proof).
    do {
        var header = [UInt8](repeating: 0, count: 16)
        let vsLen = UInt64(vsProg.count * 4), fsLen = UInt64(fsProg.count * 4)
        for k in 0..<8 {
            header[k] = UInt8((vsLen >> UInt64(k * 8)) & 0xFF)
            header[8 + k] = UInt8((fsLen >> UInt64(k * 8)) & 0xFF)
        }
        try memory.loadBytes(Data(header), at: 0x22000)
        try memory.loadBytes(Data(NovaGPUStandard.programBytes(vsProg)), at: 0x22010)
        try memory.loadBytes(Data(NovaGPUStandard.programBytes(fsProg)), at: 0x22800)
        try memory.loadBytes(Data(NovaGPUStandard.indexBytes(cube)), at: 0x23000)
        try memory.loadBytes(driverBin, at: 0)
    } catch {
        check(false, "loading guest memory", "\(error)")
    }

    // resetCPUState, mirroring NovaVMEngine via the project harness.
    cpu.dsp = memory.dataStackBase
    cpu.csp = memory.callStackBase
    cpu.vsp = memory.vectorStackBase
    cpu.esp = memory.vectorStackBase &- 1024 * 1024
    cpu.dslim = memory.dataStackBase &- 8 * 1024 * 1024
    cpu.cslim = memory.callStackBase &- 8 * 1024 * 1024
    cpu.eslim = cpu.esp &- 1024 * 1024
    cpu.vslim = memory.vectorStackBase &- 8 * 1024 * 1024
    cpu.fp = 0
    cpu.pc = 0
    cpu.status = (3 << 2) | 1
    memory.timerInterval = 0
    memory.timerCount = 0

    let t0 = Date()
    // Run in slices, servicing the GPU between slices: while the
    // CPU executes, NovaMemory's RAM buffer is under a modify
    // access, so the device does its DMA at these service points
    // (see NovaBus.swift).
    var budget: UInt64 = 0
    var slices = 0
    if let gov = gov {
        repeat {
            let slice = gov.chip.cpuSliceSteps
            let s0 = Date()
            cpu.run(maxSteps: slice)
            let sliceMs = Date().timeIntervalSince(s0) * 1000
            budget += slice
            slices += 1
            gpuDevice.servicePendingWork()
            gov.chip.tickCPU(sliceMs: sliceMs, steps: slice)
        } while cpu.running && budget < 200_000_000
    } else {
        repeat {
            cpu.run(maxSteps: 100_000)
            budget += 100_000
            slices += 1
            gpuDevice.servicePendingWork()
        } while cpu.running && budget < 200_000_000
    }
    let dt = Date().timeIntervalSince(t0)
    return DriverRun(console: consoleText, wallSeconds: dt,
                     descriptors: gpuDevice.descriptorsExecuted,
                     slices: slices, trap: trapDesc,
                     governorActions: gov?.chip.actionLog.map { $0.text } ?? [])
}

func checkRun(_ r: DriverRun, _ label: String) {
    print("  .. \(label) run finished in \(String(format: "%.2f", r.wallSeconds)) s (CI host), \(r.slices) slices, trap: \(r.trap ?? "none (WFI halt)")")
    print("  .. guest console output:")
    for line in r.console.split(separator: "\n") { print("     | \(line)") }
    check(r.console.contains("GPU OK"), "\(label): driver printed GPU OK")
    check(r.descriptors >= 20, "\(label): driver submitted a real command stream",
          "\(r.descriptors) descriptors executed")
    var driverSum: UInt64? = nil
    for line in r.console.split(separator: "\n") where line.hasPrefix("GPUFB ") {
        driverSum = UInt64(line.dropFirst(6).trimmingCharacters(in: .whitespaces))
    }
    check(driverSum != nil, "\(label): driver printed the framebuffer checksum (GPUFB)")
    if let ds = driverSum {
        check(ds == expectedSum, "\(label): CPU-driven GPU framebuffer matches client-rendered framebuffer",
              "driver \(ds) vs expected \(expectedSum)")
    }
}

let fixedRun = runDriver(governed: false)
checkRun(fixedRun, "fixed 100k slices (v5 baseline)")
let govRun = runDriver(governed: true)
checkRun(govRun, "governor-supervised slices")
print("  .. governor actions during the supervised run:")
for a in govRun.governorActions { print("     | \(a)") }
print("  .. wall time: fixed \(String(format: "%.2f", fixedRun.wallSeconds)) s vs governor \(String(format: "%.2f", govRun.wallSeconds)) s (CI host)")

print(failures == 0 ? "\nCPU PROOF PASSED" : "\n\(failures) CPU PROOF CHECK(S) FAILED")
exit(failures == 0 ? 0 : 1)
