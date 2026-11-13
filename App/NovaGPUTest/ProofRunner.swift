// ProofRunner.swift — the CPU Proof page: boots the real NovaCPU
// core on gpu_driver.bin (bundled; assembled from Driver/
// gpu_driver.nova by the project's novas.py). The Nova assembly
// driver renders a cube through NovaBus and services the GPU's
// asset request itself; this page shows its console output and
// cross-checks the framebuffer checksum it prints.

import SwiftUI
import Foundation

enum ProofRunner {
    static func run() -> [String] {
        var out: [String] = []
        guard let url = Bundle.main.url(forResource: "gpu_driver", withExtension: "bin"),
              let driverBin = try? Data(contentsOf: url) else {
            return ["FAIL  gpu_driver.bin missing from the app bundle"]
        }
        let fbW = 256, fbH = 192
        let cube = NovaGPUStandard.proofCube()
        let vsProg = NovaGPUStandard.vertexLitProgram()
        let fsProg = NovaGPUStandard.fragmentLitProgram()
        let proofMVP: [Float] = [0.8, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 1]

        // Expected checksum via the Swift bus client.
        var expected: UInt64 = 0
        do {
            let mem = NovaMemory(totalSize: 64 * 1024 * 1024)
            let device = NovaGPUDevice()
            let client = NovaBusClient(memory: mem, device: device)
            let fb = try client.alloc(UInt64(fbW * fbH * 4))
            let vb = try client.alloc(UInt64(cube.vertices.count * 4))
            let ib = try client.alloc(UInt64(cube.indices.count * 4))
            let sh = try client.alloc(8192)
            try client.uploadData(sh, data: NovaGPUStandard.programBytes(vsProg), guestStaging: 0x40000)
            try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fsProg), guestStaging: 0x40000)
            try client.setShader(.setVShader, programHandle: sh, off: 0, len: vsProg.count * 4)
            try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fsProg.count * 4)
            try client.uploadData(ib, data: NovaGPUStandard.indexBytes(cube), guestStaging: 0x40000)
            try client.setFBuf(fb, fbW, fbH)
            try client.setVBuf(vb)
            try client.setIBuf(ib)
            try NovaGPUStandard.setLitUniforms(client, mvp: proofMVP, light: (0.5, 1.0, 0.25))
            try client.clear(0xFF200810)
            client.serviceHandler = { words in
                mem.storeHostBytes(NovaGPUStandard.vertexBytes(cube), at: words[2])
            }
            try client.draw(indexCount: cube.indices.count, flags: 1,
                            assetStaging: 0x30000, assetLen: UInt64(cube.vertices.count * 4),
                            assetTarget: vb)
            try client.present(fb)
            let px = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: 0x80000)
            expected = px.reduce(UInt64(0)) { $0 + UInt64($1) }
            out.append("client reference framebuffer sum: \(expected)")
        } catch {
            return ["FAIL  client reference render: \(error)"]
        }

        // Boot the real NovaCPU on the assembly driver.
        let memory = NovaMemory(totalSize: 64 * 1024 * 1024)
        let cpu = NovaCPU()
        cpu.memory = memory
        let device = NovaGPUDevice()
        device.attach(to: memory)
        // v6: the supervisor chip watches this CPU<->GPU pair. The
        // run-slice size below comes from the chip (SET_CPU_SLICE),
        // and every slice's wall time is reported back to it.
        let governor = NovaGovernorDevice()
        governor.attach(to: memory)
        governor.supervise(device)
        device.governor = governor
        governor.chip.setEnabled(true)
        var consoleText = ""
        memory.consoleOutput = { s in consoleText += s }
        memory.netHandler = { _, _, _, _ in -1 }
        cpu.trapHandler = { _, _ in cpu.running = false }
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
            return out + ["FAIL  loading guest memory: \(error)"]
        }
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
        // Run in slices, servicing the GPU between slices (the device
        // does its DMA at service points, never inside an MMIO write —
        // see NovaBus.swift). v6: the slice size is the governor's
        // SET_CPU_SLICE actuator — each slice is timed and reported
        // to the chip, which steers the size toward its 2 ms target.
        var budget: UInt64 = 0
        var slices = 0
        let runT0 = Date()
        repeat {
            let slice = governor.chip.cpuSliceSteps
            let s0 = Date()
            cpu.run(maxSteps: slice)
            let sliceMs = Date().timeIntervalSince(s0) * 1000
            budget += slice
            slices += 1
            device.servicePendingWork()
            governor.chip.tickCPU(sliceMs: sliceMs, steps: slice)
        } while cpu.running && budget < 200_000_000
        let runWall = Date().timeIntervalSince(runT0)
        out.append("CPU run: \(String(format: "%.2f", runWall)) s wall, \(slices) governor-supervised slices, final slice \(governor.chip.cpuSliceSteps) steps")
        for a in governor.chip.actionLog {
            out.append("  governor: \(a.text)")
        }

        out.append("guest console:")
        for l in consoleText.split(separator: "\n") { out.append("  | \(l)") }
        out.append(consoleText.contains("GPU OK") ? "PASS  driver printed GPU OK" : "FAIL  no GPU OK from driver")
        var driverSum: UInt64? = nil
        for l in consoleText.split(separator: "\n") where l.hasPrefix("GPUFB ") {
            driverSum = UInt64(l.dropFirst(6).trimmingCharacters(in: .whitespaces))
        }
        if let ds = driverSum {
            out.append(ds == expected
                ? "PASS  CPU-driven framebuffer matches the client render (\(ds))"
                : "FAIL  checksum mismatch: driver \(ds) vs expected \(expected)")
        } else {
            out.append("FAIL  driver printed no GPUFB checksum")
        }
        out.append("descriptors executed by the GPU: \(device.descriptorsExecuted)")
        return out
    }
}

struct CPUProofView: View {
    @State private var lines: [String] = ["Tap Run to boot the real NovaCPU core on the Nova assembly GPU driver (Driver/gpu_driver.nova). The driver renders a cube over NovaBus and answers the GPU's asset request itself."]
    @State private var running = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("NovaCPU <-> NovaGPU Proof").font(.headline)
            Button(running ? "Running…" : "Run CPU Proof") {
                running = true
                DispatchQueue.global(qos: .userInitiated).async {
                    let result = ProofRunner.run()
                    DispatchQueue.main.async { lines = result; running = false }
                }
            }.disabled(running)
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(lines.enumerated()), id: \.offset) { _, l in
                        Text(l).font(.system(.footnote, design: .monospaced))
                    }
                }
            }
        }.padding()
    }
}
