// SelfTests.swift — the in-app self-test page. Runs the same checks
// as the CI host tests, over NovaBus only, on a fresh device, and
// returns human-readable result lines.

import SwiftUI
import Foundation

enum SelfTests {
    static func run() -> [String] {
        var out: [String] = []
        func line(_ ok: Bool, _ s: String) { out.append("\(ok ? "PASS" : "FAIL")  \(s)") }
        let mem = NovaMemory(totalSize: 64 * 1024 * 1024)
        let device = NovaGPUDevice()
        let client = NovaBusClient(memory: mem, device: device)
        let STAGE: UInt64 = 0x40000, DSTAGE: UInt64 = 0x80000
        let base = NovaMemory.gpuBaseAddr
        do {
            let id = mem.mmioRead(base + GPURegister.id.rawValue)
            line(id == 0x4E47_5055_0001_0000, String(format: "device ID 0x%016X", id))
            line(NovaGPU.computeUnits == 1536, "1536 compute units (48 clusters x 32 lanes)")
            line(NovaGPU.vramSize == 2 * 1024 * 1024 * 1024, "2 GiB VRAM address space")

            // SAXPY vs CPU reference
            let n = 200_001
            let alpha: Float = 1.75
            let h = try client.alloc(UInt64(n * 8))
            var data = [UInt8](repeating: 0, count: n * 8)
            func setF(_ idx: Int, _ v: Float) {
                let b = v.bitPattern
                data[idx * 4] = UInt8(b & 0xFF); data[idx * 4 + 1] = UInt8((b >> 8) & 0xFF)
                data[idx * 4 + 2] = UInt8((b >> 16) & 0xFF); data[idx * 4 + 3] = UInt8((b >> 24) & 0xFF)
            }
            func getF(_ buf: [UInt8], _ idx: Int) -> Float {
                Float(bitPattern: UInt32(buf[idx * 4]) | (UInt32(buf[idx * 4 + 1]) << 8)
                    | (UInt32(buf[idx * 4 + 2]) << 16) | (UInt32(buf[idx * 4 + 3]) << 24))
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
            try client.launch(lanes: n, buffer: h)
            let back = try client.downloadData(h, len: n * 8, guestStaging: DSTAGE)
            var maxErr: Float = 0
            for i in 0..<n {
                let expect = alpha * (Float(i % 97) * 0.25 - 12) + Float(i % 13) * 0.5
                maxErr = max(maxErr, abs(getF(back, n + i) - expect))
            }
            line(maxErr < 0.01, "compute SAXPY \(n) lanes vs CPU reference (max err \(maxErr))")
            try client.free(h); try client.free(ph)

            // Memory: 128 MiB fill + verify; sparse 1 GiB reservation
            let big = try client.alloc(128 * 1024 * 1024)
            try client.fill(big, off: 0, len: 128 * 1024 * 1024, pattern: 0xA5A5A5A5)
            var okFill = true
            for off in [0, 64 * 1024 * 1024, 128 * 1024 * 1024 - 4096] {
                let w = try client.downloadData(big, off: UInt64(off), len: 4096, guestStaging: DSTAGE)
                if !w.allSatisfy({ $0 == 0xA5 }) { okFill = false }
            }
            line(okFill, "128 MiB alloc + fill + verify")
            try client.free(big)
            let sparse = try client.alloc(1024 * 1024 * 1024)
            try client.uploadData(sparse, off: 1024 * 1024 * 1024 - 4,
                                  data: [0xDE, 0xAD, 0xBE, 0xEF], guestStaging: STAGE)
            let rb = try client.downloadData(sparse, off: 1024 * 1024 * 1024 - 4, len: 4, guestStaging: DSTAGE)
            line(rb == [0xDE, 0xAD, 0xBE, 0xEF],
                 "sparse: write/read at 1 GiB offset, committed \(device.gpu.vramCommitted / 1024) KiB")
            try client.free(sparse)

            // Colour depth from a shaded sphere frame
            let sphere = NovaGPUStandard.icosphere(subdivisions: 4)
            let fbW = 480, fbH = 360
            let fb = try client.alloc(UInt64(fbW * fbH * 4))
            let vs = NovaGPUStandard.vertexLitProgram()
            let fs = NovaGPUStandard.fragmentLitProgram()
            let sh = try client.alloc(8192)
            try client.uploadData(sh, data: NovaGPUStandard.programBytes(vs), guestStaging: STAGE)
            try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fs), guestStaging: STAGE)
            try client.setShader(.setVShader, programHandle: sh, off: 0, len: vs.count * 4)
            try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fs.count * 4)
            try client.setFBuf(fb, fbW, fbH)
            let vb = try client.alloc(UInt64(sphere.vertices.count * 4))
            let ib = try client.alloc(UInt64(sphere.indices.count * 4))
            try client.uploadData(vb, data: NovaGPUStandard.vertexBytes(sphere), guestStaging: STAGE)
            try client.uploadData(ib, data: NovaGPUStandard.indexBytes(sphere), guestStaging: STAGE)
            try client.setVBuf(vb); try client.setIBuf(ib)
            let proj = NovaGPUStandard.matPerspective(k: 0.45)
            let mvp = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
                NovaGPUStandard.matRotX(0.5),
                NovaGPUStandard.matMul(NovaGPUStandard.matRotY(0.8), NovaGPUStandard.matScale(0.75))))
            try NovaGPUStandard.setLitUniforms(client, mvp: mvp)
            try client.clear(0xFF200810)
            try client.draw(indexCount: sphere.indices.count)
            try client.present(fb)
            let px = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
            var distinct = Set<UInt32>()
            for i in stride(from: 0, to: px.count, by: 4) {
                distinct.insert(UInt32(px[i]) | (UInt32(px[i + 1]) << 8) | (UInt32(px[i + 2]) << 16) | (UInt32(px[i + 3]) << 24))
            }
            line(distinct.count > 1024, "colour depth: \(distinct.count) distinct RGBA8888 colours (> 1024)")
            line(device.gpu.stats.activeComputeUnits > 0,
                 "compute units active during render: \(device.gpu.stats.activeComputeUnits)")
            out.append("done — \(device.gpu.stats.trianglesIn) triangles in, \(device.gpu.stats.fragmentsShaded) fragments shaded")
        } catch {
            out.append("FAIL  suite aborted: \(error)")
        }
        return out
    }
}

struct SelfTestView: View {
    @State private var lines: [String] = ["Tap Run to execute the bus self-tests."]
    @State private var running = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("NovaGPU Self-Tests").font(.headline)
            Button(running ? "Running…" : "Run Self-Tests") {
                running = true
                DispatchQueue.global(qos: .userInitiated).async {
                    let result = SelfTests.run()
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
