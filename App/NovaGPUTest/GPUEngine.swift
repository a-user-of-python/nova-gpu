// GPUEngine.swift — the live scene. Every GPU interaction goes
// through NovaBusClient (descriptors in guest RAM + MMIO doorbell);
// the engine never touches NovaGPU internals. The frame is pulled
// back with a bus DOWNLOAD after PRESENT, like a real driver would.

import SwiftUI
import Foundation

final class GPUEngine: ObservableObject {
    @Published var frame: CGImage?
    @Published var fps: Double = 0
    @Published var frameMs: Double = 0
    @Published var cusActive: Int = 0
    @Published var vramUsedKiB: Int = 0
    @Published var distinctColours: Int = 0
    @Published var triangles: Int = 0
    @Published var statusLine: String = "setting up…"

    private let fbW = 640, fbH = 480
    private let mem = NovaMemory(totalSize: 128 * 1024 * 1024)
    private let device = NovaGPUDevice()
    private var client: NovaBusClient!
    private var fb: UInt64 = 0
    private var sphereV: UInt64 = 0, sphereI: UInt64 = 0
    private var torusV: UInt64 = 0, torusI: UInt64 = 0
    private var sphereIdx = 0, torusIdx = 0
    private var ready = false
    private var angle: Float = 0
    private var frameCount = 0
    private var lastTick = Date()
    private let queue = DispatchQueue(label: "novagpu.engine", qos: .userInitiated)
    private var timer: DispatchSourceTimer?

    private let STAGE: UInt64 = 0x40000
    private let DSTAGE: UInt64 = 0x80000

    func start() {
        queue.async { [weak self] in self?.setup() }
    }

    private func setup() {
        do {
            client = NovaBusClient(memory: mem, device: device)
            let sphere = NovaGPUStandard.icosphere(subdivisions: 5)
            let torus = NovaGPUStandard.torus(rings: 96, slices: 48)
            sphereIdx = sphere.indices.count
            torusIdx = torus.indices.count
            fb = try client.alloc(UInt64(fbW * fbH * 4))
            let vs = NovaGPUStandard.vertexLitProgram()
            let fs = NovaGPUStandard.fragmentLitProgram()
            let sh = try client.alloc(8192)
            try client.uploadData(sh, data: NovaGPUStandard.programBytes(vs), guestStaging: STAGE)
            try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fs), guestStaging: STAGE)
            try client.setShader(.setVShader, programHandle: sh, off: 0, len: vs.count * 4)
            try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fs.count * 4)
            try client.setFBuf(fb, fbW, fbH)
            sphereV = try client.alloc(UInt64(sphere.vertices.count * 4))
            sphereI = try client.alloc(UInt64(sphere.indices.count * 4))
            try client.uploadData(sphereV, data: NovaGPUStandard.vertexBytes(sphere), guestStaging: STAGE)
            try client.uploadData(sphereI, data: NovaGPUStandard.indexBytes(sphere), guestStaging: STAGE)
            torusV = try client.alloc(UInt64(torus.vertices.count * 4))
            torusI = try client.alloc(UInt64(torus.indices.count * 4))
            try client.uploadData(torusV, data: NovaGPUStandard.vertexBytes(torus), guestStaging: STAGE)
            try client.uploadData(torusI, data: NovaGPUStandard.indexBytes(torus), guestStaging: STAGE)
            ready = true
            DispatchQueue.main.async {
                self.statusLine = "running — all traffic over NovaBus"
                self.triangles = sphere.triangleCount + torus.triangleCount
            }
            startLoop()
        } catch {
            DispatchQueue.main.async { self.statusLine = "setup failed: \(error)" }
        }
    }

    private func startLoop() {
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: .milliseconds(16))
        t.setEventHandler { [weak self] in self?.renderFrame() }
        t.resume()
        timer = t
    }

    private func renderFrame() {
        guard ready else { return }
        let t0 = Date()
        do {
            angle += 0.02
            let proj = NovaGPUStandard.matPerspective(k: 0.45)
            try client.clear(0xFF200810)
            var mvp = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
                NovaGPUStandard.matRotX(0.5),
                NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle), NovaGPUStandard.matScale(0.72))))
            try NovaGPUStandard.setLitUniforms(client, mvp: mvp)
            try client.setVBuf(sphereV); try client.setIBuf(sphereI)
            try client.draw(indexCount: sphereIdx)
            mvp = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
                NovaGPUStandard.matTranslate(-0.05, -0.1, 0.35),
                NovaGPUStandard.matMul(NovaGPUStandard.matRotX(1.1),
                                       NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle * 0.6),
                                                              NovaGPUStandard.matScale(0.62)))))
            try NovaGPUStandard.setLitUniforms(client, mvp: mvp)
            try client.setVBuf(torusV); try client.setIBuf(torusI)
            try client.draw(indexCount: torusIdx)
            try client.present(fb)
            let px = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
            let img = makeImage(px)
            frameCount += 1
            var colours = distinctColours
            if frameCount % 20 == 0 {
                var set = Set<UInt32>()
                for i in stride(from: 0, to: px.count, by: 4) {
                    set.insert(UInt32(px[i]) | (UInt32(px[i + 1]) << 8) | (UInt32(px[i + 2]) << 16) | (UInt32(px[i + 3]) << 24))
                }
                colours = set.count
            }
            let now = Date()
            let dt = now.timeIntervalSince(lastTick)
            lastTick = now
            let ms = now.timeIntervalSince(t0) * 1000
            let stats = device.gpu.stats
            DispatchQueue.main.async {
                self.frame = img
                self.frameMs = ms
                if dt > 0 { self.fps = self.fps * 0.8 + (1 / dt) * 0.2 }
                self.cusActive = stats.activeComputeUnits
                self.vramUsedKiB = Int(self.device.gpu.vramReserved / 1024)
                self.distinctColours = colours
            }
        } catch {
            DispatchQueue.main.async { self.statusLine = "frame failed: \(error)" }
        }
    }

    private func makeImage(_ px: [UInt8]) -> CGImage? {
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
        guard let provider = CGDataProvider(data: Data(px) as CFData) else { return nil }
        return CGImage(width: fbW, height: fbH, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: fbW * 4, space: cs, bitmapInfo: info,
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)
    }
}

struct LiveSceneView: View {
    @StateObject private var engine = GPUEngine()
    var body: some View {
        VStack(spacing: 12) {
            Text("NovaGPU — Live Scene").font(.headline)
            if let img = engine.frame {
                Image(img, scale: 1, label: Text("NovaGPU framebuffer"))
                    .resizable().aspectRatio(4 / 3, contentMode: .fit)
                    .border(Color.secondary)
            } else {
                Rectangle().fill(Color.black).aspectRatio(4 / 3, contentMode: .fit)
                    .overlay(Text("warming up…").foregroundColor(.white))
            }
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                GridRow { Text("Compute units active"); Text("\(engine.cusActive) / 1536").bold() }
                GridRow { Text("Triangles per frame"); Text("\(engine.triangles)").bold() }
                GridRow { Text("Frame time"); Text(String(format: "%.1f ms", engine.frameMs)).bold() }
                GridRow { Text("FPS"); Text(String(format: "%.1f", engine.fps)).bold() }
                GridRow { Text("VRAM reserved"); Text("\(engine.vramUsedKiB) KiB").bold() }
                GridRow { Text("Distinct colours"); Text("\(engine.distinctColours)").bold() }
            }.font(.system(.body, design: .monospaced))
            Text(engine.statusLine).font(.footnote).foregroundColor(.secondary)
            Spacer()
        }
        .padding()
        .onAppear { engine.start() }
    }
}
