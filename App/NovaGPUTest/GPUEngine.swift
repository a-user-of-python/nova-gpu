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
    // v5 VPU test HUD state (written from the frame loop below).
    @Published var vpuTestOn = false
    @Published var vpuTriangles = 0
    @Published var vpuMismatches = 0
    @Published var vpuMaxErr: Double = 0
    @Published var vpuTriPerSec: Double = 0
    @Published var vpuClusters = 0
    @Published var vpuPassMs: Double = 0
    private var vpuWanted = false   // queue-confined: the toggle's wish
    private var vpuArmed = false    // queue-confined: device state as last sent

    /// The VPU test toggle: the frame loop arms the test at the top
    /// of every frame batch while this is on (arming also resets
    /// the device's per-frame result window) and disarms it once
    /// when it goes off.
    func setVPUTest(_ on: Bool) {
        DispatchQueue.main.async { self.vpuTestOn = on }
        queue.async { self.vpuWanted = on }
    }

    // v6/v7 governor HUD state (written from the frame loop below).
    @Published var govOn = false
    @Published var govCeilingMs: Double = 0
    @Published var govPacingText = "off"
    @Published var govMedianMs: Double = 0
    @Published var govP95Ms: Double = 0
    @Published var govJitterMs: Double = 0
    @Published var govWidth = 0
    @Published var govGrain = 0
    @Published var govLastAction = ""
    private var govWanted = false   // queue-confined
    private var govSent = false     // queue-confined: state last sent over the ring
    private var governor: NovaGovernorDevice?
    private var paceNextStartMs: Double = 0   // NovaGPU.nowMs() clock; 0 = unset

    /// The governor toggle: enable/disable travel over the bus
    /// (ring opcode 21) at the top of the next frame batch, and
    /// while the chip's pacing is ENGAGED the frame loop holds
    /// frame starts to the chip's pace ceiling (the p25 of recent
    /// frame work — v7; the chip disengages pacing itself when
    /// the loop is work-bound). The chip ticks on every PRESENT.
    func setGovernor(_ on: Bool) {
        DispatchQueue.main.async { self.govOn = on }
        queue.async { self.govWanted = on }
    }

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
            // v6: the supervisor chip is attached from boot. Its
            // telemetry window fills on every PRESENT whether or
            // not it is enabled; it only ACTS (issues commands)
            // once the toggle enables it over the ring.
            let gov = NovaGovernorDevice()
            gov.attach(to: mem)
            gov.supervise(device)
            device.governor = gov
            governor = gov
            let sphere = NovaGPUStandard.icosphere(subdivisions: 5)
            let torus = NovaGPUStandard.torus(rings: 96, slices: 48)
            sphereIdx = sphere.indices.count
            torusIdx = torus.indices.count
            fb = try client.alloc(UInt64(fbW * fbH * 4))
            let vs = NovaGPUStandard.vertexLitProgramV4()   // v4: VMAT4 transform (byte-identical output)
            let fs = NovaGPUStandard.fragmentLitProgram()
            let sh = try client.alloc(8192)
            try client.uploadData(sh, data: NovaGPUStandard.programBytes(vs), guestStaging: STAGE)
            try client.uploadData(sh, off: 4096, data: NovaGPUStandard.programBytes(fs), guestStaging: STAGE)
            try client.setShader(.setVShader, programHandle: sh, off: 0, len: vs.count * 4)
            try client.setShader(.setFShader, programHandle: sh, off: 4096, len: fs.count * 4)
            try client.setFBuf(fb, fbW, fbH)
            // Round 3: the constant lit uniforms — mask rows 4–7,
            // light row 8, ones row 9 — are sent ONCE here. Frames
            // update only the MVP rows 0–3; v2 resent all ten rows
            // per object per frame (12 wasted round trips a frame).
            try client.setUniform(4, (1, 0, 0, 0))
            try client.setUniform(5, (0, 1, 0, 0))
            try client.setUniform(6, (0, 0, 1, 0))
            try client.setUniform(7, (0, 0, 0, 1))
            try client.setUniform(8, (0.577, 0.577, 0.577, 0))
            try client.setUniform(9, (1, 1, 1, 1))
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
        // v7 governor pacing: while the chip's pacing is ENGAGED,
        // frame starts are held to the chip's pace ceiling — the
        // p25 of recent frame work, never a fixed tier — for a
        // steady cadence instead of the raw 16 ms timer firing
        // whenever. A frame that overruns its slot starts the
        // next slot from now (no catch-up debt). While pacing is
        // disengaged (work-bound loop, guard lockout, calibrating)
        // there is no wait at all: the governor may never trade
        // speed for stability. The sleep is before t0, so frameMs
        // below still measures work, not pacing.
        if govWanted, let gov = governor, gov.chip.pacingEngaged {
            let interval = gov.chip.paceCeilingMs
            let now = NovaGPU.nowMs()
            if paceNextStartMs > 0, now < paceNextStartMs {
                governorWait(untilMs: paceNextStartMs)
            }
            let start = NovaGPU.nowMs()
            paceNextStartMs = max(start, paceNextStartMs) + interval
        } else {
            paceNextStartMs = 0
        }
        let t0 = Date()
        do {
            angle += 0.02
            let proj = NovaGPUStandard.matPerspective(k: 0.45)
            let mvpSphere = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
                NovaGPUStandard.matRotX(0.5),
                NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle), NovaGPUStandard.matScale(0.72))))
            let mvpTorus = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
                NovaGPUStandard.matTranslate(-0.05, -0.1, 0.35),
                NovaGPUStandard.matMul(NovaGPUStandard.matRotX(1.1),
                                       NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle * 0.6),
                                                              NovaGPUStandard.matScale(0.62)))))
            // Round 3: the whole frame is ONE batch — 14 descriptors,
            // one doorbell write, one device drain. v2 paid a full
            // descriptor round trip (write, doorbell MMIO, drain,
            // status poll) for each of its ~15 frame commands, on
            // top of the six per-object uniform resends now avoided.
            var batch: [(opcode: GPUCommand, fence: UInt64, args: [UInt64])] = []
            // v6 governor: enable/disable travel over the ring
            // (opcode 21) on the frame after the toggle changes.
            if govWanted != govSent {
                batch.append((opcode: .governorCmd, fence: 0,
                              args: [govWanted ? 1 : 2, 0, 0, 0, 0]))
                govSent = govWanted
            }
            // v5 VPU test: arm at the top of the batch while the
            // toggle is on (each arm resets the result window, so
            // the HUD numbers below cover exactly this frame's two
            // draws); disarm once when the toggle goes off.
            if vpuWanted {
                batch.append((opcode: .vpuTest, fence: 0, args: [1, 0, 0, 0, 0]))
                vpuArmed = true
            } else if vpuArmed {
                batch.append((opcode: .vpuTest, fence: 0, args: [0, 0, 0, 0, 0]))
                vpuArmed = false
            }
            batch.append((opcode: .clear, fence: 0, args: [UInt64(0xFF200810), 0, 0, 0, 0]))
            appendMVP(&batch, mvpSphere)
            batch.append((opcode: .setVBuf, fence: 0, args: [sphereV, 0, 0, 0, 0]))
            batch.append((opcode: .setIBuf, fence: 0, args: [sphereI, 0, 0, 0, 0]))
            batch.append((opcode: .draw, fence: 0, args: [UInt64(sphereIdx), 0, 0, 0, 0]))
            appendMVP(&batch, mvpTorus)
            batch.append((opcode: .setVBuf, fence: 0, args: [torusV, 0, 0, 0, 0]))
            batch.append((opcode: .setIBuf, fence: 0, args: [torusI, 0, 0, 0, 0]))
            batch.append((opcode: .draw, fence: 0, args: [UInt64(torusIdx), 0, 0, 0, 0]))
            batch.append((opcode: .present, fence: 0, args: [fb, 0, 0, 0, 0]))
            try client.submitBatch(batch)
            let px = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
            let img = makeImage(px)
            frameCount += 1
            // Round 3: the distinct-colour census (a Set build over
            // 307,200 pixels — ~6 ms on the hosted bench) ran on
            // THIS queue every 20th frame in v2: a periodic hitch
            // inside the frame time being measured. It now runs on
            // a utility queue over the already-downloaded bytes and
            // publishes when done; the displayed count trails by one
            // census period.
            if frameCount % 20 == 0 {
                DispatchQueue.global(qos: .utility).async { [weak self] in
                    var set = Set<UInt32>()
                    for i in stride(from: 0, to: px.count, by: 4) {
                        set.insert(UInt32(px[i]) | (UInt32(px[i + 1]) << 8) | (UInt32(px[i + 2]) << 16) | (UInt32(px[i + 3]) << 24))
                    }
                    let n = set.count
                    DispatchQueue.main.async { self?.distinctColours = n }
                }
            }
            let now = Date()
            let dt = now.timeIntervalSince(lastTick)
            lastTick = now
            let ms = now.timeIntervalSince(t0) * 1000
            let stats = device.gpu.stats
            let vt = device.gpu.vpuTest
            let vtArmed = vpuArmed
            let chip = governor?.chip
            DispatchQueue.main.async {
                self.frame = img
                self.frameMs = ms
                if dt > 0 { self.fps = self.fps * 0.8 + (1 / dt) * 0.2 }
                self.cusActive = stats.activeComputeUnits
                self.vramUsedKiB = Int(self.device.gpu.vramReserved / 1024)
                if vtArmed {
                    self.vpuTriangles = vt.trianglesTested
                    self.vpuMismatches = vt.mismatches
                    self.vpuMaxErr = Double(vt.maxAbsError)
                    self.vpuTriPerSec = vt.trianglesPerSecond
                    self.vpuClusters = vt.clustersEngaged
                    self.vpuPassMs = vt.passMs
                }
                if let chip = chip {
                    self.govCeilingMs = chip.paceCeilingMs
                    self.govPacingText = chip.pacingStateText
                    self.govMedianMs = chip.intervalMedianMs
                    self.govP95Ms = chip.intervalP95Ms
                    self.govJitterMs = chip.jitterMs
                    self.govWidth = chip.gpuStageWidth
                    self.govGrain = chip.rasterChunkRows
                    self.govLastAction = chip.actionLog.last?.text ?? ""
                }
            }
        } catch {
            DispatchQueue.main.async { self.statusLine = "frame failed: \(error)" }
        }
    }

    /// Appends the four setUniform descriptors for an MVP's rows
    /// 0–3 — the only uniforms that change per frame.
    private func appendMVP(_ batch: inout [(opcode: GPUCommand, fence: UInt64, args: [UInt64])], _ mvp: [Float]) {
        for r in 0..<4 {
            batch.append((opcode: .setUniform, fence: 0,
                          args: [UInt64(r),
                                 UInt64(mvp[r * 4].bitPattern), UInt64(mvp[r * 4 + 1].bitPattern),
                                 UInt64(mvp[r * 4 + 2].bitPattern), UInt64(mvp[r * 4 + 3].bitPattern)]))
        }
    }

    // Round 3: persistent double-buffered image storage. v2 built a
    // fresh Data copy and CGDataProvider for every frame; the two
    // buffers here alternate, so the provider backing the published
    // image is never the one being filled.
    private var imageBuffers: [NSMutableData] = []
    private var imageProviders: [CGDataProvider] = []
    private var imageFlip = 0

    private func makeImage(_ px: [UInt8]) -> CGImage? {
        let n = fbW * fbH * 4
        if imageBuffers.isEmpty {
            for _ in 0..<2 {
                let d = NSMutableData(length: n)!
                imageBuffers.append(d)
                if let p = CGDataProvider(data: d as CFData) { imageProviders.append(p) }
            }
            if imageProviders.count != 2 { return nil }
        }
        imageFlip = 1 - imageFlip
        px.withUnsafeBytes { src in
            if let base = src.baseAddress {
                imageBuffers[imageFlip].replaceBytes(in: NSRange(location: 0, length: n), withBytes: base)
            }
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
        return CGImage(width: fbW, height: fbH, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: fbW * 4, space: cs, bitmapInfo: info,
                       provider: imageProviders[imageFlip], decode: nil, shouldInterpolate: false,
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
            Toggle("VPU test — position + render via the VPUs", isOn: Binding(
                get: { engine.vpuTestOn },
                set: { engine.setVPUTest($0) }))
                .font(.system(.body, design: .monospaced))
            Toggle("\(NovaGovernor.chipName) — supervisor chip (pace + stabilize)", isOn: Binding(
                get: { engine.govOn },
                set: { engine.setGovernor($0) }))
                .font(.system(.body, design: .monospaced))
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                GridRow { Text("Compute units active"); Text("\(engine.cusActive) / \(NovaGPU.computeUnits)").bold() }
                GridRow { Text("Triangles per frame"); Text("\(engine.triangles)").bold() }
                GridRow { Text("Frame time"); Text(String(format: "%.1f ms", engine.frameMs)).bold() }
                GridRow { Text("FPS"); Text(String(format: "%.1f", engine.fps)).bold() }
                GridRow { Text("VRAM reserved"); Text("\(engine.vramUsedKiB) KiB").bold() }
                GridRow { Text("Distinct colours"); Text("\(engine.distinctColours)").bold() }
                if engine.vpuTestOn {
                    GridRow {
                        Text("VPU verdict")
                        Text(engine.vpuMismatches == 0 ? "BIT-EXACT" : "MISMATCH")
                            .bold().foregroundColor(engine.vpuMismatches == 0 ? .green : .red)
                    }
                    GridRow { Text("VPU triangles tested"); Text("\(engine.vpuTriangles)").bold() }
                    GridRow { Text("VPU mismatched corners"); Text("\(engine.vpuMismatches)").bold() }
                    GridRow { Text("VPU max position error"); Text(String(format: "%.3g", engine.vpuMaxErr)).bold() }
                    GridRow { Text("VPU throughput"); Text("\(Int(engine.vpuTriPerSec)) tri/s").bold() }
                    GridRow { Text("MGVPU clusters engaged"); Text("\(engine.vpuClusters) / \(NovaGPU.mgvpuCount)").bold() }
                    GridRow { Text("VPU pass time"); Text(String(format: "%.2f ms", engine.vpuPassMs)).bold() }
                }
                if engine.govOn {
                    GridRow { Text("Governor pacing"); Text(engine.govPacingText).bold() }
                    GridRow { Text("Governor pace ceiling"); Text(String(format: "%.1f ms", engine.govCeilingMs)).bold() }
                    GridRow { Text("Frame interval p50 / p95"); Text(String(format: "%.1f / %.1f ms", engine.govMedianMs, engine.govP95Ms)).bold() }
                    GridRow { Text("Frame jitter (p95 − p50)"); Text(String(format: "%.1f ms", engine.govJitterMs)).bold() }
                    GridRow { Text("GPU width / raster grain"); Text(engine.govWidth == 0 ? "uncapped / \(engine.govGrain)" : "\(engine.govWidth) / \(engine.govGrain)").bold() }
                }
            }.font(.system(.body, design: .monospaced))
            if engine.govOn && !engine.govLastAction.isEmpty {
                Text("Governor: \(engine.govLastAction)")
                    .font(.footnote).foregroundColor(.secondary)
            }
            Text(engine.statusLine).font(.footnote).foregroundColor(.secondary)
            Spacer()
        }
        .padding()
        .onAppear { engine.start() }
    }
}

// MARK: - v4 Vectors page (plot unit)
//
// A vector-display scene rendered ONLY by the plot unit: a
// rotating wireframe cube (12 edges as plotted line vectors) with
// its 8 corners plotted as points. No triangle pipeline is
// involved — the grab unit fetches the corner positions, VPUs run
// the 3-axis position math (uniforms c0..c3), endpoints land on
// the 16-bit-per-axis screen grid, and the DDA rasteriser feeds
// the ordinary fragment path with the c10 colour in slot 1.

final class VectorsEngine: ObservableObject {
    @Published var frame: CGImage?
    @Published var fps: Double = 0
    @Published var frameMs: Double = 0
    @Published var vectorsPlotted: Int = 0
    @Published var statusLine: String = "setting up…"

    private let fbW = 640, fbH = 480
    private let mem = NovaMemory(totalSize: 64 * 1024 * 1024)
    private let device = NovaGPUDevice()
    private var client: NovaBusClient!
    private var fb: UInt64 = 0
    private var cubeV: UInt64 = 0, lineI: UInt64 = 0, pointI: UInt64 = 0
    private var ready = false
    private var angle: Float = 0
    private var lastTick = Date()
    private let queue = DispatchQueue(label: "novagpu.vectors", qos: .userInitiated)
    private var timer: DispatchSourceTimer?

    private let STAGE: UInt64 = 0x40000
    private let DSTAGE: UInt64 = 0x80000

    func start() {
        queue.async { [weak self] in self?.setup() }
    }

    private func setup() {
        do {
            client = NovaBusClient(memory: mem, device: device)
            fb = try client.alloc(UInt64(fbW * fbH * 4))
            // Flat fragment program: OUT 0 = varying slot 1 (the
            // plot colour), verbatim.
            var flat = ShaderBuilder()
            flat.ldattr(0, 1); flat.out(0, 0); flat.end()
            let sh = try client.alloc(4096)
            try client.uploadData(sh, data: NovaGPUStandard.programBytes(flat.build()), guestStaging: STAGE)
            try client.setShader(.setFShader, programHandle: sh, off: 0, len: flat.build().count * 4)
            try client.setFBuf(fb, fbW, fbH)
            // Cube corners (±0.6) as an interleaved vertex buffer
            // (only positions are read by the plot unit).
            let c: Float = 0.6
            let corners: [(Float, Float, Float)] = [
                (-c, -c, -c), (c, -c, -c), (c, c, -c), (-c, c, -c),
                (-c, -c, c), (c, -c, c), (c, c, c), (-c, c, c),
            ]
            var vdata = [UInt8]()
            for v in corners {
                var rec = [Float](repeating: 0, count: 12)
                rec[0] = v.0; rec[1] = v.1; rec[2] = v.2
                for f in rec { withUnsafeBytes(of: f.bitPattern) { vdata.append(contentsOf: $0) } }
            }
            cubeV = try client.alloc(UInt64(vdata.count))
            try client.uploadData(cubeV, data: vdata, guestStaging: STAGE)
            let edges: [UInt32] = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
            var edata = [UInt8]()
            for i in edges { withUnsafeBytes(of: i) { edata.append(contentsOf: $0) } }
            lineI = try client.alloc(UInt64(edata.count))
            try client.uploadData(lineI, data: edata, guestStaging: STAGE)
            var pdata = [UInt8]()
            for i in 0..<8 { withUnsafeBytes(of: UInt32(i)) { pdata.append(contentsOf: $0) } }
            pointI = try client.alloc(UInt64(pdata.count))
            try client.uploadData(pointI, data: pdata, guestStaging: STAGE)
            ready = true
            DispatchQueue.main.async { self.statusLine = "plotting — plot unit only, no triangles" }
            let t = DispatchSource.makeTimerSource(queue: queue)
            t.schedule(deadline: .now(), repeating: .milliseconds(16))
            t.setEventHandler { [weak self] in self?.renderFrame() }
            t.resume()
            timer = t
        } catch {
            DispatchQueue.main.async { self.statusLine = "setup failed: \(error)" }
        }
    }

    private func renderFrame() {
        guard ready else { return }
        let t0 = Date()
        do {
            angle += 0.03
            let proj = NovaGPUStandard.matPerspective(k: 0.45)
            let mvp = NovaGPUStandard.matMul(proj, NovaGPUStandard.matMul(
                NovaGPUStandard.matRotX(0.6),
                NovaGPUStandard.matMul(NovaGPUStandard.matRotY(angle), NovaGPUStandard.matScale(0.8))))
            var batch: [(opcode: GPUCommand, fence: UInt64, args: [UInt64])] = []
            batch.append((opcode: .clear, fence: 0, args: [UInt64(0xFF0A0A12), 0, 0, 0, 0]))
            for r in 0..<4 {
                batch.append((opcode: .setUniform, fence: 0,
                              args: [UInt64(r),
                                     UInt64(mvp[r * 4].bitPattern), UInt64(mvp[r * 4 + 1].bitPattern),
                                     UInt64(mvp[r * 4 + 2].bitPattern), UInt64(mvp[r * 4 + 3].bitPattern)]))
            }
            batch.append((opcode: .setVBuf, fence: 0, args: [cubeV, 0, 0, 0, 0]))
            // Edges in green.
            batch.append((opcode: .setUniform, fence: 0,
                          args: [10, UInt64(Float(0.25).bitPattern), UInt64(Float(1.0).bitPattern),
                                 UInt64(Float(0.35).bitPattern), UInt64(Float(1.0).bitPattern)]))
            batch.append((opcode: .setIBuf, fence: 0, args: [lineI, 0, 0, 0, 0]))
            batch.append((opcode: .plotVectors, fence: 0, args: [24, 0, 0, 0, 0]))
            // Corners as white points.
            batch.append((opcode: .setUniform, fence: 0,
                          args: [10, UInt64(Float(1.0).bitPattern), UInt64(Float(1.0).bitPattern),
                                 UInt64(Float(1.0).bitPattern), UInt64(Float(1.0).bitPattern)]))
            batch.append((opcode: .setIBuf, fence: 0, args: [pointI, 0, 0, 0, 0]))
            batch.append((opcode: .plotVectors, fence: 0, args: [8, 1, 0, 0, 0]))
            batch.append((opcode: .present, fence: 0, args: [fb, 0, 0, 0, 0]))
            try client.submitBatch(batch)
            let px = try client.downloadData(fb, len: fbW * fbH * 4, guestStaging: DSTAGE)
            let img = makeImage(px)
            let now = Date()
            let dt = now.timeIntervalSince(lastTick)
            lastTick = now
            let ms = now.timeIntervalSince(t0) * 1000
            let plotted = Int(device.gpu.stats.vectorsPlotted % 100000)
            DispatchQueue.main.async {
                self.frame = img
                self.frameMs = ms
                if dt > 0 { self.fps = self.fps * 0.8 + (1 / dt) * 0.2 }
                self.vectorsPlotted = plotted
            }
        } catch {
            DispatchQueue.main.async { self.statusLine = "frame failed: \(error)" }
        }
    }

    private var imageBuffers: [NSMutableData] = []
    private var imageProviders: [CGDataProvider] = []
    private var imageFlip = 0

    private func makeImage(_ px: [UInt8]) -> CGImage? {
        let n = fbW * fbH * 4
        if imageBuffers.isEmpty {
            for _ in 0..<2 {
                let d = NSMutableData(length: n)!
                imageBuffers.append(d)
                if let p = CGDataProvider(data: d as CFData) { imageProviders.append(p) }
            }
            if imageProviders.count != 2 { return nil }
        }
        imageFlip = 1 - imageFlip
        px.withUnsafeBytes { src in
            if let base = src.baseAddress {
                imageBuffers[imageFlip].replaceBytes(in: NSRange(location: 0, length: n), withBytes: base)
            }
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
        return CGImage(width: fbW, height: fbH, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: fbW * 4, space: cs, bitmapInfo: info,
                       provider: imageProviders[imageFlip], decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)
    }
}

struct VectorsView: View {
    @StateObject private var engine = VectorsEngine()
    var body: some View {
        VStack(spacing: 12) {
            Text("NovaGPU — Vectors (plot unit)").font(.headline)
            if let img = engine.frame {
                Image(img, scale: 1, label: Text("NovaGPU vector framebuffer"))
                    .resizable().aspectRatio(4 / 3, contentMode: .fit)
                    .border(Color.secondary)
            } else {
                Rectangle().fill(Color.black).aspectRatio(4 / 3, contentMode: .fit)
                    .overlay(Text("warming up…").foregroundColor(.white))
            }
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                GridRow { Text("VPU pool"); Text("\(NovaGPU.mgvpuCount) MGVPU × \(NovaGPU.vpusPerMgvpu) = \(NovaGPU.vpuCount) VPUs").bold() }
                GridRow { Text("Vectors per frame"); Text("20 (12 edges + 8 points)").bold() }
                GridRow { Text("Frame time"); Text(String(format: "%.1f ms", engine.frameMs)).bold() }
                GridRow { Text("FPS"); Text(String(format: "%.1f", engine.fps)).bold() }
            }.font(.system(.body, design: .monospaced))
            Text(engine.statusLine).font(.footnote).foregroundColor(.secondary)
            Spacer()
        }
        .padding()
        .onAppear { engine.start() }
    }
}
