// NovaGovernor.swift — the v6 supervisor chip.
//
// Sean's design (2026-11-13): "a new chip that is very basic. It
// has 1 goal which is to maintain the gpu and cpu to its best
// ability. It will change the gpu and cpu priority (what would be
// amount of power in real life) and can change memory allocation
// and move special commands to the cpu and gpu. These special
// commands would tell the gpu and cpu how to act — this is kind
// of like a bios chip but not just memory."
//
// The chip is a first-class device, not app glue:
//   * It SENSES both sides: per-frame telemetry (frame work time
//     and its rolling median / p95 / jitter, bus descriptors,
//     VRAM committed vs reserved) and CPU telemetry (run-slice
//     wall time from the host loop that drives the NovaCPU).
//   * It ACTS through a small set of named special commands —
//     SET_PACE_TIER, SET_RASTER_GRAIN, SET_GPU_WIDTH,
//     SET_CPU_SLICE, SET_SCRATCH_RETENTION — each issued through
//     the single issue() path, applied to real knobs (the GPU's
//     operating knobs, the CPU host loop's slice size, the frame
//     pacing target the host schedules against), and written to
//     an action log with the numbers that caused it. A command
//     that changes nothing is never issued; disabling the chip
//     restores the v5 operating point through the same path.
//   * It is reachable over the bus two ways: its own MMIO
//     register block (0xFFFF_FFFF_FFFF_0180–0xFFFF_FFFF_FFFF_0200,
//     beside the GPU's — the NovaCPU can read the chip's state
//     and tick it like any other device) and GPU ring opcode 21.
//   * Its policy is a few dozen lines of deterministic rules over
//     the named thresholds below. No learning, no hidden state
//     beyond the telemetry window and the streak counters.
//
// Frame pacing is the headline behaviour: the chip holds a pace
// tier (a target frame interval) and steps it down when sustained
// frame work overruns the tier, back up when headroom persists —
// with hysteresis (asymmetric streaks + a cooldown), so the
// target itself never oscillates. It never drops work and never
// changes the picture: every actuator changes only how work is
// partitioned, scheduled or retained, and the framebuffer
// checksums police that on every CI run.

import Foundation

/// Precise wait until a deadline on the NovaGPU.nowMs() clock:
/// a coarse Thread.sleep for the bulk, then a short spin for the
/// last stretch. The v6 bench convicted the plain sleep: on the
/// CI host its overshoot (waking 1–8 ms late on a loaded runner)
/// ADDED more frame-interval variance than the pacing removed,
/// so the pacer sleeps only down to the last ~1 ms and spins the
/// remainder — one busy core for at most that stretch, in
/// exchange for a deadline kept to ~0.1 ms.
func governorWait(untilMs deadline: Double, spinMs: Double = 1.0) {
    let remaining = deadline - NovaGPU.nowMs()
    if remaining > spinMs + 0.5 {
        Thread.sleep(forTimeInterval: (remaining - spinMs) / 1000)
    }
    while NovaGPU.nowMs() < deadline { }
}

/// One frame's telemetry, sampled at a frame boundary. Filled by
/// the bus device on every PRESENT (the inter-present interval is
/// the frame pacing the guest actually experiences) or by a host
/// ticking the chip explicitly over MMIO / the ring.
struct GovernorTelemetry {
    var frameWorkMs: Double = 0      // frame interval / work time
    var descriptors: UInt64 = 0      // bus descriptors executed (cumulative)
    var vramCommitted: Int = 0       // host bytes committed in VRAM blocks
    var vramReserved: UInt64 = 0     // live VRAM allocations, bytes
}

/// The chip's special commands — the complete vocabulary it can
/// issue to the CPU and GPU. Each names a real actuator.
enum GovernorCommand: String {
    case enable = "ENABLE"
    case disable = "DISABLE"
    case setPaceTier = "SET_PACE_TIER"
    case setRasterGrain = "SET_RASTER_GRAIN"
    case setGPUWidth = "SET_GPU_WIDTH"
    case setCPUSlice = "SET_CPU_SLICE"
    case setScratchRetention = "SET_SCRATCH_RETENTION"
}

/// One entry of the chip's action log: what changed, at which
/// tick, and the measured numbers that caused it.
struct GovernorAction {
    var tick: UInt64 = 0
    var command: GovernorCommand
    var value: String
    var reason: String
    var text: String { "#\(tick) \(command.rawValue) \(value) — \(reason)" }
}

final class NovaGovernor {
    /// The chip's name. Sean names his hardware — this is the ONE
    /// place the name lives; rename it here and the register
    /// banner, HUD and logs follow.
    static let chipName = "NovaGovernor"

    // MARK: Policy constants (the whole policy is these + tick())

    static let windowSize = 24                       // frames of telemetry kept
    static let paceTiersMs: [Double] = [16.7, 20.8, 25.0, 33.3, 50.0]
    static let cooldownTicks = 30                    // min ticks between frame commands
    static let overrunFactor = 1.02                  // median > tier × this = overrun
    static let headroomFactor = 0.80                 // median < lower tier × this = headroom
    static let overrunStreakToStep = 2               // windows of overrun to step down
    static let headroomStreakToStep = 4              // windows of headroom to step up
    static let jitterFactor = 0.30                   // jitter > this × median = unstable
    static let stableFactor = 0.08                   // jitter < this × median = stable
    static let contentionWidth = 6                   // GPU width cap under contention
    static let grainDefault = 4                      // v5 raster grain (rows/chunk)
    static let grainMin = 2                          // sweep: rows=1 costs ~70% mean
    static let grainMax = 8
    static let sliceTargetMs = 2.0                   // CPU slice wall steered toward
    static let sliceMin: UInt64 = 10_000
    // Capped at 200k after the attempt-2 proof log: at 400k the
    // slice's own wall hit 3.1 ms and every GPU service point the
    // driver waited on paid the longer latency — the cap keeps a
    // slice inside roughly one target band on the M-class host.
    static let sliceMax: UInt64 = 200_000
    static let sliceDefault: UInt64 = 100_000        // the v5 proof slice
    static let scratchShedBytes = 256 * 1024 * 1024  // VRAM commit level that sheds scratch
    static let scratchRestoreBytes = 128 * 1024 * 1024

    // MARK: State

    private(set) var enabled = false
    private(set) var tickCount: UInt64 = 0
    private(set) var actionsIssued: UInt64 = 0
    private(set) var actionLog: [GovernorAction] = []

    // Actuator state — the values the chip currently commands.
    private(set) var paceTierIndex = 0
    private(set) var gpuStageWidth = 0
    private(set) var rasterChunkRows = NovaGovernor.grainDefault
    private(set) var cpuSliceSteps: UInt64 = NovaGovernor.sliceDefault
    private(set) var scratchRetention = true

    // Telemetry readouts (HUD + registers).
    private(set) var lastFrameMs: Double = 0
    private(set) var windowMedianMs: Double = 0
    private(set) var windowP95Ms: Double = 0
    private(set) var descriptorsPerFrame: UInt64 = 0
    var jitterMs: Double { max(0, windowP95Ms - windowMedianMs) }
    var paceTargetMs: Double { Self.paceTiersMs[paceTierIndex] }
    var windowCount: Int { window.count }

    private var window: [Double] = []
    private var overStreak = 0
    private var underStreak = 0
    private var ticksSinceCommand = NovaGovernor.cooldownTicks
    private var lastDescriptors: UInt64 = 0

    /// The GPU whose knobs the chip drives (set by the device
    /// wrapper's supervise()). Weak: the chip never owns the GPU.
    weak var targetGPU: NovaGPU?

    private func fmt(_ v: Double) -> String { String(format: "%.1f", v) }

    // MARK: Enable / disable

    func setEnabled(_ on: Bool) {
        guard on != enabled else { return }
        if on {
            enabled = true
            issue(.enable, Self.chipName, "enabled by host")
        } else {
            // Restore the v5 operating point through the same
            // command path, so OFF provably means "as v5".
            if paceTierIndex != 0 {
                paceTierIndex = 0
                issue(.setPaceTier, "\(fmt(paceTargetMs)) ms",
                      "governor disabled — restoring the v5 operating point")
            }
            if gpuStageWidth != 0 {
                gpuStageWidth = 0
                issue(.setGPUWidth, "uncapped",
                      "governor disabled — restoring the v5 operating point")
            }
            if rasterChunkRows != Self.grainDefault {
                rasterChunkRows = Self.grainDefault
                issue(.setRasterGrain, "\(rasterChunkRows) rows/chunk",
                      "governor disabled — restoring the v5 operating point")
            }
            if cpuSliceSteps != Self.sliceDefault {
                cpuSliceSteps = Self.sliceDefault
                issue(.setCPUSlice, "\(cpuSliceSteps) steps",
                      "governor disabled — restoring the v5 operating point")
            }
            if !scratchRetention {
                scratchRetention = true
                issue(.setScratchRetention, "on",
                      "governor disabled — restoring the v5 operating point")
            }
            enabled = false
            record(.disable, Self.chipName, "disabled by host")
            applyToTarget()
        }
    }

    /// Clears the telemetry window and streaks (actuator state is
    /// kept). The MMIO CTRL register's reset.
    func resetWindow() {
        window.removeAll()
        overStreak = 0; underStreak = 0
        windowMedianMs = 0; windowP95Ms = 0; lastFrameMs = 0
    }

    // MARK: Command issue path

    private func record(_ cmd: GovernorCommand, _ value: String, _ reason: String) {
        actionsIssued += 1
        actionLog.append(GovernorAction(tick: tickCount, command: cmd,
                                        value: value, reason: reason))
        if actionLog.count > 64 { actionLog.removeFirst(actionLog.count - 64) }
    }

    /// The single path every actuator change takes: log it with
    /// its reason, push it into the supervised GPU, restart the
    /// cooldown.
    private func issue(_ cmd: GovernorCommand, _ value: String, _ reason: String) {
        record(cmd, value, reason)
        applyToTarget()
        ticksSinceCommand = 0
    }

    /// Writes the chip's current actuator state into the GPU it
    /// supervises.
    func applyToTarget() {
        guard let gpu = targetGPU else { return }
        gpu.rasterChunkRows = rasterChunkRows
        gpu.stageWidth = gpuStageWidth
        gpu.scratchRetentionEnabled = scratchRetention
    }

    // MARK: Frame tick (the control loop)

    /// Frame-boundary tick. Sensing always runs (the window and
    /// readouts stay live while disabled); acting requires the
    /// chip enabled, a filled window and an expired cooldown, and
    /// issues at most one command per tick.
    func tick(_ t: GovernorTelemetry) {
        tickCount += 1
        lastFrameMs = t.frameWorkMs
        if t.descriptors >= lastDescriptors {
            descriptorsPerFrame = t.descriptors - lastDescriptors
        }
        lastDescriptors = t.descriptors
        window.append(t.frameWorkMs)
        if window.count > Self.windowSize { window.removeFirst() }
        let s = window.sorted()
        windowMedianMs = s[s.count / 2]
        windowP95Ms = s[min(s.count - 1, Int(0.95 * Double(s.count)))]
        guard enabled else { return }
        ticksSinceCommand += 1
        guard window.count >= 8, ticksSinceCommand >= Self.cooldownTicks else { return }
        let med = windowMedianMs, jit = jitterMs, tier = paceTargetMs

        // 1. Pace: sustained overrun steps the target tier down
        //    (a longer, achievable slot); sustained headroom steps
        //    it back up. Asymmetric streaks = the hysteresis.
        if med > tier * Self.overrunFactor {
            overStreak += 1; underStreak = 0
            if overStreak >= Self.overrunStreakToStep,
               paceTierIndex < Self.paceTiersMs.count - 1 {
                paceTierIndex += 1
                overStreak = 0
                issue(.setPaceTier, "\(fmt(paceTargetMs)) ms",
                      "median frame work \(fmt(med)) ms overruns the \(fmt(tier)) ms tier")
                return
            }
        } else if paceTierIndex > 0,
                  med < Self.paceTiersMs[paceTierIndex - 1] * Self.headroomFactor {
            underStreak += 1; overStreak = 0
            if underStreak >= Self.headroomStreakToStep {
                let old = tier
                paceTierIndex -= 1
                underStreak = 0
                issue(.setPaceTier, "\(fmt(paceTargetMs)) ms",
                      "median frame work \(fmt(med)) ms leaves headroom under the \(fmt(old)) ms tier")
                return
            }
        } else { overStreak = 0; underStreak = 0 }

        // 2. Jitter: a wide p95–median gap is a ragged stage tail —
        //    finer raster grain smooths it (more, smaller chunks).
        if jit > Self.jitterFactor * med, rasterChunkRows > Self.grainMin {
            rasterChunkRows = max(Self.grainMin, rasterChunkRows / 2)
            issue(.setRasterGrain, "\(rasterChunkRows) rows/chunk",
                  "frame jitter \(fmt(jit)) ms is over 30% of the \(fmt(med)) ms median")
            return
        }
        // 3. Jitter persisting at minimum grain is contention:
        //    cap the GPU's host width so the CPU, the download
        //    path and the UI stop fighting the stages.
        if jit > Self.jitterFactor * med, gpuStageWidth == 0 {
            gpuStageWidth = Self.contentionWidth
            issue(.setGPUWidth, "\(gpuStageWidth)",
                  "frame jitter \(fmt(jit)) ms persists at grain \(rasterChunkRows) — capping GPU stage width")
            return
        }
        // 4. Stable and at/under target: hand the width back.
        if jit < Self.stableFactor * med, gpuStageWidth != 0, med <= tier {
            gpuStageWidth = 0
            issue(.setGPUWidth, "uncapped",
                  "frame jitter \(fmt(jit)) ms is under 8% of the median — GPU width restored")
            return
        }
        // 5. Stable at fine grain: the jitter that forced the
        //    fine grain is gone — restore the default grain (its
        //    per-chunk setup is cheaper; the sweeps measured it).
        if jit < Self.stableFactor * med, rasterChunkRows < Self.grainDefault {
            rasterChunkRows = min(Self.grainDefault, rasterChunkRows * 2)
            issue(.setRasterGrain, "\(rasterChunkRows) rows/chunk",
                  "frame jitter \(fmt(jit)) ms is under 8% of the median — default grain restored")
            return
        }
        // 6. Memory: a large committed VRAM footprint sheds the
        //    retained draw scratch (host RAM back to the system);
        //    it is restored when the footprint falls well below.
        if scratchRetention, t.vramCommitted > Self.scratchShedBytes {
            scratchRetention = false
            issue(.setScratchRetention, "off",
                  "VRAM committed \(t.vramCommitted / (1024 * 1024)) MB passes the 256 MB shed line")
            return
        }
        if !scratchRetention, t.vramCommitted < Self.scratchRestoreBytes {
            scratchRetention = true
            issue(.setScratchRetention, "on",
                  "VRAM committed \(t.vramCommitted / (1024 * 1024)) MB is back under the 128 MB restore line")
            return
        }
    }

    // MARK: CPU tick

    /// CPU-side tick: called by a NovaCPU host loop after each run
    /// slice with the slice's measured wall time. The chip steers
    /// the slice toward sliceTargetMs — slices that overrun shrink
    /// (the GPU gets serviced sooner; the CPU yields the host more
    /// often), slices far under target grow (less state sync per
    /// instruction). The deadband between the two thresholds is
    /// the hysteresis; the caller reads cpuSliceSteps back.
    func tickCPU(sliceMs: Double, steps: UInt64) {
        guard enabled else { return }
        if sliceMs > Self.sliceTargetMs * 1.5, cpuSliceSteps > Self.sliceMin {
            cpuSliceSteps = max(Self.sliceMin, cpuSliceSteps / 2)
            issue(.setCPUSlice, "\(cpuSliceSteps) steps",
                  "CPU slice of \(steps) steps took \(fmt(sliceMs)) ms — over the \(fmt(Self.sliceTargetMs)) ms target")
        } else if sliceMs < Self.sliceTargetMs * 0.5, cpuSliceSteps < Self.sliceMax {
            cpuSliceSteps = min(Self.sliceMax, cpuSliceSteps * 2)
            issue(.setCPUSlice, "\(cpuSliceSteps) steps",
                  "CPU slice of \(steps) steps took \(fmt(sliceMs)) ms — under half the target; growing it cuts sync overhead")
        }
    }
}

// MARK: - The device (MMIO register block)

enum GovRegister: UInt64 {
    case id          = 0x00   // RO: 0x4E474F56_00060000 ("NGOV", v6.0)
    case status      = 0x08   // RO: bit0 enabled, bit1 window full, bit2 scratch retained
    case enable      = 0x10   // RW: write 1 = enable, 0 = disable
    case paceTierUs  = 0x18   // RO: current pace target, microseconds
    case gpuWidth    = 0x20   // RO: GPU stage width (0 = uncapped)
    case rasterGrain = 0x28   // RO: raster rows per chunk
    case cpuSlice    = 0x30   // RO: CPU run-slice size, steps
    case lastFrameUs = 0x38   // RO: last frame interval, microseconds
    case jitterUs    = 0x40   // RO: window p95 − median, microseconds
    case actionCount = 0x48   // RO: commands issued since boot
    case medianUs    = 0x50   // RO: window median frame, microseconds
    case ctrl        = 0x58   // WO: 1 = reset telemetry window
    case tick        = 0x60   // WO: tick with a frame-work time in microseconds
}

/// The governor as a bus device: a register block in the MMIO
/// page beside the GPU's, exactly in the style of the other
/// NovaMemory devices. The host attaches it to a NovaMemory, then
/// points it at the GPU device it supervises. The GPU device also
/// calls notePresent() on every completed PRESENT, so the chip
/// ticks on real frame boundaries without the CPU doing anything.
final class NovaGovernorDevice {
    static let deviceID: UInt64 = 0x4E47_4F56_0006_0000

    let chip = NovaGovernor()
    weak var memory: NovaMemory?
    private weak var gpuDevice: NovaGPUDevice?
    private var lastPresentMs: Double = 0

    func attach(to memory: NovaMemory) {
        self.memory = memory
        memory.govDevice = self
    }

    /// Point the chip at the GPU device it maintains, and push
    /// its current actuator state into that GPU.
    func supervise(_ device: NovaGPUDevice) {
        gpuDevice = device
        chip.targetGPU = device.gpu
        chip.applyToTarget()
    }

    /// A PRESENT completed on the supervised GPU: tick the chip
    /// on the inter-present interval, with the device's own
    /// telemetry. Intervals over 250 ms are a resync (tab switch,
    /// first frame after attach), not a sample.
    func notePresent() {
        let now = NovaGPU.nowMs()
        defer { lastPresentMs = now }
        guard lastPresentMs > 0 else { return }
        let dt = now - lastPresentMs
        guard dt <= 250 else { return }
        guard let dev = gpuDevice else { return }
        chip.tick(GovernorTelemetry(
            frameWorkMs: dt,
            descriptors: dev.descriptorsExecuted,
            vramCommitted: dev.gpu.vramCommitted,
            vramReserved: dev.gpu.vramReserved))
    }

    // MARK: MMIO

    func mmioRead(offset: UInt64) -> UInt64 {
        guard let reg = GovRegister(rawValue: offset) else { return 0 }
        switch reg {
        case .id: return Self.deviceID
        case .status:
            var s: UInt64 = 0
            if chip.enabled { s |= 1 }
            if chip.windowCount >= NovaGovernor.windowSize { s |= 2 }
            if chip.scratchRetention { s |= 4 }
            return s
        case .enable: return chip.enabled ? 1 : 0
        case .paceTierUs: return UInt64(chip.paceTargetMs * 1000)
        case .gpuWidth: return UInt64(chip.gpuStageWidth)
        case .rasterGrain: return UInt64(chip.rasterChunkRows)
        case .cpuSlice: return chip.cpuSliceSteps
        case .lastFrameUs: return UInt64(chip.lastFrameMs * 1000)
        case .jitterUs: return UInt64(chip.jitterMs * 1000)
        case .actionCount: return chip.actionsIssued
        case .medianUs: return UInt64(chip.windowMedianMs * 1000)
        case .ctrl, .tick: return 0
        }
    }

    func mmioWrite(offset: UInt64, value: UInt64) {
        guard let reg = GovRegister(rawValue: offset) else { return }
        switch reg {
        case .enable: chip.setEnabled(value != 0)
        case .ctrl: if value == 1 { chip.resetWindow() }
        case .tick:
            chip.tick(GovernorTelemetry(
                frameWorkMs: Double(value) / 1000,
                descriptors: gpuDevice?.descriptorsExecuted ?? 0,
                vramCommitted: gpuDevice?.gpu.vramCommitted ?? 0,
                vramReserved: gpuDevice?.gpu.vramReserved ?? 0))
        default: break
        }
    }
}
