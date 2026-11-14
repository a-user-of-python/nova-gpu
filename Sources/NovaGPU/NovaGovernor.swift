// NovaGovernor.swift — the supervisor chip (v6 design, v7 policy).
//
// Design brief (2026-11-13): "a new chip that is very basic. It
// has 1 goal which is to maintain the gpu and cpu to its best
// ability. It will change the gpu and cpu priority (what would be
// amount of power in real life) and can change memory allocation
// and move special commands to the cpu and gpu. These special
// commands would tell the gpu and cpu how to act — this is kind
// of like a bios chip but not just memory."
//
// The chip is a first-class device, not app glue:
//   * It SENSES both sides: per-frame telemetry (pure frame WORK
//     time and the inter-present INTERVAL, kept in separate
//     windows — v7, see below), bus descriptors, VRAM committed
//     vs reserved, and CPU run-slice wall times.
//   * It ACTS through a small set of named special commands —
//     SET_PACING, SET_RASTER_GRAIN, SET_GPU_WIDTH, SET_CPU_SLICE,
//     SET_SCRATCH_RETENTION — each issued through the single
//     issue() path, applied to real knobs, and written to an
//     action log with the numbers that caused it. A command that
//     changes nothing is never issued; disabling the chip
//     restores the v5 operating point through the same path.
//   * It is reachable over the bus two ways: its own MMIO
//     register block (0xFFFF_FFFF_FFFF_0180–0xFFFF_FFFF_FFFF_0200,
//     beside the GPU's — the NovaCPU can read the chip's state
//     and tick it like any other device) and GPU ring opcode 21.
//   * Its policy is a few dozen lines of deterministic rules over
//     the named thresholds below. No learning, no hidden state
//     beyond the telemetry windows and the streak counters.
//
// v7 — the never-slower law. The v6 pacing policy convicted
// itself on the tester's iPad: a stable 19.2 FPS where the scene
// ran ~35 ungoverned. Its three faults, in order:
//   1. Its only frame sensor was the inter-present INTERVAL —
//      once pacing was on, that number included the pacer's own
//      wait, so the chip could not see the work underneath.
//   2. It stepped a fixed tier ladder (16.7…50 ms) off the
//      interval MEDIAN: sustained work above a tier stepped the
//      target down, and the ladder's top was 50 ms.
//   3. Stepping back up required the median to fall below the
//      next tier × 0.80 — impossible while the pacer itself was
//      holding every interval at the current tier. A ratchet.
// The v7 rules that replace it:
//   * Work and interval are separate sensors. The pace ceiling
//     derives from WORK ONLY: it is the p25 of the recent work
//     window — a frame time the hardware demonstrably achieves —
//     recomputed every tick. It can never sit above speed.
//   * Work-limited disengagement: when the interval IS the work
//     (no slack in the loop), pacing stays disengaged; there is
//     nothing to pad, and stability work falls to the grain /
//     width / priority levers.
//   * A/B self-guard: a 24-frame ungoverned probe at enable
//     time measures the loop's slop (interval − work). After
//     that, the chip continuously checks two cost channels —
//     the governed interval median against the ungoverned
//     estimate (work median + slop), and the work median
//     against the snapshot taken when the newest lever was
//     applied (lever costs like a finer grain hide inside the
//     work itself). Either channel running more than 5% hot
//     reverts the newest lever, logged in plain words
//     ("pacing disengaged: it was costing frames"). After a
//     pacing trip, a fresh probe + a lockout must pass before
//     pacing may engage again.
// It still never drops work and never changes the picture: every
// actuator changes only how work is partitioned, scheduled or
// retained, and the framebuffer checksums police that on every
// CI run.

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
/// the bus device on every PRESENT or by a host ticking the chip
/// explicitly over MMIO / the ring. v7 keeps the two frame times
/// separate: frameWorkMs is the device's busy time (it excludes
/// every wait by construction); frameIntervalMs is the
/// present-to-present interval the viewer experiences. A sender
/// that only knows one number leaves frameIntervalMs at 0, which
/// the chip reads as "interval == work".
struct GovernorTelemetry {
    var frameWorkMs: Double = 0      // pure device work time
    var frameIntervalMs: Double = 0  // inter-present interval (0 = same as work)
    var descriptors: UInt64 = 0      // bus descriptors executed (cumulative)
    var vramCommitted: Int = 0       // host bytes committed in VRAM blocks
    var vramReserved: UInt64 = 0     // live VRAM allocations, bytes
}

/// The chip's special commands — the complete vocabulary it can
/// issue to the CPU and GPU. Each names a real actuator.
enum GovernorCommand: String {
    case enable = "ENABLE"
    case disable = "DISABLE"
    case setPacing = "SET_PACING"
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
    /// The chip's name, kept in exactly ONE place so it can be
    /// renamed with a single edit; the register banner, HUD and
    /// logs all read it from here.
    static let chipName = "NovaGovernor"

    // MARK: Policy constants (the whole policy is these + tick())

    static let windowSize = 24                       // frames of telemetry kept (each window)
    static let probeFrames = 24                      // ungoverned frames that calibrate the baseline
    static let cooldownTicks = 30                    // min ticks between frame commands
    static let guardFactor = 1.05                    // governed > estimate × this = costing frames
    static let guardStreakToTrip = 2                 // consecutive checks before a revert
    static let guardLockoutTicks: UInt64 = 240       // no re-engage after a pacing trip
    static let workBoundAbsMs = 1.0                  // interval − work ≤ this …
    static let workBoundFrac = 0.03                  // … or ≤ this fraction of the interval = work-bound
    static let jitterFactor = 0.30                   // jitter > this × median = unstable
    static let stableFactor = 0.08                   // jitter < this × median = stable
    static let contentionWidth = 6                   // GPU width cap under contention
    static let grainDefault = 4                      // v5 raster grain (rows/chunk)
    static let grainMin = 2                          // sweep: rows=1 costs ~70% mean
    static let grainMax = 8
    static let sliceTargetMs = 2.0                   // CPU slice wall steered toward
    static let sliceMin: UInt64 = 10_000
    // Capped at 200k after the v6 attempt-2 proof log: at 400k the
    // slice's own wall hit 3.1 ms and every GPU service point the
    // driver waited on paid the longer latency — the cap keeps a
    // slice inside roughly one target band on the M-class host.
    static let sliceMax: UInt64 = 200_000
    static let sliceDefault: UInt64 = 100_000        // the v5 proof slice
    static let scratchShedBytes = 256 * 1024 * 1024  // VRAM commit level that sheds scratch
    static let scratchRestoreBytes = 128 * 1024 * 1024

    /// The levers the A/B guard can revert, in application order.
    private enum Lever { case pacing, width, grain }

    // MARK: State

    private(set) var enabled = false
    private(set) var tickCount: UInt64 = 0
    private(set) var actionsIssued: UInt64 = 0
    private(set) var actionLog: [GovernorAction] = []

    // Actuator state — the values the chip currently commands.
    private(set) var pacingEngaged = false
    private(set) var paceCeilingMs: Double = 0
    private(set) var gpuStageWidth = 0
    private(set) var rasterChunkRows = NovaGovernor.grainDefault
    private(set) var cpuSliceSteps: UInt64 = NovaGovernor.sliceDefault
    private(set) var scratchRetention = true

    // Telemetry readouts (HUD + registers).
    private(set) var lastFrameMs: Double = 0         // last work sample
    private(set) var lastIntervalMs: Double = 0      // last interval sample
    private(set) var workMedianMs: Double = 0
    private(set) var workP25Ms: Double = 0
    private(set) var intervalMedianMs: Double = 0
    private(set) var intervalP95Ms: Double = 0
    private(set) var descriptorsPerFrame: UInt64 = 0
    var jitterMs: Double { max(0, intervalP95Ms - intervalMedianMs) }
    var windowCount: Int { workWindow.count }
    /// The ungoverned interval the guard currently expects
    /// (work median + probed slop); nil until calibrated.
    var ungovernedEstimateMs: Double? {
        guard let slop = baselineSlopMs else { return nil }
        return workMedianMs + slop
    }
    /// Plain-words pacing state for the HUD: what pacing is
    /// doing and, when it is off, why.
    var pacingStateText: String {
        if !enabled { return "off" }
        if baselineSlopMs == nil { return "calibrating — baseline probe running" }
        if pacingEngaged { return "engaged" }
        if tickCount < guardLockoutUntilTick {
            return "disengaged — it was costing frames (guard lockout)"
        }
        if lastWorkBound { return "disengaged — frames are work-bound" }
        return "disengaged — no jitter + headroom to smooth"
    }

    private var workWindow: [Double] = []
    private var intervalWindow: [Double] = []
    private var ticksSinceCommand = 0
    private var lastDescriptors: UInt64 = 0

    // v7 policy state.
    private var baselineSlopMs: Double? = nil        // probed interval − work median
    private var probeTicks = 0
    private var guardStreak = 0
    private var guardWorkStreak = 0
    private var guardLockoutUntilTick: UInt64 = 0
    private var widthLockoutUntilTick: UInt64 = 0   // guard-convicted levers stay off a while
    private var grainLockoutUntilTick: UInt64 = 0
    private var lastWorkBound = false
    private var leverStack: [Lever] = []             // applied levers, oldest first
    private var leverWorkBaselineMs: Double? = nil   // work median when the newest lever was applied

    /// The GPU whose knobs the chip drives (set by the device
    /// wrapper's supervise()). Weak: the chip never owns the GPU.
    weak var targetGPU: NovaGPU?

    private func fmt(_ v: Double) -> String { String(format: "%.1f", v) }

    // MARK: Enable / disable

    func setEnabled(_ on: Bool) {
        guard on != enabled else { return }
        if on {
            enabled = true
            // Fresh calibration: the chip watches an ungoverned
            // probe window before it touches anything.
            baselineSlopMs = nil
            probeTicks = 0
            guardStreak = 0
            guardWorkStreak = 0
            guardLockoutUntilTick = 0
            widthLockoutUntilTick = 0
            grainLockoutUntilTick = 0
            ticksSinceCommand = 0
            issue(.enable, Self.chipName, "enabled by host")
        } else {
            // Restore the v5 operating point through the same
            // command path, so OFF provably means "as v5".
            if pacingEngaged {
                pacingEngaged = false
                issue(.setPacing, "disengaged",
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
            leverStack.removeAll()
            leverWorkBaselineMs = nil
            baselineSlopMs = nil
            probeTicks = 0
            guardStreak = 0
            guardWorkStreak = 0
            enabled = false
            record(.disable, Self.chipName, "disabled by host")
            applyToTarget()
        }
    }

    /// Clears the telemetry windows and the calibration (actuator
    /// state other than pacing is kept; pacing drops, since its
    /// evidence is gone). The MMIO CTRL register's reset.
    func resetWindow() {
        workWindow.removeAll()
        intervalWindow.removeAll()
        workMedianMs = 0; workP25Ms = 0
        intervalMedianMs = 0; intervalP95Ms = 0
        lastFrameMs = 0; lastIntervalMs = 0
        paceCeilingMs = 0
        baselineSlopMs = nil
        probeTicks = 0
        guardStreak = 0
        guardWorkStreak = 0
        if pacingEngaged {
            pacingEngaged = false
            leverStack.removeAll { $0 == .pacing }
            if leverStack.isEmpty { leverWorkBaselineMs = nil }
        }
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

    // MARK: The A/B self-guard

    /// Reverts the most recently applied lever because governing
    /// is measurably costing frames. A pacing trip also voids
    /// the calibration and starts the re-engagement lockout: the
    /// chip must re-prove the baseline before it paces again.
    private func revertTopLever(reason: String) {
        guard let lever = leverStack.popLast() else { return }
        if leverStack.isEmpty { leverWorkBaselineMs = nil }
        switch lever {
        case .pacing:
            pacingEngaged = false
            baselineSlopMs = nil
            probeTicks = 0
            guardLockoutUntilTick = tickCount + Self.guardLockoutTicks
            issue(.setPacing, "disengaged",
                  "pacing disengaged: it was costing frames (\(reason))")
        case .width:
            gpuStageWidth = 0
            widthLockoutUntilTick = tickCount + Self.guardLockoutTicks
            issue(.setGPUWidth, "uncapped",
                  "GPU width cap reverted: it was costing frames (\(reason))")
        case .grain:
            rasterChunkRows = Self.grainDefault
            grainLockoutUntilTick = tickCount + Self.guardLockoutTicks
            issue(.setRasterGrain, "\(rasterChunkRows) rows/chunk",
                  "raster grain restored: it was costing frames (\(reason))")
        }
    }

    // MARK: Frame tick (the control loop)

    /// Frame-boundary tick. Sensing always runs (the windows and
    /// readouts stay live while disabled); acting requires the
    /// chip enabled, a completed baseline probe and an expired
    /// cooldown, and issues at most one command per tick.
    func tick(_ t: GovernorTelemetry) {
        tickCount += 1
        lastFrameMs = t.frameWorkMs
        lastIntervalMs = t.frameIntervalMs > 0 ? t.frameIntervalMs : t.frameWorkMs
        if t.descriptors >= lastDescriptors {
            descriptorsPerFrame = t.descriptors - lastDescriptors
        }
        lastDescriptors = t.descriptors
        workWindow.append(t.frameWorkMs)
        if workWindow.count > Self.windowSize { workWindow.removeFirst() }
        intervalWindow.append(lastIntervalMs)
        if intervalWindow.count > Self.windowSize { intervalWindow.removeFirst() }
        let ws = workWindow.sorted()
        workMedianMs = ws[ws.count / 2]
        workP25Ms = ws[Int(0.25 * Double(ws.count))]
        let is_ = intervalWindow.sorted()
        intervalMedianMs = is_[is_.count / 2]
        intervalP95Ms = is_[min(is_.count - 1, Int(0.95 * Double(is_.count)))]
        // The ceiling is a readout, not a command: it re-derives
        // from the work window every tick, so it tracks current
        // demonstrated speed with no hysteresis at all.
        if workWindow.count >= 8 { paceCeilingMs = workP25Ms }
        guard enabled else { return }
        ticksSinceCommand += 1

        // Baseline probe: for the first probeFrames ticks after
        // enable (or after a pacing guard trip) the chip only
        // watches. The probe measures the loop as it runs
        // UNGOVERNED, which is what makes the slop — and with it
        // the A/B guard — honest.
        if baselineSlopMs == nil {
            probeTicks += 1
            if probeTicks >= Self.probeFrames, intervalWindow.count >= Self.windowSize {
                baselineSlopMs = max(0, intervalMedianMs - workMedianMs)
            }
            return
        }
        let med = intervalMedianMs, jit = jitterMs
        lastWorkBound = (med - workMedianMs) <= max(Self.workBoundAbsMs,
                                                    Self.workBoundFrac * med)

        // 0. A/B self-guard: if governing is measurably costing
        //    frames, revert the newest lever before anything else
        //    compounds it. Two trip channels, because lever costs
        //    hide in two places: pacing's cost shows in the
        //    INTERVAL (padding is not work), while a grain/width
        //    cost shows inside the WORK time itself — so the
        //    guard watches the governed interval against the
        //    ungoverned estimate AND the work median against the
        //    snapshot taken when the newest lever was applied.
        if !leverStack.isEmpty, let expected = ungovernedEstimateMs {
            let intervalCost = med > expected * Self.guardFactor
            guardStreak = intervalCost ? guardStreak + 1 : 0
            let workCost = leverWorkBaselineMs.map {
                workMedianMs > $0 * Self.guardFactor
            } ?? false
            guardWorkStreak = workCost ? guardWorkStreak + 1 : 0
            if guardStreak >= Self.guardStreakToTrip {
                guardStreak = 0
                revertTopLever(reason: "governed \(fmt(med)) ms vs ~\(fmt(expected)) ms ungoverned estimate")
                return
            }
            if guardWorkStreak >= Self.guardStreakToTrip, let wb = leverWorkBaselineMs {
                guardWorkStreak = 0
                revertTopLever(reason: "frame work rose \(fmt(wb)) ms to \(fmt(workMedianMs)) ms under the lever")
                return
            }
        } else { guardStreak = 0; guardWorkStreak = 0 }

        // 1. Pacing. Engages only with real slack (the loop is
        //    not work-bound), real jitter to smooth, no guard
        //    lockout, and an expired cooldown. The wait it arms
        //    targets the work-p25 ceiling — never a fixed tier.
        //    Disengages the moment the loop turns work-bound.
        if pacingEngaged {
            if lastWorkBound {
                pacingEngaged = false
                leverStack.removeAll { $0 == .pacing }
                if leverStack.isEmpty { leverWorkBaselineMs = nil }
                issue(.setPacing, "disengaged",
                      "frames are work-bound (work \(fmt(workMedianMs)) ms of the \(fmt(med)) ms interval) — pacing has nothing to pad")
                return
            }
        } else if !lastWorkBound,
                  tickCount >= guardLockoutUntilTick,
                  jit > Self.stableFactor * med,
                  ticksSinceCommand >= Self.cooldownTicks {
            pacingEngaged = true
            leverStack.append(.pacing)
            leverWorkBaselineMs = workMedianMs
            issue(.setPacing, "engaged @ \(fmt(paceCeilingMs)) ms ceiling",
                  "interval median \(fmt(med)) ms leaves slack over the work p25 of \(fmt(paceCeilingMs)) ms — ceiling tracks demonstrated work only")
            return
        }

        // Rules 2–6 share the v6 command cadence: at most one
        //    lever command per cooldown. (The guard above and the
        //    work-bound pacing exit are safety paths and are not
        //    cadence-gated.)
        guard ticksSinceCommand >= Self.cooldownTicks else { return }

        // 2. Jitter: a wide p95–median gap is a ragged stage tail —
        //    finer raster grain smooths it (more, smaller chunks).
        if jit > Self.jitterFactor * med, rasterChunkRows > Self.grainMin,
           tickCount >= grainLockoutUntilTick {
            rasterChunkRows = max(Self.grainMin, rasterChunkRows / 2)
            if !leverStack.contains(.grain) { leverStack.append(.grain) }
            leverWorkBaselineMs = workMedianMs
            issue(.setRasterGrain, "\(rasterChunkRows) rows/chunk",
                  "frame jitter \(fmt(jit)) ms is over 30% of the \(fmt(med)) ms median")
            return
        }
        // 3. Jitter persisting at minimum grain is contention:
        //    cap the GPU's host width so the CPU, the download
        //    path and the UI stop fighting the stages.
        if jit > Self.jitterFactor * med, gpuStageWidth == 0,
           tickCount >= widthLockoutUntilTick {
            gpuStageWidth = Self.contentionWidth
            if !leverStack.contains(.width) { leverStack.append(.width) }
            leverWorkBaselineMs = workMedianMs
            issue(.setGPUWidth, "\(gpuStageWidth)",
                  "frame jitter \(fmt(jit)) ms persists at grain \(rasterChunkRows) — capping GPU stage width")
            return
        }
        // 4. Stable: hand the width back.
        if jit < Self.stableFactor * med, gpuStageWidth != 0 {
            gpuStageWidth = 0
            leverStack.removeAll { $0 == .width }
            if leverStack.isEmpty { leverWorkBaselineMs = nil }
            issue(.setGPUWidth, "uncapped",
                  "frame jitter \(fmt(jit)) ms is under 8% of the median — GPU width restored")
            return
        }
        // 5. Stable at fine grain: the jitter that forced the
        //    fine grain is gone — restore the default grain (its
        //    per-chunk setup is cheaper; the sweeps measured it).
        if jit < Self.stableFactor * med, rasterChunkRows < Self.grainDefault {
            rasterChunkRows = min(Self.grainDefault, rasterChunkRows * 2)
            if rasterChunkRows == Self.grainDefault {
                leverStack.removeAll { $0 == .grain }
                if leverStack.isEmpty { leverWorkBaselineMs = nil }
            }
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
    case id            = 0x00   // RO: 0x4E474F56_00070000 ("NGOV", v7.0)
    case status        = 0x08   // RO: bit0 enabled, bit1 window full, bit2 scratch retained, bit3 pacing engaged
    case enable        = 0x10   // RW: write 1 = enable, 0 = disable
    case paceCeilingUs = 0x18   // RO: current pace ceiling (work p25), microseconds
    case gpuWidth      = 0x20   // RO: GPU stage width (0 = uncapped)
    case rasterGrain   = 0x28   // RO: raster rows per chunk
    case cpuSlice      = 0x30   // RO: CPU run-slice size, steps
    case lastFrameUs   = 0x38   // RO: last frame work time, microseconds
    case jitterUs      = 0x40   // RO: interval p95 − median, microseconds
    case actionCount   = 0x48   // RO: commands issued since boot
    case medianUs      = 0x50   // RO: window interval median, microseconds
    case ctrl          = 0x58   // WO: 1 = reset telemetry window
    case tick          = 0x60   // WO: tick with a frame-work time in microseconds
    case pacingState   = 0x68   // RO: 0 off, 1 calibrating, 2 engaged, 3 disengaged
    case workMedianUs  = 0x70   // RO: window work median, microseconds
    case workP25Us     = 0x78   // RO: window work p25, microseconds
}

/// The governor as a bus device: a register block in the MMIO
/// page beside the GPU's, exactly in the style of the other
/// NovaMemory devices. The host attaches it to a NovaMemory, then
/// points it at the GPU device it supervises. The GPU device also
/// calls notePresent() on every completed PRESENT, so the chip
/// ticks on real frame boundaries without the CPU doing anything.
final class NovaGovernorDevice {
    static let deviceID: UInt64 = 0x4E47_4F56_0007_0000

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
    /// with the inter-present interval AND the device's busy time
    /// since the previous present — v7 keeps the two apart, so
    /// the pacing policy reads work that no wait can inflate.
    /// Intervals over 250 ms are a resync (tab switch, first
    /// frame after attach), not a sample.
    func notePresent() {
        let now = NovaGPU.nowMs()
        defer { lastPresentMs = now }
        guard lastPresentMs > 0 else { return }
        let dt = now - lastPresentMs
        guard dt <= 250 else { return }
        guard let dev = gpuDevice else { return }
        chip.tick(GovernorTelemetry(
            frameWorkMs: dev.consumeBusyMs(),
            frameIntervalMs: dt,
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
            if chip.pacingEngaged { s |= 8 }
            return s
        case .enable: return chip.enabled ? 1 : 0
        case .paceCeilingUs: return UInt64(chip.paceCeilingMs * 1000)
        case .gpuWidth: return UInt64(chip.gpuStageWidth)
        case .rasterGrain: return UInt64(chip.rasterChunkRows)
        case .cpuSlice: return chip.cpuSliceSteps
        case .lastFrameUs: return UInt64(chip.lastFrameMs * 1000)
        case .jitterUs: return UInt64(chip.jitterMs * 1000)
        case .actionCount: return chip.actionsIssued
        case .medianUs: return UInt64(chip.intervalMedianMs * 1000)
        case .ctrl, .tick: return 0
        case .pacingState:
            if !chip.enabled { return 0 }
            if chip.pacingStateText.hasPrefix("calibrating") { return 1 }
            return chip.pacingEngaged ? 2 : 3
        case .workMedianUs: return UInt64(chip.workMedianMs * 1000)
        case .workP25Us: return UInt64(chip.workP25Ms * 1000)
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
