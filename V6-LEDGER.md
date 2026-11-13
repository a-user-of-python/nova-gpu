# NovaGPU v6 — attempt ledger

Commission (2026-11-13 15:59 ET), after device-testing v5:
"with the vpus I get roughly 35 [FPS] but the fps is all over the
place — see if you can stabilize it with a new thing I want to
introduce to both projects. For now this will be only on the gpu
app but since the gpu app has a cpu in it it will be able to test.
Design a new chip that is very basic. It has 1 goal which is to
maintain the gpu and cpu to its best ability. This basically means
it will change the gpu and cpu priority (what would be amount of
power in real life) and can change memory allocation and move
special commands to the cpu and gpu. These special commands would
tell the gpu and cpu how to act — this is kind of like a bios chip
but not just memory. This chip should help increase performance on
both ends." Plus: "increase gpu performance and cpu performance in
the gpu app so we can try to hit 60fps on the test scene."

Device anchor: **v5 with the VPU test armed ≈ 35 FPS on the
iPad Air M3, unstable** ("all over the place"). v3 baseline on the
same scene (icosphere + torus, 640×480, 29,696 tris) was ~27 FPS.
Stabilizing the frame rate is a headline goal alongside raw speed.

## The chip (working name `NovaGovernor`)

Renaming: the name lives in exactly ONE place
(`NovaGovernor.chipName`) so he can rename it with one edit.

A first-class device, not app glue:
- **Senses** (per-frame telemetry): frame work time + rolling
  window (median / p95 / jitter), GPU stage times, CPU slice time
  + steps, bus descriptors serviced, VRAM committed / reserved.
- **Actuators** (all real, all in its authority):
  a. Priority / power share — GPU stage dispatch width (caps how
     many host threads GPU stages may occupy) and the CPU's run
     slice size (how much uninterrupted host time the CPU gets
     between GPU service points).
  b. Memory allocation — the GPU's draw-scratch retention policy
     (retain-and-reuse vs allocate-per-draw; shed under VRAM
     pressure).
  c. Work partitioning — raster chunk grain (rows per chunk) and
     clip/bin fan-out ranges: how GPU work is divided across the
     host, which is what moves the stage tail (and therefore
     jitter) without touching a single pixel.
- **Special commands**: every change the chip makes is a named
  command (SET_GPU_WIDTH, SET_RASTER_GRAIN,
  SET_CPU_SLICE, SET_PACE_TIER, SET_SCRATCH_RETENTION) issued
  through one path, applied to the real knobs, and written to an
  action log with the numbers that caused it ("frame jitter
  9.4 ms > 30% of median 28.1 ms"). The chip is reachable over
  the bus two ways: its own MMIO register block (0x…0180–0x…0200,
  beside the GPU's) that the NovaCPU can read/write, and GPU ring
  opcode 21 (enable / disable / tick).
- **Frame pacing**: a deterministic control loop over named pace
  tiers (16.7 / 20.8 / 25 / 33.3 / 50 ms) with hysteresis
  (asymmetric streaks + a 30-frame cooldown, one command per
  tick): it steps the target down when sustained work overruns
  the tier and back up when headroom persists. It never drops
  work and never changes the picture — it spends or saves budget
  through the actuators and the frame-start deadline.
- **Legibility**: the whole policy is a few dozen lines of named
  thresholds. No learning, no hidden state beyond the window and
  the streak counters the HUD shows.

Hard invariant (unchanged from v2–v5): every framebuffer checksum
in the host tests stays byte-identical. Every knob above changes
only partitioning / scheduling / retention — work-list order,
per-pixel expressions, and accumulation order are untouched, and
host-test section 13 proves frame equality across knob settings.

60 FPS honesty: the target is 60 on the test scene. Host numbers
are labelled host; whatever the host lands at, BENCH-v6.md says
where it landed and what the remaining sink is. The device
verdict is the tester's.

## Attempt 0 — measurement + governor-ready knobs (no behaviour change)
- NovaGPU gains instance knobs with v5 defaults: `rasterChunkRows`
  (4; draw raster + plot pass 2), `clipRangeCount` (32), and
  `stageWidth` (0 = uncapped) applied through a `parallelFor`
  helper at the draw stages, the VPU test pass, plot passes and
  compute launch. stageWidth 0 calls DispatchQueue.concurrentPerform
  exactly as v5 did, so an untouched device is v5.
- BenchTests gains: a bus-level engine-replica frame series
  (90 frames, the GPUEngine loop shape: one submitBatch + one
  1.2 MB download per frame) printing mean / stdev / p95 / max of
  frame work — the jitter profile the governor design answers to;
  grain / width / clip-range sweeps on the app scene.
- Expected: all host tests + proof unchanged; bench gives the v6
  baseline (frame series + stage medians) in BENCH-v6.md.
- Measured (CI run 43507604592, all jobs green; full numbers in
  BENCH-v6.md): engine-series jitter baseline — 85 frames, mean
  24.0 ms, stdev 3.93 ms, p95 32.6, min 16.5, max 36.7 (jitter
  37% of median — the governor's 30% trigger sits on it). Grain
  sweep: finer grain costs host mean (rows 1 = 40.6 ms vs 4 =
  23.7) → policy grain floor is 2, default stays 4. Width sweep:
  caps cost mean on an unloaded runner → width is a contention
  lever only, applied last. Clip ranges 8 vs 32: no signal →
  default 32 stands and the governor does not manage that knob
  (SET_CLIP_RANGES was dropped from the command set for exactly
  this reason — a command with no measurable effect is theatre).
- Status: DONE.

## Attempt 1 — the governor chip
- Sources/NovaGPU/NovaGovernor.swift: telemetry window, policy
  core, command issue path, action log, NovaGovernorDevice (MMIO
  block + present hook + ring forwarding).
- NovaMemory: governor block routing (0x…0180–0x…0200).
- NovaBus: GPU ring opcode 21 (governorCmd); NovaGPUDevice holds
  an optional governor and forwards; NovaBusClient gains a
  governorControl convenience.
- Host tests section 13: register identity/readback, ring
  enable/disable/tick, policy determinism on scripted telemetry
  (pace tier steps down under sustained overrun and back up under
  headroom; grain halves under scripted jitter; CPU slice adapts
  toward its target band), pixel invariance across knob settings.
- CI workflow + Xcode project gain the new source file.
- Expected: all pre-v6 checksums unchanged; section 13 green.
- Measured: CI 43509439822 all green; §13 fully PASS (see
  BENCH-v6.md). Status: DONE.

## Attempt 2 — app integration
- GPUEngine (Live Scene): governor created + attached + supervising
  the GPU device; toggle next to the VPU test toggle; per-frame
  tick with real telemetry; deadline pacing while on; HUD rows —
  governor state + pace tier, frame-interval jitter (stdev + p95),
  active width/grain, last governor action in plain words.
- ProofRunner: the CPU slice loop is governor-supervised (slice
  size from the chip, tickCPU per slice, action log + wall time +
  slice count in the page output).
- ProofTests: the driver runs twice — fixed 100k slices (v5
  baseline, timed) and governor-supervised slices (timed) — with
  the full checks (GPU OK + framebuffer match) applied to BOTH.
- Expected: proof + host tests green; the proof page and CI print
  both wall times and the governor's actions.
- Measured: CI 43512829206 all green after the UInt64 fix. The
  bench A/B convicted the first pacer: Thread.sleep overshoot
  made governed intervals WORSE on the host (stdev 9.87 vs 2.12
  unpaced) while every chip command fired correctly. Pacer
  rebuilt as governorWait (sleep to the last ~1 ms, then spin),
  shared verbatim by the app and the bench; slice cap lowered
  400k → 200k from the proof log. Proof A/B honestly recorded
  in BENCH-v6.md (supervised run slower on wall at toy scale —
  mechanism proven, no wall claim).
- Status: DONE (pacer re-measurement lands in attempt 4's run).

## Attempt 3 — draw() scratch reuse (allocation churn)
- draw() allocates and re-initializes its big scratch every call
  (shaded-vertex buffer, 1.2 MB depth buffer re-filled per draw,
  snapshot arrays, per-stage op arrays, WorkBox/BinBox objects).
  On device that allocator traffic is both mean cost and jitter.
  The device gains a grow-only scratch cache reused across draws
  (freed on reset); the governor's SET_SCRATCH_RETENTION command
  switches it (off = v5 allocate-per-draw behaviour; the chip
  sheds it under VRAM pressure).
- Expected: checksums unchanged (scratch contents are fully
  rewritten per draw — depth is re-initialized exactly as now);
  bench A/B on frame mean + series stdev.
- Implemented as a grow-only ScratchBox set on the device:
  draw()'s vertex/index snapshots, shaded-vertex buffer, depth
  buffer, per-warp ops and per-chunk stat arrays, plus the VPU
  test pass's reference/output/range arrays. Retention off (the
  governor's shed, or the flag cleared directly) releases all of
  it at the next draw() entry and that draw allocates exactly as
  v5. reset() releases it too. plotVectors keeps the v5
  allocate-per-call behaviour deliberately (the Vectors scene is
  20 vectors; it is not the frame path under test).
- Measured: CI 43512829206 all green; same-run A/B 13.09 ms
  (on) vs 13.31 ms (off) — see BENCH-v6.md. Status: DONE.

## Attempt 4 — defaults from evidence + close-out
- Shipped defaults for grain/width set from the attempt-0 sweeps;
  governor thresholds re-read against the measured jitter profile.
- Final CI, IPA binary verified to contain the governor paths,
  README v6 section (the chip in plain language), BENCH-v6.md,
  this ledger completed, NOVA tracking entry.
- Measured: final CI run 43515631697, ALL JOBS SUCCESS on the
  tip tree. governorWait re-measurement (same run): unpaced
  interval stdev 5.19 ms → governed 3.88 ms; governed median
  interval exactly 20.800 ms = the tier the chip selected;
  max 39.3 → 34.3 ms. Proof: fixed 0.03 s vs supervised
  0.04 s, full battery green on both. Host tests: 76 PASS,
  0 FAIL — every pre-v6 checksum intact.
- IPA: NovaGPU-Test-v6.ipa, 255,052 bytes, md5
  51148a234d87f5bd2738b68c0b64fa39, bundle
  com.user.novagputest. Shipped binary strings-verified to
  contain the v6 paths (NovaGovernor, SET_PACE_TIER /
  SET_RASTER_GRAIN / SET_CPU_SLICE, governorWait symbol, the
  supervisor toggle label).
- Status: DONE. Device verdict owed from the tester (test steps in
  the delivery note).
