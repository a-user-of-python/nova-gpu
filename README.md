# NovaGPU

A custom software GPU for the NOVA project, designed and built from
scratch in the same spirit as NovaCPU: its own architecture, its own
shader instruction set, and a two-way bus (NovaBus) to the CPU.

- **1536 compute units** — 48 SIMT clusters x 32 lanes, FP32 + INT32.
- **RGBA8888 framebuffer** — 16,777,216 colours, FP32 per-channel shading.
- **2 GiB GPU-addressable VRAM**, sparsely committed in 64 KiB blocks,
  with a first-fit allocator; plus DMA into NovaCPU guest RAM.
- **3D pipeline** — vertex fetch, vertex shader, clipping,
  rasteriser, fragment shader, framebuffer. Vertex/fragment/compute
  programs share one documented shader ISA.
- **NovaBus** — MMIO register block at `0xFFFF_FFFF_FFFF_0100` (inside
  NovaCPU's existing MMIO page; the CPU core needs no changes), a
  command ring of 64-byte descriptors in guest RAM, completion/error
  IRQ bits, and a GPU->CPU service channel: when a DRAW needs an
  asset the GPU suspends the command, writes a request record into a
  guest page and raises the service bit; the CPU supplies the data
  and ACKs.

Proven by the **NovaGPU Test** app (live rotating scene rendered only
through the bus, a self-test page, and a CPU-proof page that boots
the real NovaCPU core on a Nova assembly driver) and by the CI host
tests in this repo. The full design document — architecture, ISA,
pipeline, memory map, the complete NovaBus protocol, and the entire
GPU source code — is `NovaGPU-Design.md` in the project deliverables.

NovaGPU ships standalone first; it merges into NovaVM as Nova v0.1
later (the bus is the compatibility contract for that merge).

## v4 — optimisation + vector hardware

- **Hot core completed in C**: every shader-executor op now runs
  in the NEON C core under the same FP exactness contract (one
  rounding per op, accumulation order unchanged) — all v3
  framebuffer checksums pass unchanged. Plus: a **grab unit**
  (wide vertex fetch, no per-lane closures), allocation-free
  inline varyings, and analytic rasteriser span narrowing. Host
  (CI) executor micro-benches vs v3: fragment warp 0.80 → 0.72 µs,
  vertex warp 2.76 → 2.09 µs. The device baseline is v3 at ~27 FPS
  on Sean's iPad Air M3; the v4 device verdict is his to measure.
  Full record: `BENCH-v4.md`.
- **Vector unit ISA** (appended, nothing renumbered): VCROSS
  (0x1E), VNORM (0x1F), VMAT4 (0x24) — FP32 world-space vector
  math, bit-exact against the DOT-row expressions they replace.
  `NovaGPUStandard.vertexLitProgramV4()` renders byte-identical
  frames to the v1 program.
- **VPU + MGVPU clusters**: a VPU positions one triangle (3
  grabbed vertices, 9 axis values, one 3-axis calculator). The
  pool is grouped into 50 **MGVPU** clusters of 131,072 VPUs —
  6,553,600 addressable VPUs, virtual and demand-materialized
  (zero idle cost); scheduling distributes MGVPU sub-ranges
  across host threads.
- **Plot unit**: bus command 19 plots 3D vectors as lines/points
  without the triangle pipeline — grab, VPU-position, project to
  the 16-bit-per-axis screen grid (Q10.6), DDA-rasterise with
  triangle depth rules, shade through the bound fragment program
  (colour = uniform c10 in varying slot 1). Budget: 65,536
  vectors per frame. The test app gains a **Vectors** page
  (wireframe cube + corner points, plot unit only). Host tests
  prove plotted pixel sets against a Double-precision reference.

## v5 — in-scene VPU test + optimisation

- **Direct VPU test inside the Live Scene** (bus command 20):
  toggle it and every scene triangle is positioned through the
  real VPU path — grab unit → 3-axis calculator, scheduled over
  the MGVPU clusters — bit-compared against the shader stage's
  own clip positions (snapshotted pristine before the pass),
  and the frame renders **from the VPU positions**, so a wrong
  VPU visibly breaks the picture. The HUD shows the verdict
  (BIT-EXACT, or MISMATCH stated plainly — never rounded into
  a pass), triangles tested, mismatched corners, max position
  error, triangles/sec and MGVPU clusters engaged, next to the
  normal FPS readout. Host tests prove it both directions:
  armed frames byte-identical to unarmed, and an injected +0.5
  fault is caught at exactly 12 mismatched corners with the
  faulted frame differing from the clean one.
- **Wide VPU issue**: each triangle is still grabbed
  individually (its own VPU workload), but the calculator now
  issues in groups (`VPU.positionWide`, bit-identical per
  lane). HOST micro, same-run A/B: 52.6M → 140.4M triangles/s
  serial (2.67×) for identical grab + position work; the
  in-scene test pass runs at 34–38M triangles/s (HOST), about
  1 ms per frame at the app scene's 29,696 triangles. The
  plot unit's positioning pass batches endpoints the same way
  (output proven batch-size invariant at 1 / 7 / 1024); the
  2,048-vector parallel threshold was re-swept and stays —
  its 65,536-vector minimum sits exactly at 2048.
- **C fragment writeback** (`ngp_flush_writeback`): colour
  conversion + block-table stores in the C core with the
  identical clamp/convert expressions; lanes on uncommitted
  blocks replay through the original locked fallback. Gain
  is small (the flush floor is mostly warp machinery, not
  conversion) — recorded honestly in `BENCH-v5.md`.
- **Clip fast path**: fully-inside triangles append their
  work-triangle in place (same plane expressions, identical
  work list), skipping the per-triangle array round-trip:
  clip stage ≈2.5× faster on HOST (2.30 → 0.81 ms).
- The bit-exact invariant holds throughout: every pre-v5
  framebuffer checksum and frame comparison passes unchanged
  in CI. All figures above are HOST (CI runner) numbers; the
  device verdict is Sean's. Full record: `BENCH-v5.md`,
  attempts in `V5-LEDGER.md`.

## v6 — the supervisor chip (NovaGovernor) + stabilisation

- **A new chip whose only job is to look after the CPU and
  the GPU.** Working name **NovaGovernor** — the name lives in
  exactly one place in the code (`NovaGovernor.chipName`) so
  it can be renamed with a single edit. It is a real device on
  the bus, not app glue: its own MMIO register block
  (0x…0180–0x…0200, beside the GPU's, device ID "NGOV") that
  the NovaCPU can read and tick like any other hardware, plus
  GPU ring command 21 for enable / disable / tick from a host.
- **What it senses:** every frame boundary (it ticks itself on
  every PRESENT) — frame interval and its rolling median /
  p95 / jitter, bus descriptors serviced, VRAM committed vs
  reserved — and, from the CPU's host loop, each run slice's
  wall time.
- **What it does about it — special commands, all real:**
  `SET_PACE_TIER` (the frame-pacing target the host schedules
  against: 16.7 / 20.8 / 25 / 33.3 / 50 ms), `SET_RASTER_GRAIN`
  (rows per raster chunk), `SET_GPU_WIDTH` (how much of the
  host the GPU stages may occupy at once), `SET_CPU_SLICE`
  (how long the NovaCPU runs between GPU service points),
  `SET_SCRATCH_RETENTION` (whether the GPU keeps its big draw
  buffers between frames or frees them). Every command goes
  through one issue path, lands on a real knob, and is written
  to an action log with the numbers that caused it ("frame
  jitter 7.8 ms is over 30% of the 17.6 ms median"). Turning
  the chip off restores the exact v5 operating point through
  the same path. The policy is a few dozen lines of named
  thresholds with hysteresis (asymmetric streaks + a cooldown)
  — no learning, nothing hidden. It never drops work and never
  changes a pixel: the framebuffer checksums police that, and
  host tests prove frames byte-identical with the knobs
  twisted and with the chip live.
- **Frame pacing, measured honestly:** the first pacer slept
  until each frame deadline — and the bench convicted it
  (sleep overshoot *added* variance on the host). The shipped
  pacer sleeps only to the last ~1 ms and spins the remainder.
  Same-run host A/B after the fix: interval stdev 5.19 →
  3.88 ms, and the governed median interval landed on exactly
  20.800 ms — the tier the chip itself had picked. Pacing pads
  short frames up to the tier; a frame whose work exceeds the
  tier stays long, and the log says so.
- **Retained draw scratch:** draw()'s big buffers (snapshots,
  shaded vertices, the 1.2 MB depth buffer, stage statistics)
  are kept between draws in a grow-only cache — several MB of
  allocator churn leaves the frame. Same-run host A/B: ~0.2 ms
  per frame on a healthy runner, direction-consistent across
  stages; the governor's scratch command is what frees it
  under VRAM pressure.
- **In the app:** a toggle next to the VPU test toggle (one
  tap A/B), HUD rows for pace target, frame-interval p50/p95,
  jitter, GPU width/grain, and the chip's last action in plain
  words. The CPU Proof page runs its slices under the chip and
  prints its action log; CI runs the proof twice (fixed v5
  slices vs supervised) with the full checksum battery on both.
- All figures are HOST (CI runner) numbers; the device anchor
  is ~35 FPS unstable for v5 with the VPU test armed, and the
  v6 device verdict is the tester's. Full record:
  `BENCH-v6.md`, attempts in `V6-LEDGER.md`.

## v7 — the Governor fix: pacing can never cost frames again

- **The v6 device verdict, plainly:** with the Governor
  enabled, the Live Scene ran at a stable **19.2 FPS** on the
  iPad Air M3 — stable, but much slower than the ~35 FPS the
  scene ran ungoverned. The stability arrived; the speed did
  not. v7 exists because of that result.
- **What was wrong (convicted from the code, then reproduced
  on the host before any fix):** the chip's only frame sensor
  was the present-to-present interval — a number its own
  pacing wait inflates. It stepped a fixed tier ladder
  (16.7 / 20.8 / 25 / 33.3 / 50 ms) off that interval's
  median, and stepping back up required the median to fall
  below the next tier × 0.80 — impossible while the pacer
  itself held every interval at the current tier. A one-way
  ratchet to 50 ms. The reproduction (bench §11, device-like
  work pattern) walked the same ladder to 50.0 and stayed
  there at 15.9 FPS while the identical ungoverned loop ran
  50.1 FPS.
- **The never-slower law, as now implemented:**
  - Work and interval are **separate sensors**. The pace
    ceiling is the **p25 of recent frame WORK** (device busy
    time, which no wait can inflate), re-derived every tick.
    There is no ladder; a ceiling can never sit above speed
    the hardware has demonstrably achieved.
  - **Work-limited disengagement:** when the interval *is*
    the work — the loop has no slack — pacing stays
    disengaged and the log says why. Stability work falls to
    the grain / width / priority levers.
  - **A/B self-guard:** a 24-frame ungoverned probe at enable
    time measures the loop's baseline slop. Afterwards the
    chip continuously compares governed throughput against
    that estimate (and the work median against the snapshot
    taken when a lever was applied — grain/width costs hide
    inside the work). Any lever that costs more than ~5%
    is reverted, the log says so in plain words ("pacing
    disengaged: it was costing frames"), and a convicted
    lever stays off for 240 frames.
- **HUD honesty:** the Live Scene now shows the Governor's
  pacing state in words (engaged / calibrating / disengaged —
  and why), the current pace ceiling in ms, and the frame
  jitter, next to the existing toggles — the chip's behaviour
  is checkable at a glance, one tap from the A/B.
- **Host proof (CI, same-run A/B on the device-like series):**
  phase A governed 71.96 ms vs control 71.93 (a dead heat —
  pacing out of the way, as designed); phase B governed
  16.14 ms / stdev 3.41 vs control 21.07 / 6.91, where v6 had
  stayed locked at 62.79 ms. Host tests 82 PASS / 0 FAIL with
  every framebuffer checksum byte-identical. The honest
  summary: at device-like speeds the scene is work-bound, so
  the shipped governor stabilizes only when there is headroom
  and otherwise stays out of the way — what it guarantees is
  the floor, that enabling it can no longer cost frames. The
  device verdict is the tester's. Full record: `BENCH-v7.md`,
  attempts in `V7-LEDGER.md`.

## Layout

- `Sources/NovaGPU/` — the GPU core (shader ISA + executor, the GPU,
  the bus device, the host bus client, standard shaders/geometry).
- `App/NovaGPUTest/` — the test app (includes byte-identical copies
  of the NovaCPU core files for the proof page).
- `Driver/gpu_driver.nova` — the Nova assembly driver; assemble with
  `Driver/novas.py` (the NOVA project's assembler).
- `HostTests/`, `ProofTests/` — CI test mains.

## CI

`.github/workflows/ci.yml` (free standard macos-26 runners) assembles
the driver, builds and runs the host tests and the CPU proof, then
builds the unsigned IPA as a workflow artifact.
