# NovaGPU v6 — benchmark record

All numbers are **HOST numbers** from the CI bench job (free
standard macos-26 runner, Apple Silicon) unless explicitly
labelled device. The device anchor for this round is the tester's:
**v5 with the VPU test armed ≈ 35 FPS on his iPad Air M3, with
the FPS unstable** ("all over the place"); v3 on the same scene
was ~27 FPS. CI shared-runner variance is large (this baseline
run read ~2× slower than v5's healthiest runs across the board)
— only same-run comparisons and the tight micro-benches are
cited as evidence.

## v6 baseline (CI run 43507604592, attempt 0: knobs + measurement only)

App scene stage medians (v1 vertex program, fragments 290,906):

| stage | ms |
|---|---|
| frame (wall) | 24.01 |
| snapshot | 0.21 |
| vertex | 0.94 |
| clip | 1.64 |
| bin | 3.09 |
| raster | 17.16 |
| fragExecCpu (CPU-sum) | 17.02 |

Raster/fragment is still ~71% of the frame; the END-only
fragment program still leaves fragExecCpu 9.50 ms (the
FragBatcher flush floor, as convicted in v5). Nothing in v6
changes the arithmetic — the round's levers are partitioning,
scheduling, retention and pacing.

### The jitter profile (new: engine-replica frame series)

The Live Scene's exact bus shape — one submitBatch (clear + 2
draws + present) + one 1.2 MB download per frame, 85 frames
after warm-up:

| mean | stdev | p50 | p95 | min | max |
|---|---|---|---|---|---|
| 24.00 ms | **3.93 ms** | 23.76 | 32.63 | 16.46 | 36.66 |

Frame work swings from 16.5 to 36.7 ms around a 23.8 ms median
**on the host** — stdev is 16% of the mean and p95 − median is
37% of the median. That spread, in the engine's own loop shape,
is the measured counterpart of the reported "fps all over the place":
the work itself is unevenly scheduled frame to frame, before
any display cadence touches it. The governor's thresholds are
set against this: its jitter trigger (p95 − median > 30% of
median) sits right at this run's observed 37%.

### Knob sweeps (same run, app scene)

Raster grain (rows per chunk) — mean frame / raster ms:
rows 1 → 40.6 / 32.2; rows 2 → 33.1 / 25.1; rows 4 → 23.7 /
15.5; rows 8 → 23.6 / 17.3. Finer grain costs mean speed on the
host (per-chunk setup: a FragBatcher + warp context per chunk)
and buys tail smoothness only. Governor consequence, baked
into the policy: the grain floor is **2**, the default stays 4,
and grain only moves under proven jitter.

Stage width cap — width 0 (uncapped) → 20.3 ms; 3 → 44.5;
6 → 32.7; 12 → 29.0. Capping width always costs mean speed on
an unloaded many-core runner: it is a **contention** tool (on
the device the GPU stages share 4P+4E cores with the engine
thread, the download path, the UI, and any running NovaCPU
work), not a host speed tool. The governor applies it only
when jitter persists at minimum grain.

Clip ranges 8 vs 32: 24.8 vs 24.4 ms frame — no signal; the
32 default stands and the governor does not manage this knob.

Executor micros (slow-runner readings): lit fragment warp
1.032 µs; lit vertex warp (closure path) 3.124 µs. VPU micro:
wide issue 44.7M tri/s serial / 122.2M MGVPU-parallel.
CPU proof: driver run 0.03 s (the guest program is short —
the governor's CPU slice A/B in the proof is a mechanism
demonstration at this workload size, reported as such).

## Attempt 1–2 — governor chip (host evidence)

CI run 43509439822 (chip + host tests): section 13 all PASS —
MMIO identity/enable/tick/readback, ring opcode 21 enable and
disable, and the scripted policy traces behave exactly as
designed: 30 ms frames walk the pace tier 16.7 → 33.3 ms, 8 ms
frames walk it back to 16.7; scripted jitter refines the grain
4 → 2 then caps GPU width at 6, stability restores both;
disable mid-adaptation restores the v5 operating point; the
CPU slice halves 100k → 12.5k under 9 ms slices, doubles back
inside the rule, and holds in the deadband; scratch sheds at
300 MB VRAM committed and restores at 10 MB. Frames render
byte-identically with raw knobs twisted (grain 1, width 1,
clip 4) and with the chip attached, enabled and ticked.

CI run 43512829206 (app integration + bench A/B; a healthy
runner — the app scene read frame 13.7 ms / host 73 FPS this
run vs 24.0 ms on the loaded baseline run, so only same-run
comparisons follow):

**Governor-paced series vs unpaced control, same run:**
unpaced work series — mean 14.85 ms, stdev 2.12, p95 18.99.
Governor-paced intervals — mean 28.14, **stdev 9.87**, p95
45.98, with the chip acting as designed (grain → 2 at tick 30
on a measured 9.3 ms jitter over a 23.3 ms median; pace tier
→ 20.8 ms at tick 60). **The pacing made host intervals worse,
not better — convicted cause: Thread.sleep overshoot.** The
first pacer slept until each deadline; on a loaded host the
wake lands 1–8 ms late, which is larger than the variance
being removed. The pacer now sleeps only down to the last
~1 ms and spins the remainder (`governorWait`), in both the
app and this bench (identical code path); the bench also
prints the settled tail of the series (post window+cooldown)
alongside the full run. Re-measurement lands with the final
CI run below.

**CPU proof A/B, same run:** fixed 100k slices 0.03 s / 66
slices; governor-supervised 0.06 s / 36 slices — both runs
pass the full battery (GPU OK + checksum match). The chip's
log shows it steering (100k → 200k → 400k on sub-ms slices,
halving at a 3.1 ms slice). At this workload the supervised
run is *slower* on wall: the driver spin-waits for GPU
service, so longer slices mean longer waits. Two honest
consequences: the slice cap came down 400k → 200k (a slice
stays within roughly one 2 ms latency band), and the claim
for the CPU actuator is the mechanism + log, not a wall win
on a 30 ms toy run — its value is under real CPU/GPU
contention, which the proof workload barely creates.

## Attempt 3 — scratch reuse (host evidence)

Same-run A/B (CI run 43512829206), app scene: retention ON
13.09 ms frame (snapshot 0.161, raster 9.675) vs OFF 13.31 ms
(snapshot 0.180, raster 10.263) — about 0.2 ms per frame
(~1.6%), consistent across all three reported stages. The
final run (43515631697, a more loaded runner) read the same
pair as 15.37 vs 20.42 ms — same direction, but that pair
straddles an obvious load shift, so the healthy-run figure
is the cleaner read. The bigger device-side expectation is
not the mean: it is the removal of several MB of malloc/mmap
churn per draw, a classic per-frame jitter source that a
shared CI runner cannot price. Checksums unchanged with
retention on (it is the default in every test above).

## Final measurement — governorWait pacer (CI run 43515631697)

The rebuilt pacer (sleep to the last ~1 ms, then spin), same
run, a noisier runner than the attempt-2 run (unpaced work
mean 19.46 ms):

| series | mean | stdev | p50 | p95 | max |
|---|---|---|---|---|---|
| unpaced control (work = interval) | 19.46 | **5.19** | 18.82 | 28.50 | 39.28 |
| governor-paced intervals | 21.26 | **3.88** | **20.80** | 28.03 | 34.31 |

The governed median interval is **exactly 20.800 ms — the
pace tier the chip selected** (it stepped 16.7 → 20.8 at tick
60 on a measured 20.2 ms median, and refined the grain to 2
at tick 30 on a measured 7.8 ms jitter) — the sleep pacer's
median had floated ~4 ms above its tier. Interval stdev fell
25% (5.19 → 3.88) and the max fell 39.3 → 34.3 ms. What
pacing cannot do is also visible honestly: the governed
run's remaining tail is work spikes above the tier (work
stdev 3.96 inside the governed run) — pacing pads short
frames up to the tier; it never shortens a long one, and it
never drops work.

CPU proof, final run: fixed 0.03 s vs governor-supervised
0.04 s (46 slices, slice cap now 200k), full battery green
on both.

**Where 60 FPS stands (host):** on the healthy attempt-2/3
runner the app scene read 13.7 ms (≈73 host FPS) and the
engine series 14.85 ms mean (≈67 FPS) — over 60 on host,
with the fragment executor still the largest stage (the
FragBatcher flush floor convicted in v5 is untouched by
this round's scheduling/memory levers). On loaded runners
the same code reads 19–24 ms — CI runner variance, stated
plainly. The device anchor remains the tester's: v5 armed ≈ 35
FPS unstable; v6's device verdict is his to give, with the
governor toggle one tap away from the VPU toggle for an
on/off A/B.
