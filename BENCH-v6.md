# NovaGPU v6 — benchmark record

All numbers are **HOST numbers** from the CI bench job (free
standard macos-26 runner, Apple Silicon) unless explicitly
labelled device. The device anchor for this round is Sean's:
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
is the measured counterpart of Sean's "fps all over the place":
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

## Attempt 1 — governor chip (host evidence)

(To be filled from the attempt-1 CI run: section 13 results,
governor-on vs governor-off engine series.)

## Attempt 3 — scratch reuse (host evidence)

(To be filled from the attempt-3 CI run.)
