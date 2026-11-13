# NovaGPU v5 — benchmark record

All numbers are **HOST numbers** from the CI bench job (free
standard macos-26 runner, Apple Silicon) unless explicitly
labelled device. The device baseline remains Sean's iPad Air M3:
v3 measured ~27 FPS on the Live Scene; the v4 device verdict is
still his to give, and v5's will be too. CI shared-runner variance
is large (±20% and worse between runs; the cluster sweep in the
baseline run below inverts v4's ordering purely from runner
load) — only same-run comparisons and the tight micro-benches
are cited as evidence.

## v5 baseline (CI run 43481543441, v4 code + measurement only)

App scene stage medians (both draws summed), shipped v4 vertex
program (`app-scene-lit-v4vert`), fragments 290,906:

| stage | ms |
|---|---|
| frame (wall) | 25.50 |
| snapshot | 0.20 |
| vertex | 0.78 |
| clip | 3.39 |
| bin | 2.79 |
| raster | 19.09 |

Flush-cost evidence (same run): with an END-only fragment
program (`app-scene-nofrag`), fragExecCpu is still **11.79 ms**
— that is the FragBatcher flush floor (default-colour init +
RGBA conversion + block-table writeback + warp bookkeeping)
with essentially zero shader execution. The lit program's
fragExecCpu is 17.5–22.5 ms (CPU-sum across chunk threads), so
roughly half the flush cost is writeback-side, not ALU. This
convicts the C writeback (attempt 3).

Plot-unit pass split (same run; pass 1 = grab + VPU position +
DDA + depth merge, pass 2 = fragment shading of accepted
pixels):

| vectors | total ms | pass 1 | pass 2 | fragments |
|---|---|---|---|
| 1,024 | 14.7 | 9.3 | 5.4 | 243,616 |
| 8,192 | 73.0 | 38.3 | 31.4 | 1,942,084 |
| 32,768 | 311.2 | 176.1 | 147.7 | 7,805,233 |
| 65,536 | 672.6 | 402.6 | 289.0 | 15,658,069 |

Pass 1 is the larger half at every scale → the wide-issue VPU
path (attempt 2) aims at the right half; pass 2 also inherits
the attempt-3 writeback.

Threshold sweep (same run): at 65,536 vectors every threshold
≤ 8,192 takes the parallel path and reads 671–841 ms — pure
runner noise; the threshold only gates serial vs parallel. At
1,024–8,192 vectors the serial/parallel difference is inside
the noise band too (parallel never clearly loses). Verdict:
keep 2,048 until attempt 2 lands, then re-read the sweep with
the faster pass 1.

VPU micro-bench (same run): 29,696 random triangles through
the per-triangle VPU path — **serial 0.642 ms (46.3M tri/s)**,
MGVPU-range parallel 0.373 ms (79.6M tri/s). The per-triangle
path issues the 3-axis calculator at lanes = 3, below its NEON
group width of 4, so it takes the scalar tail throughout —
the headroom attempt 2 goes after.

Executor micro (same run): lit fragment warp 0.884 µs (v4's
healthy-runner reading was 0.717 — cross-run noise; the
same-run scene A/Bs are the evidence used below).

CU sweep: NOT re-convicting anything — this run's sweep is
runner-load garbage (24 → 40.2, 48 → 52.0, 96 → 66.2 ms, while
the same run's own app scene at 48 clusters read 25.5 ms).
1536 CUs stay, per v4's cleaner sweeps.
