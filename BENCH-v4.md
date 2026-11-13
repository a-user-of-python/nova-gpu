# NovaGPU v4 — benchmark record

All numbers in this file are **HOST numbers** from the CI bench job
(free standard macos-26 runner, Apple Silicon), unless explicitly
labelled device. The device baseline that v4 is judged against is
Sean's iPad Air M3 running v3 at **~27 FPS** (reported 2026-11-13;
v2's device reading was 26.9 FPS). Host gains that don't move the
device don't count — v3 proved that — so every optimisation below
had to win on the stage profile, not on a single frame number.

Bench scenes (BenchTests/main.swift):
- `app-scene-lit` — the test app's live scene exactly: icosphere
  (5 subdivisions, 20,480 tris) + torus (96x48, 9,216 tris) at
  640x480, lit vertex/fragment programs, clear + 2 draws. Stage
  walls are the median of both draws' `lastDrawProfile` sections.
- `app-scene-lit-v4vert` — same scene with the v4 (VMAT4) vertex
  program: a same-run A/B against `app-scene-lit`.
- `app-scene-flatfrag` — same geometry/coverage, fragment program
  reduced to colour pass-through (isolates fragment ALU cost).
- `app-scene-minvert` — same scene, vertex program reduced to
  position pass-through (isolates vertex-stage floor).
- `saxpy` — the host test's 1,000,003-lane SAXPY launch.
- `clusters=N` — app scene at cluster counts 24 / 48 / 96.
- `plot-lines` — the plot unit's saturation sweep: random line
  vectors (NDC [-0.9, 0.9]², z = 0.5) at 1k / 8k / 32k / 65,536
  vectors on a 640×480 framebuffer.

CI shared-runner variance is large: single-run frame times move
±20% between identical commits, and only **same-run** comparisons
are cited as evidence below. The executor micro-benches (20,000
warps, tight loop) are the least noisy signal.

Design constraints for the M3 target applied throughout: the worker
decomposition targets the performance cores (4P + 4E on M3) without
oversubscription, tiles/chunks are sized to stay cache-resident,
and no optimisation may rely on thread behaviour the host profile
can't demonstrate.

## v3 baseline (CI run 43472970593, v3 + profiling only)

App scene, refined same-run A/B profile (fragments: 290,906):

| stage | ms |
|---|---|
| frame (wall) | 26.07 |
| snapshot | 0.16 |
| vertex | 1.49 |
| clip | 3.09 |
| bin | 2.32 |
| raster | 17.90 |
| fragment exec (CPU sum) | 17.86 |

Executor micro-benches: lit fragment warp **0.799 µs**, lit vertex
warp **2.764 µs** (the vertex warp spends most of its issues in
per-lane closure LDATTR/OUT calls — ~128 + ~224 per warp).

Profile conclusions that drove v4:
1. Fragment shader execution is the largest identified CPU sink
   (17.9 ms CPU-sum lit vs 5.6 ms with an END-only fragment
   program) → move every remaining executor op into the C core.
2. The vertex stage pays hundreds of closure calls per warp →
   grab unit + pointer fast paths.
3. Raster coverage tests the full bounding box of every triangle
   → analytic span narrowing (the per-pixel test stays the judge).
4. Cluster sweep: 48 clusters beat 24 and 96 in both baseline
   runs → **1536 CUs stay**; the count is configurable and
   documented as accounting granularity, execution parallelism
   comes from the pipeline's work decomposition.

## v4 step A — C core completion + vector ISA ops (CI run 43474077343)

All executor ops moved from Swift scalar loops into the C core
(NEON plane-wise DOT3/DOT4/VCROSS/VMAT4; identical accumulation
order — every v3 framebuffer checksum in the host tests passed
unchanged, and VMAT4/VNORM/VCROSS are bit-exact against their
reference expressions):

| metric | v3 | v4 step A |
|---|---|---|
| executor lit fragment warp | 0.799 µs | **0.717 µs** |
| executor lit vertex warp | 2.764 µs | **2.090 µs** |
| app-scene-lit frame (host) | 26.07 ms | 19.19 ms |
| vertex stage | 1.49 ms | 0.96 ms |
| fragment exec (CPU sum) | 17.86 ms | 11.34 ms |

(Frame/stage rows are cross-run and indicative; the micro-benches
are the tight signal.)

## v4 full build — grab unit, inline varyings, span narrowing

Final CI run 43478322398 (all host tests + CPU proof + IPA build
green). That run's runner was slow (frame times ~25 ms where the
healthy step-A run read ~19 ms), so scene rows are quoted from
the step-A run above; the tight executor signals and the plot
sweep are from the final run:

- Same-run A/B, v1 vs v4 (VMAT4) vertex program
  (`app-scene-lit` vs `app-scene-lit-v4vert`): frame 25.81 vs
  25.26 ms — within runner noise at scene level; the vertex
  stage itself is 0.6–1.0 ms of a ~20 ms frame, so VMAT4's win
  shows in the executor micro-bench (2.76 → 2.09 µs/warp), not
  in scene wall time. Both programs' frames are byte-identical
  (host test 10).
- Cluster sweep re-run: 24 → 21.5, 48 → 23.2, 96 → 25.2 ms —
  order between 24/48 flips between runs (noise), 96 is
  consistently worst. 1536 CUs (48 clusters) stay.

### Plot-unit saturation (final run, pixel-step DDA)

Random screen-spanning line vectors, 640×480, flat shading:

| vectors | ms | µs/vector | fragments shaded |
|---|---|---|---|
| 1,024 | 13.2 | 12.9 | 243,616 |
| 8,192 | 90.5 | 11.1 | 1,942,084 |
| 32,768 | 269.6 | 8.2 | 7,805,233 |
| 65,536 (full budget) | 476.8 | 7.3 | 15,658,069 |

Per-vector cost DROPS with count (the MGVPU-parallel path
amortizes); at full budget the pass is bounded by genuinely
shading 15.7M covered pixels, not by sampling. An earlier
grid-step DDA sampled ~64× more points than pixels and took
3,558 ms for the same 65,536 vectors — replaced before release
(see V4-LEDGER attempt 4); the pixel set is unchanged in kind
and still matches the Double reference exactly on the crafted
suite.

## Vector hardware (v4)

- **Grab unit** — per-warp wide fetch of vertex attributes into
  [lane][vec4] records (vec3→vec4 expansion identical to the v3
  closure), read by the executor through a raw pointer: the
  vertex stage's ~350 closure calls per warp are gone.
- **Vector ops** — VCROSS (0x1E), VNORM (0x1F), VMAT4 (0x24)
  appended to the ISA (no renumbering), executed as NEON ops in
  the C core; world-space math is **FP32** (chosen over 16.16
  fixed-point: it matches the existing pipeline bit-for-bit,
  which is what makes the plot path's pixel-exactness provable).
- **VPU** — one VPU positions one triangle: 3 grabbed vertices,
  9 axis values (3 points × X/Y/Z), one 3-axis calculator
  (ngp_mat4 — bit-identical to the shader stage's VMAT4, proven
  in host tests). VPUs are grouped into **MGVPU clusters**:
  MGVPU_COUNT = 50 × VPUS_PER_MGVPU = 131,072 = **6,553,600
  addressable VPUs**. The pool is virtual — a VPU is a stateless
  48-byte work slot materialized on demand on a host thread;
  scheduling distributes contiguous MGVPU sub-ranges across
  threads. Full-occupancy position data (~315 MB) lives in the
  caller's vertex buffers inside the sparse 2 GiB VRAM.
- **Plot unit** — bus command `plotVectors` (19): lines (index
  pairs) or points, transform = uniforms c0..c3, colour =
  uniform c10 delivered as varying slot 1. Endpoints convert to
  the screen grid: **SCREEN_AXIS_BITS = 16 bits per axis**, Q10.6
  (64 sub-pixel steps per pixel). The DDA steps at pixel
  resolution over the grid; depth testing is the triangle rule.
  Per-frame budget MAX_SCREEN_VECTORS = 65,536 (excess dropped).
  Pass 1 runs serially below 2,048 vectors and distributes
  MGVPU sub-ranges concurrently above it; both paths produce
  byte-identical frames (host-tested). Pixel-exactness: plotted
  lines match a Double-precision implementation of the same DDA
  spec exactly on the crafted suite; endpoints land on the same
  pixels as the full pipeline's projection.

**Device verdict: pending Sean's measurement** — same Live Scene
(and the new Vectors page) on the iPad Air M3, against the v3
baseline of ~27 FPS.
