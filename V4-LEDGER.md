# NovaGPU v4 — attempt ledger

Commission (Sean, 2026-11-13): "optimize the hell out of it. Maybe
add more cu or vector units specifically designed for grabbing and
plotting 3d vectors." Device baseline: v3 at ~27 FPS on iPad Air M3
(v2 was 26.9 — host gains did not move the device; the device is
the judge).

## Attempt 0 — profiling groundwork (no behaviour change)
- Added DrawProfile stage timing to draw() (snapshot / vertex /
  clip / bin / raster walls), a BenchTests harness (app scene +
  isolation scenes + SAXPY + cluster sweep), a CI bench job, and
  made clusterCount configurable (default unchanged: 48x32 = 1536).
- Expected: host tests unchanged (framebuffer checksums police
  bit-exactness); bench log gives the v3 stage profile.
- Status: pushed, awaiting CI.

## Baseline profile (CI bench, v3 code, runs 43472103839 + 43472970593)
- App scene (host): frame ~26-36 ms across runs (shared-runner
  variance); stage medians: snapshot 0.16, vertex 1.3-1.5,
  clip 3.1-3.7, bin 2.3, raster 17.9-25.1 ms. Fragments 290,906.
- Executor micro: lit fragment warp 0.80 us; lit vertex warp
  2.76 us (~38 issues, closure LDATTR/OUT per lane).
- Fragment ALU+issue is the largest identified raster CPU sink
  (flush CPU-sum 17.9 ms lit vs 5.6 ms END-only).
- Cluster sweep: 48 (28.3/21.7 ms) beats 24 and 96 in both runs —
  1536 CUs stay the default; the knob is documented as
  measured-neutral for execution (accounting granularity).

## Spec changes absorbed during the round
- 6-bit vector idea (12:23 ET) was REPLACED at 12:28 ET before any
  code was written: world-space math FP32 (chosen over 16.16
  fixed-point — bit-matches the existing pipeline, which is what
  makes plot-vs-triangle pixel equivalence provable), screen-space
  16 bits per axis (SCREEN_AXIS_BITS=16, Q10.6) in the plot unit's
  datapath only. No 6-bit design ships.
- VPU shape (12:30 ET): one VPU = one triangle workload (3 fetched
  vertices, 9 FP32 axis values, 3-axis calculator, positioned
  triangle out). VPU_COUNT = 2 x MAX_SCREEN_VECTORS, placeholder
  MAX_SCREEN_VECTORS = 4096 until Sean's number lands.

## Attempt 1 — C core completion + vector ISA ops
- NovaGPUFast.c/.h: min/max/sqrt/rsq/cmplt/sel/f2i/i2f + all int
  ops moved from Swift scalar loops to the C core; new NEON
  plane-wise DOT3/DOT4/VCROSS/VMAT4 (accumulation order identical
  to the scalar loops, zero-start so the -0 corner matches) and
  VNORM; scalar fallbacks are alias-safe (locals first).
- ISA appended (no renumbering): VCROSS 0x1E, VNORM 0x1F,
  VMAT4 0x24 + ShaderBuilder support. Executor routes the moved
  ops to C; SIN/COS stay scalar libm (no vector libm without
  fast-math).
- HostTests section 8: VCROSS/VNORM bit-exact vs same-expression
  references (5 lanes = NEON group + tail), in-place aliasing,
  VMAT4 bit-identical to the manual DOT4 chain, out-of-range
  uniform rows read as zero.
- Expected: all v3 checksums unchanged; executor micro-benches
  improve; frame time improves in raster/vertex.
- Status: pushed, awaiting CI.
- CI run 43474077343: SUCCESS — all v3 checksums unchanged
  (bit-exact through the C move) + section 8 green.
  Bench vs refined v3 baseline: executor-fraglit 0.799 -> 0.717
  us/warp; executor-vertlit 2.764 -> 2.090 us/warp; app-scene-lit
  frame ~26.1 -> ~19.2 ms, fragExecCpu 17.86 -> 11.34 (host,
  cross-run; executor micros are the tight signal).

## Attempt 2 — grab unit + vertex fast paths + inline varyings + span narrowing
- GrabUnit (NovaGPUVector.swift): per-warp wide fetch of the
  snapshot into [lane][16] records (vec3->vec4 expansion identical
  to the v3 closure); executor reads via attrPtr/attrStride=16.
- Vertex outputs via outPtr + outFull ([lane][32], all slots),
  scattered into ShadedVertex (slot 0 -> clip, 1..7 -> varyings,
  varyings slot 0 stays zero as always). ~350 closure calls per
  warp eliminated from the vertex stage.
- ShadedVertex varyings: [Float] heap array -> inline VarBlock
  (8 SIMD4s) — kills ~15k array allocations/draw + COW traffic in
  clip copies; lerp/rasterise read through raw pointers.
- rasteriseBody span narrowing: per-row analytic inside-interval
  (both sign cases, hull, +2 px margin) skips certainly-outside
  bbox pixels; the per-pixel inside test is unchanged and remains
  the judge — drawn set provably identical.
- Executor generalized: attrStride (fragment 32 default, vertex
  16), outFull flag; fragment behaviour unchanged.
- NovaGPU.maxScreenVectors = 4096 placeholder + vpuCount = 2x;
  GPUStats.vectorsPlotted added.
- Expected: all checksums identical; vertex stage and raster CPU
  down. Status: pushed, awaiting CI.
