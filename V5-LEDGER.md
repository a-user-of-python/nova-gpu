# NovaGPU v5 — attempt ledger

Commission (Sean, 2026-11-13 13:13 ET): "implement a direct test
into the live scene for the vpus and optimize them and the rest of
the gpu more."

Two jobs:
1. A **direct VPU test inside the Live Scene**: while the mode is
   on, every triangle of the scene is ALSO positioned through the
   actual VPU path (grab unit -> one VPU per triangle -> the
   3-axis calculator, sub-ranges scheduled over the 50 MGVPU
   clusters), the VPU clip positions are compared live against the
   shader stage's clip positions (VMAT4 on uniforms c0..c3), and
   the scene is RENDERED FROM the VPU positions — a wrong VPU
   visibly breaks the picture. The HUD shows triangles tested,
   mismatched corners, max position error (0 required — a nonzero
   error is reported as a failure, never rounded into a pass),
   VPU throughput, MGVPU clusters engaged, alongside the scene's
   normal FPS readout. A host-only fault-injection hook proves the
   test can fail loudly (see attempt 1).
2. **Optimise the VPUs and the rest of the GPU more**, profile-led
   from the v4 baseline (BENCH-v4.md): the v4 sink is raster /
   fragment; the VPU path currently runs the 3-axis calculator one
   triangle at a time (lanes = 3 — below the calculator's NEON
   group width of 4, so it takes the scalar tail path) and the
   plot unit re-derives that cost per vector.

Hard invariant (unchanged from v2-v4): every v3 framebuffer
checksum in the host tests passes byte-identical. Any optimisation
that cannot keep output pixels identical is dropped, or flagged
in this ledger as a proposed trade and NOT shipped.

## Attempt 0 — measurement only (no behaviour change)
- NovaGPU.plotVectors records a pass profile (pass 1 = grab +
  VPU position + DDA + depth merge; pass 2 = fragment shading of
  accepted pixels) into `lastPlotProfile`, two clock reads, same
  pattern as DrawProfile.
- BenchTests gains: pass1/pass2 split on the plot saturation
  sweep; a plot parallel-threshold sweep (256..8192 x 1k/8k/65k
  vectors, same run) to convict the 2,048 default by measurement;
  a VPU micro-bench (29,696 random triangles positioned by the
  per-triangle VPU path, serial vs MGVPU-range parallel) — the
  number attempt 2 must beat.
- Expected: all host tests unchanged; bench log gives the v5
  baseline (stage medians + the new splits) in BENCH-v5.md.
- Status: in progress.

## Attempt 1 — in-scene VPU test
- As planned below: bus command 20 (arm resets the window,
  disarm stops accumulation), draw() test pass between the
  vertex and clip stages (MGVPU sub-ranges, per-triangle grab +
  VPU.position, bit-compare vs the shaded clip, gather overwrites
  the shaded clips), window + lifetime accumulators, static
  fault hook (default 0, host tests only), Live Scene toggle +
  HUD rows, host-test section 11 (armed == unarmed frames on
  cube + sphere, exact counts, v1-program reference, fault
  detection + frame change, disarm), bench scene
  app-scene-vputest + vputest-result line.
- Expected: all v3 checksums unchanged (correct VPUs make the
  pass observational); section 11 green; bench shows the armed
  frame cost and the pass throughput.
- CI run 43482002661: bench SUCCESS (armed frame 26.11 ms vs
  24.44 unarmed, pass 12.2M tri/s, 0 mismatches over 237,568
  triangles); host tests 1 FAILURE — the fault check expected
  12 mismatched corners, got 6. Root cause is in the TEST PASS,
  found by its own fault hook: the compare read the reference
  from the shaded vertex itself, so once a triangle's gather
  overwrote a shared vertex, the next triangle touching it
  compared VPU-vs-VPU and the fault was masked (one mismatch
  per cube face instead of per triangle). Fix: the pass now
  snapshots the shader clips into a pristine reference buffer
  before positioning, and every corner compares against it —
  the test is strictly stronger (and the honest design: a
  reference the gather cannot contaminate).
- CI run 43482787469 (fix): **SUCCESS, all jobs green.**
  Section 11 passes in full — fault now detected at exactly
  12 corners / max error 0.5, faulted frame differs, armed ==
  unarmed frames on cube + sphere, lifetime accumulates.
  Bench: pass 14.4M tri/s; armed frame 19.21 ms vs 19.56
  unarmed same run (pass cost ≈2.1 ms/frame by its own
  accounting, inside scene noise). Attempt 1 DONE.

## Attempt 1 plan (as written before implementation)
- Device: draw() gains an optional VPU test pass after the vertex
  stage (armed over the bus, command 20): per triangle, grab +
  VPU.position vs the shaded vertex's clip; bit-compare all 4
  components; per-window + lifetime accumulators (triangles,
  corners, mismatches, max abs error, pass ms, clusters engaged);
  the shaded clips are then OVERWRITTEN with the VPU clips, so
  the frame renders from the VPU path. Static fault hook
  `NovaGPU.vpuTestFault` (host tests only, default 0) perturbs
  corner-0 clip x before compare/gather.
- App: Live Scene toggle + HUD rows (verdict, triangles, max
  error, tri/s, clusters, pass ms); FPS row stays.
- Host tests (section 11): armed renders are frame-identical to
  unarmed; counts exact on cube (12 tris / 12 clusters) and
  sphere (20,480 / 50); fault injection produces mismatches,
  max error > 0 and a DIFFERENT frame; disarm stops accumulation.
- Expected: checksums unchanged (the pass is observational when
  the VPUs are correct — and they are the same ngp_mat4).

## Attempt 2 — VPU / plot optimisation
- Implemented as planned below: VPU.positionWide (one ngp_mat4
  over grouped grabbed records), GrabUnit.grabPosition for
  endpoints; the test pass now grabs per range in 256-triangle
  groups, issues wide, compares in-range (per-range counters,
  serial reduce + serial gather); plot pass 1 batches endpoints
  (NovaGPU.plotBatchSize = 128; lines fetch 2 records/vector —
  the duplicate third vertex v4 computed is gone). Host tests:
  section 12 (positionWide bit-identical to per-triangle VPU
  for groups 1..250 tris) + 9f (plot frame invariant at batch
  sizes 1 / 7 / 1024). Bench: vpu-micro-wide A/B.
- Expected: section 9c/9d/9e pixel-exactness unchanged; test
  pass tri/s up from 14.4M; plot pass-1 ms down at all scales.
- CI run 43484455833: **host tests + bench SUCCESS** (all host
  tests pass — 9f batch invariance at 1/7/1024, section 12 wide
  bitwise proof, sections 9/11 unchanged).
  Measured (HOST, same-run micro A/B): vpu-micro-wide serial
  **140.4M tri/s** vs per-triangle 52.6M — **2.67×** for the
  same grab + position work; parallel wide 135.6M ≈ serial
  (0.2 ms scale: dispatch overhead saturates). In-scene test
  pass: **27.5M tri/s** (from 14.4M), ≈1.08 ms/frame at the
  app scene's 29,696 triangles. Plot pass 1: 65,536-vector
  total 625.8 ms vs 666.1 in the attempt-1 run, pass 1 379 vs
  387 — inside cross-run noise; the honest read is the wide
  issue's plot effect is small (pass 1 is Swift bookkeeping-
  bound, not calculator-call-bound) and its VPU effect is
  large. Threshold sweep, same run: at 65,536 vectors the
  minimum is exactly threshold **2048** (419.5 ms; 1024 →
  566.9, 4096 → 437.8); at 8,192, 1024 (56.8) ≈ 2048 (60.0).
  **plotParallelThreshold stays 2048** — now sweep-backed.
  Attempt 2 DONE.

## Attempt 3 — C fragment writeback
- Change: `ngp_flush_writeback` (C) takes over FragBatcher
  flush's conversion + block-table write: identical min/max
  Comparable-form clamps (NaN propagates), `t * 255` in Float,
  truncating cast, same table rule; unwritable lanes return as
  skip indices and replay through the original Swift
  writePixel (locked fallback commits the block). NaN channels
  write 0 in C — Swift trapped on UInt8(NaN), so no defined
  output changes; documented in the header.
- Evidence: `app-scene-nofrag` fragExecCpu 5.2–11.8 ms across
  runs for 290,906 fragments is pure flush + writeback (the
  nofrag program writes constants) — the writeback loop is a
  convicted floor under every scene.
- Expected: all framebuffer checksums byte-identical (host
  tests are the gate); fragExecCpu down on nofrag/flatfrag.
- Status: pushed, awaiting CI.

## Attempt 2 plan (as written before implementation)
- Wide calculator issue: grabbed triangle records are positioned
  in groups through one ngp_mat4 call (lanes independent, order
  per lane unchanged — bit-identical by the v4 argument, and
  host-proven against the per-triangle VPU). Used by the plot
  pass (which also stops grabbing a duplicate third endpoint for
  lines) and by the test pass.
- Plot parallel threshold set from the attempt-0 sweep.
- Expected: plot pass-1 ms and VPU tri/s improve; sections
  9c/9d/9e (pixel-exactness, parallel == serial) unchanged.

## Attempt 3 — raster/fragment optimisation (planned, evidence-led)
- Convicted by the attempt-0 baseline (nofrag fragExec = flush
  writeback cost; clip wall): candidates are a C writeback for
  FragBatcher.flush (identical clamp/convert expressions) and/or
  allocation-free clip scratch. Only what the profile convicts.
