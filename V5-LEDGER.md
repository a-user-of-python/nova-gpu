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

## Attempt 1 — in-scene VPU test (planned)
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

## Attempt 2 — VPU / plot optimisation (planned)
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
