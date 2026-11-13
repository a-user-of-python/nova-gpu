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
