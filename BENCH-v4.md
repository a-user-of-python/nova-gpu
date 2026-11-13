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
- `app-scene-flatfrag` — same geometry/coverage, fragment program
  reduced to colour pass-through (isolates fragment ALU cost).
- `app-scene-minvert` — same scene, vertex program reduced to
  position pass-through (isolates vertex-stage floor).
- `saxpy` — the host test's 1,000,003-lane SAXPY launch.
- `clusters=N` — app scene at cluster counts 24 / 48 / 96.

Design constraints for the M3 target applied throughout: the worker
decomposition targets the performance cores (4P + 4E on M3) without
oversubscription, tiles/chunks are sized to stay cache-resident,
and no optimisation may rely on thread behaviour the host profile
can't demonstrate.

## v3 baseline (commit 67f652c + profiling/bench only)

_(filled from the first v4 CI bench run)_

## v4 results

_(filled at the end of the round)_
