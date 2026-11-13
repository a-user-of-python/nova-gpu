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
