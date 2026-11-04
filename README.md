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
