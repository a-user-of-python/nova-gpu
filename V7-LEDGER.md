# NovaGPU v7 — attempt ledger (the Governor fix round)

Commission (2026-11-13, after device-testing v6): **with the
Governor enabled, the Live Scene runs at a stable 19.2 FPS on the
iPad Air M3.** The stability arrived; the speed did not. v5 with
the VPU test armed was ~35 FPS unstable on the same scene.
Governor-off v6 device FPS was not reported and is not assumed.

## Diagnosis — the mechanism, convicted from source

19.2 FPS is a 52.1 ms frame interval. No tier but one sits
anywhere near it: the v6 ladder is 16.7 / 20.8 / 25.0 / 33.3 /
50.0 ms, so the governed scene was sitting on the **50.0 ms
tier + ~2 ms of loop overhead**. The walk there, and why it
could never walk back:

1. **The sensor reads the padded interval.** The chip's only
   frame sample is `notePresent()`'s inter-present interval
   (NovaGovernor.swift / NovaBus.swift) — once pacing is on,
   that interval *includes the pacer's own wait*. The chip
   never sees the work time underneath its own padding.
2. **The statistic is the median of that interval**, over a
   24-frame window (needs ≥ 8), against **overrun = median >
   tier × 1.02**, streak 2, with a 30-tick cooldown between
   commands. Whenever the median work sits within 2% above a
   tier, the median guarantees a step down. Device work under
   the test crossed the 33.3-tier overrun line (34.0 ms
   median), so the ladder walked 16.7 → 20.8 → 25 → 33.3 → 50.
3. **The headroom test is unreachable once padded.** Stepping
   back up requires median < lower tier × 0.80 — at tier 50,
   median < 26.6 ms. But the pacer holds every frame start to
   the 50 ms grid, so the interval median can never fall below
   ~50. The chip is blind to the headroom it created: a
   one-way ratchet, by construction.
4. **Why the host missed it:** every v6 host series ran work
   medians of ~19–24 ms, comfortably under the 33.3 tier, so
   the ladder stopped at 20.8 and looked virtuous (stdev
   5.19 → 3.88) — while even there the governed mean interval
   (21.26) was already *above* the unpaced control (19.46):
   the chip was spending ~9% of throughput to buy smoothness.
   At device speeds the same trade is catastrophic.

Attempt 0 reproduces the failure on host before any fix: a
device-like slow series (scripted work pattern, median 35 ms —
the regime the device verdict implies — then a fast phase)
through the exact app loop shape with the shipped v6 governor.
Expected: tier walks to 50, governed phase-A tail ≈ 50 ms vs
control ≈ 37.5, and phase B (work dropped to ~12 ms) stays
pinned at ~50 ms in the governed run while the control drops —
the ratchet, in one run.

## The fix (attempt 1) — the never-slower law

- **Two sensors, never mixed.** Telemetry now carries the pure
  work time (device busy time between presents, measured in
  NovaGPUDevice.servicePendingWork — it excludes every wait by
  construction) AND the inter-present interval. The chip keeps
  separate work and interval windows.
- **Pace ceiling from work only.** The ladder is gone. The
  pace ceiling is the **p25 of the recent work window** — a
  frame time the hardware demonstrably achieves — recomputed
  every tick from the short window. A ceiling can never sit
  above demonstrated speed.
- **Work-limited disengagement.** If the loop is work-bound
  (interval median − work median ≤ max(1 ms, 3%) — the
  interval *is* the work; there is no slack), pacing stays
  disengaged and the log says why. Stability work falls to
  grain / width / priority only.
- **Engagement** requires a calibrated baseline, real slack
  (not work-bound), and no active guard cooldown.
- **A/B self-guard.** On enable, a 24-frame probe runs
  ungoverned and records the loop slop (interval median −
  work median). Afterwards the chip continuously estimates
  the ungoverned interval as current work median + slop; if
  the governed interval median exceeds that estimate × 1.05
  for 2 consecutive checks, it reverts its most recent lever
  (pacing first, then width cap, then fine grain) and logs it
  in plain words — "pacing disengaged: it was costing frames".
  After a pacing trip, a fresh probe must complete before
  pacing may re-engage.
- **HUD honesty.** The app shows governor on/off, pacing
  ENGAGED / DISENGAGED + the reason, the current pace ceiling
  in ms, and the interval jitter — the chip's behaviour is
  visible at a glance.
- Everything else from v6 that did not hurt stays: scratch
  reuse, CPU slice steering, grain/width levers (now also
  under the guard). All invariants bind: framebuffer
  checksums byte-identical, ISA appends only, 1536 CUs.

## Attempt 0 — reproduce the v6 failure on host (no fix yet)
- BenchTests §11: device-like slow series, governed (shipped
  v6 chip, app-identical attach + pacing) vs unpaced control,
  slow phase (scripted work, median 35 ms) then fast phase
  (~12 ms), per-phase interval stats + final tier + action log.
- Expected: governed tier → 50; governed slow-phase tail
  ≈ 50 ms vs control ≈ 37.5 ms; governed fast phase stays
  ≈ 50 ms (lock-in) vs control ≈ work speed.
- Measured: (pending CI)

## Attempt 1 — the fix
- (pending)

## Attempt 2 — close-out
- (pending)
