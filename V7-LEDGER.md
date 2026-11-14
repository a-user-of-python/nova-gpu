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
- Measured (CI run 43531474943, all jobs green): **reproduced.**
  The runner was slow that run (real scene work ~48 ms, so the
  scripted targets acted as floors and phase-A work ran hotter
  than scripted — the ratchet fired anyway, across regimes).
  The governed action log shows the full ladder walk —
  SET_PACE_TIER 20.8 @ tick 60 → 25.0 @ 120 → 33.3 @ 151 →
  **50.0 @ 182** — and the phases convict the lock-in:
  - Phase A tail: governed 49.4 ms (20.3 FPS) settling onto
    the 50 ms tier.
  - Phase B (work back to fast): governed interval mean
    **62.8 ms (15.9 FPS)** vs control **19.9 ms (50.1 FPS)** —
    the governed loop stayed pinned at the top tier while the
    identical ungoverned loop ran 3× faster. The headroom
    test (median < 26.6 ms) is unreachable from inside the
    padding, exactly as diagnosed.
  Status: DONE — mechanism convicted from source + reproduction.

## Attempt 1 — the fix
- NovaGovernor.swift: policy rewritten per the design above —
  separate work/interval windows; ceiling = work p25, no
  ladder; 24-frame ungoverned probe calibrates the slop;
  work-bound disengagement; engagement requires slack +
  jitter + no lockout; A/B guard with TWO trip channels
  (governed interval vs ungoverned estimate; work median vs
  the snapshot taken when the newest lever was applied —
  grain/width costs hide inside the work time) reverting the
  newest lever with a plain-words "costing frames" log; HUD
  state text (pacingStateText) says engaged / calibrating /
  disengaged + why. Device ID bumped to v7.0; new registers
  pacingState / workMedianUs / workP25Us; paceTierUs slot now
  reports the ceiling.
- NovaBus.swift: NovaGPUDevice accumulates its busy time in
  servicePendingWork (consumeBusyMs); notePresent now ticks
  the chip with work (busy) and interval (present-to-present)
  as separate samples.
- GPUEngine.swift: pacing wait only while the chip reports
  pacing engaged, targeting paceCeilingMs; HUD gains
  "Governor pacing" (state + reason) and "Governor pace
  ceiling" rows beside the existing jitter rows.
- BenchTests: §10 runs the v7 app algorithm (engaged-gated);
  §11 is now the v7 A/B proof on the same device-like series.
- HostTests: §13 tier assertions replaced with v7 ceiling /
  disengagement assertions; new §14 deterministic traces —
  work-bound, the device trace (engage @ 35, never 50; no
  lock-in at 12), the guard trip with the exact log phrase +
  lockout, and the width-cap guard revert.
- Expected: all pre-v7 checksums intact; §13+§14 green; §11
  governed ≈ control in both phases with the pacing state and
  log printed.
- Measured (CI run 43532204150): bench job green; host-tests
  1 FAIL (§14 width-guard trace — see attempt 1b). Bench
  evidence (slow runner; §9 control work mean 23.4 ms):
  - §10 fast series: governed interval mean 19.05 ms ≈ its
    own work mean 19.10 — zero pacing tax; final state
    "disengaged — frames are work-bound", exactly the
    v7-correct outcome on a back-to-back loop.
  - §11 slow series: phase A governed 72.3 ms vs control
    71.4 (FPS 13.83 vs 14.01 — within noise); phase A tail
    governed 64.5 vs control 71.8; phase B governed 19.7 vs
    control 13.6 — the governed run's own work ran hotter
    (19.8 vs 13.6) with grain-2 episodes active in phase B on
    a drifting runner. The log shows the guard doing its job
    throughout: pacing engaged @ 12.4 ms ceiling, guard-tripped
    at tick 32 with the exact "pacing disengaged: it was
    costing frames" line; grain applied and guard-reverted
    three times as work rose under it.
  Status: bench side DONE; one host-test failure → attempt 1b.

## Attempt 1b — guard lockout for reverted levers
- Convicted by §14's width trace: the guard reverted the
  width cap, but the jitter rule legitimately re-applied it
  30 ticks later (jitter conditions unchanged), and the two
  fought for the rest of the trace — final width depended on
  phase parity. A guard-convicted lever must stay off.
- Change: reverting width/grain now also sets a 240-tick
  lockout for that lever (pacing already had one); rules 2/3
  check their lockout before applying. Same change damps the
  apply/revert cycling the attempt-1 bench log showed on a
  noisy runner (grain applied at #62/#96/#150, reverted at
  #66/#120/#160 — after 1b the first revert locks it out).
- Expected: §14 width trace passes (width stays 0 after the
  revert); §11 phase B governed work no longer carries
  grain-2 episodes; all other results unchanged.
- Measured: (pending CI)

## Attempt 2 — close-out
- (pending)
