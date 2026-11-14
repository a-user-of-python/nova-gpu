# NovaGPU v7 — benchmark record (the Governor fix)

All numbers are **HOST numbers** from the CI bench job (free
standard macos-26 runner, Apple Silicon) unless explicitly
labelled device. CI shared-runner variance is large — the three
runs cited here read the same scene's work at ~48, ~35 and
~20 ms across the round — so only same-run comparisons are
cited as evidence.

## The device result that commissioned this round

**v6 with the Governor enabled: a stable 19.2 FPS on the
tester's iPad Air M3** (2026-11-13) — stable, but far below
v5's ~35 FPS (VPU test armed, unstable) on the same scene.
Governor-off v6 device FPS was not reported.

19.2 FPS is a 52.1 ms interval, and the v6 ladder's top tier
was 50.0 ms. Diagnosis (full version in V7-LEDGER.md): the
chip sensed only the inter-present interval — a number its
own pacing padding inflates — stepped a fixed tier ladder off
that interval's median, and could never step back up, because
its headroom test (median < lower tier × 0.80) is unreachable
while the pacer holds every interval at the current tier.

## Attempt 0 — the failure, reproduced on host (CI 43531474943)

Bench §11 (new): the app loop shape with work padded to a
scripted device-like pattern (phase A median 35 ms, phase B
12 ms), the shipped v6 governor attached exactly as the app
attaches it. The runner was slow (~48 ms real work, pattern
targets acting as floors) and the ratchet fired anyway:

- Action log: SET_PACE_TIER 20.8 → 25.0 → 33.3 → **50.0 ms**.
- Phase A tail governed: 49.4 ms (20.3 FPS), settling on the
  50 ms tier.
- Phase B (work fast again): governed interval mean
  **62.8 ms (15.9 FPS)** vs control **19.9 ms (50.1 FPS)** —
  pinned at the top tier while the identical ungoverned loop
  ran 3× faster. The lock-in, reproduced in one run.

## The fix (attempts 1 + 1b)

Separate work/interval sensors (work = device busy time,
which no wait can inflate); pace ceiling = **p25 of the work
window**, re-derived every tick, no ladder; work-bound
disengagement; engagement only with slack + jitter + no
lockout; an A/B self-guard with two cost channels (governed
interval vs the ungoverned estimate from a 24-frame probe;
work median vs the snapshot at lever application) that
reverts the newest lever with a plain-words "costing frames"
log; and (1b) a 240-tick lockout for any guard-convicted
lever, after §14 caught the jitter rule re-applying a width
cap the guard had just reverted.

## Final measurement (CI 43532564896, all jobs success)

**Host tests: 82 PASS, 0 FAIL** — every pre-v7 framebuffer
checksum intact, plus §14's deterministic traces: work-bound
loops never engage pacing (ceiling = work p25); the device
trace (work 35, intervals ~52) engages at a 35 ms ceiling —
never 50 — and disengages with the ceiling re-derived to 12
when the work drops; the guard trips on inflated intervals
with the exact log line "pacing disengaged: it was costing
frames", and a lockout + fresh probe follow; the width cap
is guard-reverted the same way. CPU proof: fixed slices
0.02 s vs governor-supervised 0.03 s, full battery green on
both (toy-scale mechanism demonstration, as in v6).

**§11 slow series, same run — the never-slower bar:**

| phase | control | governed (v7) | v6 (attempt 0) |
|---|---|---|---|
| A interval mean | 71.93 ms (13.90 FPS) | **71.96 ms (13.90 FPS)** | — |
| A interval stdev | 26.87 ms | **25.87 ms** | — |
| A tail mean | 65.73 ms | 68.89 ms | 49.36 ms (on the 50 tier) |
| B interval mean | 21.07 ms (47.5 FPS) | **16.14 ms (62.0 FPS)** | **62.79 ms (15.9 FPS)** |
| B interval stdev | 6.91 ms | **3.41 ms** | 41.86 ms |

Phase A is a dead heat (the governed loop's work is the
interval; pacing correctly stayed out of the way, and the
guard spent the run reverting its own grain/width attempts —
see the log below). Phase B is the headline: where v6 stayed
locked at ~16 FPS, v7's governed loop runs at least as fast
as the control, with half the interval stdev. The phase-A
tail governed reading (68.9 vs 65.7) is the one governed
number above its control; the full-phase means are identical
and both runs' work means match (72.0 vs 71.8), so it reads
as window placement on a drifting runner, stated plainly.

§11 governed action log (abridged): pacing engaged @ 19.7 ms
ceiling (tick 30) → grain 2 (60) → **guard reverts grain**
(62, work channel: "frame work rose 16.1 ms to 17.0 ms under
the lever") → **guard reverts pacing** (64, same channel) →
width cap (94) → **guard reverts width** (97) → lockouts hold
for the remaining ~100 frames; no lever is re-applied. Final
pacing state: "disengaged — it was costing frames (guard
lockout)".

**§10 fast series, same run:** governed interval mean
21.81 ms = its own work mean 21.80 (zero pacing tax), settled
stdev 4.58 ms; final state "disengaged — frames are
work-bound", ceiling 17.5 ms — the v7-correct outcome on a
back-to-back host loop, replacing v6's governed 21.26 vs
19.46 unpaced (a ~9% tax for smoothness).

## What v7 does and does not claim

At device-like speeds the scene is work-bound, and the honest
governor is one that stabilizes only when there is headroom
and otherwise stays out of the way — that is the governor
that shipped. It cannot make a work-bound scene faster; what
it guarantees is the floor: **enabling it can no longer cost
frames**, every lever it applies is on probation under the
guard, and the HUD shows its pacing state, its ceiling and
its reasons so the behaviour is checkable at a glance. The
device verdict — governor-on FPS at or above governor-off,
steady or with pacing visibly disengaged — is the tester's.
