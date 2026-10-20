# Plan L full-calibration result: No-Go (2026-09-03)

## Decision

Plan L v1 is **No-Go** under its pre-registered ten-frequency contract. The
implementation, depth-convergence gate, branch diagnostics, Hessian checks,
support contract, and query-speed smoke checks succeeded. The full calibration
hard gate failed only for Love root order 0 at 0.0250--0.0300 Hz.

No threshold, frequency, sample, or domain was changed after seeing this
result. Full development collection, gate freeze, sealed holdout seed
`20270509`, and Plan S were not run.

## Frozen scope and provenance

The evidence was generated from clean commit
`5eac1f036b02867c105cb3512ec94fc945312f7d` with plan hash
`9c01e843a46848176bae00ae7be4a4f095e003c12bfafc1f014a4bbee9ce0c99`.
It used the fixed 2400 km ocean-free PREM reference, the 18-parameter
0--60 km Vsv/Vsh basis, 32 prior-whitened directions with both signs, and the
seven 0.5--10% physical-anomaly levels. This produced 448 exact models: 384 in
the primary domain and 64 at the 10% stress level.

| evidence | SHA-256 |
|---|---|
| depth-convergence report | `2d5f76d818c164fc62a23b1ae5487347de920eb65a1b4d5972bf767a726ba8b0` |
| calibration split manifest | `6068224afe86a7f405ad7e9123817150b84eb85427d416fed4ba24503cf975fd` |
| candidate artifact, not frozen | `97b2ac84f742246b8f288db9381c7690c9f838ee9a2fba831f93953c00784f2d` |
| calibration report | `f744def94523bd8cea737adc5e6f582f358b7d4c9d6c35b5106d28e42a1e70ab` |

The large arrays remain under ignored `build/plan_l/`; this document records
their immutable identities and summary metrics.

## Depth and numerical checks

All Rayleigh/Love root orders 0--2 existed at every primary frequency for the
2200, 2400, and 2600 km references. Exact root/trace failure, cross-depth mode-
character disagreement, nonfinite derivative, and Phase-1 trace mismatch
counts were all zero.

- 2400--2600 km Taylor-2 prediction difference at 8%:
  `p99(|delta c|/sigma) = 2.486e-08`, versus the `1/30` limit.
- Maximum exact phase-velocity relative difference: `5.580e-07`.
- Maximum adjacent-gap absolute difference: `5.498e-07`.
- Largest relative Jacobian/Hessian difference among the four target curves:
  `3.045e-10` / `6.174e-09`.

The independent Love Hessian test compared Vsv/Vsh columns against central
finite differences of the separate full-AD gradient path at 0.0075, 0.0200,
and 0.0300 Hz for root orders 0 and 1. Its largest column difference was
`1.577e-08` relative and its largest scaled Vsv--Vsh cross-term difference was
`7.251e-10`, both far below the pre-existing `5e-3`/`1e-2` tolerances.

## Full calibration result

Errors below are normalized by the fixed synthetic observation uncertainty
`sigma/c = 0.1%`. The p90/p99 columns combine all primary amplitudes; every
frequency was also required to pass independently.

| curve | result | point p90 | point p99 | curve-RMS p90 | T2/T1 p90 | minimum T2 order |
|---|:---:|---:|---:|---:|---:|---:|
| Rayleigh 0 | Pass | 0.0132 | 0.0590 | 0.0202 | 0.0698 | 2.973 |
| Rayleigh 1 | Pass | 0.00828 | 0.0599 | 0.0186 | 0.0967 | 2.966 |
| Love 0 | **Fail** | 0.0946 | 0.9748 | 0.2331 | 0.0773 | 2.953 |
| Love 1 | Pass | 0.0627 | 0.5785 | 0.1434 | 0.1190 | 2.964 |

Love 0 failed the per-frequency criteria as follows:

| frequency (Hz) | p90 error/sigma | p99 error/sigma | T2/T1 p90 | T2 residual order | failed criterion |
|---:|---:|---:|---:|---:|---|
| 0.0250 | 0.2457 | **1.0477** | 0.1080 | 2.983 | p99 > 1 |
| 0.0275 | **0.3694** | **1.2590** | 0.1719 | 3.010 | p90 > 1/3; p99 > 1 |
| 0.0300 | **0.4694** | **1.3967** | 0.2411 | 3.043 | p90 > 1/3; p99 > 1 |

For Love 0, exact branch/trace failures, accepted mode-identity failures,
nonfinite predictions, and unsupported primary points were all zero. Support
was 100% over all ten frequencies. The near-cubic residual orders and strong
Taylor-2/Taylor-1 improvement show that the failure is the magnitude of the
third-order remainder near the high-frequency edge, not a missing Hessian or
root-label failure.

The pre-registered one-bin recipe would diagnose a candidate band ending at
0.0200 Hz: 0.0225 Hz passes but is removed as the safety bin before the first
failure at 0.0250 Hz. Plan L v1 nevertheless fixes all ten frequencies, so this
diagnostic cannot rescue the result. A shorter band would require a new plan,
version, artifact, and unused holdout; it is not implemented here.

## Runtime smoke and stopping rule

A non-promotion quick run with the fixed representative batch size 8 measured
the following speedups against branch-confirmed exact curves. It also verified
zero exact-solver counter changes during surrogate queries.

| curve | scalar speedup | batch-8 speedup |
|---|---:|---:|
| Rayleigh 0 | 5245x | 9387x |
| Rayleigh 1 | 5604x | 9936x |
| Love 0 | 1336x | 2425x |
| Love 1 | 1576x | 2714x |

These timings establish implementation feasibility but are not promotion
evidence because the accuracy gate had already failed. A two-direction quick
development smoke also failed Love root 0, but was not used as full evidence.
Following the fixed stopping rule, no full development exact collection or
full promotion benchmark was performed, no gate lock or holdout-open marker
was written, and the candidate artifact hash was not frozen. The sealed
holdout remains unopened.

## Post-decision empirical characterization

After recording the No-Go, the same 32 opened calibration directions and both
signs were evaluated at fixed 8%, 10%, 15%, and 20% maximum physical shear
anomaly. Each row below pools 64 models and ten frequencies (640 point errors).
The 8% rows come from the original calibration archive; 10/15/20% use an
opened stress diagnostic derived from seed `20270507`. The duplicate 10% exact
calculation agreed with the original archive to zero reported relative
difference.

Values are absolute phase-velocity relative errors in percent. `max observed`
is only the largest value in this finite, direction-dependent cohort; it is
not a mathematical worst-case bound or an accuracy guarantee.

| curve | amplitude | p50 | p90 | p99 | max observed |
|---|---:|---:|---:|---:|---:|
| Rayleigh 0 | 8% | 0.00102% | 0.00461% | 0.0108% | 0.0183% |
| Rayleigh 0 | 10% | 0.00194% | 0.00880% | 0.0228% | 0.0384% |
| Rayleigh 0 | 15% | 0.00642% | 0.0311% | 0.0963% | 0.154% |
| Rayleigh 0 | 20% | 0.0155% | 0.0849% | 0.290% | 0.434% |
| Rayleigh 1 | 8% | 0.000242% | 0.00458% | 0.0115% | 0.0149% |
| Rayleigh 1 | 10% | 0.000469% | 0.00890% | 0.0242% | 0.0313% |
| Rayleigh 1 | 15% | 0.00159% | 0.0307% | 0.0994% | 0.138% |
| Rayleigh 1 | 20% | 0.00376% | 0.0814% | 0.299% | 0.440% |
| Love 0 | 8% | 0.00338% | 0.0643% | 0.143% | 0.242% |
| Love 0 | 10% | 0.00630% | 0.124% | 0.284% | 0.475% |
| Love 0 | 15% | 0.0209% | 0.399% | 0.956% | 1.46% |
| Love 0 | 20% | 0.0493% | 0.976% | 2.20% | 3.21% |
| Love 1 | 8% | 0.00288% | 0.0391% | 0.0982% | 0.162% |
| Love 1 | 10% | 0.00565% | 0.0739% | 0.194% | 0.319% |
| Love 1 | 15% | 0.0187% | 0.241% | 0.617% | 0.920% |
| Love 1 | 20% | 0.0430% | 0.521% | 1.29% | 1.65% |

At 10/15/20%, all exact roots and traces succeeded for both waves and root
orders 0--2. For the target roots 0/1, accepted mode-identity failures and
nonfinite predictions were also zero. The error growth, especially Love 0 at
the high-frequency end, is consistent with the already measured near-cubic
Taylor-2 remainder. It does not justify extending the online input contract:
10/15/20% remain offline stress extrapolations, while `predict` and
`predict_batch` remain capped at 8%.

The extended report SHA-256 is
`bc404d2c172a801c4e27db1d662612c68ba44cf088d411d7c62dc9859fd77f8d`.
It records `sealed_holdout_used=false`; sealed holdout seed `20270509` remains
unopened. No full development run, gate freeze, artifact promotion, or Plan S
work followed from this characterization.
