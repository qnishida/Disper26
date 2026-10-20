# Plan L v2 mixed-response result: Go (2026-09-03)

## Decision

Plan L v2 is **Go** under its frozen ten-frequency, 8% primary contract.
Rayleigh root orders 0 and 1 expand phase velocity, while Love root orders 0
and 1 expand normalized squared slowness
\(y=(c_0/c)^2\) and map the result back to phase velocity. All four curves
passed independent development, the gate was frozen, and the one-time sealed
holdout passed without changing a threshold, response coordinate, frequency,
or sample.

The promoted claim is deliberately local. It applies only to the fixed
ocean-free, sediment-free, solid-VTI reference; the 18-parameter Vsv/Vsh
basis; the listed ten frequencies; root orders 0/1; and a maximum forward-node
shear anomaly of 8%. It is finite-sample evidence, not a mathematical
pointwise error bound. The exact solver remains the authoritative route
outside this contract. Plan S was not started.

The plan file remains byte-for-byte frozen at the hash below. Its planning
status and unchecked execution boxes are therefore a historical snapshot;
this result document records the post-freeze outcome without invalidating the
gate lock.

## Frozen scope and provenance

The evidence was produced from clean implementation commit
`c72cf1d961f2bc50118fd69869b88dbc5c015629`. The fixed frequency vector was
0.0075--0.0300 Hz in 0.0025 Hz increments. Each split contained 32 whitened
prior directions, both signs, and seven amplitudes from 0.5% through 10%: 384
primary models at or below 8% and 64 separate 10% stress models.

| evidence | SHA-256 |
|---|---|
| frozen plan | `13c4ed729479b8cfed7ebe733a3d6bb640d296758e512175c660ee8d77ce45b4` |
| promoted artifact | `4e92574a29a6a1398052b4a7744fc3c4131de4ae779906e348c05719fe555bb5` |
| depth-convergence report | `a400ac6f94a0db6634b45ede3880874f9d5e8ce0bf5b214a7d6e79580fd38452` |
| opened calibration report | `1f5a800d1c20db88dcb53f023cf8a83178d035a7c4f1c657fe35eb0c2accf2d3` |
| independent development report | `622f79541505cb67ac32fea4a81b7cd15c718586c8841d154b97bddb2d36b04d` |
| benchmark report | `1f4a3fc2603bfe6ba61c440a8a00ce3a2fbeef33d1e994412c3f3fbc5e5e7393` |
| frozen gate lock | `560b037394f206ac036c1ee5e8ad20a8d57a8cf57e75a0985eff1ca142388650` |
| holdout-open marker | `644fed068b046ee346fecd0d41b36c1f9bde5f6568f5c77865db2e0b304cd06c` |
| sealed holdout manifest | `227694d211580822575da4bb079ba485b8ab63ad9f0dd83e97deebbd95519907` |
| sealed holdout report | `2a5b2270f753e0a710002e402cab4805ec3a25e78b2f182935c0849a13c4e5ab` |
| promotion receipt | `33b7f185a3123aaadc6fd26f4e7a2797af75fe6c145e41983001ccaf840fa7bb` |

The calibration, development, and sealed-holdout seeds were 20270507,
20270510, and 20270511. Calibration was marked `opened_selection_only`;
development was `independent_development`; and the holdout was
`sealed_holdout`. The earlier v1 sealed seed 20270509 remains unopened and was
not repurposed.

## Numerical and implementation checks

The 2200/2400/2600 km reference-depth gate passed. Across both waves, root
orders 0--2, and all ten frequencies, exact-root failure, branch-identity
disagreement, nonfinite derivative, and reference trace mismatch counts were
zero. The 2400--2600 km mapped Taylor-2 difference had
`p99(|delta c|/sigma)=2.486e-08`, versus the fixed `1/30` limit. The largest
phase-velocity and adjacent-gap differences were `5.580e-07` relative and
`5.498e-07` absolute.

An independent Love Hessian check compared Vsv/Vsh columns with central finite
differences of the separate full-AD gradient path at 0.0075, 0.0200, and
0.0300 Hz for roots 0/1. The largest column-relative difference was
`1.577e-08`; the largest scaled Vsv--Vsh cross-term difference was
`7.251e-10`.

The v2 schema stores an explicit response kind and cannot silently load or
reinterpret a v1 artifact. At the reference, the Love inverse-square mapping
recovers the original phase velocity, gradient, and Hessian. The query path
contains no exact-solver call. A nonpositive Love response, nonfinite model or
output, frequency mismatch, fingerprint mismatch, unsupported root, or
amplitude above 8% fails closed.

## Sealed-holdout primary result

All values in the next table are absolute phase-velocity relative errors in
percent, pooled over the six primary amplitude levels (0.5--8%) and all ten
frequencies. `mean` and `maximum` are descriptive finite-sample statistics;
the frozen hard gates were applied separately at every frequency using p90
and p99, not the pooled maximum.

| curve | response | mean | p50 | p90 | p99 | maximum observed |
|---|---|---:|---:|---:|---:|---:|
| Rayleigh 0 | \(c\) | 0.000580% | 0.0000392% | 0.001448% | 0.008285% | 0.01551% |
| Rayleigh 1 | \(c\) | 0.000486% | 0.0000128% | 0.001112% | 0.008406% | 0.02210% |
| Love 0 | \((c_0/c)^2\) | 0.002386% | 0.000130% | 0.006028% | 0.03535% | 0.1153% |
| Love 1 | \((c_0/c)^2\) | 0.002126% | 0.0000992% | 0.004853% | 0.03483% | 0.1096% |

At the 8% contract boundary alone, the errors were:

| curve | mean | p50 | p90 | p99 | maximum observed |
|---|---:|---:|---:|---:|---:|
| Rayleigh 0 | 0.002232% | 0.001032% | 0.006421% | 0.01309% | 0.01551% |
| Rayleigh 1 | 0.001880% | 0.000428% | 0.005786% | 0.01642% | 0.02210% |
| Love 0 | 0.009153% | 0.004103% | 0.02430% | 0.06693% | 0.1153% |
| Love 1 | 0.008143% | 0.002549% | 0.02216% | 0.06920% | 0.1096% |

Thus, a Go result does not mean every observed point was below 0.1%. Two Love
maximum observations were slightly above 0.1%; the preregistered criterion was
per-frequency p99 no greater than 0.1%, and both curves passed it. The worst
per-frequency p90/p99 values in units of the fixed 0.1% observation sigma were
0.0348/0.1189 (Rayleigh 0), 0.0518/0.1691 (Rayleigh 1), 0.1604/0.6912
(Love 0), and 0.1510/0.7011 (Love 1).

Curve-RMS p90 errors in sigma units were 0.0236, 0.0234, 0.1078, and 0.0893,
respectively, all below `1/3`. The minimum Taylor-2 residual orders across
frequency were 2.964, 2.950, 2.956, and 2.959, all above 2.7.

Response-order 2 improved substantially over response-order 1 after both were
mapped to phase velocity:

| curve | order-1 p90 | order-2 p90 | order-2/order-1 p90 | pointwise improvement median |
|---|---:|---:|---:|---:|
| Rayleigh 0 | 0.02604% | 0.001448% | 0.0556 | 42.6x |
| Rayleigh 1 | 0.01224% | 0.001112% | 0.0909 | 32.7x |
| Love 0 | 0.1079% | 0.006028% | 0.0559 | 52.8x |
| Love 1 | 0.03859% | 0.004853% | 0.1258 | 26.8x |

For Love 0, minimum \(\widehat y_1/\widehat y_2\) was 0.9007/0.9111; for
Love 1 it was 0.9830/0.9799. Nonpositive response and inverse-square-root
mapping failure counts were zero.

For every curve, exact branch/trace failure, accepted mode-identity failure,
prediction nonfinite, nonpositive response, and unsupported primary-point
counts were zero. There were 3,840 assessed primary points per curve. Given
zero observed mode failures, the one-sided 95% binomial upper probability was
`7.798e-04` per curve.

The signed-error medians were effectively zero. Signed p90/p99/maximum values
in percent were 0.000649/0.006713/0.01551 (Rayleigh 0),
0.000314/0.007144/0.02210 (Rayleigh 1),
0.002460/0.02665/0.1153 (Love 0), and
0.001694/0.02210/0.07150 (Love 1). Signed-error cancellation was not used for
any gate.

## Same-holdout v1 versus v2 response-coordinate ablation

The v1 all-\(c\) and v2 mixed-response jets were evaluated on exactly the same
sealed-holdout models and frequencies. Rayleigh is identical by construction.
The table uses only the 8% boundary cohort and reports phase-velocity error.

| curve | response | p90 | p99 | maximum observed | gate result on full primary cohort |
|---|---|---:|---:|---:|:---:|
| Rayleigh 0 v1/v2 | \(c\) | 0.006421% | 0.01309% | 0.01551% | Pass / Pass |
| Rayleigh 1 v1/v2 | \(c\) | 0.005786% | 0.01642% | 0.02210% | Pass / Pass |
| Love 0 v1 | \(c\) | 0.04315% | 0.1144% | 0.1630% | **Fail** |
| Love 0 v2 | \((c_0/c)^2\) | 0.02430% | 0.06693% | 0.1153% | **Pass** |
| Love 1 v1 | \(c\) | 0.02458% | 0.07216% | 0.1126% | Pass |
| Love 1 v2 | \((c_0/c)^2\) | 0.02216% | 0.06920% | 0.1096% | Pass |

The decisive change is Love fundamental: squared slowness reduced its 8% p90
by about 44% and p99 by about 41%, moving the curve from v1 failure to v2
pass. Love first overtone improved more modestly. This supports the selected
response coordinate for this fixed reference and domain, not a universal
claim that \(1/c^2\) is optimal.

## Stress and extension results

The 10% cohort was evaluated offline and did not affect promotion. Errors are
again percent of phase velocity.

| curve | p50 | p90 | p99 | maximum observed |
|---|---:|---:|---:|---:|
| Rayleigh 0 | 0.002005% | 0.01275% | 0.02727% | 0.03269% |
| Rayleigh 1 | 0.000833% | 0.01182% | 0.03493% | 0.04713% |
| Love 0 | 0.008039% | 0.04748% | 0.1353% | 0.2404% |
| Love 1 | 0.005005% | 0.04327% | 0.1336% | 0.2251% |

As expected for a predominantly cubic remainder, the Love tails grow beyond
the 8% contract. The online evaluator therefore continues to reject amplitudes
above 8%.

The 20--30 s extension was evaluated only as a post-promotion, outside-primary
diagnostic on the pre-existing broad-band grid 0.035, 0.040, 0.045, and
0.050 Hz. It used the same sealed-holdout model cohort, a fresh 2400 km
reference jet, and no frequency selection. At the 8% boundary:

| curve | p50 | p90 | p99 | maximum observed | minimum residual order |
|---|---:|---:|---:|---:|---:|
| Rayleigh 0 | 0.006455% | 0.02278% | 0.04081% | 0.05158% | 3.003 |
| Rayleigh 1 | 0.007096% | 0.02920% | 0.05469% | 0.06748% | 2.983 |
| Love 0 | 0.007348% | 0.02887% | 0.05236% | 0.09063% | 2.982 |
| Love 1 | 0.003150% | 0.02139% | 0.05367% | 0.07978% | 2.891 |

Exact branch/trace, accepted mode identity, nonfinite prediction, and Love
nonpositive-response counts were all zero in this extension. These favorable
diagnostics do not extend the artifact: 0.035--0.050 Hz remains
`EXACT_REQUIRED` and would need a separately frozen version and unused
holdout for promotion.

## Runtime, storage, and break-even

The fixed benchmark used the same branch-confirmed exact path, a representative
batch of 8, and the minimum of five repetitions. Times are per curve and
include all ten primary frequencies.

| curve | exact scalar | surrogate scalar | scalar speedup | batch-8 speedup |
|---|---:|---:|---:|---:|
| Rayleigh 0 | 199.5 ms | 31.45 us | 6,344x | 9,363x |
| Rayleigh 1 | 215.4 ms | 31.59 us | 6,817x | 9,986x |
| Love 0 | 44.70 ms | 38.81 us | 1,152x | 2,037x |
| Love 1 | 56.68 ms | 38.24 us | 1,482x | 2,399x |

All four scalar and batch measurements passed the fixed 1,000x gate, and the
surrogate exact-solver call count was zero. The Love inverse square root does
not change the dense quadratic asymptotic cost.

A cold build of the single 2400 km, four-curve, ten-frequency reference jet
took 10.40 s with 8 processes on the evaluation machine. A complete
four-curve scalar query took 516.24 ms exact versus 140.09 us surrogate, so
the measured cold-build break-even is about 20.2 model queries (round up to 21).
This excludes the much larger one-time certification cost of collecting the
calibration, development, and holdout cohorts. The deterministic artifact is
79,298 bytes on disk and contains 166,560 bytes of numerical arrays.

## Reproduction and limitations

The generated arrays and reports remain under ignored
`build/plan_l_v2_c72cf1d/`. Starting from evaluation commit `c72cf1d`, the
workflow is driven by:

```bash
python tests/python/run_plan_l_long_period_taylor.py \
  --profile mixed-response-v2 <reference|split|build|validate|benchmark|freeze|promote> ...
```

This command describes the historical driver, which is not bundled with this
distribution. Run it only in a dedicated environment with its dependencies.
For the public replay workflow, see [paper/README.md](../../README.md).
The original command arguments and sequencing were described in
`long_period_taylor_surrogate.md`; the gate lock refuses a dirty or different
code revision, a changed plan hash, mismatched reports, or holdout access
before freeze. The v1 sealed holdout is not needed for this reproduction and
must remain unopened.

The evidence establishes forward approximation accuracy, branch safety, and
query economics only. It does not establish deeper Earth resolution,
posterior invariance, performance for other references or parameterizations,
or validity outside the 8%/ten-frequency contract. Plan S and all crustal-
thickness anchor work remain unimplemented.
