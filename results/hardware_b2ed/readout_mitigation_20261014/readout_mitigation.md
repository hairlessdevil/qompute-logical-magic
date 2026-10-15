## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-15T03:58:55.443070+00:00
- Verification Status: ANALYZED (archived hardware; postprocessing independently checked)
- Version Label: ibm_b2ed_readout_mitigation_v1

# Readout mitigation: no DD and XY4

The primary correction estimates the fidelity of the actual recorded-000 accepted ensemble. It inverts only final data-bit assignment errors. It does not alter acceptance or repair the prepared quantum state.

| Fidelity estimate | No DD | XY4 | XY4 minus no DD (percentage points; 95% sampling interval) |
|---|---:|---:|---:|
| Raw; recorded 000 | 69.709% | 74.219% | +4.511 [+2.834, +6.187] |
| Data corrected; recorded 000 (primary) | 77.659% | 83.011% | +5.352 [+3.427, +7.278] |
| Data + syndrome corrected; true 000 (diagnostic) | 77.944% | 83.315% | +5.371 [+3.429, +7.313] |

## Primary estimates and uncertainty

| Job | Data-corrected fidelity | 95% sampling interval | Paired change from raw (points) |
|---|---:|---:|---:|
| no DD (`db84dj8331ic73fu5jfg`) | 77.659% | [76.264%, 79.054%] | +7.950 (paired bootstrap [+7.759, +8.139]) |
| XY4 (`db84p30331ic73fu5tm0`) | 83.011% | [81.684%, 84.338%] | +8.792 (paired bootstrap [+8.612, +8.973]) |

All intervals condition on the saved assignment matrices being exact and applicable. They include finite experimental shot uncertainty only, not calibration uncertainty, drift, basis-rotation errors, leakage or correlated readout. Each job has 23 Pauli settings plus one dedicated acceptance circuit, each with 2,048 shots. The 10,000-replicate bootstrap resamples each full joint histogram, preserving syndrome/data correlations and pairing raw/corrected estimates. Jobs use distinct random streams. Intervals are pointwise and exploratory.

The XY4 and no-DD jobs ran about 24.5 minutes apart. Both use the same saved calibration, mapping and circuit durations. A between-job difference does not isolate a causal DD effect from drift.

## What correcting the syndrome changes

Full joint inversion uses every recorded syndrome to estimate the distribution conditioned on latent (pre-assignment) syndrome 000. This is an additional diagnostic, not the fidelity of the actually retained ensemble and not a way to identify which individual shots were misread. It does not undo syndrome-circuit gate faults or guarantee codespace membership.

| Job | Actual recorded-000 acceptance | Estimated true-000 probability (diagnostic) |
|---|---:|---:|
| no DD | 48.438% | 48.545% |
| XY4 | 50.977% | 51.146% |

## Method and calibration

For qubit q the row-stochastic matrix is `A_q[true, recorded] = [[1-e01,e01],[e10,1-e10]]`. The observable weights solve `A_q w = [1,-1]`. Tensor-product weights are applied to the four measured data bits using their final physical mapping. For syndrome correction, weights solve `A_s w = indicator(000)`; a ratio of weighted joint moments gives each conditional Pauli expectation. The target-projector expansion is unchanged. The ratio variance retains numerator/denominator covariance. No negative quasiprobabilities or estimates are clipped. A full probability-distribution solve independently checks each setting.

This is the assignment-matrix model described in [IBM's readout-mitigation tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/readout-error-mitigation-sampler). We use a small tensor-product solve; no M3 package or new dependencies were needed.

The backend calibration fields are `prob_meas1_prep0` and `prob_meas0_prep1`, measured around 2026-10-14 19:34 Pacific (about 48 and 72 minutes before the two runs). These were not dedicated calibration circuits embedded in these jobs. Their sampling counts and covariance are unavailable, so calibration uncertainty cannot be included honestly. The independent classical channel model also assumes syndrome misassignment is uncorrelated with the subsequent data dynamics.

| Role / bit | Physical qubit | P(record 1 given 0) | P(record 0 given 1) |
|---|---:|---:|---:|
| data 0 | 5 | 0.244% | 0.439% |
| data 1 | 3 | 0.732% | 1.416% |
| data 2 | 2 | 2.344% | 4.443% |
| data 3 | 16 | 3.125% | 2.490% |
| syndrome 0 | 1 | 0.098% | 0.781% |
| syndrome 1 | 4 | 0.684% | 1.270% |
| syndrome 2 | 23 | 0.342% | 1.025% |

## Sensitivity to calibration strength

Multiply all directional assignment-error rates by the factor below. This is a model sensitivity check, not a calibration confidence interval. The same factor is used for both jobs.

| Rate multiplier | Data-corrected no DD | Data-corrected XY4 | Difference (points) |
|---|---:|---:|---:|
| 0.0 | 69.709% | 74.219% | +4.511 |
| 0.5 | 73.491% | 78.400% | +4.909 |
| 1.0 | 77.659% | 83.011% | +5.352 |
| 1.5 | 82.267% | 88.115% | +5.849 |

This processing can reduce measurement bias if the model holds; a higher number alone does not validate the correction or demonstrate improved state preparation. Final data readout is corrected statistically, while basis-gate and preparation errors remain.

## Validation and statistical interpretation

Raw data hashes were checked against the prior validated archive. The original fidelity and SE are reproduced. Independent full-matrix and weighted-observable solves agree to numerical precision. Synthetic asymmetric-channel tests verify recovery and the distinction between recorded and true syndrome conditioning. Identity-channel controls reproduce the raw estimator. All bootstrap denominators remain positive; diagnostics are saved in JSON.

Statistical fallacy scan: 11/11 considered. Simpson: per-setting terms retained; ecological: no individual-fault inference; Berkson/collider: recorded and latent selected ensembles separated; base rates/survivorship: actual acceptance reported and all syndrome outcomes retained; regression to mean: no causal improvement claim from sequential runs; look-elsewhere: all 23 preselected terms included; forking paths: correction and sensitivity analysis labeled exploratory; causation/reverse causality: postprocessing changes estimates, not quantum states. Overall confidence: CAUTION because calibration applicability and correlated errors are unvalidated.

## Reproduce

```bash
conda run -n quantum-jax python -B scripts/mitigate_ibm_b2ed_readout.py
```

Outputs are separate from the original raw reports. No hardware jobs were submitted or packages modified. JSON includes input hashes, per-Pauli estimates, codespace scores, matrix condition numbers, negative-mass diagnostics and bootstrap results.
