## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-14 (America/Los_Angeles)
- Verification Status: ANALYZED
- Version Label: ibm_b2ed_hardware_analysis_v1

# IBM Aachen: four-qubit B2_ED pilot (XY4)

Job `db84p30331ic73fu5tm0`, instance `qzh-eu`, 24 circuits × 2,048 shots = 49,152 shots.

**Raw accepted-state fidelity estimate: 0.742192**, with approximate 95% sampling interval **[0.730631, 0.753754]**. The dedicated acceptance circuit retained **1,044/2,048 = 50.977%** (Wilson 95% interval [48.812%, 53.138%]).

## Accepted, all, and rejected shots

All three rows use the same B2_ED circuit data. 'All' does not represent a separate B1/no-check hardware control.

| Shots used | Raw target-projector estimate | Approx. 95% sampling interval | Raw codespace-projector score |
|---|---:|---:|---:|
| accepted | 0.742192 | [0.730631, 0.753754] | 0.760347 |
| all | 0.555631 | [0.546305, 0.564956] | 0.565308 |
| rejected | 0.368697 | [0.354723, 0.382670] | 0.371845 |

Selecting recorded `000` is associated with a 18.66 percentage-point increase over retaining all shots (joint-bootstrap 95% interval [17.72, 19.59] points). This is a conditional-selection comparison, not evidence that adding syndrome gates improves preparation versus a no-check circuit.

The codespace score comes from the seven nonidentity stabilizer-group terms already measured. It includes measurement bias. It is not a direct measurement of an exact density matrix, and it cannot separate physical codespace loss from readout/basis errors. No extra codespace filtering is applied, and no within-codespace physical fidelity is asserted.

## Syndrome diagnostics

Across all circuits, 24,596/49,152 shots were accepted (50.041%). Per-circuit acceptance ranged from 48.877% to 51.123%. An exploratory acceptance-homogeneity test gives chi-square(23) = 9.983, p = 0.9914. This does not establish absence of drift or gate-dependent bias.

Bit order is `s2 s1 s0`, corresponding to checks `ZZII`, `IIZZ`, `XXXX`.

| Recorded syndrome | Shots | Fraction |
|---|---:|---:|
| `000` | 24,596 | 50.041% |
| `001` | 1,093 | 2.224% |
| `010` | 6,104 | 12.419% |
| `011` | 719 | 1.463% |
| `100` | 12,024 | 24.463% |
| `101` | 932 | 1.896% |
| `110` | 3,141 | 6.390% |
| `111` | 543 | 1.105% |

| Check | Syndrome bit | Physical ancilla at readout | Recorded 1 fraction | Saved assignment-error average |
|---|---:|---:|---:|---:|
| `XXXX` | 0 | 1 | 6.687% | 0.439% |
| `IIZZ` | 1 | 4 | 21.377% | 0.977% |
| `ZZII` | 2 | 23 | 33.854% | 0.684% |

A syndrome does not uniquely locate a hardware fault. The recorded-1 fractions are much larger than the saved nominal ancilla assignment-error averages, but this alone does not identify the cause.

## Comparison with simulations

Compare hardware raw scores with model raw measurement scores, not with the model's exact delivered-state fidelity.

| Source | Acceptance | Raw measurement score | Exact model state fidelity (simulation only) |
|---|---:|---:|---:|
| Hardware, this job | 50.977% | 0.742192 | unavailable |
| Earlier April 17 snapshot, no DD | 90.505% | 0.810400 | 0.943088 |
| Saved live calibration, frozen circuits | 91.165% | 0.839805 | 0.942723 |

Replay assumptions: Saved gate errors, T1/T2, asymmetric independent assignment, zero temperature. Frozen circuit delays retained; ancilla readout deferred only in this independent-channel model. No added initialization error, coherent/correlated noise, crosstalk, or unreported runtime latency. Model predictions, not hardware-corrected estimates.

## XY4 versus the previous no-DD hardware job

No-DD job: `db84dj8331ic73fu5jfg`. XY4 job: `db84p30331ic73fu5tm0`. Submission separation: 24.50 minutes. Identical archived calibration: True. Circuit mapping, original gate times, total duration, shots and Pauli settings match.

| Metric | No DD | XY4 | Change (percentage points) | Approx. 95% interval for change (points) |
|---|---:|---:|---:|---:|
| Raw accepted fidelity | 69.709% | 74.219% | +4.511 | [+2.834, +6.187] |
| Dedicated-circuit acceptance | 48.438% | 50.977% | +2.539 | [-0.522, +5.600] |
| Pooled acceptance (supplementary) | 48.873% | 50.041% | +1.168 | [+0.543, +1.793] |
| Raw fidelity, all shots | 52.520% | 55.563% | +3.043 | [+1.716, +4.370] |
| Raw fidelity, rejected shots | 36.207% | 36.870% | +0.663 | [-1.306, +2.631] |
| Raw accepted codespace score | 71.058% | 76.035% | +4.976 | [+2.936, +7.017] |

The independent-job bootstrap interval for the raw accepted-fidelity change is [+2.863, +6.194] points. The raw-fidelity increase exceeds the estimated finite-shot uncertainty. The dedicated acceptance-only circuits do not establish an acceptance increase at the approximate 95% level; the supplementary pooled counts suggest a smaller increase with more precision.

Observed difference between two independent sequential jobs. Sampling intervals are pointwise; they omit drift and systematic readout/basis error. This was not an interleaved randomized DD/no-DD comparison. Identical saved calibration does not imply identical physical noise during execution.

The higher score is consistent with a benefit from XY4, but a causal DD effect is not isolated from drift. A future interleaved no-DD/XY4 comparison would address that limitation; no further jobs were submitted.

### Simulator comparison for the two frozen circuits

The independent Markovian model does not include slow correlated/coherent noise that DD can refocus. Its difference should not be used as a prediction of DD's full hardware benefit.

| Replay | Acceptance | Raw fidelity score | Exact model fidelity |
|---|---:|---:|---:|
| No DD | 91.441% | 0.842766 | 0.946029 |
| XY4 | 91.165% | 0.839805 | 0.942723 |

The hardware/model gap is a model-validation finding. This single job does not distinguish unmodeled coherent/correlated noise, readout crosstalk, runtime timing, calibration drift, or other causes. A no-check control and dedicated measurement calibrations would be needed for a stronger diagnosis; neither was run in this pilot.

## Uncertainty and validation

Every PUB contains the requested 2,048 shots. Register widths, Pauli metadata, circuit-file hashes, and archived joint counts were checked. A separate parity calculation reproduces the retrieval script's fidelity and standard error within 1e-12. All 23 preselected Pauli settings are included.

The 10,000-replicate multinomial bootstrap gives accepted-state interval [0.730506, 0.753687]. The original conservative Hoeffding/union-bound interval is [0.610031, 0.874354]. All intervals concern sampling uncertainty, assuming independent shots and a stable preparation/measurement process. They exclude systematic measurement bias, calibration uncertainty, and drift; more shots alone do not remove those errors.

Hardware readout mitigation has not been applied. Saved calibration values are contextual diagnostics, not dedicated in-circuit measurement calibrations. Estimates are not clipped to [0,1].

### Statistical interpretation audit

Coverage: 11/11 fallacy types considered.

| Check | Assessment |
|---|---|
| Simpson's paradox | Per-setting yields retained; pooled yield is supplementary. |
| Ecological fallacy | Inferences restricted to this circuit/accepted ensemble, not individual fault locations. |
| Berkson selection | Acceptance is intentional conditioning; it does not represent all output states. |
| Collider bias | No causal attribution is made from syndrome-conditioned correlations. |
| Base-rate neglect | Attempted, accepted and rejected counts are all reported. |
| Regression to the mean | No before/after or selected-best-run claim; one pilot only. |
| Survivorship bias | Rejected shots archived and analyzed; postselection yield reported. |
| Look-elsewhere effect | All preselected terms retained; one exploratory homogeneity test, no search for significant subsets. |
| Forking paths | Primary raw estimator fixed before submission; subgroup and model comparisons labeled diagnostics. |
| Correlation/causation | Accepted/all comparison does not establish benefit over a separate B1 control. |
| Reverse causality | No directional fault-mechanism claim from these measurements. |

## Reproduction

Hardware was not rerun. Analysis is deterministic from the saved inputs; bootstrap uses seed 20261014. No additional IBM jobs or package changes were made.

```bash
conda run -n quantum-jax python scripts/analyze_ibm_b2ed.py \
  --bundle results/hardware_b2ed/four_xy4_20261014 --replay-calibration --dt 4e-9
```

Outputs: `analysis_extended.json`, `pauli_estimates.csv`, `hardware_analysis.png`, `hardware_analysis.pdf`, and this report. Raw measurements remain in `runtime_result.json` and `joint_counts.json`.
