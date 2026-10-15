## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-14 (America/Los_Angeles)
- Verification Status: ANALYZED
- Version Label: ibm_b2ed_hardware_analysis_v1

# IBM Aachen: four-qubit B2_ED pilot

Job `db84dj8331ic73fu5jfg`, instance `qzh-eu`, 24 circuits × 2,048 shots = 49,152 shots.

**Raw accepted-state fidelity estimate: 0.697085**, with approximate 95% sampling interval **[0.684944, 0.709227]**. The dedicated acceptance circuit retained **992/2,048 = 48.438%** (Wilson 95% interval [46.278%, 50.603%]).

## Accepted, all, and rejected shots

All three rows use the same B2_ED circuit data. 'All' does not represent a separate B1/no-check hardware control.

| Shots used | Raw target-projector estimate | Approx. 95% sampling interval | Raw codespace-projector score |
|---|---:|---:|---:|
| accepted | 0.697085 | [0.684944, 0.709227] | 0.710584 |
| all | 0.525200 | [0.515761, 0.534639] | 0.540527 |
| rejected | 0.362070 | [0.348204, 0.375937] | 0.379498 |

Selecting recorded `000` is associated with a 17.19 percentage-point increase over retaining all shots (joint-bootstrap 95% interval [16.24, 18.16] points). This is a conditional-selection comparison, not evidence that adding syndrome gates improves preparation versus a no-check circuit.

The codespace score comes from the seven nonidentity stabilizer-group terms already measured. It includes measurement bias. It is not a direct measurement of an exact density matrix, and it cannot separate physical codespace loss from readout/basis errors. No extra codespace filtering is applied, and no within-codespace physical fidelity is asserted.

## Syndrome diagnostics

Across all circuits, 24,022/49,152 shots were accepted (48.873%). Per-circuit acceptance ranged from 46.924% to 51.270%. An exploratory acceptance-homogeneity test gives chi-square(23) = 22.949, p = 0.4638. This does not establish absence of drift or gate-dependent bias.

Bit order is `s2 s1 s0`, corresponding to checks `ZZII`, `IIZZ`, `XXXX`.

| Recorded syndrome | Shots | Fraction |
|---|---:|---:|
| `000` | 24,022 | 48.873% |
| `001` | 1,258 | 2.559% |
| `010` | 5,491 | 11.171% |
| `011` | 610 | 1.241% |
| `100` | 13,038 | 26.526% |
| `101` | 1,028 | 2.091% |
| `110` | 3,185 | 6.480% |
| `111` | 520 | 1.058% |

| Check | Syndrome bit | Physical ancilla at readout | Recorded 1 fraction | Saved assignment-error average |
|---|---:|---:|---:|---:|
| `XXXX` | 0 | 1 | 6.950% | 0.439% |
| `IIZZ` | 1 | 4 | 19.950% | 0.977% |
| `ZZII` | 2 | 23 | 36.155% | 0.684% |

A syndrome does not uniquely locate a hardware fault. The recorded-1 fractions are much larger than the saved nominal ancilla assignment-error averages, but this alone does not identify the cause.

## Comparison with simulations

Compare hardware raw scores with model raw measurement scores, not with the model's exact delivered-state fidelity.

| Source | Acceptance | Raw measurement score | Exact model state fidelity (simulation only) |
|---|---:|---:|---:|
| Hardware, this job | 48.438% | 0.697085 | unavailable |
| Earlier April 17 snapshot | 90.505% | 0.810400 | 0.943088 |
| Saved live calibration, frozen circuits | 91.441% | 0.842766 | 0.946029 |

Replay assumptions: Saved gate errors, T1/T2, asymmetric independent assignment, zero temperature. Frozen circuit delays retained; ancilla readout deferred only in this independent-channel model. No added initialization error, coherent/correlated noise, crosstalk, or unreported runtime latency. Model predictions, not hardware-corrected estimates.

The hardware/model gap is a model-validation finding. This single job does not distinguish unmodeled coherent/correlated noise, readout crosstalk, runtime timing, calibration drift, or other causes. A no-check control and dedicated measurement calibrations would be needed for a stronger diagnosis; neither was run in this pilot.

## Uncertainty and validation

Every PUB contains the requested 2,048 shots. Register widths, Pauli metadata, circuit-file hashes, and archived joint counts were checked. A separate parity calculation reproduces the retrieval script's fidelity and standard error within 1e-12. All 23 preselected Pauli settings are included.

The 10,000-replicate multinomial bootstrap gives accepted-state interval [0.685064, 0.709262]. The original conservative Hoeffding/union-bound interval is [0.563439, 0.830731]. All intervals concern sampling uncertainty, assuming independent shots and a stable preparation/measurement process. They exclude systematic measurement bias, calibration uncertainty, and drift; more shots alone do not remove those errors.

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
  --bundle results/hardware_b2ed/four_pilot_20261014 --replay-calibration --dt 4e-9
```

Outputs: `analysis_extended.json`, `pauli_estimates.csv`, `hardware_analysis.png`, `hardware_analysis.pdf`, and this report. Raw measurements remain in `runtime_result.json` and `joint_counts.json`.
