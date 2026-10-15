# Five-qubit B2_ED: no DD versus XY4

Target: [[5,1,3]] encoded positive tetrahedral state. Both jobs used ibm_aachen, 64 circuits and 2,048 shots/circuit (131,072 shots/job). Accept recorded syndrome 0000. No active recovery.

| Run | Raw fidelity (95% sampling CI) | Data-corrected fidelity (95% sampling CI) | Dedicated acceptance (Wilson 95% CI) |
|---|---:|---:|---:|
| no DD | 60.576% [59.495%, 61.657%] | 68.378% [67.138%, 69.619%] | 37.744% [35.670%, 39.865%] |
| XY4 | 69.938% [68.934%, 70.941%] | 79.119% [77.966%, 80.271%] | 38.574% [36.489%, 40.702%] |

Data-corrected XY4 minus no DD: +10.741 percentage points; normal 95% interval [9.047, 12.434], independent-bootstrap interval [9.022, 12.443].

Readout correction uses the archived per-qubit directional assignment probabilities. Data-only correction preserves the recorded-0000 accepted ensemble; it changes the estimator, not the prepared state. No new readout-calibration job was run.

Submission order: XY4 then no DD; separation 6.95 minutes. Both share the same saved calibration, mapping, schedule duration, and Pauli settings. Separate sequential jobs do not isolate DD from time-dependent hardware variation.

Uncertainty includes finite-shot sampling only. Calibration uncertainty, correlated assignment errors, basis-rotation errors and drift are not included. Negative quasiprobabilities are retained; no clipping or extra codespace filtering is applied.

## Detailed diagnostics

| Quantity | No DD | XY4 |
|---|---:|---:|
| All-outcome raw target score | 35.582% | 42.455% |
| Rejected raw target score | 20.704% | 25.023% |
| Raw codespace score | 60.785% [59.393%, 62.178%] | 70.205% [68.977%, 71.433%] |
| Data-corrected codespace score | 68.610% [67.002%, 70.217%] | 79.366% [77.944%, 80.788%] |
| Joint-corrected fidelity (different ensemble) | 68.882% [67.626%, 70.139%] | 79.682% [78.515%, 80.850%] |

## Provenance and checks

All 128 Runtime PUBs reproduce the archived shot histograms. Raw fidelity and standard error match the retrieval estimator and an independent bit-count implementation. Assignment-corrected observable scores agree with independent full-distribution inversion. Bootstrap: 10,000 multinomial replicates per job; independent seeds between jobs, paired estimators within each job.

- no DD: `db865b5bgg3c73emfbkg`; [extended analysis](../five_pilot_20261014/analysis_extended.json).
- XY4: `db8622rkbioc73chut2g`; [extended analysis](../five_xy4_20261014/analysis_extended.json).

Reproduce offline: `python scripts/analyze_ibm_five_b2ed.py`. Material status: ANALYZED (archived hardware; independently checked postprocessing).
