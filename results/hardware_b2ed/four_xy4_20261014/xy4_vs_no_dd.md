# XY4 versus no DD

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
