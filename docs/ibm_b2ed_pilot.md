# Four-qubit B2_ED pilot on IBM Aachen

This pilot uses `QiskitRuntimeService(instance="qzh-eu")` and `ibm_aachen`.
Run every command in the `quantum-jax` environment from the project root.
The scripts do not install packages or save/change IBM account credentials.

## Protocol and budget

- Encode one H-state in the [[4,1,2]] code and measure `XXXX`, `IIZZ`, `ZZII`
  with three fresh ancillas. Keep only the recorded `000` syndrome.
- Use the existing fixed nine-qubit routing patch `[3,2,4,16,1,5,23,0,6]`,
  populated from the **live** target. Seven qubits carry data/ancillas; the
  nine-wire patch provides routing space. Physical output locations are saved.
- Freeze one ALAP preparation prefix for all settings. Data wait through the
  ordinary syndrome measurement, then undergo basis rotations and final readout.
  This preserves the existing simulation's baseline protocol. It deliberately
  does not substitute `measure_2`; that would be a different timing experiment.
- Run one acceptance-only circuit and all 23 nonidentity terms in the full
  physical target projector. At 2,048 shots each, this is **49,152 total shots**
  in one SamplerV2 job. No Session or Batch is opened.
- The baseline job uses no active correction, online readout mitigation,
  twirling, or dynamical decoupling. The second job adds explicit native XY4;
  offline readout mitigation is described below.
  The runtime execution-time cap is 300 seconds; queue waiting is separate.

The estimate includes population outside the code subspace. It is a **raw
measurement estimate**, affected by analysis-gate and readout errors, rather than
an exact delivered-state fidelity. Statistical intervals omit systematic bias
and device drift. Negative or greater-than-one finite-shot estimates are not
clipped. Each Pauli setting is conditioned on its own recorded-zero shots.

## 1. Prepare only

```bash
python scripts/prepare_ibm_b2ed.py --output results/hardware_b2ed/four_pilot --shots 2048
```

This reads backend status/calibration, compiles, checks the device instruction
set and faulty qubits/edges, verifies all settings with an exact ideal
statevector, and saves a review bundle. It does not instantiate a Sampler or
submit a job. Use a new directory for each bundle; existing bundles are not
overwritten.

Review `manifest.json` for the physical mapping, gate counts, shot budget,
options, approximate QPU usage, and ideal-validation results. `preparation.txt`
shows the local routed circuit. `circuits.qpy` contains actual device-indexed
circuits; `local_circuits.qpy` is for compact offline inspection only.

## 2. Submit only after explicit human approval

```bash
python scripts/submit_ibm_b2ed.py --bundle results/hardware_b2ed/four_pilot \
  --approve-manifest-sha256 APPROVED_MANIFEST_SHA256
```

The digest is printed by the preparation script. It ties approval to the exact
manifest; referenced circuit/calibration files are also checked by SHA256.
Submission rechecks the backend and sends one job. It then prints the job ID,
writes `submission.json`, and **exits immediately**. It never calls `status()`,
`result()`, or any wait method on that job.

`submission_attempt.json` prevents accidental repeat submissions. If the
network response is lost, the job might already exist: inspect IBM workloads
using the saved job tag before deciding what to do. Do not delete the marker
and retry blindly. If receipt writing fails, preserve the printed job ID.

## 3. Retrieve later in a separate process

```bash
python scripts/retrieve_ibm_b2ed.py --bundle results/hardware_b2ed/four_pilot
```

This checks status once. If the job is not done, it exits without waiting. A
completed result is saved to `runtime_result.json` before analysis, then joint
counts and an estimate are written to `joint_counts.json` and `analysis.json`.
Explicit register concatenation preserves shot-by-shot data/syndrome
correlations: data bits on the left, syndrome bits on the right.

IBM references: [dynamic circuits and measurement instructions](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits),
[approximate workload usage](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time).
The usage estimate includes initialization, repetition delay and one 2-second
overhead; it is not a guaranteed usage or monetary quote.

## Completed runs and offline analysis

The executed [hardware-results notebook](../notebooks/03_ibm_aachen_hardware_results.ipynb)
displays both jobs, raw and readout-corrected estimates, uncertainty, syndrome
distributions and the previously saved noise-model comparison. Its cells only
read archived files and recompute statistics from existing counts; Run All does
not contact IBM, submit jobs, or run simulations. Use the Python (quantum-jax)
kernel. Plot and table outputs are embedded for viewing on GitHub.

| Variant | Job ID | Saved bundle |
|---|---|---|
| No DD | `db84dj8331ic73fu5jfg` | [`four_pilot_20261014`](../results/hardware_b2ed/four_pilot_20261014/) |
| XY4 | `db84p30331ic73fu5tm0` | [`four_xy4_20261014`](../results/hardware_b2ed/four_xy4_20261014/) |

The XY4 preparation script, `scripts/prepare_ibm_b2ed_dd.py`, derives a review
bundle from the frozen baseline and validates current device durations. It
inserts X/Y/X/Y pulses into noninitial idle windows of at least 500 ns, including
the four data-qubit waits during syndrome readout. Y pulses are native X pulses
with virtual Rz frames. Original gates, their timestamps and the 5.020 µs
preparation duration are preserved. Runtime automatic DD remains disabled.
See the saved [DD review](../results/hardware_b2ed/four_xy4_20261014/dd_review.md).
Preparing a bundle does not authorize submission; use the separate approval and
submission procedure above for any future run.

Once results have been downloaded, the following commands are entirely offline:

```bash
python scripts/analyze_ibm_b2ed.py \
  --bundle results/hardware_b2ed/four_xy4_20261014 \
  --compare-to results/hardware_b2ed/four_pilot_20261014 \
  --replay-calibration --dt 4e-9
python scripts/mitigate_ibm_b2ed_readout.py
```

The [raw comparison](../results/hardware_b2ed/four_xy4_20261014/xy4_vs_no_dd.md)
reports 69.71% fidelity without DD and 74.22% with XY4. The
[readout-mitigation report](../results/hardware_b2ed/readout_mitigation_20261014/readout_mitigation.md)
gives 77.66% and 83.01% after correcting final data assignment using saved
calibration. This preserves the recorded-000 accepted ensemble and its yield.
An additional diagnostic corrects syndrome assignment too, estimating a
different ensemble conditioned on latent true syndrome 000.

Neither correction repairs the quantum state or removes preparation/basis-gate
errors. Sampling intervals omit calibration uncertainty, correlated readout and
drift. The two jobs were sequential, about 24.5 minutes apart. Original raw
reports are preserved; the later mitigation results live in a separate folder.
