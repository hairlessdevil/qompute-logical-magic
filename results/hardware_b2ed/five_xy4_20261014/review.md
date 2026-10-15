# Five-qubit B2_ED + XY4: submission review

Status: **PREPARED ONLY — no hardware job submitted.**

## Choice

For the first [[5,1,3]] hardware example, choose **B2_ED with explicit XY4**.
It extends the preparation-and-syndrome-screening experiment already tested on
[[4,1,2]]. B2_ED ideally accepts with probability 1; B2_BK ideally accepts with
probability 1/6 and consumes five physical magic-state copies. B2_BK is a
separate five-copy distillation/projection primitive.

In our existing April-snapshot noisy simulation, five-qubit B2_ED acceptance was
74.42%, versus 14.71% for B2_BK. The B2_ED circuit is deeper: 106 CZ gates versus
67 for the saved B2_BK circuit. These simulations do not predict the current
hardware's acceptance or establish which protocol has better hardware fidelity.
The higher ideal yield and direct protocol continuity motivate this first pilot.

The five-qubit target is the positive logical tetrahedral state, with logical
Bloch vector (1,1,1)/sqrt(3). The four-qubit experiment targeted the H-state.
The two code experiments therefore differ in target, gate count and circuit
duration; their fidelity difference is not an isolated effect of code distance.

## One proposed hardware job

| Setting | Value |
|---|---|
| Instance / backend | `qzh-eu` / `ibm_aachen` |
| Code / protocol | [[5,1,3]] / B2_ED |
| Target | Encoded positive tetrahedral magic state |
| Data / syndrome qubits | 5 / 4 |
| Routing patch | `[3, 2, 4, 16, 1, 5, 23, 0, 6]` |
| Final data physical qubits, bit order | `[6, 5, 4, 16, 2]` |
| Syndrome physical qubits, bit order | `[0, 23, 1, 3]` |
| Checks in syndrome-bit order | `XZZXI`, `IXZZX`, `XIXZZ`, `ZXIXZ` |
| Acceptance | Recorded `0000`; no active recovery or extra codespace filtering |
| Measurement circuits | 63 nonidentity target-projector terms + 1 acceptance-only circuit |
| Shots per circuit / total | 2,048 / **131,072** |
| Preparation duration | **9.992 µs** |
| Ordinary syndrome readout duration | 2.180 µs |
| CZ count | 106 |
| DD | Explicit XY4 in 28 eligible idle windows, including all 5 data readout waits |
| Added DD gates per preparation | 112 native X pulses + 112 virtual Rz gates |
| DD timing audit | Original gates, timestamps and total duration preserved |
| Runtime automatic DD / twirling | Disabled / disabled |
| Approximate QPU usage | **36.7 seconds**; not a quote or queue estimate |
| Maximum execution time | 300 seconds |

Explicit native DD follows our four-qubit implementation. Runtime automatic DD
is disabled because of its documented incompatibility with dynamic circuits:
[IBM Sampler feature compatibility](https://quantum.cloud.ibm.com/docs/en/guides/sampler-options#feature-compatibility).
The previous four-qubit XY4 increase is not a guarantee of benefit for this
longer circuit. This remains an unflagged error-detection experiment, not a
fault-tolerant preparation or active single-qubit-error recovery demonstration.

## Validation

- All 64 settings match the intended ideal target; maximum numerical error
  `4.22e-15` and ideal acceptance 1.
- Native device instructions and faulty qubits/edges checked against live Aachen.
- All original non-DD gates and full measurement circuit durations preserved.
- 22 targeted tests passed across four- and five-qubit preparation, explicit DD,
  submission/retrieval separation, readout bit order and mitigation controls.
- A dedicated test rejects syndrome `1000`; the fourth syndrome bit is not lost.
- The unsubmitted `five_pilot_20261014` bundle is a compilation source for XY4.
  Approval is requested for the XY4 bundle only, not for a second baseline job.

## Exact bundle and separate job lifecycle

Manifest SHA256: `7626ad2f232af019b8643b4956ae4e00be497674affbff0824add3b4e7d3f522`

After explicit human approval of this bundle:

```bash
conda run -n quantum-jax python -B scripts/submit_ibm_b2ed.py \
  --bundle results/hardware_b2ed/five_xy4_20261014 \
  --approve-manifest-sha256 7626ad2f232af019b8643b4956ae4e00be497674affbff0824add3b4e7d3f522
```

Submission prints/saves the job ID and exits immediately without querying its
status or results. Retrieve later in a separate process:

```bash
conda run -n quantum-jax python -B scripts/retrieve_ibm_b2ed.py \
  --bundle results/hardware_b2ed/five_xy4_20261014
```

Retrieval archives the full result and joint five-data/four-syndrome counts,
then computes the raw fidelity and finite-shot uncertainty. Readout mitigation
can subsequently use the saved calibration, retaining recorded-0000 acceptance.
The existing extended four-qubit comparison and notebook are specific to the
previous jobs; a five-qubit report will be added after this job completes.
