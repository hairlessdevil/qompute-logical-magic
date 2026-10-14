# Local noisy magic-state simulation: IBM Aachen

Calibration: **2026-04-17T12:29:46+02:00**, packaged `FakeAachen` snapshot from qiskit-ibm-runtime 0.47.0. This is a local simulation, not a live calibration fetch or hardware run.

## Results

| Code / strategy | Acceptance | Fidelity incl. delay | Fidelity omitting delay | Codespace population | Duration (µs) | CZ gates |
|---|---:|---:|---:|---:|---:|---:|
| four/B0 | 1.000000 | 0.999792 | 0.999792 | 1.000000 | 0.064 | 0 |
| four/B1 | 1.000000 | 0.985589 | 0.987774 | 0.987178 | 0.668 | 7 |
| four/B2_ED | 0.905051 | 0.943088 | 0.980060 | 0.945341 | 5.424 | 39 |
| five/B0 | 1.000000 | 0.999784 | 0.999784 | 1.000000 | 0.064 | 0 |
| five/B1 | 1.000000 | 0.908950 | 0.926142 | 0.913418 | 3.164 | 33 |
| five/B2_ED | 0.744221 | 0.886818 | 0.948951 | 0.894925 | 10.396 | 106 |
| five/B2_BK | 0.147074 | 0.854524 | 0.913466 | 0.856809 | 6.224 | 67 |

Fidelity is the exact density-matrix overlap with the physical encoded target, conditioned only on the **recorded all-zero syndrome**. B0 uses the corresponding single-qubit target and has a different output size. Codespace population is a diagnostic, not an additional acceptance filter. These are model predictions, not measured hardware fidelities or finite-shot tomography estimates.

B1/B2_ED start with one magic input; B2_BK starts with five antipodal tetrahedral inputs. These are the existing baselines; no manually inserted X, Y, or Z faults and no active recovery are used.

## Timing and noise

- Native gates and connectivity come from physical qubits `[3, 2, 4, 16, 1, 5, 23, 0, 6]` (local wire order). This is a fixed connected patch rooted at qubit 3, selected without calibration/fidelity optimization. Initial placement, routed final placement, gate counts, and every wire's idle time are saved in JSON.
- ALAP scheduling uses calibrated durations and dt = 4 ns. A terminal barrier aligns syndrome readout. Every data wire explicitly waits **2.584 µs during syndrome readout**, in addition to routing and synchronization idles.
- `NoiseModel.from_backend` includes calibrated depolarizing/thermal gate channels, readout assignment error, and its built-in Delay-only `RelaxationNoisePass`. Idle amplitude damping/dephasing use each physical qubit's T1/T2 at zero temperature, with T2 capped at 2T1. No second delay pass is added.
- The no-delay ablation removes only Delay instructions from the already compiled sequence. Native gates, routing and gate/readout error channels stay fixed; gate-duration relaxation remains enabled.
- In this BackendV2 snapshot Aer builds independent symmetric readout assignment channels from each Target's average readout error. No extra ancilla relaxation is added during measurement, matching Aer's standard backend measurement model; the unmeasured data's readout wait is explicitly noisy.
- Initial qubits are ideal |0〉; no explicit reset/SPAM model beyond compiled preparation gates and syndrome assignment is added. Data are delivered at readout completion; no subsequent classical processing, storage, or decoding latency is assumed.

## Acceptance and logical output

Let ρ_s be the unnormalized data block for true ancilla outcome s, after routing is undone in the analysis and spectator qubits are traced out. The recorded-zero output is

$$\rho_{\rm acc}=\frac{\sum_s P(0\mid s)\rho_s}{p_{\rm acc}},\qquad p_{\rm acc}=\sum_s P(0\mid s)\operatorname{Tr}\rho_s.$$

Readout errors can therefore accept a nonzero true syndrome or reject a zero true syndrome. Errors after a qubit's last relevant check, including relaxation while readout finishes, can leave accepted data outside the codespace. The JSON includes codespace population, logical Bloch coordinates conditioned on that subspace (diagnostic only), and the fraction of accepted runs with a nonzero true syndrome.

## Sampling and checks

Each mode uses 8,192 Aer shots for recorded syndrome counts, seed 20261014. Exact density matrices set the fidelities/acceptances in the table; finite-shot acceptance and Wilson intervals are saved separately.

| Code / strategy | Exact acceptance | Sampled acceptance | Wilson 95% interval |
|---|---:|---:|---|
| four/B0 | 1.000000 | 1.000000 | [0.999531, 1.000000] |
| four/B1 | 1.000000 | 1.000000 | [0.999531, 1.000000] |
| four/B2_ED | 0.905051 | 0.899048 | [0.892336, 0.905386] |
| five/B0 | 1.000000 | 1.000000 | [0.999531, 1.000000] |
| five/B1 | 1.000000 | 1.000000 | [0.999531, 1.000000] |
| five/B2_ED | 0.744221 | 0.740112 | [0.730504, 0.749495] |
| five/B2_BK | 0.147074 | 0.151489 | [0.143889, 0.159416] |

All seven routed circuits recover ideal fidelity 1 with noise disabled; ideal B2_BK acceptance is 1/6. A clean |1〉/|+〉 followed by a 50 µs delay matches exp(−t/T1) / exp(−t/T2) to 3.3e-16 absolute error. This checks the presence, units, and single application of delay relaxation.

## Interpretation and limitations

For **four**, B2_ED has accepted fidelity 0.9431 at acceptance 0.9051, versus B1 fidelity 0.9856. Omitting idle noise overstates B2_ED fidelity by 3.70 percentage points for this schedule.

For **five**, B2_ED has accepted fidelity 0.8868 at acceptance 0.7442, versus B1 fidelity 0.9090. Omitting idle noise overstates B2_ED fidelity by 6.21 percentage points for this schedule.

This first comparison uses each strategy's own completion time, so B1 is delivered earlier than B2. A common downstream consumption time would require adding the corresponding B1/B0 storage waits. The fixed patch includes physical qubit 2 with an 8.67% average readout error; its effect depends on the routed ancilla placement. No claim of optimal placement or hardware advantage follows from these rows. The calibrated local Markovian model omits crosstalk, leakage, drift, correlated noise, and control-stack effects. The bare syndrome circuits remain non-fault-tolerant.

## Reproduce

```bash
conda run -n quantum-jax python scripts/run_noisy.py
```

Saved alongside this report: `noisy_results.json`, `accepted_states.npz`, PNG/PDF comparison, the exact scheduled circuits in QPY/text, and copies of the calibration files with SHA-256 provenance in JSON. QPY is the reference for replay of this run; re-synthesizing Clifford circuits in another process/version can change gate order and routing.

References: [FakeAachen](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider-fake-aachen), [Aer NoiseModel](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html), [RelaxationNoisePass](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.RelaxationNoisePass.html).
