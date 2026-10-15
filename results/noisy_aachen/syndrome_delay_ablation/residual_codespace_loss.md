# Residual codespace loss with extraction delay noise disabled

The matched controls identify extraction-gate noise as the dominant source of the remaining outside-code population. The separate CZ-only control identifies the extraction CZ channels as a large contributor. Readout assignment errors account for a smaller additional effect; ideal preparation alone leaves most of the residual population outside the code.

All controls use the same frozen stage-separated circuits from the latest delay-ablation study. Extraction-phase Delay relaxation remains off throughout. Each additional control is compared directly with that baseline unless explicitly combined; control differences are not additive source fractions because the acceptance probability and accepted ensemble can change.

## Accepted population outside the code subspace

| Protocol | Extraction delay noise off | Also perfect readout | Also perfect extraction gates | Also perfect extraction gates + readout |
|---|---:|---:|---:|---:|
| four/B2_ED | 1.5606% | 1.5393% | 0.0048% | 0.0000% |
| five/B2_ED | 5.5355% | 4.8317% | 0.1464% | 0.0000% |
| five/B2_BK | 8.6928% | 7.9865% | 0.6976% | 0.0000% |

Removing Delay noise does not remove calibrated noise on gates. The gate channels still combine depolarizing errors and T1/T2 relaxation over each gate's duration. The controls below disable the existing calibrated channels as whole channels, without refitting their depolarizing strengths or changing ideal gates, routing, durations, or readout timing.

A faulty extraction gate can leave a data error after a relevant stabilizer has already been mapped onto an ancilla. Later checks need not detect that error, and the earlier check is not repeated. Ancilla gate faults can also corrupt the mapped syndrome or propagate to data. Consequently, recorded zero is not an exact final codespace projection for the noisy extraction. Perfect readout alone cannot repair this quantum-circuit error.

With ideal extraction gates and ideal syndrome assignment, retained preparation noise is followed by a complete ideal stabilizer projection. The accepted density matrix lies in the code subspace to numerical precision, though its logical state may remain distorted. This is a control of the circuit implementation and a direct explanation of why preparation noise alone need not yield accepted codespace loss.

## Additional controls and state quality

| Protocol | Control | Outside code | Within-code fidelity | Full fidelity | Acceptance |
|---|---|---:|---:|---:|---:|
| four/B2_ED | Extraction delay noise off (latest result) | 1.5606% | 99.7803% | 98.2231% | 91.2159% |
| four/B2_ED | Also perfect syndrome assignment | 1.5393% | 99.7806% | 98.2446% | 92.4610% |
| four/B2_ED | Also perfect extraction gates | 0.0048% | 99.8391% | 99.8343% | 97.3666% |
| four/B2_ED | Also perfect extraction gates + syndrome assignment | 0.0000% | 99.8391% | 99.8391% | 98.7178% |
| four/B2_ED | Also perfect preparation (gates + idles) | 1.5452% | 99.9439% | 98.3995% | 92.3796% |
| four/B2_ED | Also perfect extraction CZ gates only | 0.2437% | 99.8351% | 99.5918% | 96.2100% |
| five/B2_ED | Extraction delay noise off (latest result) | 5.5355% | 99.1527% | 93.6641% | 68.5313% |
| five/B2_ED | Also perfect syndrome assignment | 4.8317% | 99.1534% | 94.3626% | 75.7762% |
| five/B2_ED | Also perfect extraction gates | 0.1464% | 99.1823% | 99.0371% | 81.7999% |
| five/B2_ED | Also perfect extraction gates + syndrome assignment | 0.0000% | 99.1823% | 99.1823% | 91.0896% |
| five/B2_ED | Also perfect preparation (gates + idles) | 5.2548% | 99.9891% | 94.7348% | 74.9836% |
| five/B2_ED | Also perfect extraction CZ gates only | 0.8156% | 99.1789% | 98.3700% | 79.4456% |
| five/B2_BK | Extraction delay noise off (latest result) | 8.6928% | 99.7196% | 91.0512% | 14.9318% |
| five/B2_BK | Also perfect syndrome assignment | 7.9865% | 99.7217% | 91.7575% | 15.1246% |
| five/B2_BK | Also perfect extraction gates | 0.6976% | 99.9999% | 99.3023% | 16.4084% |
| five/B2_BK | Also perfect extraction gates + syndrome assignment | 0.0000% | 99.9999% | 99.9999% | 16.6366% |
| five/B2_BK | Also perfect preparation (gates + idles) | 8.6841% | 99.7213% | 91.0614% | 14.9570% |
| five/B2_BK | Also perfect extraction CZ gates only | 1.9094% | 99.9583% | 98.0497% | 16.1236% |

Perfect preparation removes only preparation gate channels and preparation idle channels. Perfect extraction CZ removes the entire calibrated channel on extraction CZ gates, including its gate-duration relaxation; it retains extraction single-qubit gate noise and assignment errors.

Here codespace loss is within the modeled two-level qubit Hilbert space. Physical leakage into higher transmon levels is not included.

Validation: the explicit-gate baseline matches the saved accepted density matrix and recorded syndrome distribution to absolute tolerance 1e-11 for all three circuits. Ideal extraction plus ideal assignment gives unit codespace population to tolerance 1e-11. QPY input hashes and calibration hashes are checked. All state quantities are exact density-matrix diagnostics; no per-shot good/bad labeling is inferred.

Artifacts: `residual_codespace_loss.json`, `residual_codespace_states.npz`. Reproduce: `conda run -n quantum-jax python scripts/analyze_residual_codespace_loss.py`.

References: [Aer device noise channels](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html), [IBM stabilizer codes](https://quantum.cloud.ibm.com/learning/en/courses/foundations-of-quantum-error-correction/stabilizer-formalism/stabilizer-codes).
