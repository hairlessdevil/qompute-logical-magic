# Disable delay noise only during syndrome extraction

## Matched experiment and phase definition

The original optimized noisy circuits have no preserved preparation/extraction boundary. This experiment therefore uses the previously saved stage-separated diagnostic QPY circuits from the timeline study. **Their full-noise baseline is recomputed here; it is different from the original noisy-result baseline.** Every comparison below uses the same fixed circuit, routing, gates, delays, and measurement duration within that row. No new transpilation is performed.

The extraction window starts at the barrier after input preparation/encoding and ends at the completion of terminal syndrome readout. Delay T1/T2 relaxation is disabled on all physical wires within this window, including ancilla idles and the data wait during readout. Preparation-phase delay relaxation, gate depolarization, gate-duration relaxation, and syndrome readout assignment errors remain enabled. This disables only scheduled Delay channels; it does not remove gate-duration noise or alter calibration times.

All Delay instructions remain in the simulation. Their retained relaxation is applied explicitly once, using the same thermal relaxation parameters as Aer. The calibrated gate/readout model uses Aer's public device helpers without adding its automatic Delay pass. Delays crossing a boundary retain noise only for the portion outside the disabled window. This is an idealized channel-removal experiment; no DD pulses are added.

## Requested quantities

For the state $\rho_A$ conditioned on the **recorded** zero syndrome and codespace projector $P$,

$$p_{\rm code}=\mathrm{Tr}(P\rho_A),\quad \rho_{\rm code}=P\rho_A P/p_{\rm code},\quad F_{\rm code}=\langle\psi_L|\rho_{\rm code}|\psi_L\rangle.$$

The full accepted-state fidelity obeys $F=p_{\rm code}F_{\rm code}$. Codespace conditioning is a diagnostic, not an extra physical acceptance step.

Arrows below mean **full noise → delay noise off during extraction and readout**, on the same stage-separated circuit.

| Protocol | Codespace population | Fidelity within codespace | Full target fidelity | Acceptance |
|---|---:|---:|---:|---:|
| four/B2_ED | 94.172% → 98.439% | 99.761% → 99.780% | 93.948% → 98.223% | 89.863% → 91.216% |
| five/B2_ED | 88.567% → 94.464% | 99.134% → 99.153% | 87.800% → 93.664% | 65.481% → 68.531% |
| five/B2_BK | 82.929% → 91.307% | 99.656% → 99.720% | 82.643% → 91.051% | 14.630% → 14.932% |

Changes in the requested quantities: four/B2_ED: codespace population +4.267 percentage points; within-code fidelity +0.019 percentage points; five/B2_ED: codespace population +5.898 percentage points; within-code fidelity +0.018 percentage points; five/B2_BK: codespace population +8.378 percentage points; within-code fidelity +0.064 percentage points.

## Readout-only and all-delay controls

| Protocol | Model | Codespace population | Fidelity within codespace | Full target fidelity | Acceptance |
|---|---|---:|---:|---:|---:|
| four/B2_ED | Full noise | 94.172% | 99.761% | 93.948% | 89.863% |
| four/B2_ED | Readout-window delay noise off | 97.588% | 99.774% | 97.368% | 89.863% |
| four/B2_ED | Extraction + readout delay noise off | 98.439% | 99.780% | 98.223% | 91.216% |
| four/B2_ED | All delay noise off (control) | 98.442% | 99.813% | 98.258% | 91.385% |
| five/B2_ED | Full noise | 88.567% | 99.134% | 87.800% | 65.481% |
| five/B2_ED | Readout-window delay noise off | 92.041% | 99.137% | 91.246% | 65.481% |
| five/B2_ED | Extraction + readout delay noise off | 94.464% | 99.153% | 93.664% | 68.531% |
| five/B2_ED | All delay noise off (control) | 94.486% | 99.230% | 93.758% | 69.578% |
| five/B2_BK | Full noise | 82.929% | 99.656% | 82.643% | 14.630% |
| five/B2_BK | Readout-window delay noise off | 86.938% | 99.666% | 86.647% | 14.630% |
| five/B2_BK | Extraction + readout delay noise off | 91.307% | 99.720% | 91.051% | 14.932% |
| five/B2_BK | All delay noise off (control) | 91.307% | 99.720% | 91.051% | 14.932% |

Readout-only suppression isolates the final 2.584 µs wait. Its exact acceptance is unchanged, because it affects data after mapping and no ancilla is waiting during its own simultaneous terminal readout. Suppression throughout extraction can also change syndrome probabilities. The all-delay control additionally removes preparation idles and is distinct from the requested extraction-only experiment.

## Timing and retained preparation noise

| Protocol | Extraction start (µs) | Readout start (µs) | Delivery (µs) | Preparation idle duration retaining noise (sum over wires, µs) |
|---|---:|---:|---:|---:|
| four/B2_ED | 0.668 | 3.504 | 6.088 | 4.612 |
| five/B2_ED | 3.068 | 9.276 | 11.860 | 21.204 |
| five/B2_BK | 0.064 | 5.808 | 8.392 | 0.256 |

## Validation and artifacts

Each explicit-channel full-noise accepted density matrix and syndrome distribution agrees with the existing automatic-delay model to absolute tolerance 1e-11. The all-delay-off control agrees with the existing no-delay mode. Ideal references retain fidelity one and the expected acceptance. Original instructions are checked unchanged, and per-delay active/suppressed durations are recorded in JSON. All reported state quantities are exact density-matrix diagnostics, with no finite-shot fidelity estimator involved. 8,192 sampled syndromes per mode are saved separately as an acceptance check.

Files: `results.json`, `accepted_states.npz`, and `circuits/*_stage_diagnostic.qpy` in this folder. The frozen circuit copies have saved SHA-256 hashes. The original noisy-result circuits and the timeline-study inputs remain unchanged.

Reproduce: `conda run -n quantum-jax python scripts/analyze_syndrome_delays.py`.

Channel definition: [Aer thermal_relaxation_error](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.thermal_relaxation_error.html).
