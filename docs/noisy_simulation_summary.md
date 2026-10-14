# Noisy simulation summary: IBM Aachen

We simulated the existing circuits locally using the **IBM Aachen calibration snapshot dated April 17, 2026**, on a fixed connected nine-qubit patch.

## Noise considered

- **Gate noise:** calibrated single- and two-qubit errors, including relaxation during gate execution.
- **Idle noise:** T1 relaxation and T2 dephasing during all scheduled delays—including the **2.584 µs data-qubit wait during syndrome readout**.
- **Syndrome readout error:** independent bit-assignment errors, which can falsely reject or accept a preparation.
- **For finite-shot quality measurements:** additional basis-rotation noise, associated delays, and **data readout error**.

The initial qubits are ideal \(|0\rangle\). The model excludes crosstalk, correlated noise, calibration drift, and leakage into higher physical energy levels. No Pauli faults were manually inserted.

## Resulting state and protocols

The resulting output is a **noisy mixed state**. For B2 protocols, we keep runs with a **recorded all-zero syndrome**, including any readout mistakes. There is no active recovery or additional codespace filtering.

We evaluate its fidelity as

$$
F=\langle\psi_{\mathrm{target}}|\rho_{\mathrm{accepted}}|\psi_{\mathrm{target}}\rangle.
$$

The four-qubit target is the encoded H state; the five-qubit target is the encoded tetrahedral state. B0 uses the corresponding unencoded single-qubit target.

| Protocol | Construction |
|---|---|
| B0 | Prepare one physical magic state. |
| B1 | Prepare and encode one magic state. |
| B2_ED | B1 followed by syndrome screening. |
| B2_BK | Five-input projection, keeping an encoded output. |

## Density-matrix and finite-shot results

| Protocol | Acceptance | Density-matrix fidelity | Raw finite-shot estimate | Data-readout-corrected estimate |
|---|---:|---:|---:|---:|
| Four / B0 | 100% | 99.98% | 99.32% | 99.64% ± 0.78 pp |
| Four / B1 | 100% | 98.56% | 84.02% | 98.47% ± 0.45 pp |
| Four / B2_ED | 90.51% | 94.31% | 80.96% | 94.14% ± 0.48 pp |
| Five / B0 | 100% | 99.98% | 99.13% | 99.44% ± 0.90 pp |
| Five / B1 | 100% | 90.90% | 77.08% | 90.83% ± 0.36 pp |
| Five / B2_ED | 74.42% | 88.68% | 74.75% | 88.51% ± 0.42 pp |
| Five / B2_BK | 14.71% | 85.45% | 72.07% | 85.35% ± 0.97 pp |

Here, **pp means percentage points**, and ± denotes an approximate 95% statistical interval.

The density-matrix column is the exact fidelity **within this simulation model**. The finite-shot columns estimate that quality through noisy measurements. We measure the full target-projector Pauli expansion on separate copies: **23 settings for four-qubit outputs and 63 for five-qubit outputs**, with **8,192 attempted shots per setting**. The two B0 targets require two and three settings, respectively.

Raw estimates are biased downward by data readout errors. The corrected estimates account for those errors using the snapshot's assignment matrices; **they do not improve the prepared state or change syndrome acceptance**. Their intervals cover shot noise only. Remaining measurement-gate/delay bias is approximately **0.07–0.13 percentage points for the B2 protocols**, and calibration uncertainty is excluded. Linear estimates and intervals are not clipped to [0, 1], so an interval near unit fidelity can extend above 100%.

## How the accepted states differ from the target

| Accepted output | Population inside the code subspace | Fidelity within that subspace |
|---|---:|---:|
| Four / B2_ED | 94.53% | 99.76% |
| Five / B2_ED | 89.49% | 99.09% |
| Five / B2_BK | 85.68% | 99.73% |

Much of the fidelity loss comes from accepted population **outside the code subspace**. Passing the recorded syndrome does not guarantee that the delivered state remains inside it, especially after relaxation during readout. The within-subspace values are diagnostics; we do not apply that extra filtering. Population outside the code subspace here remains within the physical qubits' computational Hilbert space; it is distinct from leakage into higher physical energy levels.

## Interpretation

For this placement and implementation, **B2_ED has lower delivered-state fidelity than B1**. The extra checks, routing, and waiting introduce substantial noise. Omitting idle noise would overstate B2_ED fidelity by **3.70 percentage points for four qubits and 6.21 for five qubits**.

This comparison uses each protocol's own completion time; a common downstream use time would require additional storage waits for the faster protocols. B0 produces a single physical qubit rather than an encoded output, and B2_BK consumes five magic inputs. These results do not establish optimal placement, fault tolerance, or a hardware performance advantage.

## Saved results

- [Executed notebook](/Users/runzhemo/Desktop/Zhuang_group/hackathon/notebooks/02_noisy_aachen.ipynb)
- [Simulation report](/Users/runzhemo/Desktop/Zhuang_group/hackathon/results/noisy_aachen/noisy_report.md)
- [Finite-shot measurement report](/Users/runzhemo/Desktop/Zhuang_group/hackathon/results/noisy_aachen/readout_and_fidelity.md)
- [Exact accepted density matrices](/Users/runzhemo/Desktop/Zhuang_group/hackathon/results/noisy_aachen/accepted_states.npz)
- [Simulation results and metadata](/Users/runzhemo/Desktop/Zhuang_group/hackathon/results/noisy_aachen/noisy_results.json)
- [Finite-shot counts and uncertainty estimates](/Users/runzhemo/Desktop/Zhuang_group/hackathon/results/noisy_aachen/readout_and_fidelity.json)
