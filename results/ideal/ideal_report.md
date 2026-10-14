# Ideal simulation results

## Material Passport

- Stage: local ideal model construction and validation.
- Source plan: `docs/project_plan_original.md` (unaltered copy supplied by user).
- Evidence: generated exact states, Qiskit Aer shot counts, circuit fault enumeration.
- Scope: no noisy performance claim, hardware execution, or B3 optimization.
- Reproduction: `conda run -n quantum-jax python scripts/run_ideal.py --shots 8192 --seed 20261014`.

## Baselines

Fidelity means squared pure-target overlap of the actual accepted data state. Only recorded syndrome bits define acceptance; no extra codespace projection is applied.

| Code / strategy | Exact fidelity | Acceptance | Attempts/accepted | Qubits | 2q gates | Shot estimate ± SE |
|---|---:|---:|---:|---:|---:|---:|
| four/B0 | 1.000000000000 | 1.000000000 | 1.000 | 1 | 0 | 0.996926 ± 0.003930 |
| four/B1 | 1.000000000000 | 1.000000000 | 1.000 | 4 | 4 | 1.001177 ± 0.001378 |
| four/B2_ED | 1.000000000000 | 1.000000000 | 1.000 | 7 | 12 | 1.001177 ± 0.001378 |
| five/B0 | 1.000000000000 | 1.000000000 | 1.000 | 1 | 0 | 0.999825 ± 0.004511 |
| five/B1 | 1.000000000000 | 1.000000000 | 1.000 | 5 | 12 | 1.002380 ± 0.001125 |
| five/B2_ED | 1.000000000000 | 1.000000000 | 1.000 | 9 | 28 | 1.002380 ± 0.001125 |
| five/B2_BK | 1.000000000000 | 0.166666667 | 6.000 | 9 | 16 | 0.999140 ± 0.002771 |

B0 = bare physical state; B1 = ordinary encoding; B2_ED = encoding followed by all stabilizer checks; B2_BK = five antipodal tetrahedral inputs projected onto the five-qubit codespace, retaining the encoded output. B2_ED is a screening baseline, not verified fault-tolerant magic-state preparation. B2_BK assumes ideal Clifford checks for its distillation property; the implemented unflagged extraction is not fault tolerant.

## Fault injection

| Circuit | Enumerated cases | Fully rejected | Accepted with target intact | Can accept a damaged state |
|---|---:|---:|---:|---:|
| four/B2_ED | 228 | 157 | 24 | 47 |
| five/B2_ED | 559 | 454 | 38 | 67 |
| five/B2_BK | 325 | 15 | 89 | 221 |

These are counts of modeled fault locations and Pauli operators, not error rates or a statistically weighted noise model. An accepted category can include partial acceptance; each row in the CSV records its exact acceptance probability and conditional fidelity.

All 12 single-qubit data Pauli errors for the four-qubit code and all 15 for the five-qubit code are detected when inserted after encoding and before otherwise ideal checks. Circuit faults during encoding or extraction can escape. In particular, a Y fault just after four-qubit H-state input preparation becomes a logical Y: acceptance remains 1, codespace population remains 1, and target fidelity becomes 0.

## Interpretation

The ideal circuits and measurement conventions are validated. Noiseless fidelity ties do not establish an error-suppression advantage. The four-qubit model is the small primary screening experiment; the five-qubit BK primitive is the literature-based distillation reference. Final code/protocol selection requires the next noise study.

The BK ideal acceptance is 1/6, so six attempts and thirty input magic states are needed per accepted encoded state on average. Gate/qubit costs are per attempt and exclude analysis copies, routing, timing, reset, and calibration overhead. Durations are unset because no physical backend or timing model has been selected.

Finite-shot linear estimates can exceed 1; they are deliberately not clipped. The plot's error bars are 1.96 times the plug-in standard error, not exact confidence intervals. `ideal_results.json` also records conservative 95% Hoeffding intervals, Wilson intervals for acceptance, all raw counts, seeds, settings, and total shot cost.

## Sources

- [Leung et al., four-qubit amplitude-damping code, Eqs. 9–10](https://arxiv.org/html/quant-ph/9704002v1).
- [Laflamme et al., five-qubit code](https://arxiv.org/abs/quant-ph/9602019).
- [Bravyi–Kitaev, five-copy projection, Sec. V and Eqs. 15–22](https://arxiv.org/html/quant-ph/0403025v2).
- [Qiskit Clifford API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Clifford).
- [AerSimulator API](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html).
