# Qompute: local logical magic-state models

This workspace implements the **model, ideal simulation, and local Aachen noisy simulation** of the supplied
[project plan](docs/project_plan_original.md). All execution is local. The first
reference results are in [the ideal report](results/ideal/ideal_report.md), and an
executable walkthrough is in [the notebook](notebooks/01_ideal_magic_states.ipynb).
Section 6 injects X/Y/Z on every data qubit and runs actual syndrome-controlled
recovery: the five-qubit code corrects all 15 errors; the four-qubit code detects
all 12 but has ambiguous syndromes and cannot universally correct them.
That is a restricted storage-error check. **Section 7** compares a few faults
during magic preparation, encoding, and syndrome extraction, plus late data
errors and a wrong syndrome bit. It shows why the five-qubit storage result
does not imply that the preparation circuit corrects every single fault.

The executed [Aachen notebook](notebooks/02_noisy_aachen.ipynb) adds calibrated
gate/readout noise and scheduled T1/T2 delay relaxation, with **no manually
inserted Pauli faults**. It uses the packaged `FakeAachen` calibration dated
2026-04-17, on a fixed connected nine-qubit patch. Data-qubit waits during the
2.584 µs syndrome readout are explicitly noisy. See the
[noisy report](results/noisy_aachen/noisy_report.md) for results and assumptions.
Sections 7–8 of the noisy notebook isolate syndrome readout error and estimate
target fidelity using finite-shot Pauli measurements, with raw and data-readout-
mitigated estimates. The [readout and measurement report](results/noisy_aachen/readout_and_fidelity.md)
separates statistical uncertainty from residual analysis-gate bias.

## Run in the existing environment

```bash
cd /Users/runzhemo/Desktop/Zhuang_group/hackathon
conda run -n quantum-jax python -m pytest -q
conda run -n quantum-jax python scripts/run_ideal.py
conda run -n quantum-jax python scripts/run_noisy.py
conda run -n quantum-jax python scripts/analyze_readout.py --shots 8192
```

The script defaults to 8,192 shots per Pauli measurement setting, seed 20261014,
and a 600-second hard timeout. It does not require installing this project.

```bash
conda run -n quantum-jax python scripts/run_ideal.py --shots 16384 --seed 7 --output results/ideal_seed7
```

Open the notebook using the **Python (quantum-jax)** kernel. Its first cell locates
the project from either the repository root or the `notebooks` directory.

Qiskit 2.4.1 was already installed. Aer **0.17.2** was added with `--no-deps`;
no existing package versions changed, and `pip check` passed. See
[the installation audit](results/environment_aer_install.json) and
[validated dependency versions](requirements-validated.txt). Do not upgrade the
shared environment's dependencies without the user's approval. The early unused
`.venv` directory is not used by any command, script, or notebook in this project.
The noisy stage uses the already installed qiskit-ibm-runtime **0.47.0**; it
requires no package installation, IBM credentials, or hardware submission.

## Models and baselines

| Model | Target | Role |
|---|---|---|
| `four`: [[4,1,2]] | H eigenstate, Bloch vector (1,0,1)/√2 | Primary small screening experiment |
| `five`: [[5,1,3]] | Tetrahedral state, Bloch vector (1,1,1)/√3 | Code comparison and five-copy projection reference |

The tetrahedral state is not the equatorial state obtained by applying a pi/8
phase gate to |+⟩. Pauli labels and basis kets use **Qiskit little-endian ordering**:
the rightmost character acts on q0. Both encoders take their logical input on q0.

| Strategy | Construction | Ideal acceptance |
|---|---|---:|
| B0 | Prepare one physical magic state | 1 |
| B1 | Prepare and encode one magic state | 1 |
| B2_ED | B1 followed by all stabilizer measurements | 1 |
| B2_BK (five only) | Prepare five antipodal tetrahedral states; measure all stabilizers | 1/6 |

All accepted ideal outputs have fidelity 1 to their respective targets. B2_BK
keeps the output **encoded**; starting with antipodal input states accounts for
the pole reversal in the Bravyi–Kitaev projection. Its target is the same positive
tetrahedral logical state used by five-qubit B1/B2_ED.

**B2_ED is a provisional error-detection baseline, not a complete protected
magic-state preparation protocol.** Passing stabilizers only establishes the
codespace. B2_BK is a known distillation primitive under ideal Clifford operations;
our bare ancilla checks are not fault tolerant against their own faults. No B3
improvement or hardware result is claimed. The first noisy comparison does not
show a B2_ED fidelity improvement over B1 at each circuit's own completion time.

## What is validated

- Independent algebraic code projectors, logical bases, Pauli conventions, and distances.
- Gate-level encoding of arbitrary complex one-qubit states, including decoding.
- Nondestructive syndrome extraction and every measurement branch against matrix projectors.
- Ideal accepted-state fidelity, codespace population, logical observables, and success probability.
- Full target-projector fidelity measurements on separate output copies, with the same acceptance rule.
- All post-encoding single data-Pauli faults, plus a circuit-level audit of initialization,
  every gate's Pauli faults, and syndrome readout flips.
- Finite-shot Aer counts, explicit shot budgets, uncertainty estimates, and saved random seeds.

There is one fresh ancilla per check: 7 total qubits for four-qubit B2_ED and 9
for five-qubit B2_ED/B2_BK. Ancillas are measured at the end; data qubits are
left intact in preparation circuits. Destructive fidelity measurements run on
separate copies. The ideal notebook assumes all-to-all connectivity and unset
durations. The noisy notebook instead routes to Aachen connectivity and uses
calibrated durations plus explicit delay channels.

## Files

| Path | Contents |
|---|---|
| `src/logical_magic/codes.py` | Mathematical definitions and Clifford encoders |
| `src/logical_magic/circuits.py` | Preparation, check selection/order, and analysis circuits |
| `src/logical_magic/simulation.py` | Exact states, Aer sampling, and fault audits |
| `src/logical_magic/error_correction.py` | Single-Pauli storage faults and syndrome-controlled recovery |
| `src/logical_magic/fault_examples.py` | Representative insertion points and actual versus ideal-next-round recovery |
| `src/logical_magic/aachen_noise.py` | Calibrated target patch, scheduling, idle noise, exact recorded-syndrome conditioning |
| `src/logical_magic/noisy_fidelity.py` | Count-based Pauli fidelity, data assignment correction, uncertainty and measurement-bias diagnostics |
| `scripts/run_ideal.py` | Reproducible experiment, exports, plots, report |
| `scripts/run_noisy.py` | Reproducible Aachen noisy experiment and delay-noise ablation |
| `scripts/analyze_readout.py` | Readout-only ablation and finite-shot quality measurements of the saved preparations |
| `docs/model.md` | Equations, assumptions, fidelity definition, limitations |
| `notebooks/01_ideal_magic_states.ipynb` | Executed, editable walkthrough |
| `notebooks/02_noisy_aachen.ipynb` | Executed noisy walkthrough, calibration, delay checks, and QPY replay |
| `results/noisy_aachen/` | Report, exact states, raw counts, calibration snapshots, QPY circuits, and plots |
| `results/ideal/ideal_results.json` | Exact results, finite-shot estimates, raw counts, metadata |
| `results/ideal/reference_states.npz` | Independent logical bases, targets, codespace projectors |
| `results/ideal/faults_*.csv` | Individual circuit-fault outcomes |
| `results/ideal/storage_faults_*.csv` | Single-data-error syndrome table |
| `results/ideal/circuits/` | QPY circuits and plain-text diagrams |
| `results/single_pauli_errors.json` | Detection/correction sweep, controls, and raw syndrome counts |
| `results/representative_fault_locations.json` | Selected circuit faults, insertion locations, and syndrome counts |

## Noisy simulation and further work

Circuit construction is separate from simulation, so these same circuits can be
passed to an Aer density-matrix simulator or transpiled to a hardware target.
The `checks` argument already supports subsets and order changes for future
controlled comparisons; no order has been labeled optimal from ideal results.

The Aachen implementation retains recorded-syndrome acceptance, compares exact
conditional density matrices with finite-shot syndrome statistics, and includes
gate/readout noise plus scheduled idles. It checks the routed circuits against
the ideal references and validates delay noise against analytic T1/T2 decay.
The no-delay ablation isolates the effect of idle relaxation while preserving
the compiled gate sequence and gate/readout noise.

Further comparisons could optimize physical placement or check order, or align
the strategies to a common downstream consumption time. The present rows deliver
B0/B1 at preparation completion and B2 after syndrome readout, so delivery times
differ. The snapshot model does not include crosstalk, leakage, drift, correlated
noise, or downstream control latency.
Shared circuit builders simplify migration; they do not establish hardware
compatibility or fault tolerance by themselves.

The final primary protocol should be selected after these noise/cost comparisons.
The four-qubit model minimizes initial circuit overhead; the five-qubit BK model
provides a stronger theoretical preparation reference at a lower ideal yield.

Primary sources and the limits of each attribution are recorded in
[the model document](docs/model.md).
