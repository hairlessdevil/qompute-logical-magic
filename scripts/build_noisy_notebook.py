#!/usr/bin/env python3
"""Build the Aachen-noise walkthrough; optionally execute in quantum-jax."""

import argparse
from pathlib import Path

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    notebook = nb.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python (quantum-jax)", "language": "python", "name": "quantum-jax"}
    notebook.cells = [
        md("""# Logical magic states with IBM Aachen noise

This notebook runs the existing B0, B1, B2_ED, and five-qubit B2_BK circuits with
the **packaged IBM Aachen calibration snapshot**. All computation is local in
Aer. There are **no manually inserted Pauli faults** in this experiment.

The circuits are routed onto a fixed connected patch, compiled to Aachen's native
gates, and scheduled. The model includes gate errors, syndrome readout error,
and **T1/T2 relaxation during every explicit delay**, including the data-qubit
wait while syndrome readout finishes. Use the **Python (quantum-jax)** kernel.
The ideal and manual-fault experiments remain in `01_ideal_magic_states.ipynb`."""),
        code("""from pathlib import Path
import sys, json
import numpy as np
from IPython.display import display, Markdown, Image
from qiskit import qpy

ROOT = Path.cwd().resolve()
if not (ROOT / 'src' / 'logical_magic').exists():
    ROOT = ROOT.parent
assert (ROOT / 'src' / 'logical_magic').exists()
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from logical_magic import CODES, build_preparation
from logical_magic.aachen_noise import (
    AachenPatch, ScheduledPreparation, simulate_scheduled, validate_idle_relaxation,
)
from run_noisy import run_experiment, summary_table

OUTPUT = ROOT / 'results' / 'noisy_aachen'
SHOTS = 8192
SEED = 20261014
backend = AachenPatch()
provenance = backend.provenance()
print('Python:', sys.executable)
print('Calibration:', provenance['calibration_date'])
print('Physical qubits in local wire order:', provenance['physical_patch'])
print('dt:', provenance['dt_seconds'], 'seconds')"""),
        md("""## 1. Calibration and assumptions

`FakeAachen` supplies the local snapshot; it is not a current online calibration.
The patch is a fixed breadth-first neighborhood rooted at physical qubit 3,
without noise-aware placement optimization. Initial placement uses the listed
wire order. Routing may change which physical wires carry data and syndrome.

The reduced backend preserves the snapshot's native connectivity, instruction
errors/durations, and qubit T1/T2 values. Aer uses its standard BackendV2 model:
calibrated depolarizing plus gate-duration relaxation, independent symmetric
readout assignment from the Target's average readout errors, and delay relaxation
at zero temperature. The readout error is especially high on physical qubit 2;
check the final routed placement before interpreting its effect.

The initial state is ideal all-zero. There is no additional reset, crosstalk,
leakage, drift, or correlated-noise model. No installed package is changed."""),
        code("""table = ['| Local wire | IBM qubit | T1 (µs) | T2 used (µs) | Readout error | Measurement (µs) |',
         '|---:|---:|---:|---:|---:|---:|']
for q in provenance['qubits']:
    table.append(f\"| {q['local']} | {q['physical']} | {q['t1_seconds']*1e6:.2f} | {q['effective_t2_seconds']*1e6:.2f} | {q['readout_error']:.5f} | {q['readout_duration_seconds']*1e6:.3f} |\")
display(Markdown('\\n'.join(table)))"""),
        md(r"""## 2. Verify that delay relaxation is actually active

`NoiseModel.from_backend(..., thermal_relaxation=True)` in Aer 0.17.2 already
adds a Delay-only `RelaxationNoisePass`. We do **not** add another pass.
Scheduling creates the delays on which this pass acts.

As a unit check, prepare clean simulator states |1〉 and |+〉, then apply only a
50 µs delay on the patch's first qubit. The expected excited population and X
coherence are $e^{-t/T_1}$ and $e^{-t/T_2}$. This checks time units and detects a
missing or duplicated relaxation channel; it is not a fault-insertion experiment."""),
        code("""idle_check = validate_idle_relaxation(backend)
display(idle_check)
assert idle_check['max_absolute_error'] < 1e-12"""),
        md(r"""## 3. Run the scheduled circuits with the Aachen model

Three modes use the same routed native circuit:

1. **Ideal control:** all noise disabled.
2. **No-delay ablation:** calibrated gate/readout noise, including gate-duration
   relaxation; only Delay instructions are omitted from channel evolution.
3. **Full:** gate/readout noise plus relaxation during all scheduled delays.

ALAP scheduling and a terminal barrier align syndrome readout. In B2 the output
is delivered at readout completion, so every data qubit waits **2.584 µs** during
syndrome readout, in addition to earlier idles. The standard Aer measurement
model applies assignment error on measured ancillas; it adds no separate
measurement-duration relaxation channel on those ancillas. Data remain unmeasured.

For true ancilla blocks $\rho_s$, the actual accepted output is

$$p_{\rm acc}=\sum_s P(\text{recorded }0\mid s)\operatorname{Tr}\rho_s,\qquad
\rho_{\rm acc}=\frac{\sum_s P(\text{recorded }0\mid s)\rho_s}{p_{\rm acc}}.$$

No active recovery or extra codespace postselection is performed. Fidelity is
$\langle\psi_{\rm target}|\rho_{\rm acc}|\psi_{\rm target}\rangle$ from exact
noisy density-matrix evolution. This section uses simulator diagnostics;
Sections 7–8 below add readout ablation and finite-shot fidelity measurements.
8,192 Aer shots per mode provide an independent sampled syndrome distribution.

The following cell displays the saved run. Set `RERUN=True` to rerun the
experiment and update `results/noisy_aachen`."""),
        code("""RERUN = False
if RERUN or not (OUTPUT / 'noisy_results.json').exists():
    report = run_experiment(OUTPUT, shots=SHOTS, seed=SEED)
else:
    report = json.loads((OUTPUT / 'noisy_results.json').read_text())

for row in report['rows']:
    assert np.isclose(row['ideal']['fidelity'], 1)
    assert np.isclose(row['ideal']['acceptance'], 1/6 if row['strategy'] == 'B2_BK' else 1)
display(Markdown(summary_table(report)))"""),
        code("""display(Image(filename=str(OUTPUT / 'noisy_comparison.png')))"""),
        md("""## 4. Separate loss from logical distortion

A recorded zero syndrome does not guarantee the final state is in the
codespace. Check faults and late relaxation, especially during readout, can
leave accepted data outside it. Codespace population and within-codespace
logical fidelity below are **diagnostics**. The reported main fidelity includes
both codespace loss and logical distortion; no simulator-only filter improves
the delivered output.

B0/B1 accept every shot. B2_ED rejects recorded nonzero syndromes. B2_BK has
ideal acceptance 1/6 even without errors. All rows use each strategy's own
completion time; B0/B1 have no syndrome-readout wait. A shared downstream use
time would require additional storage waits for the faster preparations."""),
        code("""table = ['| Code / strategy | Full fidelity | Codespace population | Logical fidelity within codespace | Accepted fraction with nonzero true syndrome |',
         '|---|---:|---:|---:|---:|']
for row in report['rows']:
    r = row['full']
    table.append(f\"| {row['code']}/{row['strategy']} | {r['fidelity']:.6f} | {r['codespace_population']:.6f} | {r['fidelity_conditioned_on_codespace_diagnostic']:.6f} | {r['accepted_fraction_from_nonzero_true_syndrome']:.6f} |\")
display(Markdown('\\n'.join(table)))
for row in report['rows']:
    if row['strategy'].startswith('B2'):
        delta = row['no_delay']['fidelity'] - row['full']['fidelity']
        print(f\"{row['code']}/{row['strategy']}: omitting delays overstates fidelity by {100*delta:.2f} percentage points\")"""),
        md("""## 5. Recorded syndrome counts and uncertainty

These Wilson intervals apply to the finite-shot acceptance estimate. The exact
model fidelities and probabilities above have no sampling error; model mismatch
with a physical device is a separate limitation. Bitstrings are s[m−1]…s[0]."""),
        code("""table = ['| Code / strategy | Exact acceptance | Sampled acceptance | Wilson 95% interval |',
         '|---|---:|---:|---|']
for row in report['rows']:
    r = row['full']
    lo, hi = r['acceptance_wilson_95_interval']
    table.append(f\"| {row['code']}/{row['strategy']} | {r['acceptance']:.6f} | {r['acceptance_sampled']:.6f} | [{lo:.6f}, {hi:.6f}] |\")
display(Markdown('\\n'.join(table)))
selected = next(r for r in report['rows'] if r['code'] == 'four' and r['strategy'] == 'B2_ED')
print('four/B2_ED raw recorded counts:', selected['full']['syndrome_counts'])"""),
        md("""## 6. Inspect and replay the saved schedule

The QPY files preserve the exact native circuits used for these results.
Re-synthesizing a Clifford circuit in another process/version may change gate
ordering and routing. The replay below uses the saved circuit, physical mapping,
and seed. Calibration file hashes are checked against the current local snapshot.
The original measured circuit is saved; simulator snapshots are added only for
analysis. The saved state is reordered into logical data order, with spectator
wires traced out."""),
        code("""for name, info in report['backend']['snapshot_files'].items():
    assert info['sha256'] == provenance['snapshot_files'][name]['sha256']
res = selected['resources']
print('Data physical qubits at delivery:', res['final_data_physical_qubits'])
print('Syndrome physical qubits in bit order:', res['syndrome_physical_qubits_in_bit_order'])
print('Total duration (µs):', res['duration_seconds']*1e6)
print('Data wait during syndrome readout (µs):', res['syndrome_readout_wait_seconds']*1e6)
print('Idle time by local physical wire (µs):', np.array(res['idle_seconds_by_local_wire'])*1e6)
with (OUTPUT / 'circuits' / 'four_B2_ED.qpy').open('rb') as f:
    saved_circuit = qpy.load(f)[0]
replay = ScheduledPreparation(
    build_preparation(CODES['four'], 'B2_ED'), saved_circuit,
    tuple(res['final_data_local_wires']), tuple(res['syndrome_local_wires_in_bit_order']), res,
)
replayed, accepted_rho = simulate_scheduled(replay, backend, mode='full', shots=SHOTS, seed=SEED)
assert np.isclose(replayed['fidelity'], selected['full']['fidelity'])
assert np.isclose(replayed['acceptance'], selected['full']['acceptance'])
assert replayed['syndrome_counts'] == selected['full']['syndrome_counts']
print('Saved-circuit replay agrees, including seeded counts.')
print(saved_circuit.draw(output='text', fold=120))"""),
        md("""## 7. Isolate syndrome readout error

Readout assignment error was already included in the preparation results above.
It can reject a true zero syndrome and accept a nonzero one. The latter changes
the accepted data ensemble. It is a classical assignment channel, not a new
quantum operation on the data. A true zero syndrome is not proof of a perfect
logical output.

The following ablation disables **only syndrome assignment error**. Gate noise,
delay noise, and the full 2.584 µs data wait during readout remain unchanged.
B0/B1 have no syndrome readout. Their exact preparation fidelity therefore does
not include any destructive data-readout error.

The analysis reuses the saved QPY circuits. Set `RERUN_QUALITY=True` to repeat
the ablation and measurement experiments; otherwise the saved results are used."""),
        code("""from analyze_readout import run_analysis, readout_table, quality_table

RERUN_QUALITY = False
quality_path = OUTPUT / 'readout_and_fidelity.json'
if RERUN_QUALITY or not quality_path.exists():
    quality = run_analysis(OUTPUT, shots=SHOTS, seed=SEED)
else:
    quality = json.loads(quality_path.read_text())

for original, measured in zip(report['rows'], quality['rows']):
    assert (original['code'], original['strategy']) == (measured['code'], measured['strategy'])
    assert np.isclose(original['full']['fidelity'], measured['exact_preparation_fidelity'])
display(Markdown(readout_table(quality)))"""),
        md("""## 8. Estimate state quality from finite-shot Pauli measurements

The earlier sections used finite shots only for acceptance. Here **fidelity is
estimated from measurement counts**. Expand the full physical target projector
as a weighted sum of Pauli strings and sum their measured expectations:
`F_est = c_I + sum_P c_P * mean(P | recorded syndrome = 0)`.

For each nonidentity term, prepare fresh copies, apply local basis rotations,
and measure data parity. X uses H; Y uses S† then H; Z needs no rotation.
Keep the original recorded-zero syndrome shots. There are **23 settings** for
the four-qubit target and **63** for the five-qubit target. Each setting uses
8,192 attempted shots; rejection reduces the accepted sample count. Analysis
copies are consumed, while preparation circuits still leave reusable outputs.

The existing routing and timing are frozen. Analysis rotations begin after the
original output delivery boundary. Their native-gate noise, padding delays, and
data assignment errors are included. The target-projector expansion includes
codespace loss; logical X/Y/Z alone would not suffice.

Two estimates use the same counts: **raw parity** and **data-readout-corrected
parity**. Correction uses the snapshot's known assignment matrices. Syndrome
bits are never corrected, so the estimate refers to the actually accepted state."""),
        code("""display(Markdown(quality_table(quality)))
display(Image(filename=str(OUTPUT / 'finite_shot_fidelity.png')))
print('Total attempted quality-measurement shots:', quality['total_quality_measurement_shots'])"""),
        md("""### Sampling uncertainty versus measurement bias

For data assignment A[true, recorded], the corrected parity weights solve
`A @ w = [1, -1]`. Products of single-qubit weights give each Pauli-string score.
For symmetric assignment error e, this divides parity by `(1 - 2e)` on each
nonidentity factor. This removes data assignment bias under the stated model.

The error bars are approximate 95% statistical intervals from independent
settings. Conservative Hoeffding/union-bound intervals are also saved. Neither
interval includes calibration uncertainty or basis-rotation/delay bias. The
assignment matrices are treated as exact in this simulation; a hardware
measurement would need calibration and its uncertainty. Estimates and intervals
are left unclipped, so a statistical fluctuation can exceed one.

The simulator-only infinite-shot measurement score below separates residual
measurement bias from finite-shot fluctuations. It is not used to compute the
estimate. The remaining difference from the exact delivered-state fidelity is
due to the noisy analysis rotations and waits."""),
        code("""table = ['| Protocol | Exact delivered-state fidelity | Infinite-shot raw score | Infinite-shot data-readout-corrected score | Remaining corrected bias |',
         '|---|---:|---:|---:|---:|']
for row in quality['rows']:
    r = row['finite_shot']
    raw = r['raw']['infinite_shot_measurement_score_diagnostic']
    corrected = r['data_readout_mitigated']['infinite_shot_measurement_score_diagnostic']
    bias = corrected - row['exact_preparation_fidelity']
    table.append(f\"| {row['code']}/{row['strategy']} | {row['exact_preparation_fidelity']:.6f} | {raw:.6f} | {corrected:.6f} | {bias:+.6f} |\")
display(Markdown('\\n'.join(table)))"""),
        md("""### Reconstruct the estimate using counts alone

This cell recomputes both four-qubit B2_ED estimates using only raw counts,
Pauli coefficients, and assignment matrices. It does not read an output density
matrix or simulator expectation value. QPY files in `quality_circuits` store
the measurement circuits without any simulator state-save instructions."""),
        code("""from logical_magic.noisy_fidelity import accepted_score, parity_weights

measured = next(r for r in quality['rows'] if r['code'] == 'four' and r['strategy'] == 'B2_ED')
sample = measured['finite_shot']
assignment = [np.asarray(a) for a in sample['data_assignment_matrices']]
for name, matrices in [('raw', None), ('data_readout_mitigated', assignment)]:
    estimate = sample['identity_coefficient']
    variance = 0.0
    for term in sample['terms']:
        weights = parity_weights(term['pauli'], matrices)
        score = accepted_score(term['counts'], n_syndrome=3, weights=weights)
        estimate += term['coefficient'] * score['expectation']
        variance += term['coefficient']**2 * score['variance_of_mean']
    assert np.isclose(estimate, sample[name]['fidelity_estimate'])
    print(f'{name}: {estimate:.6f} ± {1.96*np.sqrt(variance):.6f} (approx. 95% statistical interval)')
print('Accepted shots across settings:', sample['total_accepted_shots'])
print('First setting:', sample['terms'][0]['pauli'])
print('Its raw joint data/syndrome counts:', sample['terms'][0]['counts'])"""),
        md("""## Scope and saved artifacts

This is a first noisy comparison of the existing bare circuits on one fixed
patch. It does not establish fault tolerance, optimal placement, or a hardware
advantage. B0 is a physical single-qubit output; encoded rows have different
resource costs and delivery times. B2_BK consumes five magic inputs. The model
omits drift, crosstalk, leakage, correlations, and downstream control latency.

Results are in `results/noisy_aachen`: a Markdown report, JSON metadata and raw
counts, accepted density matrices, a PNG/PDF comparison, exact QPY/text circuits,
and copied calibration files. `readout_and_fidelity.json` / `.md` add the
readout-only ablation, finite-shot measurement results, raw counts, statistical
intervals, and systematic-bias diagnostics. No IBM job is submitted and no
credentials are used.

References: [FakeAachen](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider-fake-aachen),
[Aer NoiseModel](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html),
[RelaxationNoisePass](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.RelaxationNoisePass.html)."""),
    ]
    path = ROOT / "notebooks" / "02_noisy_aachen.ipynb"
    nb.write(notebook, path)
    if args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(notebook, timeout=600, kernel_name="quantum-jax",
                                resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute()
        finally:
            nb.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
