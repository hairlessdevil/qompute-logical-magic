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
        md("""## 9. Visualize the full circuit timing with Qiskit

The figures below use **Qiskit's `timeline_drawer`** with the exact saved native
circuits and calibrated Aachen durations. They show gates, delays, and readout;
the local snapshot does not supply analog pulse waveforms. Qiskit 2.4.1 no longer
includes Qiskit Pulse.

QPY preserves the gates and explicit delays, but not their start-time metadata.
`ALAPScheduleAnalysis` reconstructs those timestamps without recompilation.
Instruction identity, total duration, per-wire idle totals, and source-file
hashes are checked. Durations are in **microseconds**.

Blue bars are CZ gates, teal bars are native one-qubit gates, gray bars are
delays, and orange bars are ancilla readout. Virtual Rz updates have zero
duration and their symbols are hidden for readability. Each row is a physical
IBM qubit; D/S labels give its data/syndrome role **at output**, because routing
moves states between physical wires.

The first timing column includes input preparation/encoding and coherent
syndrome mapping. It is **not syndrome extraction alone**: the optimized
original circuit does not retain boundaries between those source stages.
All ancillas are measured together at the end. The data wait throughout that
**2.584 µs** readout window, so their state can change after syndrome mapping."""),
        code("""from draw_timelines import run_timelines

TIMELINES = OUTPUT / 'timelines'
timing_path = TIMELINES / 'timing_summary.json'
if not timing_path.exists():
    timing = run_timelines(OUTPUT)
else:
    timing = json.loads(timing_path.read_text())
table = ['| Protocol | Preparation + syndrome mapping (µs) | Readout / data wait (µs) | Total (µs) |',
         '|---|---:|---:|---:|']
for row in timing['rows']:
    r = row['original']
    table.append(f\"| {row['code']}/{row['strategy']} | {r['coherent_phase_us']:.3f} | {r['readout_duration_us']:.3f} | {r['total_us']:.3f} |\")
display(Markdown('\\n'.join(table)))
for row in timing['rows']:
    label = f\"{row['code']}_{row['strategy']}\"
    display(Image(filename=str(TIMELINES / (label + '_original.png'))))"""),
        md("""### Time each stabilizer check in a separate diagnostic compilation

To expose the stages, this diagnostic inserts barriers after preparation and
after every check, then compiles and schedules again. The P band is preparation;
S0, S1, etc. are the coherent stabilizer-mapping blocks. These extra barriers
prevent overlap/fusion across stages and can change routing, so **these durations
describe a different compilation**. They must not replace the timings of the
original noisy results. Section 10 runs a separate matched noise experiment on
these circuits.

The diagnostic's syndrome-extraction duration is the sum of the check blocks
plus the terminal readout. Ideal simulation verifies fidelity 1 and acceptance
1 for B2_ED, or 1/6 for B2_BK. The original result circuits remain unchanged.
Full plots, gate-region zooms, and PDF/SVG exports are in `results/noisy_aachen/timelines`."""),
        code("""table = ['| Protocol | Preparation (µs) | Check blocks (µs) | Readout (µs) | Checks + readout (µs) | Total (µs) |',
         '|---|---:|---:|---:|---:|---:|']
for row in timing['rows']:
    r = row['stage_diagnostic']
    assert np.isclose(r['ideal_validation']['fidelity'], 1)
    assert np.isclose(r['ideal_validation']['acceptance'], 1/6 if row['strategy'] == 'B2_BK' else 1)
    table.append(f\"| {row['code']}/{row['strategy']} | {r['stages'][0]['duration_us']:.3f} | {r['syndrome_mapping_us']:.3f} | {r['readout_duration_us']:.3f} | {r['syndrome_mapping_and_readout_us']:.3f} | {r['total_us']:.3f} |\")
display(Markdown('\\n'.join(table)))
for row in timing['rows']:
    label = f\"{row['code']}_{row['strategy']}\"
    display(Image(filename=str(TIMELINES / (label + '_staged.png'))))
    checks = ['| Source stage | Duration (µs) |', '|---|---:|']
    for stage in row['stage_diagnostic']['stages']:
        checks.append(f\"| {stage['stage']} | {stage['duration_us']:.3f} |\")
    display(Markdown('\\n'.join(checks)))"""),
        md(r"""## 10. Remove delay noise only during syndrome extraction

This test disables **T1/T2 relaxation on Delay instructions from the end of
preparation/encoding through completion of syndrome readout**, on all physical
wires. It includes ancilla idles and the final data-qubit readout wait.
Preparation-phase delay noise, gate errors, gate-duration relaxation, and
syndrome readout assignment errors remain enabled. All gates and delays retain
their scheduled durations; no dynamical-decoupling pulses are inserted.

**Matched baseline:** the optimized original circuits have no clean preparation
boundary, so this test uses the **saved stage-separated circuits from Section 9**.
Their full-noise baseline is recomputed. The baseline and each ablation use the
same QPY circuit, routing, and timings; they are not compared against the earlier
original-circuit numbers. These additional stage barriers can change the noise
and performance of the circuit.

For the accepted state $\rho_A$ (recorded zero syndrome), define

$$p_{\rm code}=\mathrm{Tr}(P\rho_A),\qquad
F_{\rm code}=\frac{\langle\psi_L|P\rho_A P|\psi_L\rangle}{p_{\rm code}}.$$

These are the requested codespace population and fidelity within the codespace.
The full accepted-state fidelity is $F=p_{\rm code}F_{\rm code}$. This extra
codespace projection is a diagnostic only; it does not change the acceptance
rule or physically correct the delivered state.

**Arrows mean full noise → extraction/readout delay noise off.** These state
metrics come from exact density matrices, not finite-shot fidelity estimates."""),
        code("""from analyze_syndrome_delays import run_analysis as run_delay_analysis, comparison_table, modes_table

DELAY_ABLATION = OUTPUT / 'syndrome_delay_ablation'
RERUN_EXTRACTION_DELAYS = False
if RERUN_EXTRACTION_DELAYS or not (DELAY_ABLATION / 'results.json').exists():
    delay_study = run_delay_analysis(OUTPUT)
else:
    delay_study = json.loads((DELAY_ABLATION / 'results.json').read_text())
display(Markdown(comparison_table(delay_study)))
for row in delay_study['rows']:
    for mode in row['modes'].values():
        assert np.isclose(mode['fidelity'], mode['codespace_population'] * mode['fidelity_conditioned_on_codespace_diagnostic'])
    assert max(row['validation'].values()) < 1e-11
print('Explicit delay channels agree with the existing full/no-delay implementations.')"""),
        md("""### Separate the final readout wait from earlier extraction idles

The readout-only control disables Delay relaxation only during the final
2.584 µs window. The extraction-only ablation additionally disables idle noise
during all stabilizer-mapping blocks. The all-delay-off control also removes
preparation idles, which remain noisy in the requested experiment.

The explicit-channel implementation is checked against the existing full-noise
and all-delay-off models, including density matrices and syndrome distributions.
Boundary-crossing delays retain the correct amount of preparation noise.
Readout-only suppression leaves exact acceptance unchanged. All 8,192-shot
syndrome counts per mode, per-delay noise durations, accepted density matrices,
and the frozen QPY inputs are saved in `results/noisy_aachen/syndrome_delay_ablation`.
This is a channel-removal diagnostic, not a prediction of DD performance."""),
        code("""display(Markdown(modes_table(delay_study)))
for row in delay_study['rows']:
    full = row['modes']['full']
    readout_off = row['modes']['readout_delay_off']
    assert np.isclose(full['acceptance'], readout_off['acceptance'], atol=1e-11, rtol=0)
    print(f\"{row['code']}/{row['strategy']}: extraction starts at {row['timing']['extraction_start_us']:.3f} µs; output delivered at {row['timing']['delivery_us']:.3f} µs\")"""),
        md("""### Why does some accepted population still lie outside the code?

Extraction-delay suppression leaves noise on the gates themselves, including
gate-duration T1/T2 relaxation. The following controls use exactly the same
frozen circuits and start from the extraction-delay-off model in Section 10.
Each column removes the additional named channel; preparation noise remains
unless explicitly disabled in the detailed report.

**Every table entry below is the accepted population outside the code subspace.**
The controls identify extraction-gate noise as the main source of the residual
loss. A later faulty gate can damage data after a relevant stabilizer has already
been mapped onto an ancilla. That earlier parity is not measured again. Ancilla
gate faults can also corrupt the syndrome or propagate to data. Perfect classical
assignment cannot repair these quantum-circuit faults.

Ideal extraction gates plus ideal syndrome assignment restore unit codespace
population even with noisy preparation, because complete ideal stabilizer
projection rejects the outside-code component. The logical state inside the
code can still be distorted. These control differences are not additive source
fractions: changing noise can also change acceptance.

Here codespace loss stays inside the modeled two-level qubit Hilbert space;
physical leakage into higher transmon levels is not modeled."""),
        code("""from analyze_residual_codespace_loss import run_analysis as run_residual_analysis, summary_table as residual_table

residual_path = DELAY_ABLATION / 'residual_codespace_loss.json'
if residual_path.exists():
    residual = json.loads(residual_path.read_text())
else:
    residual = run_residual_analysis(DELAY_ABLATION)
display(Markdown(residual_table(residual)))
for row in residual['rows']:
    assert row['baseline_max_density_matrix_difference'] < 1e-11
    assert abs(row['modes']['perfect_extraction_gates_and_readout']['outside_code_population']) < 1e-11
    cz = row['modes']['perfect_extraction_cz']['outside_code_population']
    print(f\"{row['code']}/{row['strategy']}: making extraction CZ gates ideal leaves {100*cz:.4f}% outside code (other gate noise and assignment errors retained).\")"""),
        md("""## 11. Restore full noise and test dynamical decoupling

This experiment uses **the original optimized saved Aachen circuits**, restoring
the baseline from Sections 3–8. The stage-separated circuits used for the phase
ablations have different baselines. Every variant here runs with **full noise**:
gate depolarization, gate-duration T1/T2 relaxation, relaxation during every
remaining Delay, and syndrome readout assignment errors. No ablation remains
active in this experiment.

Qiskit's `PadDynamicalDecoupling` inserts balanced **2-, 4-, or 8-X sequences**
inside sufficiently long idles. Native X pulses last 32 ns and include their
calibrated errors. The original gates, measurement times, and overall duration
are preserved. Initial ground-state idles are skipped, and short windows that
cannot fit the chosen sequence are left alone. All data qubits receive DD
during the 2.584 µs syndrome-readout window. The settings specify a pulse count
per eligible idle, not a uniform repetition rate across the full circuit.

The table gives **full accepted-state fidelity**, including codespace loss.
The no-DD replay matches the original saved density matrix, and all inserted
sequences preserve ideal target fidelity and acceptance."""),
        code("""from analyze_dd import run_analysis as run_dd_analysis, fidelity_table as dd_fidelity_table, quality_table as dd_quality_table

DD_OUTPUT = OUTPUT / 'dd_full_noise'
RERUN_DD = False
if RERUN_DD or not (DD_OUTPUT / 'results.json').exists():
    dd_study = run_dd_analysis(OUTPUT)
else:
    dd_study = json.loads((DD_OUTPUT / 'results.json').read_text())
assert dd_study['noise_mode'] == 'full'
display(Markdown(dd_fidelity_table(dd_study)))
display(Image(filename=str(DD_OUTPUT / 'comparison.png')))
for row in dd_study['rows']:
    for name, variant in row['variants'].items():
        assert variant['metrics']['mode'] == 'full'
        assert np.isclose(variant['ideal_validation']['fidelity'], 1)
        assert np.isclose(variant['ideal_validation']['acceptance'], 1/6 if row['strategy'] == 'B2_BK' else 1)
print('Full noise enabled for every DD variant; ideal references verified.')"""),
        md(r"""### Interpret the DD result in the noise model we actually have

The current calibrated idle model uses **memoryless exponential T1/T2 decay**.
It has no explicit slow detuning or temporal noise correlations. Those are
important for the phase-refocusing mechanism of DD. In these runs, additional
DD pulses lower fidelity; increasing pulse density adds further calibrated
gate noise. This is not evidence that DD cannot help the physical Aachen device.

For a one-qubit |+〉 state, ideal instantaneous X pulses leave the transverse
coherence at $\langle X\rangle=e^{-t/T_2}$ under this memoryless channel. The
sanity check below verifies this for zero, two, four, and eight pulses. A separate
constant-Z-detuning toy model refocuses exactly, showing that the echo sequence
works when the relevant coherent noise is present. **The toy model is not added
to the Aachen experiment and assumes no hardware detuning value.**

To assess DD's expected hardware benefit, the next modeling step is to separate
refocusable, correlated idle noise from irreversible relaxation, with measured
parameters or clearly stated sensitivity assumptions. Removing all Delay noise
is not a physical model of DD.

Sources: [IBM explanation of DD and slow noise](https://quantum.cloud.ibm.com/learning/en/courses/tools-for-quantum-advantage/error-suppression),
[Markovian limits of DD](https://arxiv.org/abs/1710.11133)."""),
        code("""display(Markdown(dd_quality_table(dd_study)))
sanity = dd_study['echo_sanity']
table = ['| Ideal instantaneous X pulses | Memoryless T1/T2: X coherence | Constant-detuning toy: X coherence |',
         '|---:|---:|---:|']
for row in sanity['rows']:
    assert np.isclose(row['markovian_x_coherence'], sanity['expected_markovian_x_coherence'], atol=1e-12, rtol=0)
    table.append(f\"| {row['ideal_instantaneous_x_pulses']} | {row['markovian_x_coherence']:.8f} | {row['constant_detuning_toy_x_coherence']:.8f} |\")
display(Markdown('\\n'.join(table)))
display(Image(filename=str(DD_OUTPUT / 'four_B2_ED_x8_timeline.png')))
print('All three DD timelines and native QPY circuits are saved in:', DD_OUTPUT)"""),
        md("""## 12. Earlier readout and representative check-order changes

Ordinary `measure` and IBM `MidCircuitMeasure` implement the same logical
Z-basis measurement. A measurement is *mid-circuit* when computation continues
after it; ordinary `measure` can also be used in that position. IBM's special
`MidCircuitMeasure` maps to `measure_2` and is tuned for lower overhead during
ongoing computation. The actual duration and errors depend on calibration.
Our Aachen snapshot only exposes ordinary `measure`, lasting 2.584 µs on this
patch. We retain that duration; no faster measurement is assumed.
[IBM measurement documentation](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits).

First, the exact original native circuits are retimed without changing gates,
physical mapping, or per-wire operation order:

1. Original simultaneous terminal readout.
2. Read each ancilla after its last gate, keeping all gate times fixed.
3. Move gates as early as dependencies allow (ASAP), then read ancillas early.
4. Keep those early readout deadlines but move gates as late as possible (ALAP).

All data idles through final delivery, pre-readout ancilla idles, gate noise, and
assignment errors remain enabled. After an ancilla is measured, its subsequent
relaxation cannot alter the stored bit. The exact deferred-measurement analysis
therefore discards only those unused, already-measured ancilla idles and rejects
any circuit that reuses a measured ancilla.

These are local-model candidate schedules. The model omits measurement-induced
crosstalk and does not certify hardware controller/readout-group constraints on
overlapping measurements with other gates. No hardware job is submitted."""),
        code("""from analyze_syndrome_schedules import (
    run_analysis as run_schedule_analysis, summary_table as schedule_table,
    timing_table as schedule_timing_table, check_order_table,
)

SCHEDULE_OUTPUT = OUTPUT / 'syndrome_schedules'
RERUN_SCHEDULES = False
if RERUN_SCHEDULES or not (SCHEDULE_OUTPUT / 'results.json').exists():
    schedule_study = run_schedule_analysis(OUTPUT)
else:
    schedule_study = json.loads((SCHEDULE_OUTPUT / 'results.json').read_text())
assert schedule_study['noise_mode'] == 'full'
display(Markdown(schedule_table(schedule_study)))
for row in schedule_study['rows']:
    original = row['variants']['original']
    for variant in row['variants'].values():
        assert np.isclose(variant['ideal_validation']['fidelity'], 1, atol=1e-11, rtol=0)
        assert np.isclose(variant['ideal_validation']['acceptance'], 1/6 if row['strategy'] == 'B2_BK' else 1, atol=1e-11, rtol=0)
        assert np.isclose(variant['resources']['duration_seconds'], original['resources']['duration_seconds'], atol=1e-13, rtol=0)
    assert np.isclose(row['variants']['fixed']['metrics']['unconditional_fidelity'],
                      original['metrics']['unconditional_fidelity'], atol=1e-11, rtol=0)
print('Exact state metrics at delivery; 8,192 sampled syndromes per schedule are saved separately.')"""),
        md("""### What earlier readout changes

The last ancilla remains on the critical path in the frozen native circuits,
so early readout does not shorten their total delivery time. With fixed gate
times it mainly changes how faithfully the earlier syndrome is stored; the
unconditional data state is unchanged. Earlier checks still cannot detect later
data faults. Moving every gate earlier can prepare fragile states too soon and
worsen their exposure to noise. The deadline-based ALAP schedule avoids some of
that exposure; it can also move independent data-only gates into readout waits.
The timeline below shows the original and deadline-based ALAP schedules."""),
        code("""display(Markdown(schedule_timing_table(schedule_study)))
display(Image(filename=str(SCHEDULE_OUTPUT / 'five_B2_ED_timelines.png')))"""),
        md("""### Change whole-check order: a separate matched compilation comparison

For each protocol we freeze one source preparation and try three whole-check
orders: reference, reverse, and a one-step rotation. Source preparation gates,
source ancilla assignments, initial placement, optimization level, and compiler
seed stay fixed. Gate count, routing, and final physical placement may change.
Whole stabilizer checks commute ideally; we do not arbitrarily interleave their
individual controlled-Pauli gates.

The newly compiled reference is the matched baseline for this table. It can
differ from the original archived circuit because encoder synthesis was not
frozen before the original run. The new source QPY is cached for reproduction.
Check indices refer to the stabilizer lists printed below. Each order has a
terminal-readout and an early deadline-based ALAP variant, both with full noise.
Fidelity is the complete accepted-state fidelity, including codespace loss.
Shorter circuits need not have better fidelity because routing changes which
physical gate and readout errors affect the data and acceptance."""),
        code("""display(Markdown(check_order_table(schedule_study)))
for row in schedule_study['order_study']:
    print(row['code'], row['strategy'], 'check indices:', dict(enumerate(row['checks_in_bit_order'])))
    for case in row['orders'].values():
        for variant in case['variants'].values():
            assert np.isclose(variant['ideal_validation']['fidelity'], 1, atol=1e-11, rtol=0)
            assert np.isclose(variant['ideal_validation']['acceptance'], 1/6 if row['strategy'] == 'B2_BK' else 1, atol=1e-11, rtol=0)
print('Saved report, states, native schedules, source circuits, and timelines:', SCHEDULE_OUTPUT)"""),
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

`timelines/timing_report.md` and `timing_summary.json` add the Qiskit timing
figures and instruction timestamps. Diagnostic circuits with stage barriers
are saved separately in that folder.

`syndrome_delay_ablation/report.md` and `results.json` contain the matched
extraction-phase delay-noise ablation; its frozen inputs and accepted density
matrices are stored alongside them.
`syndrome_delay_ablation/residual_codespace_loss.md` adds separate extraction-gate,
CZ-gate, readout-assignment, and preparation-noise controls.
`dd_full_noise/report.md` adds full-noise DD results on the original circuits,
native DD circuit QPY files, pulse timestamps, timing plots, and model checks.
`syndrome_schedules/report.md` adds earlier readout, native-gate retiming, and
three representative whole-check orders, with full noise and saved timelines.

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
