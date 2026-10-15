#!/usr/bin/env python3
"""Identify residual codespace loss after extraction-delay noise is disabled."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from qiskit import qpy
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.device import basic_device_gate_errors, basic_device_readout_errors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logical_magic import CODES, build_preparation
from logical_magic.aachen_noise import AachenPatch, ScheduledPreparation, recorded_syndrome_state, state_metrics, syndrome_blocks
from logical_magic.delay_ablation import delay_ablation_snapshot
from draw_timelines import timing_record

MODES = {
    "baseline": "Extraction delay noise off (latest result)",
    "perfect_readout": "Also perfect syndrome assignment",
    "perfect_extraction_gates": "Also perfect extraction gates",
    "perfect_extraction_gates_and_readout": "Also perfect extraction gates + syndrome assignment",
    "perfect_preparation": "Also perfect preparation (gates + idles)",
    "perfect_extraction_cz": "Also perfect extraction CZ gates only",
}


def simulate_control(scheduled, backend, boundary, end, gate_errors, mode):
    perfect_readout = mode in {"perfect_readout", "perfect_extraction_gates_and_readout"}
    all_gates_off = mode in {"perfect_extraction_gates", "perfect_extraction_gates_and_readout"}
    perfect_prep = mode == "perfect_preparation"
    # Existing helper preserves every Delay and explicitly applies only its
    # retained noise. Extraction-delay noise is off in EVERY control below.
    base, _ = delay_ablation_snapshot(scheduled, backend, (0 if perfect_prep else boundary, end))
    qc = base.copy_empty_like()
    records = list(zip(scheduled.circuit.op_start_times, scheduled.circuit.data))
    records = [r for r in records if r[1].operation.name != "measure"] + [r for r in records if r[1].operation.name == "measure"]
    cursor = 0
    active_gates, disabled_gates = [], []
    for inst in base.data:
        qc.append(inst)
        if inst.operation.name in {"quantum_channel", "save_density_matrix"}:
            continue
        start, original = records[cursor]
        cursor += 1
        assert inst == original
        wires = tuple(base.find_bit(q).index for q in inst.qubits)
        key = (inst.operation.name, wires)
        error = gate_errors.get(key)
        if error is None:
            continue
        extracting = start >= boundary
        off = (perfect_prep and not extracting) or (extracting and
              (all_gates_off or (mode == "perfect_extraction_cz" and inst.operation.name == "cz")))
        record = {"operation": inst.operation.name, "local_wires": list(wires),
                  "start_dt": int(start), "stage": "extraction" if extracting else "preparation"}
        (disabled_gates if off else active_gates).append(record)
        if not off:
            qc.append(error.to_instruction(), inst.qubits)
    assert cursor == len(records)
    # All quantum channels are explicit and fixed at their original calibrated
    # values; switching a channel off never refits another channel's strength.
    noise = NoiseModel(basis_gates=backend.operation_names)
    if not perfect_readout:
        for wires, error in basic_device_readout_errors(target=backend.target):
            noise.add_readout_error(error, wires)
    result = AerSimulator(method="density_matrix", max_parallel_threads=1, noise_model=noise).run(qc, shots=1).result()
    assert result.success
    blocks = syndrome_blocks(np.asarray(result.data(0)["before_readout"]), scheduled.data_wires, scheduled.syndrome_wires)
    matrices = backend.readout_matrices()
    assignment = [np.eye(2) if perfect_readout else matrices[q] for q in scheduled.syndrome_wires]
    state, probabilities = recorded_syndrome_state(blocks, assignment)
    metrics = {**probabilities, **state_metrics(state, scheduled.preparation),
               "outside_code_population": 1-state_metrics(state, scheduled.preparation)["codespace_population"],
               "gate_channels_retained": active_gates, "gate_channels_disabled": disabled_gates}
    assert np.isclose(np.trace(state), 1, atol=1e-11, rtol=0)
    assert np.linalg.eigvalsh(state).min() > -1e-10
    return metrics, state


def summary_table(result):
    lines = ["| Protocol | Extraction delay noise off | Also perfect readout | Also perfect extraction gates | Also perfect extraction gates + readout |",
             "|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        values = [f"{max(0, 100*row['modes'][m]['outside_code_population']):.4f}%" for m in
                  ["baseline", "perfect_readout", "perfect_extraction_gates", "perfect_extraction_gates_and_readout"]]
        lines.append(f"| {row['code']}/{row['strategy']} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(result):
    lines = ["# Residual codespace loss with extraction delay noise disabled", "",
             "The matched controls identify extraction-gate noise as the dominant source of the remaining outside-code population. "
             "The separate CZ-only control identifies the extraction CZ channels as a large contributor. Readout assignment errors account "
             "for a smaller additional effect; ideal preparation alone leaves most of the residual population outside the code.", "",
             "All controls use the same frozen stage-separated circuits from the latest delay-ablation study. Extraction-phase Delay relaxation "
             "remains off throughout. Each additional control is compared directly with that baseline unless explicitly combined; control "
             "differences are not additive source fractions because the acceptance probability and accepted ensemble can change.", "",
             "## Accepted population outside the code subspace", "", summary_table(result), "",
             "Removing Delay noise does not remove calibrated noise on gates. The gate channels still combine depolarizing errors and T1/T2 "
             "relaxation over each gate's duration. The controls below disable the existing calibrated channels as whole channels, "
             "without refitting their depolarizing strengths or changing ideal gates, routing, durations, or readout timing.", "",
             "A faulty extraction gate can leave a data error after a relevant stabilizer has already been mapped onto an ancilla. "
             "Later checks need not detect that error, and the earlier check is not repeated. Ancilla gate faults can also corrupt the mapped "
             "syndrome or propagate to data. Consequently, recorded zero is not an exact final codespace projection for the noisy extraction. "
             "Perfect readout alone cannot repair this quantum-circuit error.", "",
             "With ideal extraction gates and ideal syndrome assignment, retained preparation noise is followed by a complete ideal stabilizer "
             "projection. The accepted density matrix lies in the code subspace to numerical precision, though its logical state may remain distorted. "
             "This is a control of the circuit implementation and a direct explanation of why preparation noise alone need not yield accepted codespace loss.", "",
             "## Additional controls and state quality", "",
             "| Protocol | Control | Outside code | Within-code fidelity | Full fidelity | Acceptance |",
             "|---|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        for mode, name in MODES.items():
            r = row["modes"][mode]
            values = [f"{max(0, 100*r[k]):.4f}%" for k in
                      ["outside_code_population", "fidelity_conditioned_on_codespace_diagnostic", "fidelity", "acceptance"]]
            lines.append(f"| {row['code']}/{row['strategy']} | {name} | " + " | ".join(values) + " |")
    lines += ["", "Perfect preparation removes only preparation gate channels and preparation idle channels. "
              "Perfect extraction CZ removes the entire calibrated channel on extraction CZ gates, including its gate-duration relaxation; "
              "it retains extraction single-qubit gate noise and assignment errors.", "",
              "Here codespace loss is within the modeled two-level qubit Hilbert space. Physical leakage into higher transmon levels is not included.", "",
              "Validation: the explicit-gate baseline matches the saved accepted density matrix and recorded syndrome distribution to absolute "
              "tolerance 1e-11 for all three circuits. Ideal extraction plus ideal assignment gives unit codespace population to tolerance 1e-11. "
              "QPY input hashes and calibration hashes are checked. All state quantities are exact density-matrix diagnostics; no per-shot good/bad labeling is inferred.", "",
              "Artifacts: `residual_codespace_loss.json`, `residual_codespace_states.npz`. "
              "Reproduce: `conda run -n quantum-jax python scripts/analyze_residual_codespace_loss.py`.", "",
              "References: [Aer device noise channels](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html), "
              "[IBM stabilizer codes](https://quantum.cloud.ibm.com/learning/en/courses/foundations-of-quantum-error-correction/stabilizer-formalism/stabilizer-codes).", ""]
    return "\n".join(lines)


def run_analysis(folder=ROOT / "results" / "noisy_aachen" / "syndrome_delay_ablation"):
    folder = Path(folder)
    base = json.loads((folder / "results.json").read_text())
    backend = AachenPatch(base["backend"]["physical_patch"])
    for name, info in base["backend"]["snapshot_files"].items():
        assert info["sha256"] == backend.provenance()["snapshot_files"][name]["sha256"]
    gate_errors = {(name, tuple(wires)): error for name, wires, error in basic_device_gate_errors(
        target=backend.target, gate_error=True, thermal_relaxation=True, temperature=0)}
    output = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source": "results.json",
              "baseline": "extraction_delay_off", "calibration_date": base["backend"]["calibration_date"], "rows": []}
    states = {}
    with np.load(folder / "accepted_states.npz") as saved_states:
        for row in base["rows"]:
            label = f"{row['code']}_{row['strategy']}"
            path = folder / "circuits" / (label + "_stage_diagnostic.qpy")
            assert sha256(path.read_bytes()).hexdigest() == row["source_qpy_sha256"]
            with path.open("rb") as f:
                raw = qpy.load(f)[0]
            circuit = PassManager([ALAPScheduleAnalysis(target=backend.target)]).run(raw)
            assert list(circuit.data) == list(raw.data)
            timing = timing_record(circuit, backend)
            boundary = next(i["start_dt"] for i in timing["instructions"] if i["label"] == "end:prepare")
            scheduled = ScheduledPreparation(build_preparation(CODES[row["code"]], row["strategy"]), circuit,
                                             tuple(row["final_data_local_wires"]), tuple(row["syndrome_local_wires"]), {})
            entry = {"code": row["code"], "strategy": row["strategy"], "source_qpy_sha256": row["source_qpy_sha256"], "modes": {}}
            for mode in MODES:
                metrics, state = simulate_control(scheduled, backend, boundary, timing["total_dt"], gate_errors, mode)
                entry["modes"][mode] = metrics
                states[label + "_" + mode] = state
                if mode == "baseline":
                    diff = np.max(np.abs(state - saved_states[label + "_extraction_delay_off"]))
                    assert diff < 1e-11
                    entry["baseline_max_density_matrix_difference"] = float(diff)
                    assert np.allclose(metrics["recorded_syndrome_probabilities"], row["modes"]["extraction_delay_off"]["recorded_syndrome_probabilities"], atol=1e-11, rtol=0)
                if mode == "perfect_extraction_gates_and_readout":
                    assert abs(metrics["outside_code_population"]) < 1e-11
                print(label, mode, "outside code %", 100*metrics["outside_code_population"], flush=True)
            output["rows"].append(entry)
    (folder / "residual_codespace_loss.json").write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(folder / "residual_codespace_states.npz", **states)
    (folder / "residual_codespace_loss.md").write_text(render_report(output))
    return output


if __name__ == "__main__":
    run_analysis()
