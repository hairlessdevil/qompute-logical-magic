#!/usr/bin/env python3
"""Qiskit timelines of the saved Aachen schedules and a stage timing diagnostic.

These are calibrated instruction timelines, not analog pulse waveforms.
The original noisy-result circuits are never recompiled or overwritten.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, qpy, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit.visualization import timeline_drawer
from qiskit.visualization.timeline import IQXStandard, drawings, generators, types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logical_magic import CODES, build_preparation
from logical_magic.aachen_noise import AachenPatch, ScheduledPreparation, simulate_scheduled


COLORS = {"sx": "#238a89", "x": "#238a89", "cz": "#406bb6", "delay": "#d9dfe6",
          "measure": "#a66b13", "rz": "#292d33", "id": "#238a89"}


def instruction_duration_ticks(circuit, inst, backend):
    name = inst.operation.name
    if name == "barrier":
        return 0
    if name == "delay":
        if inst.operation.unit != "dt":
            raise ValueError("Expected dt delays in saved circuits")
        return int(inst.operation.duration)
    wires = tuple(circuit.find_bit(q).index for q in inst.qubits)
    return round(backend.target[name][wires].duration / backend.dt)


def timing_record(circuit, backend):
    rows = []
    for index, (start, inst) in enumerate(zip(circuit.op_start_times, circuit.data)):
        duration = instruction_duration_ticks(circuit, inst, backend)
        wires = [circuit.find_bit(q).index for q in inst.qubits]
        rows.append({"index": index, "operation": inst.operation.name,
                     "label": inst.operation.label, "local_wires": wires,
                     "physical_qubits": [backend.physical_qubits[q] for q in wires],
                     "start_dt": int(start), "duration_dt": duration,
                     "start_us": float(start*backend.dt*1e6),
                     "duration_us": float(duration*backend.dt*1e6)})
    measured = [r for r in rows if r["operation"] == "measure"]
    starts = {r["start_dt"] for r in measured}
    durations = {r["duration_dt"] for r in measured}
    assert len(starts) == len(durations) == 1
    total = max(r["start_dt"]+r["duration_dt"] for r in rows)
    readout = measured[0]
    return {"instructions": rows, "readout_start_dt": readout["start_dt"],
            "readout_duration_dt": readout["duration_dt"], "total_dt": total,
            "coherent_phase_us": readout["start_us"],
            "readout_duration_us": readout["duration_us"],
            "total_us": total*backend.dt*1e6}


def restore_schedule(circuit, backend, resources):
    # QPY preserves Delay instructions but not op_start_times. Analysis recovers
    # the timing metadata without synthesis, routing, or instruction changes.
    restored = PassManager([ALAPScheduleAnalysis(target=backend.target)]).run(circuit)
    assert list(restored.data) == list(circuit.data)
    assert restored.global_phase == circuit.global_phase
    record = timing_record(restored, backend)
    # QPY also drops the aggregate duration. Qiskit's timeline renderer still
    # reads this metadata, so restore it from the unchanged instructions.
    restored.duration = record["total_dt"]
    restored.unit = "dt"
    assert np.isclose(record["total_us"]*1e-6, resources["duration_seconds"], atol=1e-13)
    idle = np.zeros(restored.num_qubits)
    for row in record["instructions"]:
        if row["operation"] == "delay":
            idle[row["local_wires"][0]] += row["duration_dt"]*backend.dt
    assert np.allclose(idle, resources["idle_seconds_by_local_wire"], atol=1e-13)
    return restored, record


def staged_diagnostic(code, strategy, backend, seed):
    """Insert source-stage barriers for timing visibility in a NEW compilation.

    The barriers prevent overlap/fusion across stages and can change routing.
    This diagnostic's durations must not be substituted for the original run.
    """
    prep = build_preparation(code, strategy)
    qc = QuantumCircuit(prep.unitary.num_qubits, name=f"{code.name}_{strategy}_stage_timing")
    previous = None
    stage_names = []
    for stage, inst in zip(prep.stages, prep.unitary.data):
        group = "prepare" if stage in {"magic_input", "encoding"} else stage
        if group != previous:
            if previous is not None:
                qc.barrier(label=f"end:{previous}")
            stage_names.append(group)
            previous = group
        qc.append(inst)
    qc.barrier(label=f"end:{previous}")
    qc.add_register(ClassicalRegister(prep.n_checks, "syndrome"))
    qc.measure(range(prep.n_data, prep.n_data+prep.n_checks), range(prep.n_checks))
    compiled = transpile(qc, target=backend.target,
                         initial_layout=list(range(qc.num_qubits)),
                         seed_transpiler=seed, optimization_level=1, scheduling_method="alap")
    record = timing_record(compiled, backend)
    boundaries = {r["label"].removeprefix("end:"): r["start_dt"] for r in record["instructions"]
                  if r["operation"] == "barrier" and r["label"] and r["label"].startswith("end:")}
    assert set(boundaries) == set(stage_names)
    stage_rows = []
    start = 0
    for stage in stage_names:
        end = boundaries[stage]
        assert end >= start
        stage_rows.append({"stage": stage, "start_dt": start, "end_dt": end,
                           "duration_us": (end-start)*backend.dt*1e6})
        start = end
    record["stages"] = stage_rows
    record["syndrome_mapping_us"] = (record["readout_start_dt"]-boundaries["prepare"])*backend.dt*1e6
    record["syndrome_mapping_and_readout_us"] = record["syndrome_mapping_us"]+record["readout_duration_us"]
    final = compiled.layout.final_index_layout(filter_ancillas=True)
    data = tuple(final[:prep.n_data])
    syndrome = [None]*prep.n_checks
    for inst in compiled.data:
        if inst.operation.name == "measure":
            syndrome[compiled.find_bit(inst.clbits[0]).index] = compiled.find_bit(inst.qubits[0]).index
    # Verify semantic equivalence under ideal evolution; no noisy performance
    # estimate is made for this new, barrier-separated diagnostic circuit.
    scheduled = ScheduledPreparation(prep, compiled, data, tuple(syndrome), {})
    ideal, _ = simulate_scheduled(scheduled, backend, mode="ideal", shots=32, seed=seed)
    assert np.isclose(ideal["fidelity"], 1)
    assert np.isclose(ideal["acceptance"], 1/6 if strategy == "B2_BK" else 1)
    record["ideal_validation"] = {"fidelity": ideal["fidelity"], "acceptance": ideal["acceptance"]}
    return compiled, record, data, tuple(syndrome)


def draw_timeline(circuit, backend, record, data_wires, syndrome_wires, title, path,
                  staged=False, zoom=False, dd=False):
    end = record["readout_start_dt"] if zoom else record["total_dt"]
    window = (0, end)
    dt_us = backend.dt*1e6
    roles = {q: f"D{i}" for i, q in enumerate(data_wires)}
    roles.update({q: f"S{i}" for i, q in enumerate(syndrome_wires)})

    def labels(bit, formatter, program=None):
        q = circuit.find_bit(bit).index
        styles = {"zorder": formatter["layer.bit_name"], "color": "#27323f", "size": 10,
                  "va": "center", "ha": "right"}
        return [drawings.TextData(types.LabelType.BIT_NAME, types.AbstractCoordinate.LEFT, 0, bit,
                                 text=f"p{backend.physical_qubits[q]}  |  {roles.get(q, 'spare')}",
                                 styles=styles)]

    def axis_map(time_window):
        t0, t1 = time_window
        ticks = np.linspace(max(t0, 0), t1, 7)
        return types.HorizontalAxis(window=(t0, t1),
                                    axis_map={t: f"{t*dt_us:.2f}" for t in ticks},
                                    label="Time from circuit start (µs)")

    def gates(gate, formatter):
        # Virtual Rz updates have zero duration; omitting their overlapping
        # symbols preserves readable time bars without changing any timing.
        if gate.duration == 0:
            return []
        return generators.gen_sched_gate(gate, formatter)

    style = IQXStandard(**{
        "generator.gates": [gates], "generator.bits": [labels, generators.gen_timeslot],
        "layout.time_axis_map": axis_map, "formatter.color.gates": COLORS,
        "formatter.color.timeslot": "#f2f4f6", "formatter.alpha.timeslot": .3,
        "formatter.line_width.gate_link": .6, "formatter.alpha.gate_link": .45,
        "formatter.line_width.barrier": .8, "formatter.alpha.barrier": .6,
        "formatter.margin.left_percent": 0., "formatter.margin.right_percent": 0.,
        "formatter.box_height.gate": .55,
    })
    fig = plt.figure(figsize=(14.5, 7.8))
    grid = fig.add_gridspec(2, 1, height_ratios=[.62, 6.8], hspace=.10)
    phases = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])
    # The renderer calls tight_layout on its own. We supply a second phase axis
    # and explicitly lay out the whole figure below instead.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        timeline_drawer(circuit, target=backend.target, style=style, time_range=window,
                        show_clbits=False, idle_wires=True, plot_barriers=True,
                        show_delays=True, show_labels=False, axis=ax)
    ax.set_xlim(window)
    phases.set_xlim(window)
    phases.set_ylim(0, 1)
    phases.axis("off")
    if staged:
        intervals = [(r["start_dt"], r["end_dt"], "P" if r["stage"] == "prepare" else "S"+r["stage"].split("_")[1],
                      "#d6e5f2" if r["stage"] == "prepare" else "#dcefe9") for r in record["stages"]]
    else:
        intervals = [(0, record["readout_start_dt"], "Preparation + syndrome-mapping gates", "#d6e5f2")]
    if not zoom:
        readout_label = "Syndrome readout / DD on data" if dd else "Syndrome readout / data idle"
        intervals.append((record["readout_start_dt"], record["total_dt"], readout_label, "#f7e5c3"))
        ax.axvspan(record["readout_start_dt"], record["total_dt"], color="#fff3dc", zorder=-2)
    for left, right, name, color in intervals:
        phases.barh(.5, right-left, left=left, height=.95, color=color, edgecolor="white", linewidth=1.8)
        width_fraction = (right-left)/max(1, end)
        text = name if width_fraction < .09 else f"{name}\n{(right-left)*dt_us:.3f} µs"
        phases.text((left+right)/2, .5, text, ha="center", va="center", fontsize=9)
    if not zoom:
        ax.axvline(record["readout_start_dt"], color="#a66b13", linewidth=1.1, linestyle="--", zorder=6)
    fig.suptitle(title, fontsize=14, fontweight="medium", y=.985)
    legend = [Patch(facecolor=COLORS[k], label=label) for k, label in
              [("sx", "Native 1q gate"), ("cz", "CZ gate"), ("delay", "Delay (T1/T2 relaxation)"), ("measure", "Ancilla readout")]]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.54, .044), ncol=4, frameon=False, fontsize=10)
    foot = "Physical-wire labels; D/S denote data/syndrome roles at OUTPUT (routing moves states). Virtual Rz: zero duration, hidden."
    if dd:
        foot += "\nSaved DD circuit; original gate times and total duration preserved. Added X pulses carry calibrated gate noise. No analog waveforms available."
    elif staged:
        foot += "\nP = preparation. Extra stage barriers change compilation; this is a timing diagnostic, not the reported noisy run."
    else:
        foot += "\nExact saved noisy-run circuit; instruction start times reconstructed without changing gates or delays. No analog waveforms available."
    fig.text(.51, .013, foot, ha="center", va="bottom", fontsize=8.4, color="#465160")
    fig.subplots_adjust(left=.105, right=.985, top=.925, bottom=.17)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(Path(path).with_suffix("."+suffix), dpi=190)
    plt.close(fig)


def render_report(result):
    lines = ["# Aachen circuit timing with Qiskit", "",
             "These figures use Qiskit's `timeline_drawer` and the same calibrated instruction durations used in the noise model. "
             "They are gate/delay/readout timelines. The packaged Aachen snapshot contains no analog pulse envelopes, "
             "and Qiskit 2.4.1 does not include the removed Qiskit Pulse module. No pulse amplitudes or shapes are inferred.", "",
             "## Original schedules used for the noisy results", "",
             "Saved QPY circuits retain explicit delays, but not their start-time metadata. `ALAPScheduleAnalysis` reconstructs those timestamps. "
             "Instruction order, parameters, delay durations, total durations and per-wire idle totals are checked against the saved run; "
             "there is no recompilation or routing change.", "",
             "| Protocol | Preparation + coherent syndrome mapping (µs) | Syndrome readout / data wait (µs) | Total (µs) |",
             "|---|---:|---:|---:|"]
    for row in result["rows"]:
        r = row["original"]
        lines.append(f"| {row['code']}/{row['strategy']} | {r['coherent_phase_us']:.3f} | {r['readout_duration_us']:.3f} | {r['total_us']:.3f} |")
    lines += ["", "The first column includes input preparation and encoding where applicable; it must not be described as syndrome-extraction time alone. "
              "The optimized original circuits do not preserve source-stage boundaries. Gates can overlap or combine across those boundaries.", "",
              "The orange readout bars and the simultaneous gray data-wire delays show the interval during which the existing syndrome can become stale. "
              "Every data wire waits 2.584 µs. The timelines show all physical wires in the patch; D/S labels indicate their final roles, not fixed roles throughout routing.", "",
              "## Separate stage timing diagnostic", "",
              "To distinguish encoding, check 0, check 1, etc., a separate compilation adds barriers between those source stages. "
              "This forbids overlap/fusion across stages and can change gate synthesis and routing. "
              "Its times describe only this diagnostic compilation and are not substituted into the original noise results. "
              "Ideal fidelity and acceptance are checked for each diagnostic circuit.", "",
              "| Protocol | Preparation (µs) | Coherent syndrome mapping (µs) | Syndrome readout (µs) | Mapping + readout (µs) | Total (µs) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        r = row["stage_diagnostic"]
        lines.append(f"| {row['code']}/{row['strategy']} | {r['stages'][0]['duration_us']:.3f} | {r['syndrome_mapping_us']:.3f} | "
                     f"{r['readout_duration_us']:.3f} | {r['syndrome_mapping_and_readout_us']:.3f} | {r['total_us']:.3f} |")
    for row in result["rows"]:
        lines += ["", f"### {row['code']}/{row['strategy']}", "",
                  f"![Original full timeline]({row['code']}_{row['strategy']}_original.png)", "",
                  f"![Stage timing diagnostic]({row['code']}_{row['strategy']}_staged.png)", "",
                  "| Stage | Duration (µs) |", "|---|---:|"]
        for stage in row["stage_diagnostic"]["stages"]:
            lines.append(f"| {stage['stage']} | {stage['duration_us']:.3f} |")
    lines += ["", "PNG/PDF/SVG files include full schedules and gate-region zooms. `timing_summary.json` stores every instruction's timestamp "
              "and physical wires. Barrier-separated diagnostic QPY files are saved separately; original result files are untouched.", "",
              "Reproduce: `conda run -n quantum-jax python scripts/draw_timelines.py`.", "",
              "References: [Qiskit timeline visualization](https://quantum.cloud.ibm.com/docs/en/api/qiskit/visualization), "
              "[Qiskit 2.0 Pulse removal](https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.0).", ""]
    return "\n".join(lines)


def run_timelines(folder=ROOT / "results" / "noisy_aachen"):
    folder = Path(folder)
    output = folder / "timelines"
    output.mkdir(exist_ok=True)
    base = json.loads((folder / "noisy_results.json").read_text())
    backend = AachenPatch(base["backend"]["physical_patch"])
    for name, info in base["backend"]["snapshot_files"].items():
        assert info["sha256"] == backend.provenance()["snapshot_files"][name]["sha256"]
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "dt_seconds": backend.dt,
              "calibration_date": base["backend"]["calibration_date"], "environment": base["environment"],
              "visualizer": "qiskit.visualization.timeline_drawer", "analog_pulse_waveforms": False, "rows": []}
    for row in base["rows"]:
        if not row["strategy"].startswith("B2"):
            continue
        label = f"{row['code']}_{row['strategy']}"
        source = folder / "circuits" / f"{label}.qpy"
        original_hash = sha256(source.read_bytes()).hexdigest()
        with source.open("rb") as f:
            original = qpy.load(f)[0]
        circuit, record = restore_schedule(original, backend, row["resources"])
        for zoom in [False, True]:
            draw_timeline(circuit, backend, record, row["resources"]["final_data_local_wires"],
                          row["resources"]["syndrome_local_wires_in_bit_order"],
                          f"{row['code']}/{row['strategy']} — original Aachen schedule" + (" · gate-region zoom" if zoom else ""),
                          output / (label+"_original"+("_zoom" if zoom else "")), zoom=zoom)
        diagnostic, timing, data, syndrome = staged_diagnostic(CODES[row["code"]], row["strategy"], backend, base["seed"])
        for zoom in [False, True]:
            draw_timeline(diagnostic, backend, timing, data, syndrome,
                          f"{row['code']}/{row['strategy']} — stage timing diagnostic" + (" · gate-region zoom" if zoom else ""),
                          output / (label+"_staged"+("_zoom" if zoom else "")), staged=True, zoom=zoom)
        with (output / (label+"_stage_diagnostic.qpy")).open("wb") as f:
            qpy.dump(diagnostic, f)
        assert original_hash == sha256(source.read_bytes()).hexdigest()
        result["rows"].append({"code": row["code"], "strategy": row["strategy"], "source_qpy_sha256": original_hash,
                               "original": record, "stage_diagnostic": timing})
        print(label, "original µs", record["total_us"], "diagnostic syndrome mapping µs", timing["syndrome_mapping_us"], flush=True)
    (output / "timing_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (output / "timing_report.md").write_text(render_report(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "results" / "noisy_aachen")
    run_timelines(parser.parse_args().folder)
