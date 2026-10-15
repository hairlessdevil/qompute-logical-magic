#!/usr/bin/env python3
"""Full-noise tests of early readout on the original frozen native circuits."""

from datetime import datetime, timezone
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis

from analyze_readout import load_preparation
from logical_magic.aachen_noise import AachenPatch, schedule_preparation
from logical_magic.decoupling import timed_instructions
from logical_magic.syndrome_schedule import retime_syndrome_readout, simulate_early_readout

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [("original", "Original"), ("fixed", "Early readout; fixed gates"),
            ("asap", "Early readout; ASAP gates"), ("alap", "Early readout; ALAP gates")]


def summary_table(result):
    lines = ["| Protocol | Schedule | Delivery (µs) | Acceptance | Codespace population | Within-code fidelity | Full fidelity |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        for key, label in VARIANTS:
            variant = row["variants"][key]
            metrics = variant["metrics"]
            values = [f"{100*metrics[k]:.4f}%" for k in ["acceptance", "codespace_population",
                      "fidelity_conditioned_on_codespace_diagnostic", "fidelity"]]
            lines.append(f"| {row['code']}/{row['strategy']} | {label} | "
                         f"{variant['resources']['duration_seconds']*1e6:.3f} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def timing_table(result):
    lines = ["| Protocol | Schedule | Ancilla readout starts (µs, syndrome-bit order) | Delivery (µs) |",
             "|---|---|---|---:|"]
    for row in result["rows"]:
        for key, label in VARIANTS:
            variant = row["variants"][key]
            reads = sorted([r for r in variant["instructions"] if r["name"] == "measure"], key=lambda r: r["clbits"])
            starts = ", ".join(f"{r['start_dt']*result['backend']['dt_seconds']*1e6:.3f}" for r in reads)
            lines.append(f"| {row['code']}/{row['strategy']} | {label} | {starts} | "
                         f"{variant['resources']['duration_seconds']*1e6:.3f} |")
    return "\n".join(lines)


def check_order_table(result):
    lines = ["| Protocol | Check execution order | Delivery (µs) | Terminal-readout fidelity | Early-readout fidelity | Early-readout codespace | Early-readout acceptance |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["order_study"]:
        for key in ["reference", "reverse", "rotate"]:
            case = row["orders"][key]
            terminal, early = case["variants"]["terminal"], case["variants"]["early"]
            metrics = early["metrics"]
            lines.append(f"| {row['code']}/{row['strategy']} | {case['execution_order']} | "
                         f"{early['resources']['duration_seconds']*1e6:.3f} | "
                         f"{100*terminal['metrics']['fidelity']:.4f}% | {100*metrics['fidelity']:.4f}% | "
                         f"{100*metrics['codespace_population']:.4f}% | {100*metrics['acceptance']:.4f}% |")
    return "\n".join(lines)


def run_check_order_study(base, backend, folder, out, states):
    """Three whole-check permutations; fixed input preparation and ancilla IDs.

    These NEW compilations have their own matched reference. Cache the source
    to avoid different Clifford synthesis in subsequent processes/versions.
    """
    entries = []
    for row in base["rows"]:
        if not row["strategy"].startswith("B2"):
            continue
        label = f"{row['code']}_{row['strategy']}"
        prep = load_preparation(folder, row).preparation
        source_path = out/"circuits"/(label+"_order_source.qpy")
        stage_path = source_path.with_suffix(".json")
        if source_path.exists():
            with source_path.open("rb") as stream:
                source = qpy.load(stream)[0]
            prep = replace(prep, unitary=source, stages=tuple(json.loads(stage_path.read_text())["stages"]))
        else:
            with source_path.open("wb") as stream:
                qpy.dump(prep.unitary, stream)
            stage_path.write_text(json.dumps({"stages": prep.stages, "checks_in_bit_order": prep.checks}, indent=2)+"\n")
        assert len(prep.stages) == len(prep.unitary.data)
        stages = list(dict.fromkeys(s for s in prep.stages if s.startswith("check_")))
        entry = {"code": row["code"], "strategy": row["strategy"], "orders": {},
                 "source_sha256": sha256(source_path.read_bytes()).hexdigest(), "checks_in_bit_order": prep.checks}
        n = prep.n_checks
        for key, order in [("reference", list(range(n))), ("reverse", list(reversed(range(n)))),
                           ("rotate", [*range(1, n), 0])]:
            indices = [i for i, s in enumerate(prep.stages) if not s.startswith("check_")]
            for check in order:
                indices.extend(i for i, s in enumerate(prep.stages) if s == stages[check])
            assert sorted(indices) == list(range(len(prep.stages)))
            source = prep.unitary.copy_empty_like()
            for i in indices:
                source.append(prep.unitary.data[i])
            variant_prep = replace(prep, unitary=source, stages=tuple(prep.stages[i] for i in indices))
            terminal = schedule_preparation(variant_prep, backend, seed=base["seed"])
            early = retime_syndrome_readout(terminal, backend, "alap")
            case = {"execution_order": order, "execution_stabilizers": [prep.checks[i] for i in order], "variants": {}}
            for schedule_key, scheduled in [("terminal", terminal), ("early", early)]:
                ideal, _ = simulate_early_readout(scheduled, backend, mode="ideal", shots=1)
                assert np.isclose(ideal["fidelity"], 1, atol=1e-11, rtol=0)
                assert np.isclose(ideal["acceptance"], 1/6 if row["strategy"] == "B2_BK" else 1, atol=1e-11, rtol=0)
                metrics, state = simulate_early_readout(scheduled, backend, mode="full", shots=8192, seed=base["seed"])
                stem = label+"_order_"+key+"_"+schedule_key
                states[stem] = state
                with (out/"circuits"/(stem+".qpy")).open("wb") as stream:
                    qpy.dump(scheduled.circuit, stream)
                case["variants"][schedule_key] = {"metrics": metrics, "resources": scheduled.resources,
                                                  "ideal_validation": {"fidelity": ideal["fidelity"], "acceptance": ideal["acceptance"]}}
                if key == "reference" and schedule_key == "terminal":
                    entry["reference_vs_original_max_density_matrix_difference"] = float(
                        np.max(np.abs(state-states[label+"_original"])))
                print(stem, "F", metrics["fidelity"], "duration us", scheduled.resources["duration_seconds"]*1e6,
                      "CZ", scheduled.resources["two_qubit_gates"], flush=True)
            entry["orders"][key] = case
        entries.append(entry)
    return entries


def plot_timelines(row, circuits, backend, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from qiskit.visualization import timeline_drawer
    from qiskit.visualization.timeline import IQXStandard, drawings, generators, types

    colors = {"sx": "#238a89", "x": "#238a89", "cz": "#406bb6", "delay": "#d9dfe6",
              "measure": "#a66b13", "rz": "#292d33", "id": "#238a89"}
    def gates(gate, formatter):
        return [] if gate.duration == 0 else generators.gen_sched_gate(gate, formatter)
    roles = {q: f"D{i}" for i, q in enumerate(row["variants"]["original"]["resources"]["final_data_local_wires"])}
    roles.update({q: f"S{i}" for i, q in enumerate(row["variants"]["original"]["resources"]["syndrome_local_wires_in_bit_order"])})
    def labels(bit, formatter, program=None):
        q = circuits["original"].find_bit(bit).index
        styles = {"zorder": formatter["layer.bit_name"], "color": "#27323f", "size": 10,
                  "va": "center", "ha": "right"}
        return [drawings.TextData(types.LabelType.BIT_NAME, types.AbstractCoordinate.LEFT, 0, bit,
                                 text=f"p{backend.physical_qubits[q]} | {roles.get(q, 'spare')}", styles=styles)]
    def axis_map(time_window):
        ticks = np.linspace(0, time_window[1], 7)
        return types.HorizontalAxis(window=time_window, axis_map={t: f"{t*backend.dt*1e6:.2f}" for t in ticks}, label="Time (µs)")
    style = IQXStandard(**{"generator.gates": [gates], "formatter.color.gates": colors,
                           "generator.bits": [labels, generators.gen_timeslot], "layout.time_axis_map": axis_map,
                           "formatter.margin.left_percent": 0., "formatter.margin.right_percent": 0.,
                           "formatter.line_width.gate_link": .5, "formatter.alpha.gate_link": .4})
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    total = round(row["variants"]["original"]["resources"]["duration_seconds"]/backend.dt)
    # The ALAP variant keeps early ASAP readout deadlines, then delays gates
    # where possible. Show exactly this schedule, even if it is not the best F.
    for ax, key in zip(axes, ["original", "alap"]):
        circuit = circuits[key]
        circuit.duration = round(row["variants"][key]["resources"]["duration_seconds"]/backend.dt)
        circuit.unit = "dt"
        timeline_drawer(circuit, target=backend.target, style=style,
                        show_clbits=False, show_labels=False, show_delays=True,
                        plot_barriers=False, idle_wires=True, time_range=(0, total), axis=ax)
        ax.set_xlim(0, total)
        ax.set_xlabel("Time (µs)")
        ax.set_title("Original: simultaneous terminal readout" if key == "original"
                     else "Early ancilla readout; gates delayed subject to those readout deadlines", fontsize=11)
    fig.suptitle(f"{row['code']}/{row['strategy']} · same native gates and physical mapping", fontsize=14)
    fig.legend(handles=[Patch(facecolor=colors[k], label=label) for k, label in
                        [("sx", "Native 1q gate"), ("cz", "CZ gate"), ("delay", "Idle"), ("measure", "Ancilla readout")]],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .035))
    fig.text(.5, .009, "Physical-qubit labels; D/S are data/syndrome roles at output. Qiskit instruction timelines, not analog pulses.\n"
             "Overlap assumes independent readout. Post-readout ancilla idles cannot alter stored syndrome bits.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.93, bottom=.14, hspace=.35, left=.105, right=.98)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out/f"{row['code']}_{row['strategy']}_timelines.{ext}", dpi=170)
    plt.close(fig)


def render_report(result):
    ed = next(r for r in result["order_study"] if r["code"] == "five" and r["strategy"] == "B2_ED")
    ref, rev, rot = [ed["orders"][k]["variants"]["terminal"] for k in ["reference", "reverse", "rotate"]]
    bk = next(r for r in result["rows"] if r["strategy"] == "B2_BK")
    bk_ref, bk_early = bk["variants"]["original"], bk["variants"]["alap"]
    findings = (
        f"For five-qubit ED, the reversed order with ordinary terminal readout changes fidelity from "
        f"{100*ref['metrics']['fidelity']:.4f}% to {100*rev['metrics']['fidelity']:.4f}% and delivery from "
        f"{ref['resources']['duration_seconds']*1e6:.3f} to {rev['resources']['duration_seconds']*1e6:.3f} µs. "
        f"Acceptance changes from {100*ref['metrics']['acceptance']:.4f}% to {100*rev['metrics']['acceptance']:.4f}%. "
        f"The rotated order gives {100*rot['metrics']['fidelity']:.4f}% fidelity at "
        f"{100*rot['metrics']['acceptance']:.4f}% acceptance and {rot['resources']['duration_seconds']*1e6:.3f} µs delivery. "
        f"For BK, original-order early deadline-based ALAP gives {100*bk_early['metrics']['fidelity']:.4f}% fidelity "
        f"versus {100*bk_ref['metrics']['fidelity']:.4f}%, at unchanged delivery. "
        "These are candidate results, not proof of an optimum or isolated idle-error contributions: recompilation "
        "changes gate channels and physical mappings as well as timing. Acceptance must be considered alongside fidelity.")
    return "\n".join([
        "# Earlier syndrome readout on frozen Aachen circuits", "",
        "## What changed", "",
        "Four schedules use each original optimized native QPY circuit. There is no resynthesis, new routing, "
        "check reordering, qubit reassignment, DD, or gate removal. Per-wire native operation order, physical mappings, "
        "gate parameters and global phase are preserved. The original replay is checked against the saved full-noise density matrix.", "",
        "1. **Original:** the existing terminal barrier aligns all syndrome measurements.\n"
        "2. **Early readout; fixed gates:** remove that synchronization and measure each physical ancilla after its last gate, "
        "without moving any native gate.\n"
        "3. **Early readout; ASAP gates:** schedule native gates as early as per-wire dependencies permit, then measure each ancilla.\n"
        "4. **Early readout; ALAP gates:** retain the ASAP measurement times as deadlines, then move gates as late as possible "
        "without missing those deadlines or delivery. This differs from ordinary ALAP with all measurements aligned.", "",
        "## Full-noise result", "", summary_table(result), "",
        "All fidelities and codespace populations describe the data state **at delivery, after every syndrome readout finishes**, "
        "conditioned on recorded all-zero syndrome. Within-code fidelity is an extra diagnostic, not another physical filter. "
        "Gate depolarization, gate-duration relaxation, pre-measurement ancilla idles, all data idles, and assignment errors remain enabled. "
        "State metrics are exact density-matrix calculations; the saved 8,192 syndrome shots per variant check acceptance, "
        "not a new finite-shot fidelity estimate.", "",
        "## Timing and interpretation", "", timing_table(result), "",
        "Earlier readout protects a finished ancilla's recorded syndrome from further ancilla storage errors. It does not recheck "
        "data errors that happen afterward. For these frozen gate dependencies, the final ancilla is still on the critical path: "
        "all four schedules have the same overall delivery time. Thus early readout alone does not remove the final data-storage burden. "
        "With fixed gate times, an unchanged unconditional data fidelity verifies that any accepted-state change comes from altered "
        "syndrome selection, not less data decoherence. ASAP also moves preparation earlier and can increase exposure of fragile states. "
        "The deadline-based ALAP schedule can move independent data-only gates into readout intervals; those gates and all intervening "
        "idles retain their calibrated noise.", "",
        "## Exact measurement handling and validation", "",
        "Each ancilla is measured once and is never reused. Its measurement commutes with every later operation on other wires. "
        "For exact syndrome-block evaluation, measurements are deferred only after dropping idles on already measured ancillas: "
        "their subsequent quantum relaxation cannot rewrite a classical bit. All data evolution is retained. Any later gate on a measured "
        "ancilla is rejected by the implementation. A Bell-pair test checks this distinction analytically while retaining data T1 decay. "
        "Every schedule retains ideal fidelity one and ideal acceptance one (ED) or one sixth (BK). Source QPY and calibration hashes "
        "are verified; per-wire operation order and absence of overlapping operations on a wire are asserted.", "",
        "## Three representative check orders: separate matched compilation study", "",
        "This additional study recompiles one frozen source preparation with the reference, reversed, and one-step-rotated "
        "whole-check order. The preparation/encoding gates and assignment of a stabilizer to its source ancilla are identical "
        "within each protocol. No arbitrary interleaving of controlled-Pauli factors is attempted: commuting stabilizers do not "
        "justify arbitrarily commuting all their elementary gates. Native routing, gate count, and final physical placement can "
        "change when the order changes. Optimization level 1, initial layout, calibration, and transpiler seed stay fixed.", "",
        "The newly compiled reference order is the matched baseline for this table; it can differ from the original saved "
        "QPY because the original source encoder was not frozen at that time. Source QPY and stage metadata are now cached "
        "for exact reproduction. Check indices refer to the saved `checks_in_bit_order` list; only execution order changes, "
        "not the recorded-zero acceptance rule. Each order has both simultaneous terminal readout and early deadline-based "
        "ALAP readout, with full noise retained. Results are a small candidate comparison, not a global optimization.", "",
        check_order_table(result), "",
        findings, "",
        "## Hardware scope and ordinary versus mid-circuit measurement", "",
        "Ordinary `measure` and IBM `MidCircuitMeasure` implement the same logical computational-basis measurement. "
        "A mid-circuit measurement means a measurement while quantum computation continues; ordinary `measure` can be used there too. "
        "IBM's `MidCircuitMeasure` maps to `measure_2`, an implementation tuned for lower overhead during ongoing computation, "
        "whereas terminal-readout tuning need not prioritize keeping other quantum data alive. Actual durations and errors depend on "
        "the backend calibration. The packaged Aachen snapshot exposes only `measure`, with duration 2.584 µs on this patch. "
        "No faster measurement calibration is invented or substituted.", "",
        "These are candidate **local-model** schedules. The Target supplies per-instruction durations and alignment, but the model "
        "does not establish all controller restrictions on concurrent readout and gates, measurement groups, measurement-induced "
        "dephasing or crosstalk. Consequently native instructions and valid per-wire timing do not by themselves certify that hardware "
        "will execute these overlaps unchanged. No IBM job or live calibration request is made.", "",
        *[f"![{r['code']}/{r['strategy']} schedules]({r['code']}_{r['strategy']}_timelines.png)\n" for r in result["rows"]],
        "Reproduce: `conda run -n quantum-jax python scripts/analyze_syndrome_schedules.py`. "
        "Outputs: `results.json`, `accepted_states.npz`, native circuits in `circuits/`, and Qiskit timelines.", "",
        "Sources: [IBM measurement instructions](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits), "
        "[IBM transpilation and scheduling](https://quantum.cloud.ibm.com/docs/en/guides/transpiler-stages).", "",
    ])


def run_analysis(folder=ROOT/"results/noisy_aachen"):
    folder = Path(folder)
    out = folder/"syndrome_schedules"
    (out/"circuits").mkdir(parents=True, exist_ok=True)
    base = json.loads((folder/"noisy_results.json").read_text())
    backend = AachenPatch(base["backend"]["physical_patch"])
    for name, info in base["backend"]["snapshot_files"].items():
        assert backend.provenance()["snapshot_files"][name]["sha256"] == info["sha256"]
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "backend": base["backend"],
              "environment": base["environment"], "noise_mode": "full", "seed": base["seed"], "rows": []}
    states = {}
    with np.load(folder/"accepted_states.npz") as saved:
        for row in base["rows"]:
            if not row["strategy"].startswith("B2"):
                continue
            label = f"{row['code']}_{row['strategy']}"
            source = folder/"circuits"/(label+".qpy")
            source_hash = sha256(source.read_bytes()).hexdigest()
            original = load_preparation(folder, row)
            original.circuit = PassManager([ALAPScheduleAnalysis(target=backend.target)]).run(original.circuit)
            entry = {"code": row["code"], "strategy": row["strategy"], "source_qpy_sha256": source_hash, "variants": {}}
            circuits = {}
            for key, _ in VARIANTS:
                scheduled = original if key == "original" else retime_syndrome_readout(original, backend, key)
                ideal, _ = simulate_early_readout(scheduled, backend, mode="ideal", shots=1)
                assert np.isclose(ideal["fidelity"], 1, atol=1e-11, rtol=0)
                assert np.isclose(ideal["acceptance"], 1/6 if row["strategy"] == "B2_BK" else 1, atol=1e-11, rtol=0)
                metrics, state = simulate_early_readout(scheduled, backend, mode="full", shots=8192, seed=base["seed"])
                if key == "original":
                    assert np.allclose(state, saved[label+"_full"], atol=1e-11, rtol=0)
                    assert np.isclose(metrics["acceptance"], row["full"]["acceptance"], atol=1e-11, rtol=0)
                if key == "fixed":
                    assert np.isclose(metrics["unconditional_fidelity"], row["full"]["unconditional_fidelity"], atol=1e-11, rtol=0)
                entry["variants"][key] = {"metrics": metrics, "resources": scheduled.resources,
                                           "instructions": timed_instructions(scheduled.circuit, backend),
                                           "ideal_validation": {"fidelity": ideal["fidelity"], "acceptance": ideal["acceptance"]}}
                states[label+"_"+key] = state
                circuits[key] = scheduled.circuit
                with (out/"circuits"/(label+"_"+key+".qpy")).open("wb") as stream:
                    qpy.dump(scheduled.circuit, stream)
                print(label, key, "F", metrics["fidelity"], "Pcode", metrics["codespace_population"],
                      "accept", metrics["acceptance"], "duration us", scheduled.resources["duration_seconds"]*1e6, flush=True)
            result["rows"].append(entry)
            assert sha256(source.read_bytes()).hexdigest() == source_hash
            plot_timelines(entry, circuits, backend, out)
    result["order_study"] = run_check_order_study(base, backend, folder, out, states)
    np.savez_compressed(out/"accepted_states.npz", **states)
    (out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    (out/"report.md").write_text(render_report(result))
    return result


if __name__ == "__main__":
    run_analysis()
