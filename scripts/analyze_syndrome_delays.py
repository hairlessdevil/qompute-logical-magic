#!/usr/bin/env python3
"""Matched delay-channel ablations on saved circuits with source-stage barriers."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from qiskit import qpy
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logical_magic import CODES, build_preparation
from logical_magic.aachen_noise import AachenPatch, ScheduledPreparation, simulate_scheduled
from logical_magic.delay_ablation import simulate_delay_ablation
from draw_timelines import timing_record

MODES = {
    "full": "Full noise",
    "readout_delay_off": "Readout-window delay noise off",
    "extraction_delay_off": "Extraction + readout delay noise off",
    "all_delay_off": "All delay noise off (control)",
}


def comparison_table(result):
    lines = ["| Protocol | Codespace population | Fidelity within codespace | Full target fidelity | Acceptance |",
             "|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        a, b = row["modes"]["full"], row["modes"]["extraction_delay_off"]
        values = [f"{100*a[k]:.3f}% → {100*b[k]:.3f}%" for k in
                  ["codespace_population", "fidelity_conditioned_on_codespace_diagnostic", "fidelity", "acceptance"]]
        lines.append(f"| {row['code']}/{row['strategy']} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def modes_table(result):
    lines = ["| Protocol | Model | Codespace population | Fidelity within codespace | Full target fidelity | Acceptance |",
             "|---|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        for key, name in MODES.items():
            r = row["modes"][key]
            values = [f"{100*r[k]:.3f}%" for k in
                      ["codespace_population", "fidelity_conditioned_on_codespace_diagnostic", "fidelity", "acceptance"]]
            lines.append(f"| {row['code']}/{row['strategy']} | {name} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(result):
    changes = []
    for row in result["rows"]:
        a, b = row["modes"]["full"], row["modes"]["extraction_delay_off"]
        changes.append(f"{row['code']}/{row['strategy']}: codespace population {100*(b['codespace_population']-a['codespace_population']):+.3f} percentage points; "
                       f"within-code fidelity {100*(b['fidelity_conditioned_on_codespace_diagnostic']-a['fidelity_conditioned_on_codespace_diagnostic']):+.3f} percentage points")
    lines = ["# Disable delay noise only during syndrome extraction", "",
             "## Matched experiment and phase definition", "",
             "The original optimized noisy circuits have no preserved preparation/extraction boundary. This experiment therefore uses the "
             "previously saved stage-separated diagnostic QPY circuits from the timeline study. **Their full-noise baseline is recomputed here; "
             "it is different from the original noisy-result baseline.** Every comparison below uses the same fixed circuit, routing, gates, "
             "delays, and measurement duration within that row. No new transpilation is performed.", "",
             "The extraction window starts at the barrier after input preparation/encoding and ends at the completion of terminal syndrome readout. "
             "Delay T1/T2 relaxation is disabled on all physical wires within this window, including ancilla idles and the data wait during readout. "
             "Preparation-phase delay relaxation, gate depolarization, gate-duration relaxation, and syndrome readout assignment errors remain enabled. "
             "This disables only scheduled Delay channels; it does not remove gate-duration noise or alter calibration times.", "",
             "All Delay instructions remain in the simulation. Their retained relaxation is applied explicitly once, using the same thermal relaxation "
             "parameters as Aer. The calibrated gate/readout model uses Aer's public device helpers without adding its automatic Delay pass. "
             "Delays crossing a boundary retain noise only for the portion outside the disabled window. This is an idealized channel-removal experiment; no DD pulses are added.", "",
             "## Requested quantities", "",
             r"For the state $\rho_A$ conditioned on the **recorded** zero syndrome and codespace projector $P$,", "",
             r"$$p_{\rm code}=\mathrm{Tr}(P\rho_A),\quad \rho_{\rm code}=P\rho_A P/p_{\rm code},\quad F_{\rm code}=\langle\psi_L|\rho_{\rm code}|\psi_L\rangle.$$", "",
             r"The full accepted-state fidelity obeys $F=p_{\rm code}F_{\rm code}$. Codespace conditioning is a diagnostic, not an extra physical acceptance step.", "",
             "Arrows below mean **full noise → delay noise off during extraction and readout**, on the same stage-separated circuit.", "",
             comparison_table(result), "", "Changes in the requested quantities: " + "; ".join(changes) + ".", "",
             "## Readout-only and all-delay controls", "", modes_table(result), "",
             "Readout-only suppression isolates the final 2.584 µs wait. Its exact acceptance is unchanged, because it affects data after mapping "
             "and no ancilla is waiting during its own simultaneous terminal readout. Suppression throughout extraction can also change syndrome probabilities. "
             "The all-delay control additionally removes preparation idles and is distinct from the requested extraction-only experiment.", "",
             "## Timing and retained preparation noise", "",
             "| Protocol | Extraction start (µs) | Readout start (µs) | Delivery (µs) | Preparation idle duration retaining noise (sum over wires, µs) |",
             "|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        t = row["timing"]
        audit = row["modes"]["extraction_delay_off"]["delay_audit"]
        retained = sum(d["noise_retained_dt"] for d in audit)*result["dt_seconds"]*1e6
        lines.append(f"| {row['code']}/{row['strategy']} | {t['extraction_start_us']:.3f} | {t['readout_start_us']:.3f} | {t['delivery_us']:.3f} | {retained:.3f} |")
    lines += ["", "## Validation and artifacts", "",
              "Each explicit-channel full-noise accepted density matrix and syndrome distribution agrees with the existing automatic-delay model "
              "to absolute tolerance 1e-11. The all-delay-off control agrees with the existing no-delay mode. Ideal references retain fidelity one "
              "and the expected acceptance. Original instructions are checked unchanged, and per-delay active/suppressed durations are recorded in JSON. "
              "All reported state quantities are exact density-matrix diagnostics, with no finite-shot fidelity estimator involved. "
              "8,192 sampled syndromes per mode are saved separately as an acceptance check.", "",
              "Files: `results.json`, `accepted_states.npz`, and `circuits/*_stage_diagnostic.qpy` in this folder. "
              "The frozen circuit copies have saved SHA-256 hashes. The original noisy-result circuits and the timeline-study inputs remain unchanged.", "",
              "Reproduce: `conda run -n quantum-jax python scripts/analyze_syndrome_delays.py`.", "",
              "Channel definition: [Aer thermal_relaxation_error](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.thermal_relaxation_error.html).", ""]
    return "\n".join(lines)


def run_analysis(folder=ROOT / "results" / "noisy_aachen"):
    folder = Path(folder)
    output = folder / "syndrome_delay_ablation"
    (output / "circuits").mkdir(parents=True, exist_ok=True)
    base = json.loads((folder / "noisy_results.json").read_text())
    backend = AachenPatch(base["backend"]["physical_patch"])
    for name, info in base["backend"]["snapshot_files"].items():
        assert info["sha256"] == backend.provenance()["snapshot_files"][name]["sha256"]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "backend": base["backend"],
              "environment": base["environment"], "seed": base["seed"], "dt_seconds": backend.dt,
              "circuit_family": "saved stage-separated timing diagnostics; matched full-noise baseline",
              "disabled_scope": "Delay relaxation on all physical wires after preparation through readout completion",
              "rows": []}
    states = {}
    for row in base["rows"]:
        if not row["strategy"].startswith("B2"):
            continue
        label = f"{row['code']}_{row['strategy']}"
        filename = label + "_stage_diagnostic.qpy"
        source = output / "circuits" / filename
        if not source.exists():
            source.write_bytes((folder / "timelines" / filename).read_bytes())
        source_hash = sha256(source.read_bytes()).hexdigest()
        with source.open("rb") as stream:
            raw = qpy.load(stream)[0]
        circuit = PassManager([ALAPScheduleAnalysis(target=backend.target)]).run(raw)
        assert list(circuit.data) == list(raw.data)
        timeline = timing_record(circuit, backend)
        boundary = next(r["start_dt"] for r in timeline["instructions"] if r["label"] == "end:prepare")
        readout, end = timeline["readout_start_dt"], timeline["total_dt"]
        prep = build_preparation(CODES[row["code"]], row["strategy"])
        # Analysis-only PassManager drops layout metadata; wire instructions are
        # unchanged, so use the final logical-to-physical mapping from the QPY.
        data = tuple(raw.layout.final_index_layout(filter_ancillas=True)[:prep.n_data])
        syndrome = [None] * prep.n_checks
        for inst in circuit.data:
            if inst.operation.name == "measure":
                syndrome[circuit.find_bit(inst.clbits[0]).index] = circuit.find_bit(inst.qubits[0]).index
        scheduled = ScheduledPreparation(prep, circuit, data, tuple(syndrome), {})
        ideal, _ = simulate_scheduled(scheduled, backend, mode="ideal", shots=1)
        assert np.isclose(ideal["fidelity"], 1)
        assert np.isclose(ideal["acceptance"], 1/6 if row["strategy"] == "B2_BK" else 1)
        entry = {"code": row["code"], "strategy": row["strategy"], "source_qpy_sha256": source_hash,
                 "final_data_local_wires": data, "syndrome_local_wires": syndrome,
                 "timing": {"extraction_start_us": boundary*backend.dt*1e6,
                            "readout_start_us": readout*backend.dt*1e6, "delivery_us": end*backend.dt*1e6},
                 "modes": {}, "validation": {}}
        for mode, window in [("full", None), ("readout_delay_off", (readout, end)),
                             ("extraction_delay_off", (boundary, end)), ("all_delay_off", (0, end))]:
            metrics, state = simulate_delay_ablation(scheduled, backend, window, seed=base["seed"])
            entry["modes"][mode] = metrics
            states[label + "_" + mode] = state
            assert np.isclose(metrics["fidelity"], metrics["codespace_population"] * metrics["fidelity_conditioned_on_codespace_diagnostic"])
            if mode in {"full", "all_delay_off"}:
                old, old_state = simulate_scheduled(scheduled, backend, mode="full" if mode == "full" else "no_delay", shots=1)
                assert np.allclose(state, old_state, atol=1e-11, rtol=0)
                assert np.allclose(metrics["recorded_syndrome_probabilities"], old["recorded_syndrome_probabilities"], atol=1e-11, rtol=0)
                entry["validation"][mode + "_max_state_difference"] = float(np.max(np.abs(state - old_state)))
            if mode == "extraction_delay_off":
                for delay in metrics["delay_audit"]:
                    assert delay["noise_retained_dt"] == max(0, min(delay["start_dt"] + delay["duration_dt"], boundary) - delay["start_dt"])
            print(label, mode, {k: round(metrics[k], 8) for k in
                  ["acceptance", "codespace_population", "fidelity_conditioned_on_codespace_diagnostic", "fidelity"]}, flush=True)
        assert np.isclose(entry["modes"]["full"]["acceptance"], entry["modes"]["readout_delay_off"]["acceptance"], atol=1e-11, rtol=0)
        assert source_hash == sha256(source.read_bytes()).hexdigest()
        report["rows"].append(entry)
    np.savez_compressed(output / "accepted_states.npz", **states)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(render_report(report))
    return report


if __name__ == "__main__":
    run_analysis()
