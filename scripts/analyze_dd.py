#!/usr/bin/env python3
"""DD comparison on the ORIGINAL saved Aachen circuits, with full noise restored."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from qiskit import qpy

from analyze_readout import load_preparation
from logical_magic.aachen_noise import AachenPatch, simulate_scheduled
from logical_magic.decoupling import insert_dd, markovian_echo_sanity
from draw_timelines import draw_timeline, restore_schedule

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [("no_dd", 0), ("xx", 2), ("x4", 4), ("x8", 8)]


def fidelity_table(result):
    lines = ["| Protocol | No DD | 2 X / eligible idle | 4 X / eligible idle | 8 X / eligible idle |",
             "|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        values = [f"{100*row['variants'][key]['metrics']['fidelity']:.4f}%" for key, _ in VARIANTS]
        lines.append(f"| {row['code']}/{row['strategy']} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def quality_table(result):
    lines = ["| Protocol | DD pulses per eligible idle | Added X gates | Protected idle windows | Codespace population | Within-code fidelity | Full fidelity | Acceptance |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        for key, n in VARIANTS:
            variant = row["variants"][key]
            dd = variant["resources"].get("dd", {})
            metrics = variant["metrics"]
            values = [f"{100*metrics[k]:.4f}%" for k in ["codespace_population", "fidelity_conditioned_on_codespace_diagnostic", "fidelity", "acceptance"]]
            lines.append(f"| {row['code']}/{row['strategy']} | {n} | {dd.get('added_pulse_count', 0)} | {dd.get('protected_idle_count', 0)} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_quality(result, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), layout="constrained")
    panels = [("fidelity", "Full accepted-state fidelity"), ("codespace_population", "Accepted codespace population"),
              ("fidelity_conditioned_on_codespace_diagnostic", "Fidelity within code subspace"), ("acceptance", "Acceptance probability")]
    for ax, (key, title) in zip(axes.flat, panels):
        for row, color in zip(result["rows"], ["#19777a", "#4264b0", "#b66f23"]):
            ax.plot([n for _, n in VARIANTS], [row["variants"][v]["metrics"][key] for v, _ in VARIANTS],
                    "o-", color=color, label=f"{row['code']}/{row['strategy']}", markersize=5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([0, 2, 4, 8])
        ax.set_xlabel("X pulses per eligible idle (0 = no DD)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
    axes[1, 1].legend(fontsize=9, loc="best")
    fig.suptitle("Full Aachen noise · original schedules with native DD pulses\n"
                 "All pulse errors included; original gate times and total durations preserved", fontsize=13)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(folder / ("comparison." + ext), dpi=180)
    plt.close(fig)


def plot_saved_timelines(result, backend, folder):
    for row in result["rows"]:
        label = f"{row['code']}_{row['strategy']}"
        resources = row["variants"]["x8"]["resources"]
        with (folder / "circuits" / (label + "_x8.qpy")).open("rb") as stream:
            raw = qpy.load(stream)[0]
        circuit, record = restore_schedule(raw, backend, resources)
        draw_timeline(circuit, backend, record, resources["final_data_local_wires"],
                      resources["syndrome_local_wires_in_bit_order"],
                      f"{row['code']}/{row['strategy']} — 8 X DD pulses per eligible idle",
                      folder / (label + "_x8_timeline"), dd=True)


def render_report(result):
    lines = ["# Dynamical decoupling with the full Aachen noise model", "",
             "## Baseline and implementation", "",
             "This comparison returns to the **original optimized saved QPY circuits**, rather than the stage-separated circuits used for the "
             "phase-ablation studies. The no-DD full-noise accepted density matrix is replayed and checked against the original saved result. "
             "Every experiment calls `simulate_scheduled(..., mode='full')`: gate depolarization, gate-duration T1/T2 relaxation, all remaining "
             "Delay relaxation, and syndrome assignment error are enabled. No earlier channel-removal setting is used.", "",
             "Qiskit's `PadDynamicalDecoupling` inserts balanced sequences of 2, 4, or 8 native X gates into sufficiently long idles. "
             "Initial ground-state idles are skipped. The backend gives each X gate a 32 ns duration and its calibrated noise channel. "
             "Added pulses keep their native names with no custom labels, so ordinary Aachen X-gate noise applies. Remaining gaps remain noisy. "
             "There is no synthesis, routing change, or optimization after DD insertion.", "",
             "The original gates and measurements retain exactly their prior start times. The pulse sequences fit inside the original delays, "
             "and total duration is unchanged. All final data wires receive the chosen sequence during the 2.584 µs syndrome-readout wait. "
             "Higher pulse counts may cause shorter idle windows to be skipped, so these settings specify pulses per eligible window, "
             "not a uniform pulse repetition rate across the circuit. Actual pulse counts and timestamps are saved.", "",
             "## Full accepted-state fidelity", "", fidelity_table(result), "",
             "![Full-noise DD comparison](comparison.png)", "", "## State quality and pulse cost", "", quality_table(result), "",
             "All state quantities are exact density-matrix diagnostics conditioned on recorded-zero syndrome. "
             "The within-code fidelity additionally normalizes the projected state for diagnosis; it is not another physical selection. "
             "8,192 syndrome shots per variant are saved as an independent acceptance check, not a finite-shot fidelity estimator.", "",
             "## What this noise model can say about DD", "",
             "The calibrated idle channel is a memoryless, exponential T1/T2 relaxation channel. It has no explicit slow detuning, "
             "noise spectrum, temporal correlations, or coherent idle ZZ coupling. Echo pulses refocus coherent or sufficiently slow "
             "phase accumulation; this mechanism is absent from the current model. More pulses add calibrated gate noise and can reshape "
             "amplitude-damping bias, but they do not reverse this model's Markovian transverse decay. Consequently this test is not a "
             "prediction of DD effectiveness on Aachen hardware. The calibration snapshot alone does not determine the temporal correlations "
             "needed for that prediction.", "",
             "The stored one-qubit sanity check uses ideal instantaneous pulses on physical qubit " + str(result["echo_sanity"]["physical_qubit"]) +
             ": over 2.584 µs, X coherence is " + f"{result['echo_sanity']['expected_markovian_x_coherence']:.8f}" +
             " for no DD and for all three DD counts, matching exp(-t/T2). A separate constant-Z-detuning toy check refocuses exactly. "
             "That toy is a validation of the echo mechanism only; no detuning is added to the Aachen experiment and no hardware detuning strength is assumed.", "",
             "Before using this simulation to optimize DD, a separate calibrated or explicitly parameterized model of correlated idle noise is needed. "
             "It must distinguish refocusable dephasing from irreversible relaxation rather than treating the previous no-delay ablation as achievable DD performance.", "",
             "## Timing plots and reproducibility", ""]
    for row in result["rows"]:
        label = f"{row['code']}_{row['strategy']}"
        lines += [f"### {row['code']}/{row['strategy']} with 8 X pulses per eligible idle", "",
                  f"![DD timeline]({label}_x8_timeline.png)", ""]
    lines += ["Results include `results.json`, `accepted_states.npz`, comparison PNG/PDF/SVG, DD timelines, and native QPY circuits in `circuits/`. "
              "The original QPY hashes are unchanged. Ideal accepted-state fidelity and acceptance are checked for all inserted sequences. "
              "Reproduce: `conda run -n quantum-jax python scripts/analyze_dd.py`.", "",
              "Sources: [Qiskit DD padding](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.PadDynamicalDecoupling), "
              "[IBM explanation of echo and slow noise](https://quantum.cloud.ibm.com/learning/en/courses/tools-for-quantum-advantage/error-suppression), "
              "[Markovian limits of DD](https://arxiv.org/abs/1710.11133).", ""]
    return "\n".join(lines)


def run_analysis(folder=ROOT / "results" / "noisy_aachen"):
    folder = Path(folder)
    out = folder / "dd_full_noise"
    (out / "circuits").mkdir(parents=True, exist_ok=True)
    base = json.loads((folder / "noisy_results.json").read_text())
    backend = AachenPatch(base["backend"]["physical_patch"])
    for name, info in base["backend"]["snapshot_files"].items():
        assert info["sha256"] == backend.provenance()["snapshot_files"][name]["sha256"]
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "backend": base["backend"],
              "environment": base["environment"], "seed": base["seed"], "noise_mode": "full",
              "circuit_family": "original optimized saved QPY circuits", "echo_sanity": markovian_echo_sanity(backend), "rows": []}
    states = {}
    with np.load(folder / "accepted_states.npz") as saved:
        for row in base["rows"]:
            if not row["strategy"].startswith("B2"):
                continue
            label = f"{row['code']}_{row['strategy']}"
            source = folder / "circuits" / (label + ".qpy")
            source_hash = sha256(source.read_bytes()).hexdigest()
            original = load_preparation(folder, row)
            entry = {"code": row["code"], "strategy": row["strategy"], "source_qpy_sha256": source_hash, "variants": {}}
            for key, n in VARIANTS:
                scheduled = original if n == 0 else insert_dd(original, backend, n)
                ideal, _ = simulate_scheduled(scheduled, backend, mode="ideal", shots=32)
                assert np.isclose(ideal["fidelity"], 1, atol=1e-11, rtol=0)
                assert np.isclose(ideal["acceptance"], 1/6 if row["strategy"] == "B2_BK" else 1, atol=1e-11, rtol=0)
                metrics, state = simulate_scheduled(scheduled, backend, mode="full", shots=8192, seed=base["seed"])
                if n == 0:
                    assert np.allclose(state, saved[label + "_full"], atol=1e-11, rtol=0)
                    assert np.isclose(metrics["acceptance"], row["full"]["acceptance"], atol=1e-11, rtol=0)
                entry["variants"][key] = {"metrics": metrics, "resources": scheduled.resources,
                                           "ideal_validation": {"fidelity": ideal["fidelity"], "acceptance": ideal["acceptance"]}}
                states[label + "_" + key] = state
                with (out / "circuits" / (label + "_" + key + ".qpy")).open("wb") as f:
                    qpy.dump(scheduled.circuit, f)
                print(label, key, "F", metrics["fidelity"], "Pcode", metrics["codespace_population"],
                      "added pulses", scheduled.resources.get("dd", {}).get("added_pulse_count", 0), flush=True)
            assert sha256(source.read_bytes()).hexdigest() == source_hash
            result["rows"].append(entry)
    np.savez_compressed(out / "accepted_states.npz", **states)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_quality(result, out)
    plot_saved_timelines(result, backend, out)
    (out / "report.md").write_text(render_report(result))
    return result


if __name__ == "__main__":
    run_analysis()
