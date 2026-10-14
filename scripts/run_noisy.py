#!/usr/bin/env python3
"""Run scheduled local Aer simulations using the packaged Aachen calibration."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from qiskit import QuantumCircuit, qpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logical_magic import CODES, build_preparation
from logical_magic.circuits import Preparation, check_block
from logical_magic.aachen_noise import (
    AachenPatch, DEFAULT_SEED, schedule_preparation, simulate_scheduled,
    validate_idle_relaxation,
)


def run_experiment(output=ROOT / "results" / "noisy_aachen", shots=8192, seed=DEFAULT_SEED):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "circuits").mkdir(exist_ok=True)
    (output / "calibration").mkdir(exist_ok=True)
    backend = AachenPatch()
    provenance = backend.provenance()
    for name, info in provenance["snapshot_files"].items():
        shutil.copyfile(info["source_path"], output / "calibration" / name)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {p: version(p) for p in ["qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy", "scipy"]},
        "python_executable": sys.executable,
        "seed": seed, "shots_per_syndrome_circuit": shots,
        "backend": provenance,
        "idle_validation": validate_idle_relaxation(backend),
        "fidelity_type": "exact density-matrix target overlap, not finite-shot tomography",
        "acceptance": "recorded all-zero syndrome; B0/B1 keep every preparation",
        "timing": "ALAP, common terminal readout barrier, data delivered at end of syndrome readout",
        "noise_modes": {"full": "calibrated gate and readout noise plus all delay relaxation",
                        "no_delay": "same gate/readout noise and routing; only delay channels omitted",
                        "ideal": "same routed native circuit with noise disabled"},
        "rows": [],
    }
    assert report["idle_validation"]["max_absolute_error"] < 1e-12
    states = {}
    for name, model in CODES.items():
        strategies = ["B0", "B1", "B2_ED"] + (["B2_BK"] if name == "five" else [])
        # Use one encoder instance for B1 and B2_ED in this run. QPY stores the
        # exact synthesized/routed circuits so replay does not rely on synthesis.
        prep_b1 = build_preparation(model, "B1")
        for strategy in strategies:
            prep = prep_b1 if strategy == "B1" else build_preparation(model, strategy)
            if strategy == "B2_ED":
                circuit = QuantumCircuit(model.n + len(model.stabilizers), name=f"{name}_{strategy}")
                circuit.compose(prep_b1.unitary, range(model.n), inplace=True)
                stages = list(prep_b1.stages)
                for i, check in enumerate(model.stabilizers):
                    before = len(circuit.data)
                    circuit.compose(check_block(model.n, check), [*range(model.n), model.n+i], inplace=True)
                    stages.extend([f"check_{i}_{check}"] * (len(circuit.data) - before))
                prep = Preparation(model, strategy, circuit, tuple(stages), model.stabilizers, model.n)
            scheduled = schedule_preparation(prep, backend, seed=seed)
            label = f"{name}_{strategy}"
            with (output / "circuits" / f"{label}.qpy").open("wb") as f:
                qpy.dump(scheduled.circuit, f)
            (output / "circuits" / f"{label}.txt").write_text(str(scheduled.circuit.draw(output="text", fold=140)))
            row = {"code": name, "strategy": strategy, "resources": scheduled.resources}
            for mode in ["ideal", "no_delay", "full"]:
                metrics, state = simulate_scheduled(scheduled, backend, mode=mode, shots=shots, seed=seed)
                row[mode] = metrics
                states[f"{label}_{mode}"] = state
            assert np.isclose(row["ideal"]["fidelity"], 1)
            assert np.isclose(row["ideal"]["acceptance"], 1/6 if strategy == "B2_BK" else 1)
            report["rows"].append(row)
            print(f"{label}: F={row['full']['fidelity']:.6f}, accept={row['full']['acceptance']:.6f}, "
                  f"F without delay={row['no_delay']['fidelity']:.6f}", flush=True)
    np.savez_compressed(output / "accepted_states.npz", **states)
    (output / "noisy_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "noisy_report.md").write_text(render_report(report))
    make_figure(report, output)
    return report


def summary_table(report):
    lines = ["| Code / strategy | Acceptance | Fidelity incl. delay | Fidelity omitting delay | Codespace population | Duration (µs) | CZ gates |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in report["rows"]:
        f, nd, res = r["full"], r["no_delay"], r["resources"]
        lines.append(f"| {r['code']}/{r['strategy']} | {f['acceptance']:.6f} | {f['fidelity']:.6f} | "
                     f"{nd['fidelity']:.6f} | {f['codespace_population']:.6f} | {res['duration_seconds']*1e6:.3f} | {res['two_qubit_gates']} |")
    return "\n".join(lines)


def render_report(report):
    backend = report["backend"]
    lines = ["# Local noisy magic-state simulation: IBM Aachen", "",
        f"Calibration: **{backend['calibration_date']}**, packaged `FakeAachen` snapshot from qiskit-ibm-runtime {report['environment']['qiskit-ibm-runtime']}. "
        "This is a local simulation, not a live calibration fetch or hardware run.", "",
        "## Results", "", summary_table(report), "",
        "Fidelity is the exact density-matrix overlap with the physical encoded target, conditioned only on the **recorded all-zero syndrome**. "
        "B0 uses the corresponding single-qubit target and has a different output size. Codespace population is a diagnostic, not an additional acceptance filter. "
        "These are model predictions, not measured hardware fidelities or finite-shot tomography estimates.", "",
        "B1/B2_ED start with one magic input; B2_BK starts with five antipodal tetrahedral inputs. "
        "These are the existing baselines; no manually inserted X, Y, or Z faults and no active recovery are used.", "",
        "## Timing and noise", "",
        f"- Native gates and connectivity come from physical qubits `{backend['physical_patch']}` (local wire order). "
        "This is a fixed connected patch rooted at qubit 3, selected without calibration/fidelity optimization. "
        "Initial placement, routed final placement, gate counts, and every wire's idle time are saved in JSON.",
        "- ALAP scheduling uses calibrated durations and dt = 4 ns. A terminal barrier aligns syndrome readout. "
        "Every data wire explicitly waits **2.584 µs during syndrome readout**, in addition to routing and synchronization idles.",
        "- `NoiseModel.from_backend` includes calibrated depolarizing/thermal gate channels, readout assignment error, "
        "and its built-in Delay-only `RelaxationNoisePass`. Idle amplitude damping/dephasing use each physical qubit's T1/T2 "
        "at zero temperature, with T2 capped at 2T1. No second delay pass is added.",
        "- The no-delay ablation removes only Delay instructions from the already compiled sequence. "
        "Native gates, routing and gate/readout error channels stay fixed; gate-duration relaxation remains enabled.",
        "- In this BackendV2 snapshot Aer builds independent symmetric readout assignment channels from each Target's average readout error. "
        "No extra ancilla relaxation is added during measurement, matching Aer's standard backend measurement model; "
        "the unmeasured data's readout wait is explicitly noisy.",
        "- Initial qubits are ideal |0〉; no explicit reset/SPAM model beyond compiled preparation gates and syndrome assignment is added. "
        "Data are delivered at readout completion; no subsequent classical processing, storage, or decoding latency is assumed.", "",
        "## Acceptance and logical output", "",
        "Let ρ_s be the unnormalized data block for true ancilla outcome s, after routing is undone in the analysis and spectator qubits are traced out. "
        "The recorded-zero output is", "",
        r"$$\rho_{\rm acc}=\frac{\sum_s P(0\mid s)\rho_s}{p_{\rm acc}},\qquad "
        r"p_{\rm acc}=\sum_s P(0\mid s)\operatorname{Tr}\rho_s.$$", "",
        "Readout errors can therefore accept a nonzero true syndrome or reject a zero true syndrome. "
        "Errors after a qubit's last relevant check, including relaxation while readout finishes, can leave accepted data outside the codespace. "
        "The JSON includes codespace population, logical Bloch coordinates conditioned on that subspace (diagnostic only), "
        "and the fraction of accepted runs with a nonzero true syndrome.", "",
        "## Sampling and checks", "",
        f"Each mode uses {report['shots_per_syndrome_circuit']:,} Aer shots for recorded syndrome counts, seed {report['seed']}. "
        "Exact density matrices set the fidelities/acceptances in the table; finite-shot acceptance and Wilson intervals are saved separately.", "",
        "| Code / strategy | Exact acceptance | Sampled acceptance | Wilson 95% interval |",
        "|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        f = row["full"]
        lo, hi = f["acceptance_wilson_95_interval"]
        lines.append(f"| {row['code']}/{row['strategy']} | {f['acceptance']:.6f} | {f['acceptance_sampled']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ["", "All seven routed circuits recover ideal fidelity 1 with noise disabled; ideal B2_BK acceptance is 1/6. "
              "A clean |1〉/|+〉 followed by a 50 µs delay matches exp(−t/T1) / exp(−t/T2) to "
              f"{report['idle_validation']['max_absolute_error']:.2g} absolute error. This checks the presence, units, and single application of delay relaxation.", "",
              "## Interpretation and limitations", ""]
    for name in CODES:
        rows = {r["strategy"]: r for r in report["rows"] if r["code"] == name}
        ed, b1 = rows["B2_ED"], rows["B1"]
        delta = ed["no_delay"]["fidelity"] - ed["full"]["fidelity"]
        lines.append(f"For **{name}**, B2_ED has accepted fidelity {ed['full']['fidelity']:.4f} at acceptance "
                     f"{ed['full']['acceptance']:.4f}, versus B1 fidelity {b1['full']['fidelity']:.4f}. "
                     f"Omitting idle noise overstates B2_ED fidelity by {100*delta:.2f} percentage points for this schedule.")
        lines.append("")
    lines += ["This first comparison uses each strategy's own completion time, so B1 is delivered earlier than B2. "
              "A common downstream consumption time would require adding the corresponding B1/B0 storage waits. "
              "The fixed patch includes physical qubit 2 with an 8.67% average readout error; its effect depends on the routed ancilla placement. "
              "No claim of optimal placement or hardware advantage follows from these rows. "
              "The calibrated local Markovian model omits crosstalk, leakage, drift, correlated noise, and control-stack effects. "
              "The bare syndrome circuits remain non-fault-tolerant.", "",
              "## Reproduce", "", "```bash", "conda run -n quantum-jax python scripts/run_noisy.py", "```", "",
              "Saved alongside this report: `noisy_results.json`, `accepted_states.npz`, PNG/PDF comparison, "
              "the exact scheduled circuits in QPY/text, and copies of the calibration files with SHA-256 provenance in JSON. "
              "QPY is the reference for replay of this run; re-synthesizing Clifford circuits in another process/version can change gate order and routing.", "",
              "References: [FakeAachen](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider-fake-aachen), "
              "[Aer NoiseModel](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html), "
              "[RelaxationNoisePass](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.RelaxationNoisePass.html).", ""]
    return "\n".join(lines)


def make_figure(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["rows"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for ax, metric, title in zip(axes, ["fidelity", "acceptance"], ["Accepted target fidelity", "Recorded-zero acceptance"]):
        ax.bar(x-.18, [r["no_delay"][metric] for r in rows], width=.36, label="Gate + readout noise", color="#b4c7d9")
        ax.bar(x+.18, [r["full"][metric] for r in rows], width=.36, label="Gate + readout + delay noise", color="#226899")
        ax.set_xticks(x, [f"{r['code']}\n{r['strategy']}" for r in rows], fontsize=9)
        ax.set_ylim(0, 1.04)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Aachen snapshot · fixed physical patch · exact density-matrix diagnostics")
    for suffix in ["png", "pdf"]:
        fig.savefig(Path(output) / f"noisy_comparison.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "noisy_aachen")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    report = run_experiment(args.output, args.shots, args.seed)
    print(summary_table(report))


if __name__ == "__main__":
    main()
