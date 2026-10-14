#!/usr/bin/env python3
"""Reproduce all ideal results without installing the project or using IBM APIs."""

import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
from qiskit import qpy

from logical_magic import CODES, build_preparation, exact_result
from logical_magic.simulation import circuit_fault_audit, estimate_fidelity, storage_fault_audit


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(out, results, audits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    keys = list(results)
    labels = [key.replace("four/", "4q ").replace("five/", "5q ") for key in keys]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    axes[0].bar(x, [results[k]["exact"]["fidelity"] for k in keys], color="#3d806d", alpha=.75)
    axes[0].errorbar(x, [results[k]["finite_shots"]["fidelity_estimate"] for k in keys],
                    yerr=[1.96*results[k]["finite_shots"]["standard_error_plugin"] for k in keys],
                    fmt="o", color="#203c65", capsize=3, label="sample estimate ± 1.96 SE")
    axes[0].set(ylabel="Fidelity to the target", ylim=(.94, 1.06), title="Accepted-state quality")
    axes[0].axhline(1, color="gray", lw=.6)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].bar(x, [results[k]["exact"]["acceptance"] for k in keys], color="#4b77a1")
    axes[1].set(ylabel="Acceptance probability", ylim=(0, 1.08), title="Success probability")
    axes[2].bar(x, [results[k]["exact"]["resources"]["two_qubit_gates"] for k in keys], color="#b4884b")
    axes[2].set(ylabel="Two-qubit gates per attempt", title="Ideal circuit cost (before routing)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=55, ha="right")
    fig.suptitle("Local ideal simulation · no noise model · no hardware execution", fontsize=13)
    fig.savefig(out / "ideal_comparison.png", dpi=180)
    fig.savefig(out / "ideal_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    labels = list(audits)
    bottom = np.zeros(len(labels))
    for cat, color in [("rejected", "#4b77a1"), ("accepted_target", "#3d806d"), ("accepted_harmful", "#b65b54")]:
        values = np.array([audits[k]["categories"].get(cat, 0) for k in labels])
        ax.bar(np.arange(len(labels)), values, bottom=bottom, label=cat.replace("_", " "), color=color)
        bottom += values
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set(ylabel="Number of enumerated fault cases", title="Circuit-level audit: ideal checks do not imply fault tolerance")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.savefig(out / "fault_audit.png", dpi=180)
    fig.savefig(out / "fault_audit.pdf")
    plt.close(fig)


def write_report(out, payload):
    lines = [
        "# Ideal simulation results", "", "## Material Passport", "",
        "- Stage: local ideal model construction and validation.",
        "- Source plan: `docs/project_plan_original.md` (unaltered copy supplied by user).",
        "- Evidence: generated exact states, Qiskit Aer shot counts, circuit fault enumeration.",
        "- Scope: no noisy performance claim, hardware execution, or B3 optimization.",
        f"- Reproduction: `conda run -n quantum-jax python scripts/run_ideal.py --shots {payload['metadata']['shots_per_setting']} --seed {payload['metadata']['seed']}`.",
        "", "## Baselines", "",
        "Fidelity means squared pure-target overlap of the actual accepted data state. "
        "Only recorded syndrome bits define acceptance; no extra codespace projection is applied.", "",
        "| Code / strategy | Exact fidelity | Acceptance | Attempts/accepted | Qubits | 2q gates | Shot estimate ± SE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, r in payload["results"].items():
        e, f = r["exact"], r["finite_shots"]
        lines.append(f"| {key} | {e['fidelity']:.12f} | {e['acceptance']:.9f} | {e['attempts_per_accept']:.3f} | "
                     f"{e['resources']['total_qubits']} | {e['resources']['two_qubit_gates']} | "
                     f"{f['fidelity_estimate']:.6f} ± {f['standard_error_plugin']:.6f} |")
    lines += ["", "B0 = bare physical state; B1 = ordinary encoding; B2_ED = encoding followed by all "
              "stabilizer checks; B2_BK = five antipodal tetrahedral inputs projected onto the five-qubit "
              "codespace, retaining the encoded output. B2_ED is a screening baseline, not verified "
              "fault-tolerant magic-state preparation. B2_BK assumes ideal Clifford checks for its "
              "distillation property; the implemented unflagged extraction is not fault tolerant.", "",
              "## Fault injection", "",
              "| Circuit | Enumerated cases | Fully rejected | Accepted with target intact | Can accept a damaged state |",
              "|---|---:|---:|---:|---:|"]
    for key, r in payload["fault_audits"].items():
        c = r["categories"]
        lines.append(f"| {key} | {r['cases']} | {c.get('rejected',0)} | {c.get('accepted_target',0)} | {c.get('accepted_harmful',0)} |")
    lines += ["", "These are counts of modeled fault locations and Pauli operators, not error rates or "
              "a statistically weighted noise model. An accepted category can include partial acceptance; "
              "each row in the CSV records its exact acceptance probability and conditional fidelity.", "",
              "All 12 single-qubit data Pauli errors for the four-qubit code and all 15 for the five-qubit "
              "code are detected when inserted after encoding and before otherwise ideal checks. "
              "Circuit faults during encoding or extraction can escape. In particular, a Y fault just "
              "after four-qubit H-state input preparation becomes a logical Y: acceptance remains 1, "
              "codespace population remains 1, and target fidelity becomes 0.", "",
              "## Interpretation", "",
              "The ideal circuits and measurement conventions are validated. Noiseless fidelity ties "
              "do not establish an error-suppression advantage. The four-qubit model is the small primary "
              "screening experiment; the five-qubit BK primitive is the literature-based distillation "
              "reference. Final code/protocol selection requires the next noise study.", "",
              "The BK ideal acceptance is 1/6, so six attempts and thirty input magic states are needed "
              "per accepted encoded state on average. Gate/qubit costs are per attempt and exclude "
              "analysis copies, routing, timing, reset, and calibration overhead. Durations are unset "
              "because no physical backend or timing model has been selected.", "",
              "Finite-shot linear estimates can exceed 1; they are deliberately not clipped. The plot's "
              "error bars are 1.96 times the plug-in standard error, not exact confidence intervals. "
              "`ideal_results.json` also records conservative 95% Hoeffding intervals, Wilson intervals "
              "for acceptance, all raw counts, seeds, settings, and total shot cost.", "",
              "## Sources", "",
              "- [Leung et al., four-qubit amplitude-damping code, Eqs. 9–10](https://arxiv.org/html/quant-ph/9704002v1).",
              "- [Laflamme et al., five-qubit code](https://arxiv.org/abs/quant-ph/9602019).",
              "- [Bravyi–Kitaev, five-copy projection, Sec. V and Eqs. 15–22](https://arxiv.org/html/quant-ph/0403025v2).",
              "- [Qiskit Clifford API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Clifford).",
              "- [AerSimulator API](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html).", ""]
    (out / "ideal_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=8192, help="Shots per independent Pauli setting")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ideal")
    parser.add_argument("--timeout", type=int, default=600, help="Hard wall-clock limit in seconds")
    args = parser.parse_args()
    if args.shots <= 0 or args.timeout <= 0:
        parser.error("shots and timeout must be positive")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Ideal run exceeded hard timeout")))
    signal.alarm(args.timeout)
    start = time.monotonic()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    circuit_dir = out / "circuits"
    circuit_dir.mkdir(exist_ok=True)
    results, audits, references = {}, {}, {}
    for code in CODES.values():
        references[code.name + "_basis"] = code.basis
        references[code.name + "_target"] = code.target
        references[code.name + "_projector"] = code.projector
        strategies = ["B0", "B1", "B2_ED"] + (["B2_BK"] if code.name == "five" else [])
        for strategy in strategies:
            key = f"{code.name}/{strategy}"
            prep = build_preparation(code, strategy)
            print(f"Running {key}: exact state, {args.shots} shots/setting", flush=True)
            exact = exact_result(prep)
            expected_acceptance = 1/6 if strategy == "B2_BK" else 1
            assert abs(exact["fidelity"]-1) < 1e-10
            assert abs(exact["acceptance"]-expected_acceptance) < 1e-10
            finite = estimate_fidelity(prep, args.shots, args.seed)
            results[key] = {"exact": exact, "finite_shots": finite}
            with (circuit_dir / f"{code.name}_{strategy}.qpy").open("wb") as file:
                qpy.dump(prep.measured(), file)
            (circuit_dir / f"{code.name}_{strategy}.txt").write_text(str(prep.measured().draw(output="text", fold=110)))
            if strategy.startswith("B2"):
                print(f"Auditing {key}: every gate Pauli, initialization, and syndrome-bit fault", flush=True)
                rows = circuit_fault_audit(prep)
                write_csv(out / f"faults_{code.name}_{strategy}.csv", rows)
                audits[key] = {"cases": len(rows), "categories": dict(Counter(r["category"] for r in rows)),
                               "harmful_examples": [r for r in rows if r["category"] == "accepted_harmful"][:5]}
            if strategy == "B2_ED":
                storage = storage_fault_audit(prep)
                assert all(row["acceptance"] < 1e-10 for row in storage)
                write_csv(out / f"storage_faults_{code.name}.csv", storage)
    metadata = {"seed": args.seed, "shots_per_setting": args.shots, "python": sys.version,
                "platform": platform.platform(), "versions": {p: version(p) for p in ["qiskit", "qiskit-aer", "numpy", "scipy", "matplotlib"]},
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "src").rglob("*.py"))},
                "elapsed_seconds": time.monotonic()-start,
                "scope": "Ideal circuits only; Pauli fault injection is a separate diagnostic, not a noise benchmark"}
    payload = {"metadata": metadata, "results": results, "fault_audits": audits}
    (out / "ideal_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    np.savez(out / "reference_states.npz", **references)
    plots(out, results, audits)
    write_report(out, payload)
    signal.alarm(0)
    print(f"Completed in {time.monotonic()-start:.1f}s. Results: {out}", flush=True)


if __name__ == "__main__":
    main()
