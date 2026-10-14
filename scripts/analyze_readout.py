#!/usr/bin/env python3
"""Readout-only ablation and finite-shot fidelity of saved Aachen preparations."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from qiskit import qpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logical_magic import CODES, build_preparation
from logical_magic.aachen_noise import AachenPatch, ScheduledPreparation, simulate_scheduled
from logical_magic.noisy_fidelity import estimate_noisy_fidelity


def load_preparation(folder, row):
    label = f"{row['code']}_{row['strategy']}"
    with (Path(folder) / "circuits" / f"{label}.qpy").open("rb") as f:
        circuit = qpy.load(f)[0]
    res = row["resources"]
    return ScheduledPreparation(build_preparation(CODES[row["code"]], row["strategy"]), circuit,
                                tuple(res["final_data_local_wires"]),
                                tuple(res["syndrome_local_wires_in_bit_order"]), res)


def run_analysis(folder=ROOT / "results" / "noisy_aachen", shots=8192, seed=20261014):
    folder = Path(folder)
    base = json.loads((folder / "noisy_results.json").read_text())
    backend = AachenPatch(base["backend"]["physical_patch"])
    current = backend.provenance()
    for name, info in base["backend"]["snapshot_files"].items():
        if info["sha256"] != current["snapshot_files"][name]["sha256"]:
            raise ValueError("Current calibration differs from the saved run")
    output = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "shots_per_setting": shots,
              "seed": seed, "backend": base["backend"], "environment": base["environment"],
              "source_results": "noisy_results.json", "rows": []}
    (folder / "quality_circuits").mkdir(exist_ok=True)
    for row in base["rows"]:
        scheduled = load_preparation(folder, row)
        label = f"{row['code']}_{row['strategy']}"
        result = {"code": row["code"], "strategy": row["strategy"],
                  "with_syndrome_readout_error": row["full"],
                  "exact_preparation_fidelity": row["full"]["fidelity"]}
        if scheduled.preparation.n_checks:
            no_readout, _ = simulate_scheduled(scheduled, backend, mode="no_readout", shots=shots, seed=seed)
            result["without_syndrome_readout_error"] = no_readout
            matrices = backend.readout_matrices()
            correct_zero = float(np.prod([matrices[q][0, 0] for q in scheduled.syndrome_wires]))
            result["false_reject_probability_given_true_zero"] = 1-correct_zero
            result["false_accept_probability_per_attempt"] = (
                row["full"]["acceptance"]*row["full"]["accepted_fraction_from_nonzero_true_syndrome"])
            result["false_reject_probability_per_attempt"] = (
                no_readout["acceptance"]*(1-correct_zero))
        circuits = []
        sampled = estimate_noisy_fidelity(scheduled, backend, shots=shots, seed=seed,
                                          circuit_callback=lambda pauli, circuit: circuits.append(circuit))
        with (folder / "quality_circuits" / f"{label}.qpy").open("wb") as f:
            qpy.dump(circuits, f)
        result["finite_shot"] = sampled
        output["rows"].append(result)
        print(f"{label}: raw={sampled['raw']['fidelity_estimate']:.6f}, "
              f"readout-mitigated={sampled['data_readout_mitigated']['fidelity_estimate']:.6f}, "
              f"exact={result['exact_preparation_fidelity']:.6f}", flush=True)
    output["total_quality_measurement_shots"] = sum(r["finite_shot"]["total_attempted_shots"] for r in output["rows"])
    (folder / "readout_and_fidelity.json").write_text(json.dumps(output, indent=2)+"\n")
    (folder / "readout_and_fidelity.md").write_text(render_report(output))
    make_quality_figure(output, folder)
    return output


def readout_table(output):
    lines = ["| Protocol | Acceptance without → with readout error | Fidelity without → with readout error | Nonzero true syndrome among accepted |",
             "|---|---:|---:|---:|"]
    for row in output["rows"]:
        if "without_syndrome_readout_error" not in row:
            continue
        off, on = row["without_syndrome_readout_error"], row["with_syndrome_readout_error"]
        lines.append(f"| {row['code']}/{row['strategy']} | {100*off['acceptance']:.3f}% → {100*on['acceptance']:.3f}% | "
                     f"{100*off['fidelity']:.3f}% → {100*on['fidelity']:.3f}% | {100*on['accepted_fraction_from_nonzero_true_syndrome']:.4f}% |")
    return "\n".join(lines)


def quality_table(output):
    lines = ["| Protocol | Exact preparation fidelity | Raw finite-shot estimate | Data-readout-mitigated estimate | Approx. 95% statistical interval (mitigated) | Settings × shots |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in output["rows"]:
        r = row["finite_shot"]
        corrected = r["data_readout_mitigated"]
        lo, hi = corrected["normal_approx_95_interval"]
        lines.append(f"| {row['code']}/{row['strategy']} | {row['exact_preparation_fidelity']:.6f} | "
                     f"{r['raw']['fidelity_estimate']:.6f} | {corrected['fidelity_estimate']:.6f} | "
                     f"[{lo:.6f}, {hi:.6f}] | {r['measurement_settings']} × {r['shots_per_setting']:,} |")
    return "\n".join(lines)


def make_quality_figure(output, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = output["rows"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 4.8), layout="constrained")
    for key, offset, color, label in [
        ("raw", -.12, "#ad6d24", "Raw counts"),
        ("data_readout_mitigated", .12, "#226899", "Data readout corrected"),
    ]:
        ax.errorbar(x+offset, [r["finite_shot"][key]["fidelity_estimate"] for r in rows],
                    yerr=[1.95996398454*r["finite_shot"][key]["standard_error"] for r in rows],
                    fmt="o", capsize=4, color=color, label=label)
    ax.scatter(x, [r["exact_preparation_fidelity"] for r in rows], marker="_", s=360,
               color="#272c32", linewidths=2, label="Exact delivered-state fidelity")
    ax.set_xticks(x, [f"{r['code']}\n{r['strategy']}" for r in rows])
    ax.set_ylim(.65, 1.03)
    ax.set_ylabel("Target fidelity / measurement estimate")
    ax.set_title(f"Finite-shot Pauli measurements · {output['shots_per_setting']:,} attempts per setting\n"
                 "Error bars: approximate 95% sampling intervals; calibration/basis bias excluded", fontsize=11)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower left", fontsize=9)
    for ext in ["png", "pdf"]:
        fig.savefig(Path(folder) / f"finite_shot_fidelity.{ext}", dpi=180)
    plt.close(fig)


def render_report(output):
    lines = ["# Syndrome readout error and finite-shot state quality", "",
        "## What was included in the original noisy result?", "",
        "Yes: `NoiseModel.from_backend(..., readout_error=True)` was enabled. The existing fidelity used an exact density matrix "
        "conditioned on the **recorded** all-zero syndrome. Ancilla assignment error changes which data blocks are accepted; "
        "it is not an extra quantum gate acting on the data. False rejections reduce yield, and false acceptances mix in blocks associated "
        "with nonzero true ancilla outcomes. A true zero outcome itself does not guarantee an undamaged logical state.", "",
        "The original noisy notebook sampled acceptance only. Its quoted fidelities were simulator diagnostics, not finite-shot measurements. "
        "There were no destructive data measurements in those preparation circuits, so data readout error did not enter those fidelities.", "",
        "## Isolating syndrome assignment error", "",
        "Here only readout assignment is disabled. Routing, gate noise, every delay, and the 2.584 µs syndrome measurement wait stay fixed. "
        "This ablation does not remove measurement latency.", "", readout_table(output), "",
        "In these circuits, false rejection accounts for most of the readout-induced yield loss. "
        "The BK accepted ensemble is more affected by false acceptance. B0/B1 have no syndrome measurement, so this ablation does not change them.", "",
        "## Finite-shot target fidelity", "",
        "Expand the **full physical target projector** into Pauli strings:", "",
        r"$$|\psi\rangle\langle\psi|=\sum_P c_P P,\qquad "
        r"\widehat F=c_I+\sum_{P\ne I}c_P\,\overline{p}_{P,\mathrm{recorded\ syndrome}=0}.$ $".replace("$ $", "$$"), "",
        "For every nonidentity term, prepare fresh copies, retain the recorded-zero shots, and measure the relevant data-qubit parity. "
        "Use native decompositions of H for X and S† followed by H for Y; Z needs no rotation. "
        "Each saved preparation schedule is frozen, so analysis rotations start at its original delivery boundary. "
        "The model includes their gate noise, padding delays, and data readout error. These destructive measurements consume analysis copies. "
        "The fidelity operator includes codespace loss; measuring just logical X/Y/Z without the codespace projector would be insufficient.", "",
        quality_table(output), "",
        f"Total attempted quality-measurement shots: **{output['total_quality_measurement_shots']:,}**; "
        "shots per setting are attempts, not accepted shots. B2_BK's low yield reduces its effective sample size. "
        "Identity terms require no shots. The JSON records every count, accepted sample size, coefficient, seed, and raw/mitigated score.", "",
        "## What readout mitigation does and does not correct", "",
        "Only **data readout** is corrected, using the known assignment matrices of the simulation snapshot. "
        "For one qubit with A[true, recorded], solve A w = (1, −1); multiply the corresponding per-qubit weights for a Pauli parity. "
        "For symmetric error e this is division by 1−2e for each nonidentity factor. "
        "Syndrome bits are not mitigated, so the estimated state is still the actually accepted output. "
        "These calculations use counts alone; saved simulator probabilities are used only to diagnose systematic bias.", "",
        "The statistical intervals use independent settings and the sample variance of their weighted parity scores. "
        "Normal intervals are approximate; conservative Hoeffding/union-bound intervals are also saved. Estimates/intervals are not clipped to [0,1]. "
        "Both describe sampling uncertainty around the measurement-score mean. They do not cover calibration error, drift, correlated assignment, "
        "or residual basis-rotation/delay bias. The assignment matrices are treated as exact here; hardware use would need calibration and its uncertainty.", "",
        "| Protocol | Infinite-shot raw score | Infinite-shot mitigated score | Residual mitigated bias relative to delivery fidelity |",
        "|---|---:|---:|---:|"]
    for row in output["rows"]:
        r = row["finite_shot"]
        raw = r["raw"]["infinite_shot_measurement_score_diagnostic"]
        corrected = r["data_readout_mitigated"]["infinite_shot_measurement_score_diagnostic"]
        lines.append(f"| {row['code']}/{row['strategy']} | {raw:.6f} | {corrected:.6f} | {corrected-row['exact_preparation_fidelity']:+.6f} |")
    lines += ["", "The large raw bias in some rows comes from data assignment, especially the 8.67% readout error on physical qubit 2 "
              "when it carries a data qubit. That error did not affect the earlier ancilla-only acceptance measurement when qubit 2 was unmeasured.", "",
              "## Execution and reproduction", "",
              "The saved `quality_circuits/*.qpy` contain hardware-form measurement circuits without simulator state saves. "
              "For efficient Aer sampling, syndrome measurements are mathematically deferred across the later data-only rotations. "
              "No extra evolution is applied to the deferred ancillas. This equivalence relies on this model's independent channels; "
              "the original data waits and the physically recorded acceptance rule are preserved.", "",
              "```bash", "conda run -n quantum-jax python scripts/analyze_readout.py --shots 8192", "```", "",
              "No packages are installed or upgraded. No manual faults are inserted. No hardware job is submitted.", "",
              "Reference: [Aer NoiseModel and readout_error option](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "results" / "noisy_aachen")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    run_analysis(args.folder, args.shots, args.seed)
