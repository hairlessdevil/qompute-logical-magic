#!/usr/bin/env python3
"""Joint accept/reject and final codespace populations of saved noisy circuits."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from qiskit_aer import AerSimulator

from analyze_readout import load_preparation
from logical_magic.aachen_noise import (
    AachenPatch, recorded_syndrome_state, snapshot_circuit, syndrome_blocks,
)

ROOT = Path(__file__).resolve().parents[1]


def run_analysis(folder=ROOT / "results" / "noisy_aachen"):
    folder = Path(folder)
    base = json.loads((folder / "noisy_results.json").read_text())
    backend = AachenPatch(base["backend"]["physical_patch"])
    for name, info in base["backend"]["snapshot_files"].items():
        assert info["sha256"] == backend.provenance()["snapshot_files"][name]["sha256"]
    sim = AerSimulator(method="density_matrix", max_parallel_threads=1,
                       noise_model=backend.noise_model())
    assignment = backend.readout_matrices()
    output = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": "noisy_results.json", "calibration_date": base["backend"]["calibration_date"],
              "definition": "Bad means outside the code subspace at original output delivery time, including the readout wait.",
              "interpretation": "Joint quantum populations, equivalent to an ideal final codespace test on data plus the recorded syndrome; not labels of sampled fault trajectories.",
              "rows": []}
    for row in base["rows"]:
        if not row["strategy"].startswith("B2"):
            continue
        prep = load_preparation(folder, row)
        result = sim.run(snapshot_circuit(prep), shots=1, seed_simulator=base["seed"]).result()
        assert result.success
        blocks = syndrome_blocks(np.asarray(result.data(0)["before_readout"]),
                                 prep.data_wires, prep.syndrome_wires)
        accepted, probs = recorded_syndrome_state(blocks, [assignment[q] for q in prep.syndrome_wires])
        pacc = probs["acceptance"]
        uncond = blocks.sum(axis=0)
        acc = pacc * accepted
        rej = uncond - acc
        projector = prep.preparation.code.projector
        joint = {}
        for name, state in [("accepted", acc), ("rejected", rej)]:
            assert np.linalg.eigvalsh(state).min() > -1e-10
            inside = float(np.trace(projector @ state).real)
            outside = float(np.trace(state).real) - inside
            joint[name + "_inside_code"] = inside
            joint[name + "_outside_code"] = outside
        assert np.isclose(sum(joint.values()), 1, atol=1e-10, rtol=0)
        assert np.isclose(pacc, row["full"]["acceptance"], atol=1e-10, rtol=0)
        assert np.isclose(joint["accepted_inside_code"] / pacc, row["full"]["codespace_population"], atol=1e-10, rtol=0)
        target = prep.preparation.target
        uncond_fidelity = float(np.vdot(target, uncond @ target).real)
        accepted_fidelity = float(np.vdot(target, accepted @ target).real)
        assert np.isclose(uncond_fidelity, row["full"]["unconditional_fidelity"], atol=1e-10, rtol=0)
        assert np.isclose(accepted_fidelity, row["full"]["fidelity"], atol=1e-10, rtol=0)
        bad = joint["accepted_outside_code"] + joint["rejected_outside_code"]
        shots = row["full"]["shots"]
        accepted_shots = row["full"]["syndrome_counts"].get("0" * prep.preparation.n_checks, 0)
        source = folder / "circuits" / f"{row['code']}_{row['strategy']}.qpy"
        entry = {"code": row["code"], "strategy": row["strategy"],
                 "source_qpy_sha256": sha256(source.read_bytes()).hexdigest(),
                 "joint_population_per_attempt": joint, "acceptance": pacc, "rejection": 1-pacc,
                 "unconditional_outside_code": bad,
                 "fraction_of_outside_code_population_rejected": joint["rejected_outside_code"] / bad,
                 "outside_code_fraction_among_rejected": joint["rejected_outside_code"] / (1-pacc),
                 "outside_code_fraction_among_accepted": joint["accepted_outside_code"] / pacc,
                 "unconditional_target_fidelity": uncond_fidelity,
                 "accepted_target_fidelity": accepted_fidelity,
                 "original_sampled_rejected_shots": shots - accepted_shots,
                 "original_sampled_total_shots": shots}
        output["rows"].append(entry)
        print(json.dumps(entry, indent=2), flush=True)
    (folder / "syndrome_filtering.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = ["# How much bad population does syndrome filtering reject?", "",
             "Here bad means **outside the code subspace at the original output delivery time**, after all gates and delays, including the syndrome-readout wait. "
             "This does not classify logical errors inside the code subspace. All probabilities include syndrome assignment error. "
             "The diagnostic replays the exact saved circuits without recompilation and verifies acceptance and fidelities against the saved results.", "",
             "For unnormalized accepted/rejected data states R_A and R_R and codespace projector P, the joint populations are "
             "Tr(P R_A), Tr((I-P) R_A), Tr(P R_R), and Tr((I-P) R_R). These correspond to an ideal final codespace test on data, "
             "jointly with the recorded syndrome. No such extra test or postselection is applied in the actual preparation.", "",
             "## Joint populations: percentages of all attempts", "",
             "| Protocol | Accepted, inside code | Accepted, outside code | Rejected, inside code | Rejected, outside code |",
             "|---|---:|---:|---:|---:|"]
    for row in output["rows"]:
        j = row["joint_population_per_attempt"]
        lines.append(f"| {row['code']}/{row['strategy']} | " + " | ".join(f"{100*j[k]:.3f}%" for k in
                     ["accepted_inside_code", "accepted_outside_code", "rejected_inside_code", "rejected_outside_code"]) + " |")
    lines += ["", "## Filtering efficiency at delivery", "",
              "| Protocol | Total rejected | Outside-code population without selection | Fraction of that population rejected | Outside-code fraction among accepted | Original rejected shots |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in output["rows"]:
        lines.append(f"| {row['code']}/{row['strategy']} | " + " | ".join(f"{100*row[k]:.3f}%" for k in
                     ["rejection", "unconditional_outside_code", "fraction_of_outside_code_population_rejected", "outside_code_fraction_among_accepted"]) +
                     f" | {row['original_sampled_rejected_shots']}/{row['original_sampled_total_shots']} |")
    lines += ["", "All exact populations refer to the same completed circuit, with or without selecting its accepted branch. "
              "They are not estimates of how many faults were present before extraction, nor a comparison to omitting the extraction gates. "
              "Some noise occurs during extraction or after the relevant check. Density-matrix populations do not give a unique good/bad label to each sampled shot.", "",
              "B2_BK rejects 5/6 of attempts even ideally: its large rejection includes the intended projection of product inputs, not just removal of noise-induced errors. "
              "Inside-code does not necessarily mean the correct logical state, and rejected inside-code population is not all attributable to readout errors.", "",
              "Reproduce: `conda run -n quantum-jax python scripts/analyze_filtering.py`.", ""]
    (folder / "syndrome_filtering.md").write_text("\n".join(lines))
    return output


if __name__ == "__main__":
    run_analysis()
