#!/usr/bin/env python3
"""Offline analysis of an already downloaded B2_ED job; never contacts IBM."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp
from scipy.stats import chi2_contingency
from qiskit_ibm_runtime import RuntimeDecoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logical_magic.codes import CODES
from logical_magic.ibm_b2ed import file_hash, joint_runtime_counts, load_bundle, write_json
from logical_magic.noisy_fidelity import expected_measurement_score, parity_weights
from logical_magic.simulation import wilson_interval


def category_counts(counts, pauli):
    """Independent parity implementation: accepted+/-, rejected+/- counts."""
    support = sum(1 << q for q, p in enumerate(reversed(pauli)) if p != "I")
    out = np.zeros(4, dtype=int)
    for bits, count in counts.items():
        value = int(bits.replace(" ", ""), 2)
        negative = ((value >> 3) & support).bit_count() % 2
        out[2*int((value & 7) != 0) + negative] += count
    return out


def statistic(values, group):
    if group == "accepted":
        plus, minus = values[..., 0], values[..., 1]
    elif group == "rejected":
        plus, minus = values[..., 2], values[..., 3]
    else:
        plus, minus = values[..., 0] + values[..., 2], values[..., 1] + values[..., 3]
    n = plus + minus
    if np.any(n < 2):
        raise ValueError("Insufficient samples in a conditional group")
    mean = (plus-minus)/n
    return mean, (1-mean**2)/(n-1)


def analyze(bundle, bootstrap_samples=10000, seed=20261014):
    bundle, manifest, _ = load_bundle(bundle)
    counts = json.loads((bundle / "joint_counts.json").read_text())
    runtime = json.loads((bundle / "runtime_result.json").read_text(), cls=RuntimeDecoder)
    original = json.loads((bundle / "analysis.json").read_text())
    receipt = json.loads((bundle / "submission.json").read_text())
    if receipt["manifest_sha256"] != file_hash(bundle / "manifest.json"):
        raise ValueError("Submitted manifest hash changed")
    if len(counts) != len(runtime) or len(runtime) != manifest["number_of_circuits"]:
        raise ValueError("Missing or extra circuit results")
    acceptance_counts = []
    syndrome = Counter()
    for pub, setting, saved in zip(runtime, manifest["settings"], counts):
        if pub.data.syndrome.num_bits != 3 or pub.data.syndrome.num_shots != manifest["shots_per_circuit"]:
            raise ValueError("Unexpected syndrome register or shot count")
        if setting["kind"] == "acceptance":
            reconstructed = pub.data.syndrome.get_counts()
        else:
            if pub.metadata["circuit_metadata"]["analysis_pauli"] != setting["pauli"]:
                raise ValueError("Result PUB/Pauli order changed")
            reconstructed = joint_runtime_counts(pub)
        if saved != reconstructed or sum(saved.values()) != manifest["shots_per_circuit"]:
            raise ValueError("Archived counts do not reproduce the Runtime bit arrays")
        h = pub.data.syndrome.get_counts()
        syndrome.update(h)
        acceptance_counts.append(h.get("000", 0))

    project = {p: float(c.real) for p, c in SparsePauliOp.from_operator(CODES["four"].projector).to_list()}
    rng = np.random.default_rng(seed)
    results, rows = {}, []
    boot = {g: np.full(bootstrap_samples, manifest["identity_coefficient"]) for g in ("accepted", "all", "rejected")}
    for group in boot:
        results[group] = {"raw_fidelity": manifest["identity_coefficient"], "variance": 0.,
                          "raw_codespace_score": project["IIII"], "codespace_variance": 0.}
    for setting, saved in zip(manifest["settings"][1:], counts[1:]):
        categories = category_counts(saved, setting["pauli"])
        samples = rng.multinomial(int(categories.sum()), categories/categories.sum(), size=bootstrap_samples)
        row = {"pauli": setting["pauli"], "coefficient": setting["coefficient"],
               "accepted_shots": int(categories[:2].sum()), "total_shots": int(categories.sum())}
        for group, result in results.items():
            mean, variance = statistic(categories, group)
            result["raw_fidelity"] += setting["coefficient"]*mean
            result["variance"] += setting["coefficient"]**2*variance
            coefficient = project.get(setting["pauli"], 0.)
            result["raw_codespace_score"] += coefficient*mean
            result["codespace_variance"] += coefficient**2*variance
            boot[group] += setting["coefficient"]*statistic(samples, group)[0]
            row[f"{group}_expectation"] = float(mean)
            row[f"{group}_standard_error"] = float(np.sqrt(variance))
        rows.append(row)
    for group, result in results.items():
        se = float(np.sqrt(result.pop("variance")))
        cp_se = float(np.sqrt(result.pop("codespace_variance")))
        result.update(standard_error=se, normal_95_interval=[result["raw_fidelity"]-1.959964*se,
                      result["raw_fidelity"]+1.959964*se],
                      bootstrap_95_interval=np.quantile(boot[group], [.025, .975]).tolist(),
                      codespace_standard_error=cp_se)
    if abs(results["accepted"]["raw_fidelity"]-original["raw_fidelity_estimate"]) > 1e-12:
        raise ValueError("Independent fidelity calculation did not reproduce retrieval analysis")
    if abs(results["accepted"]["standard_error"]-original["standard_error"]) > 1e-12:
        raise ValueError("Independent uncertainty calculation did not reproduce retrieval analysis")

    total = sum(syndrome.values())
    check = chi2_contingency([acceptance_counts, [manifest["shots_per_circuit"]-a for a in acceptance_counts]])
    calibration = json.loads((bundle / "calibration.json").read_text())
    readout = []
    for role, wires in [("data", manifest["resources"]["final_data_physical_qubits"]),
                        ("syndrome", manifest["resources"]["syndrome_physical_qubits_in_bit_order"])]:
        for bit, q in enumerate(wires):
            props = {v["name"]: v for v in calibration["qubits"][q]}
            readout.append({"role": role, "bit": bit, "physical_qubit": q,
                            "readout_error": props["readout_error"]["value"],
                            "calibration_date": props["readout_error"]["date"]})
    old = json.loads((ROOT / "results/noisy_aachen/readout_and_fidelity.json").read_text())
    oldrow = next(r for r in old["rows"] if r["code"] == "four" and r["strategy"] == "B2_ED")
    return {
        "job_id": receipt["job_id"], "total_shots": total, "number_of_circuits": len(counts),
        "variant": manifest["resources"].get("dd", {}).get("sequence", "no DD"),
        "independent_archive_and_estimator_validation": "passed",
        "bootstrap": {"samples": bootstrap_samples, "seed": seed,
                      "method": "Per-setting multinomial resampling of joint acceptance/parity categories"},
        "primary_acceptance": original["acceptance_only_circuit"],
        "pooled_acceptance": {"accepted": syndrome["000"], "attempted": total,
                              "estimate": syndrome["000"]/total,
                              "wilson_95_interval": list(wilson_interval(syndrome["000"], total))},
        "acceptance_across_settings": {"minimum": min(acceptance_counts)/manifest["shots_per_circuit"],
                                       "maximum": max(acceptance_counts)/manifest["shots_per_circuit"],
                                       "chi_square": float(check.statistic), "dof": int(check.dof),
                                       "p_value": float(check.pvalue),
                                       "minimum_expected_cell_count": float(check.expected_freq.min())},
        "syndrome_counts": dict(sorted(syndrome.items())),
        "syndrome_bit_one_probabilities": [sum(v for s, v in syndrome.items() if int(s, 2) & (1 << bit))/total for bit in range(3)],
        "groups": results, "per_pauli": rows,
        "accepted_minus_all": {"difference": results["accepted"]["raw_fidelity"]-results["all"]["raw_fidelity"],
                               "bootstrap_95_interval": np.quantile(boot["accepted"]-boot["all"], [.025, .975]).tolist()},
        "conservative_primary_95_interval": original["conservative_95_interval"],
        "saved_readout_calibration": readout,
        "execution_spans": str(runtime.metadata["execution"]["execution_spans"]),
        "earlier_snapshot_simulation": {
            "calibration_date": old["backend"]["calibration_date"],
            "acceptance": oldrow["with_syndrome_readout_error"]["acceptance"],
            "raw_measurement_score": oldrow["finite_shot"]["raw"]["infinite_shot_measurement_score_diagnostic"],
            "exact_preparation_fidelity": oldrow["exact_preparation_fidelity"],
            "comparison_note": "Different calibration and schedule timing; raw compared with raw only"},
        "input_sha256": {name: file_hash(bundle/name) for name in
                         ("manifest.json", "runtime_result.json", "joint_counts.json", "analysis.json", "calibration.json")},
    }


def compare_with_baseline(bundle, analysis, baseline):
    """Compare independent jobs; uncertainty does not include inter-job drift."""
    bundle, manifest, _ = load_bundle(bundle)
    baseline, base_manifest, _ = load_bundle(baseline)
    base = json.loads((baseline/"analysis_extended.json").read_text())
    for name, expected in base["input_sha256"].items():
        if file_hash(baseline/name) != expected:
            raise ValueError(f"Baseline analysis input changed: {name}")
    if manifest["source_manifest_sha256"] != file_hash(baseline/"manifest.json"):
        raise ValueError("DD source does not match the selected no-DD baseline")
    if manifest["source_circuits_sha256"] != file_hash(baseline/"circuits.qpy"):
        raise ValueError("DD source circuits do not match the baseline")
    for key in ("code", "strategy", "target", "stabilizers_in_syndrome_bit_order", "shots_per_circuit", "settings"):
        if manifest[key] != base_manifest[key]:
            raise ValueError(f"Comparison protocol changed: {key}")
    for key in ("duration_seconds", "final_data_physical_qubits", "syndrome_physical_qubits_in_bit_order"):
        if manifest["resources"][key] != base_manifest["resources"][key]:
            raise ValueError(f"Comparison layout/timing changed: {key}")
    same_calibration = manifest["files"]["calibration.json"] == base_manifest["files"]["calibration.json"]
    before_receipt = json.loads((baseline/"submission.json").read_text())
    after_receipt = json.loads((bundle/"submission.json").read_text())
    gap = (datetime.fromisoformat(after_receipt["submitted_at_utc"])-datetime.fromisoformat(before_receipt["submitted_at_utc"])).total_seconds()

    def difference(before, after, before_se, after_se):
        delta = after-before
        se = float(np.hypot(before_se, after_se))
        return {"no_dd": before, "xy4": after, "difference": delta, "standard_error": se,
                "normal_95_interval": [delta-1.959964*se, delta+1.959964*se]}

    metrics = {}
    for group in ("accepted", "all", "rejected"):
        a, b = base["groups"][group], analysis["groups"][group]
        metrics[f"raw_fidelity_{group}"] = difference(a["raw_fidelity"], b["raw_fidelity"], a["standard_error"], b["standard_error"])
    for key in ("primary_acceptance", "pooled_acceptance"):
        a, b = base[key], analysis[key]
        estimate_key, shots_key = ("acceptance", "shots") if key == "primary_acceptance" else ("estimate", "attempted")
        x, y = a[estimate_key], b[estimate_key]
        metrics[key] = difference(x, y, np.sqrt(x*(1-x)/a[shots_key]), np.sqrt(y*(1-y)/b[shots_key]))
    a, b = base["groups"]["accepted"], analysis["groups"]["accepted"]
    metrics["accepted_codespace_score"] = difference(a["raw_codespace_score"], b["raw_codespace_score"],
                                                     a["codespace_standard_error"], b["codespace_standard_error"])
    # Distinct RNG streams are essential: the jobs have no paired shots.
    samples = []
    bootstrap_seed = 20261015
    for i, (directory, m) in enumerate(((baseline, base_manifest), (bundle, manifest))):
        rng = np.random.default_rng(bootstrap_seed+i)
        f = np.full(10000, m["identity_coefficient"])
        counts = json.loads((directory/"joint_counts.json").read_text())
        for setting, saved in zip(m["settings"][1:], counts[1:]):
            c = category_counts(saved, setting["pauli"])
            draw = rng.multinomial(int(c.sum()), c/c.sum(), size=len(f))
            f += setting["coefficient"]*statistic(draw, "accepted")[0]
        samples.append(f)
    metrics["raw_fidelity_accepted"]["independent_bootstrap_95_interval"] = np.quantile(samples[1]-samples[0], [.025, .975]).tolist()
    return {"baseline_job_id": base["job_id"], "xy4_job_id": analysis["job_id"],
            "baseline_bundle": str(baseline), "submission_gap_minutes": gap/60,
            "identical_saved_calibration": same_calibration, "matched_layout_timing_and_measurement_settings": True,
            "metrics": metrics, "baseline_summary": {k: base[k] for k in
                  ("groups", "primary_acceptance", "pooled_acceptance", "syndrome_bit_one_probabilities", "saved_calibration_replay")},
            "bootstrap_samples": 10000, "independent_bootstrap_seeds": [bootstrap_seed, bootstrap_seed+1],
            "interpretation": "Observed difference between two independent sequential jobs. Sampling intervals are pointwise; "
                              "they omit drift and systematic readout/basis error. This was not an interleaved randomized DD/no-DD comparison. "
                              "Identical saved calibration does not imply identical physical noise during execution."}


def replay_calibration(bundle, dt):
    """Replay frozen circuits with an independent-gate model, entirely offline.

    The live properties omit frequency. At zero temperature, frequency is
    unnecessary. Construct the standard gate errors and an explicit Delay-only
    relaxation pass, avoiding Aer from_backend_properties' missing-frequency bug.
    Readout assignment is applied analytically to saved probabilities.
    """
    from qiskit.circuit import Delay
    from qiskit.transpiler import PassManager
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.device.models import basic_device_gate_errors, basic_device_readout_errors
    from qiskit_aer.noise.passes import RelaxationNoisePass
    from qiskit_ibm_runtime.models import BackendProperties
    from logical_magic.aachen_noise import syndrome_blocks, recorded_syndrome_state

    bundle, m, _ = load_bundle(bundle)
    physical = m["physical_patch"]
    remap = {q: i for i, q in enumerate(physical)}
    props = json.loads((bundle/"calibration.json").read_text())
    props["qubits"] = [props["qubits"][q] for q in physical]
    gates = []
    for gate in props["gates"]:
        if all(q in remap for q in gate["qubits"]):
            gate = deepcopy(gate)
            gate["qubits"] = [remap[q] for q in gate["qubits"]]
            gates.append(gate)
    props["gates"] = gates
    props = BackendProperties.from_dict(props)
    noise = NoiseModel(basis_gates=["rz", "sx", "x", "cz", "id"])
    for name, qubits, error in basic_device_gate_errors(props, gate_error=True, thermal_relaxation=True, temperature=0):
        noise.add_quantum_error(error, name, qubits)
    matrices = {q[0]: np.asarray(error.probabilities) for q, error in basic_device_readout_errors(props)}
    relax = PassManager(RelaxationNoisePass(
        t1s=[props.t1(q) for q in range(len(physical))],
        t2s=[min(props.t2(q), 2*props.t1(q)) for q in range(len(physical))],
        dt=dt, op_types=Delay, excited_state_populations=[0.]*len(physical)))
    sim = AerSimulator(method="density_matrix", noise_model=noise,
                       max_parallel_threads=1, max_parallel_experiments=1)
    data = m["resources"]["final_data_local_wires"]
    syn = m["resources"]["syndrome_local_wires_in_bit_order"]
    syn_a = [matrices[q] for q in syn]
    data_a = [matrices[q] for q in data]
    with (bundle/"local_circuits.qpy").open("rb") as handle:
        circuits = qpy.load(handle)
    raw = m["identity_coefficient"]
    terms = []
    for index, (circuit, setting) in enumerate(zip(circuits, m["settings"])):
        qc = circuit.copy_empty_like()
        for inst in circuit.data:
            if inst.operation.name != "measure":
                qc.append(inst)
        if index == 0:
            qc.save_density_matrix(label="state")
        else:
            qc.save_probabilities([*syn, *data], label="probabilities")
        result = sim.run(relax.run(qc), shots=1).result()
        if not result.success:
            raise RuntimeError(result.status)
        if index == 0:
            blocks = syndrome_blocks(np.asarray(result.data(0)["state"]), data, syn)
            state, metrics = recorded_syndrome_state(blocks, syn_a)
            fidelity = float(np.vdot(CODES["four"].target, state @ CODES["four"].target).real)
        else:
            value = expected_measurement_score(np.asarray(result.data(0)["probabilities"]), 3,
                                               syn_a, data_a, parity_weights(setting["pauli"]))
            raw += setting["coefficient"]*value
            terms.append({"pauli": setting["pauli"], "expectation": value})
        if index % 4 == 0:
            print(f"Offline calibration replay: {index+1}/{len(circuits)} circuits", flush=True)
    return {"dt_seconds": dt, "dt_source": "Live backend inspection at preparation: 4 ns",
            "calibration_date": m["calibration_date"], "acceptance": metrics["acceptance"],
            "raw_measurement_score": raw, "exact_model_preparation_fidelity": fidelity,
            "syndrome_probabilities": metrics["recorded_syndrome_probabilities"], "terms": terms,
            "assumptions": "Saved gate errors, T1/T2, asymmetric independent assignment, zero temperature. "
                           "Frozen circuit delays retained; ancilla readout deferred only in this independent-channel model. "
                           "No added initialization error, coherent/correlated noise, crosstalk, or unreported runtime latency. "
                           "Model predictions, not hardware-corrected estimates."}


def report(a, bundle):
    primary = a["primary_acceptance"]
    accepted = a["groups"]["accepted"]
    low, high = accepted["normal_95_interval"]
    lines = ["## Material Passport", "", "- Origin Skill: academic-research-suite / experiment-agent",
             "- Origin Mode: validate", "- Origin Date: 2026-10-14 (America/Los_Angeles)",
             "- Verification Status: ANALYZED", "- Version Label: ibm_b2ed_hardware_analysis_v1", "",
             f"# IBM Aachen: four-qubit B2_ED pilot ({a['variant']})", "",
             f"Job `{a['job_id']}`, instance `qzh-eu`, 24 circuits × 2,048 shots = {a['total_shots']:,} shots.", "",
             f"**Raw accepted-state fidelity estimate: {accepted['raw_fidelity']:.6f}**, with approximate 95% "
             f"sampling interval **[{low:.6f}, {high:.6f}]**. The dedicated acceptance circuit retained "
             f"**{primary['accepted_shots']:,}/{primary['shots']:,} = {primary['acceptance']:.3%}** "
             f"(Wilson 95% interval [{primary['acceptance_wilson_95_interval'][0]:.3%}, {primary['acceptance_wilson_95_interval'][1]:.3%}]).", "",
             "## Accepted, all, and rejected shots", "",
             "All three rows use the same B2_ED circuit data. 'All' does not represent a separate B1/no-check hardware control.", "",
             "| Shots used | Raw target-projector estimate | Approx. 95% sampling interval | Raw codespace-projector score |",
             "|---|---:|---:|---:|"]
    for group, row in a["groups"].items():
        lines.append(f"| {group} | {row['raw_fidelity']:.6f} | [{row['normal_95_interval'][0]:.6f}, "
                     f"{row['normal_95_interval'][1]:.6f}] | {row['raw_codespace_score']:.6f} |")
    diff = a["accepted_minus_all"]
    lines += ["", f"Selecting recorded `000` is associated with a {100*diff['difference']:.2f} percentage-point increase "
              f"over retaining all shots (joint-bootstrap 95% interval [{100*diff['bootstrap_95_interval'][0]:.2f}, "
              f"{100*diff['bootstrap_95_interval'][1]:.2f}] points). This is a conditional-selection comparison, "
              "not evidence that adding syndrome gates improves preparation versus a no-check circuit.", "",
              "The codespace score comes from the seven nonidentity stabilizer-group terms already measured. "
              "It includes measurement bias. It is not a direct measurement of an exact density matrix, "
              "and it cannot separate physical codespace loss from readout/basis errors. "
              "No extra codespace filtering is applied, and no within-codespace physical fidelity is asserted.", "",
              "## Syndrome diagnostics", "",
              f"Across all circuits, {a['pooled_acceptance']['accepted']:,}/{a['total_shots']:,} shots were accepted "
              f"({a['pooled_acceptance']['estimate']:.3%}). Per-circuit acceptance ranged from "
              f"{a['acceptance_across_settings']['minimum']:.3%} to {a['acceptance_across_settings']['maximum']:.3%}. "
              f"An exploratory acceptance-homogeneity test gives chi-square({a['acceptance_across_settings']['dof']}) = "
              f"{a['acceptance_across_settings']['chi_square']:.3f}, p = {a['acceptance_across_settings']['p_value']:.4f}. "
              "This does not establish absence of drift or gate-dependent bias.", "",
              "Bit order is `s2 s1 s0`, corresponding to checks `ZZII`, `IIZZ`, `XXXX`.", "",
              "| Recorded syndrome | Shots | Fraction |", "|---|---:|---:|"]
    for syndrome, count in a["syndrome_counts"].items():
        lines.append(f"| `{syndrome}` | {count:,} | {count/a['total_shots']:.3%} |")
    lines += ["", "| Check | Syndrome bit | Physical ancilla at readout | Recorded 1 fraction | Saved assignment-error average |",
              "|---|---:|---:|---:|---:|"]
    synrows = [v for v in a["saved_readout_calibration"] if v["role"] == "syndrome"]
    for label, item in zip(CODES["four"].stabilizers, synrows):
        lines.append(f"| `{label}` | {item['bit']} | {item['physical_qubit']} | "
                     f"{a['syndrome_bit_one_probabilities'][item['bit']]:.3%} | {item['readout_error']:.3%} |")
    lines += ["", "A syndrome does not uniquely locate a hardware fault. The recorded-1 fractions are much larger "
              "than the saved nominal ancilla assignment-error averages, but this alone does not identify the cause.", "",
              "## Comparison with simulations", "",
              "Compare hardware raw scores with model raw measurement scores, not with the model's exact delivered-state fidelity.", "",
              "| Source | Acceptance | Raw measurement score | Exact model state fidelity (simulation only) |",
              "|---|---:|---:|---:|",
              f"| Hardware, this job | {primary['acceptance']:.3%} | {accepted['raw_fidelity']:.6f} | unavailable |"]
    old = a["earlier_snapshot_simulation"]
    lines.append(f"| Earlier April 17 snapshot, no DD | {old['acceptance']:.3%} | {old['raw_measurement_score']:.6f} | {old['exact_preparation_fidelity']:.6f} |")
    if "saved_calibration_replay" in a:
        replay = a["saved_calibration_replay"]
        lines.append(f"| Saved live calibration, frozen circuits | {replay['acceptance']:.3%} | "
                     f"{replay['raw_measurement_score']:.6f} | {replay['exact_model_preparation_fidelity']:.6f} |")
        lines += ["", "Replay assumptions: " + replay["assumptions"]]
    if "baseline_comparison" in a:
        lines += comparison_report(a)
    lines += ["", "The hardware/model gap is a model-validation finding. This single job does not distinguish "
              "unmodeled coherent/correlated noise, readout crosstalk, runtime timing, calibration drift, or other causes. "
              "A no-check control and dedicated measurement calibrations would be needed for a stronger diagnosis; "
              "neither was run in this pilot.", "", "## Uncertainty and validation", "",
              "Every PUB contains the requested 2,048 shots. Register widths, Pauli metadata, circuit-file hashes, "
              "and archived joint counts were checked. A separate parity calculation reproduces the retrieval script's "
              "fidelity and standard error within 1e-12. All 23 preselected Pauli settings are included.", "",
              f"The {a['bootstrap']['samples']:,}-replicate multinomial bootstrap gives accepted-state interval "
              f"[{accepted['bootstrap_95_interval'][0]:.6f}, {accepted['bootstrap_95_interval'][1]:.6f}]. "
              f"The original conservative Hoeffding/union-bound interval is [{a['conservative_primary_95_interval'][0]:.6f}, "
              f"{a['conservative_primary_95_interval'][1]:.6f}]. All intervals concern sampling uncertainty, "
              "assuming independent shots and a stable preparation/measurement process. They exclude systematic measurement "
              "bias, calibration uncertainty, and drift; more shots alone do not remove those errors.", "",
              "Hardware readout mitigation has not been applied. Saved calibration values are contextual diagnostics, "
              "not dedicated in-circuit measurement calibrations. Estimates are not clipped to [0,1].", "",
              "### Statistical interpretation audit", "", "Coverage: 11/11 fallacy types considered.", "",
              "| Check | Assessment |", "|---|---|"]
    for name, finding in [
        ("Simpson's paradox", "Per-setting yields retained; pooled yield is supplementary."),
        ("Ecological fallacy", "Inferences restricted to this circuit/accepted ensemble, not individual fault locations."),
        ("Berkson selection", "Acceptance is intentional conditioning; it does not represent all output states."),
        ("Collider bias", "No causal attribution is made from syndrome-conditioned correlations."),
        ("Base-rate neglect", "Attempted, accepted and rejected counts are all reported."),
        ("Regression to the mean", "No before/after or selected-best-run claim; one pilot only."),
        ("Survivorship bias", "Rejected shots archived and analyzed; postselection yield reported."),
        ("Look-elsewhere effect", "All preselected terms retained; one exploratory homogeneity test, no search for significant subsets."),
        ("Forking paths", "Primary raw estimator fixed before submission; subgroup and model comparisons labeled diagnostics."),
        ("Correlation/causation", "Accepted/all comparison does not establish benefit over a separate B1 control."),
        ("Reverse causality", "No directional fault-mechanism claim from these measurements."),
    ]:
        lines.append(f"| {name} | {finding} |")
    lines += ["", "## Reproduction", "", "Hardware was not rerun. Analysis is deterministic from the saved inputs; "
              "bootstrap uses seed 20261014. No additional IBM jobs or package changes were made.", "",
              "```bash", "conda run -n quantum-jax python scripts/analyze_ibm_b2ed.py \\",
              f"  --bundle {bundle.relative_to(ROOT)} --replay-calibration --dt 4e-9", "```", "",
              "Outputs: `analysis_extended.json`, `pauli_estimates.csv`, `hardware_analysis.png`, "
              "`hardware_analysis.pdf`, and this report. Raw measurements remain in `runtime_result.json` and `joint_counts.json`.", ""]
    return "\n".join(lines)


def comparison_report(a):
    comparison = a["baseline_comparison"]
    lines = ["", "## XY4 versus the previous no-DD hardware job", "",
             f"No-DD job: `{comparison['baseline_job_id']}`. XY4 job: `{comparison['xy4_job_id']}`. "
             f"Submission separation: {comparison['submission_gap_minutes']:.2f} minutes. "
             f"Identical archived calibration: {comparison['identical_saved_calibration']}. "
             "Circuit mapping, original gate times, total duration, shots and Pauli settings match.", "",
             "| Metric | No DD | XY4 | Change (percentage points) | Approx. 95% interval for change (points) |",
             "|---|---:|---:|---:|---:|"]
    names = {"raw_fidelity_accepted": "Raw accepted fidelity", "primary_acceptance": "Dedicated-circuit acceptance",
             "pooled_acceptance": "Pooled acceptance (supplementary)", "raw_fidelity_all": "Raw fidelity, all shots",
             "raw_fidelity_rejected": "Raw fidelity, rejected shots", "accepted_codespace_score": "Raw accepted codespace score"}
    for key, name in names.items():
        r = comparison["metrics"][key]
        lines.append(f"| {name} | {100*r['no_dd']:.3f}% | {100*r['xy4']:.3f}% | {100*r['difference']:+.3f} | "
                     f"[{100*r['normal_95_interval'][0]:+.3f}, {100*r['normal_95_interval'][1]:+.3f}] |")
    delta = comparison["metrics"]["raw_fidelity_accepted"]
    lines += ["", f"The independent-job bootstrap interval for the raw accepted-fidelity change is "
              f"[{100*delta['independent_bootstrap_95_interval'][0]:+.3f}, {100*delta['independent_bootstrap_95_interval'][1]:+.3f}] points. "
              "The raw-fidelity increase exceeds the estimated finite-shot uncertainty. "
              "The dedicated acceptance-only circuits do not establish an acceptance increase at the approximate 95% level; "
              "the supplementary pooled counts suggest a smaller increase with more precision.", "",
              comparison["interpretation"], "",
              "The higher score is consistent with a benefit from XY4, but a causal DD effect is not isolated from drift. "
              "A future interleaved no-DD/XY4 comparison would address that limitation; no further jobs were submitted.", "",
              "### Simulator comparison for the two frozen circuits", "",
              "The independent Markovian model does not include slow correlated/coherent noise that DD can refocus. "
              "Its difference should not be used as a prediction of DD's full hardware benefit."]
    if "saved_calibration_replay" in a:
        before = comparison["baseline_summary"]["saved_calibration_replay"]
        after = a["saved_calibration_replay"]
        lines += ["", "| Replay | Acceptance | Raw fidelity score | Exact model fidelity |",
                  "|---|---:|---:|---:|",
                  f"| No DD | {before['acceptance']:.3%} | {before['raw_measurement_score']:.6f} | {before['exact_model_preparation_fidelity']:.6f} |",
                  f"| XY4 | {after['acceptance']:.3%} | {after['raw_measurement_score']:.6f} | {after['exact_model_preparation_fidelity']:.6f} |"]
    return lines


def plot(a, bundle):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qompute-hardware-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    groups = ["all", "accepted", "rejected"]
    values = [a["groups"][g]["raw_fidelity"] for g in groups]
    errors = [1.959964*a["groups"][g]["standard_error"] for g in groups]
    axes[0].bar(groups, values, color=["#7895ad", "#238b8e", "#c69076"], width=.65)
    axes[0].errorbar(groups, values, yerr=errors, fmt="none", color="#263238", capsize=4)
    for i, value in enumerate(values):
        axes[0].text(i, value+.035, f"{value:.3f}", ha="center")
    axes[0].set(ylim=(0, 1), ylabel="Raw target-projector estimate", title="Effect of recorded-syndrome selection")
    labels = list(a["syndrome_counts"])
    axes[1].bar(labels, [a["syndrome_counts"][s]/a["total_shots"] for s in labels], color="#6588aa")
    axes[1].set(ylabel="Fraction of attempted shots", title="Recorded syndromes (all 49,152 shots)", xlabel="s2 s1 s0")
    axes[1].tick_params(axis="x", rotation=45)
    accept = a["primary_acceptance"]["acceptance"]
    accept_ci = a["primary_acceptance"]["acceptance_wilson_95_interval"]
    axes[2].errorbar([0, 1], [accept, values[1]],
                    yerr=[[accept-accept_ci[0], errors[1]], [accept_ci[1]-accept, errors[1]]],
                    fmt="o", color="#238b8e", capsize=4, label="Hardware (95% sampling interval)")
    old = a["earlier_snapshot_simulation"]
    axes[2].scatter([0, 1], [old["acceptance"], old["raw_measurement_score"]], marker="x", s=60, color="#8f789f", label="April snapshot model (no DD)")
    if "saved_calibration_replay" in a:
        r = a["saved_calibration_replay"]
        axes[2].scatter([0, 1], [r["acceptance"], r["raw_measurement_score"]], marker="_", s=180, color="#ca902c", label="Saved live-calibration model")
    axes[2].set(xticks=[0, 1], xticklabels=["Acceptance", "Raw fidelity"], ylim=(0, 1.03), xlim=(-.5, 1.5), title="Hardware and model predictions")
    axes[2].legend(loc="lower left", fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"IBM Aachen · [[4,1,2]] B2_ED H-state · {a['variant']} · raw measurements", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(bundle/f"hardware_analysis.{ext}", dpi=180)
    plt.close(fig)


def plot_comparison(a, bundle):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qompute-hardware-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    comparison = a["baseline_comparison"]
    base = comparison["baseline_summary"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), layout="constrained")
    for index, (name, source, color) in enumerate((("No DD", base, "#718096"), ("XY4", a, "#168a88"))):
        f = source["groups"]["accepted"]
        acceptance = source["primary_acceptance"]
        y = [f["raw_fidelity"], acceptance["acceptance"]]
        lower = [1.959964*f["standard_error"], y[1]-acceptance["acceptance_wilson_95_interval"][0]]
        upper = [lower[0], acceptance["acceptance_wilson_95_interval"][1]-y[1]]
        x = np.arange(2)+(index-.5)*.24
        axes[0].errorbar(x, y, yerr=[lower, upper], fmt="o", capsize=4, color=color, label=name, markersize=7)
        for xi, yi in zip(x, y):
            axes[0].text(xi, yi+.035, f"{100*yi:.2f}%", ha="center", fontsize=9, color=color)
        axes[2].bar(np.arange(3)+(index-.5)*.35, source["syndrome_bit_one_probabilities"], width=.33, color=color, label=name)
    axes[0].set(xticks=[0, 1], xticklabels=["Accepted raw fidelity", "Acceptance-only circuit"],
                xlim=(-.5, 1.5), ylim=(.4, .85), title="Hardware results (95% sampling intervals)")
    axes[0].legend(loc="lower left")
    keys = ["raw_fidelity_accepted", "primary_acceptance", "pooled_acceptance"]
    rows = [comparison["metrics"][k] for k in keys]
    y = [100*r["difference"] for r in rows]
    axes[1].errorbar(y, np.arange(3), xerr=[1.959964*100*r["standard_error"] for r in rows], fmt="o", color="#168a88", capsize=4)
    axes[1].axvline(0, color="#b0b0b0", linestyle="--", linewidth=1)
    axes[1].set(yticks=np.arange(3), yticklabels=["Accepted fidelity", "Dedicated acceptance", "Pooled acceptance"],
                xlabel="XY4 − no DD (percentage points)", title="Observed changes", ylim=(2.5, -.5))
    axes[2].set(xticks=np.arange(3), xticklabels=["XXXX / s0", "IIZZ / s1", "ZZII / s2"],
                ylabel="Recorded syndrome bit = 1", title="Check outcomes (pooled shots)", ylim=(0, .42))
    for axis in (axes[0], axes[2]):
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("IBM Aachen · B2_ED: XY4 versus no DD\nRaw measurements; sequential jobs, drift and measurement bias excluded from intervals", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(bundle/f"xy4_vs_no_dd.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--replay-calibration", action="store_true")
    parser.add_argument("--dt", type=float, default=4e-9, help="Device dt in seconds, verified at preparation")
    parser.add_argument("--compare-to", type=Path, help="Previously analyzed no-DD source bundle")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    analysis = analyze(bundle)
    if args.replay_calibration:
        replay = replay_calibration(bundle, args.dt)
        write_json(bundle/"calibration_replay.json", replay)
        analysis["saved_calibration_replay"] = replay
    if args.compare_to:
        comparison = compare_with_baseline(bundle, analysis, args.compare_to)
        analysis["baseline_comparison"] = comparison
        write_json(bundle/"xy4_vs_no_dd.json", comparison)
    write_json(bundle/"analysis_extended.json", analysis)
    with (bundle/"pauli_estimates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["per_pauli"][0]))
        writer.writeheader()
        writer.writerows(analysis["per_pauli"])
    (bundle/"hardware_report.md").write_text(report(analysis, bundle))
    plot(analysis, bundle)
    if args.compare_to:
        (bundle/"xy4_vs_no_dd.md").write_text("# XY4 versus no DD\n"+"\n".join(comparison_report(analysis))+"\n")
        plot_comparison(analysis, bundle)
    print(json.dumps({"job_id": analysis["job_id"], "groups": analysis["groups"],
                      "pooled_acceptance": analysis["pooled_acceptance"],
                      "report": str(bundle/"hardware_report.md")}, indent=2))


if __name__ == "__main__":
    main()
