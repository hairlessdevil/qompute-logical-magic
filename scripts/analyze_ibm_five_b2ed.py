#!/usr/bin/env python3
"""Analyze the two archived five-qubit B2_ED jobs, entirely offline.

Data-only assignment correction preserves the recorded-0000 selection.
Joint data/syndrome inversion is a separate latent-0000 diagnostic.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import RuntimeDecoder
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logical_magic.codes import CODES
from logical_magic.ibm_b2ed import file_hash, joint_runtime_counts, load_bundle, write_json
from logical_magic.noisy_fidelity import accepted_score, parity_weights
from logical_magic.simulation import wilson_interval
from mitigate_ibm_b2ed_readout import (
    MODES, calibration_matrices, histogram, ratio_stats, score_weights,
    summary, tensor_assignment,
)


def group_weights(pauli, n_syndrome, group):
    """Independent bit-count parity implementation, including all syndrome bits."""
    size = 2**(len(pauli)+n_syndrome)
    support = sum(1 << q for q, p in enumerate(reversed(pauli)) if p != "I")
    parity = np.array([(-1.)**((x >> n_syndrome) & support).bit_count() for x in range(size)])
    accepted = np.array([(x & (2**n_syndrome-1)) == 0 for x in range(size)])
    select = accepted if group == "accepted" else ~accepted if group == "rejected" else np.ones(size, bool)
    return parity*select, select.astype(float)


def analyze(bundle, samples=10000, seed=20261020):
    bundle, manifest, _ = load_bundle(bundle)
    if (manifest["n_data"], manifest["n_syndrome"]) != (5, 4):
        raise ValueError("This report requires the five-qubit B2_ED protocol")
    counts = json.loads((bundle / "joint_counts.json").read_text())
    runtime = json.loads((bundle / "runtime_result.json").read_text(), cls=RuntimeDecoder)
    original = json.loads((bundle / "analysis.json").read_text())
    receipt = json.loads((bundle / "submission.json").read_text())
    calibration = json.loads((bundle / "calibration.json").read_text())
    assert receipt["manifest_sha256"] == file_hash(bundle / "manifest.json")
    assert len(counts) == len(runtime) == manifest["number_of_circuits"] == 64
    assert original["job_id"] == receipt["job_id"]
    n = manifest["shots_per_circuit"]
    syndrome, acceptance = Counter(), []
    for pub, setting, saved in zip(runtime, manifest["settings"], counts, strict=True):
        assert pub.data.syndrome.num_bits == 4 and pub.data.syndrome.num_shots == n
        if setting["kind"] == "acceptance":
            reconstructed = pub.data.syndrome.get_counts()
        else:
            assert pub.data.quality_readout.num_bits == 5 and pub.data.quality_readout.num_shots == n
            assert pub.metadata["circuit_metadata"]["analysis_pauli"] == setting["pauli"]
            reconstructed = joint_runtime_counts(pub, n_syndrome=4, n_data=5)
        assert reconstructed == saved and sum(saved.values()) == n
        h = pub.data.syndrome.get_counts()
        syndrome.update(h)
        acceptance.append(h.get("0000", 0))

    data_a, data_rows = calibration_matrices(calibration, manifest["resources"]["final_data_physical_qubits"])
    syn_a, syn_rows = calibration_matrices(calibration, manifest["resources"]["syndrome_physical_qubits_in_bit_order"])
    da, sa = tensor_assignment(data_a), tensor_assignment(syn_a)
    project = {p: float(c.real) for p, c in SparsePauliOp.from_operator(CODES["five"].projector).to_list()}
    measured = {s["pauli"] for s in manifest["settings"][1:]}
    assert set(project) - {"IIIII"} <= measured
    f = np.full(3, manifest["identity_coefficient"])
    cp, var, cpvar = np.full(3, project["IIIII"]), np.zeros(3), np.zeros(3)
    boot, cpboot = np.tile(f, (samples, 1)), np.tile(cp, (samples, 1))
    groups = {g: {"f": f[0], "var": 0., "cp": cp[0], "cpvar": 0., "boot": np.full(samples, f[0])}
              for g in ("accepted", "all", "rejected")}
    sensitivity = {s: np.copy(f) for s in (0., .5, 1., 1.5)}
    scaled = {s: (calibration_matrices(calibration, manifest["resources"]["final_data_physical_qubits"], s)[0],
                  calibration_matrices(calibration, manifest["resources"]["syndrome_physical_qubits_in_bit_order"], s)[0]) for s in sensitivity}
    rng, terms, errors = np.random.default_rng(seed), [], []
    for index, (setting, saved) in enumerate(zip(manifest["settings"][1:], counts[1:], strict=True), 1):
        c, pauli = histogram(saved, 9), setting["pauli"]
        u, v = score_weights(pauli, data_a, syn_a)
        mean, variance, denominator = ratio_stats(c, u, v)
        for mode, matrices in enumerate((None, data_a)):
            independent = accepted_score(saved, 4, parity_weights(pauli, matrices))
            np.testing.assert_allclose(mean[mode], independent["expectation"], atol=1e-12, rtol=0)
            variance[mode] = independent["variance_of_mean"]
        draws = rng.multinomial(n, c/n, size=samples)
        d = draws @ v
        if np.any(d <= 0):
            raise ValueError("Nonpositive bootstrap denominator; no replicates may be discarded")
        means = (draws @ u)/d
        coefficient, pc = setting["coefficient"], project.get(pauli, 0.)
        f += coefficient*mean
        var += coefficient**2*variance
        cp += pc*mean
        cpvar += pc**2*variance
        boot += coefficient*means
        cpboot += pc*means
        for group, g in groups.items():
            gu, gv = group_weights(pauli, 4, group)
            selected = c @ gv
            gm = (c @ gu)/selected
            gvar = (1-gm**2)/(selected-1)
            g["f"] += coefficient*gm
            g["var"] += coefficient**2*gvar
            g["cp"] += pc*gm
            g["cpvar"] += pc**2*gvar
            g["boot"] += coefficient*(draws @ gu)/(draws @ gv)
        observed = (c/n).reshape(32, 16)
        corrected_data = np.linalg.solve(da.T, observed)
        corrected_joint = np.linalg.solve(sa.T, corrected_data.T).T
        parity = parity_weights(pauli)
        direct = [observed[:, 0] @ parity/observed[:, 0].sum(),
                  corrected_data[:, 0] @ parity/corrected_data[:, 0].sum(),
                  corrected_joint[:, 0] @ parity/corrected_joint[:, 0].sum()]
        error = float(np.max(np.abs(mean-direct)))
        assert error < 1e-11
        errors.append(error)
        terms.append({"pauli": pauli, "coefficient": coefficient, "accepted_shots": int(c.reshape(32, 16)[:, 0].sum()),
                      "expectations": dict(zip(MODES, map(float, mean))),
                      "standard_errors": dict(zip(MODES, map(float, np.sqrt(variance)))),
                      "conditional_probability": dict(zip(MODES, map(float, denominator))),
                      "minimum_joint_quasiprobability": float(corrected_joint.min()),
                      "joint_negative_mass": float(-np.minimum(corrected_joint, 0).sum()),
                      "minimum_bootstrap_joint_denominator": float(d[:, 2].min()/n)})
        for scale, matrices in scaled.items():
            su, sv = score_weights(pauli, *matrices)
            sensitivity[scale] += coefficient*ratio_stats(c, su, sv)[0]
        if index % 21 == 0:
            print(f"{bundle.name}: analyzed {index}/63 Pauli settings", flush=True)

    for key, value in [("raw_fidelity_estimate", f[0]), ("standard_error", np.sqrt(var[0]))]:
        np.testing.assert_allclose(value, original[key], atol=1e-12, rtol=0)
    np.testing.assert_allclose([groups["accepted"]["f"], groups["accepted"]["var"]], [f[0], var[0]], atol=1e-12, rtol=0)
    accept_hist = histogram(counts[0], 4)
    aw = np.column_stack([np.eye(16)[0], np.linalg.solve(sa, np.eye(16)[0])])
    a = (accept_hist/n) @ aw
    av = np.maximum(0., (accept_hist/n) @ aw**2-a**2)/(n-1)
    aboot = rng.multinomial(n, accept_hist/n, size=samples) @ aw/n
    homogeneity = chi2_contingency([acceptance, [n-x for x in acceptance]])
    result = {
        "bundle": str(bundle), "job_id": receipt["job_id"], "submitted_at_utc": receipt["submitted_at_utc"],
        "variant": manifest["resources"].get("dd", {}).get("sequence", "no DD"),
        "target": manifest["target"], "code": manifest["code"], "total_shots": sum(syndrome.values()),
        "number_of_circuits": len(counts), "execution_spans": str(runtime.metadata["execution"]["execution_spans"]),
        "input_sha256": {name: file_hash(bundle/name) for name in
                         ("manifest.json", "runtime_result.json", "joint_counts.json", "analysis.json", "calibration.json", "submission.json")},
        "bootstrap": {"samples": samples, "seed": seed, "method": "Independent per-setting multinomial draws of all 512 joint data/syndrome outcomes; paired estimators within a job"},
        "calibration": {"data": data_rows, "syndrome": syn_rows,
                        "matrix_orientation": "rows=true, columns=recorded",
                        "data_tensor_condition_number": float(np.linalg.cond(da)),
                        "joint_tensor_condition_number": float(np.linalg.cond(da)*np.linalg.cond(sa))},
        "fidelity": {m: summary(f[i], var[i], boot[:, i]) for i, m in enumerate(MODES)},
        "codespace_score": {m: summary(cp[i], cpvar[i], cpboot[:, i]) for i, m in enumerate(MODES)},
        "groups": {g: {"raw_fidelity": r["f"], "standard_error": float(np.sqrt(r["var"])),
                        "normal_95_interval": summary(r["f"], r["var"], r["boot"])["normal_95_interval"],
                        "bootstrap_95_interval": np.quantile(r["boot"], [.025, .975]).tolist(),
                        "raw_codespace_score": r["cp"], "codespace_standard_error": float(np.sqrt(r["cpvar"]))}
                   for g, r in groups.items()},
        "dedicated_acceptance": {name: summary(a[i], av[i], aboot[:, i]) for i, name in enumerate(("recorded_zero", "true_zero_diagnostic"))},
        "primary_acceptance": original["acceptance_only_circuit"],
        "pooled_acceptance": {"accepted": syndrome["0000"], "attempted": sum(syndrome.values()),
                              "estimate": syndrome["0000"]/sum(syndrome.values()),
                              "wilson_95_interval": list(wilson_interval(syndrome["0000"], sum(syndrome.values())))},
        "acceptance_across_settings": {"minimum": min(acceptance)/n, "maximum": max(acceptance)/n,
                                       "chi_square": float(homogeneity.statistic), "dof": int(homogeneity.dof),
                                       "p_value": float(homogeneity.pvalue)},
        "syndrome_counts": dict(sorted(syndrome.items())),
        "paired_mitigation_shift": {"estimate": float(f[1]-f[0]), "bootstrap_95_interval": np.quantile(boot[:, 1]-boot[:, 0], [.025, .975]).tolist()},
        "accepted_minus_all": {"estimate": float(f[0]-groups["all"]["f"]), "bootstrap_95_interval": np.quantile(boot[:, 0]-groups["all"]["boot"], [.025, .975]).tolist()},
        "assignment_rate_sensitivity": [{"rate_multiplier": s, "fidelity": dict(zip(MODES, map(float, values)))} for s, values in sensitivity.items()],
        "per_pauli": terms,
        "validation": {"runtime_bit_arrays_match_all_archived_histograms": True,
                       "retrieval_fidelity_and_se_reproduced": True,
                       "independent_bitcount_estimator_matches": True,
                       "max_distribution_vs_observable_solve_error": max(errors),
                       "minimum_bootstrap_joint_denominator": min(t["minimum_bootstrap_joint_denominator"] for t in terms),
                       "quasiprobabilities_clipped": False, "fidelity_clipped": False},
    }
    return result, boot, cpboot, aboot


def compare(baseline, xy4, samples=10000):
    bdir, bm, _ = load_bundle(baseline)
    xdir, xm, _ = load_bundle(xy4)
    assert xm["source_manifest_sha256"] == file_hash(bdir/"manifest.json")
    assert xm["source_circuits_sha256"] == file_hash(bdir/"circuits.qpy")
    for k in ("code", "strategy", "target", "stabilizers_in_syndrome_bit_order", "shots_per_circuit", "settings"):
        assert bm[k] == xm[k], k
    for k in ("duration_seconds", "final_data_physical_qubits", "syndrome_physical_qubits_in_bit_order", "syndrome_readout_wait_seconds"):
        assert bm["resources"][k] == xm["resources"][k], k
    assert file_hash(bdir/"calibration.json") == file_hash(xdir/"calibration.json")
    before, bb, bcp, ba = analyze(bdir, samples, 20261020)
    after, xb, xcp, xa = analyze(xdir, samples, 20261021)
    differences = {}
    for field, draws in (("fidelity", xb-bb), ("codespace_score", xcp-bcp), ("dedicated_acceptance", xa-ba)):
        keys = MODES if field != "dedicated_acceptance" else ("recorded_zero", "true_zero_diagnostic")
        differences[field] = {m: summary(after[field][m]["estimate"]-before[field][m]["estimate"],
                              after[field][m]["standard_error"]**2+before[field][m]["standard_error"]**2, draws[:, i])
                              for i, m in enumerate(keys)}
    time_gap = (datetime.fromisoformat(before["submitted_at_utc"])-datetime.fromisoformat(after["submitted_at_utc"])).total_seconds()/60
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "runs": [before, after], "comparison": differences,
              "matching": {"identical_saved_calibration": True, "matched_layout_timing_settings": True,
                           "submission_order": "XY4 then no DD" if time_gap > 0 else "no DD then XY4",
                           "submission_separation_minutes": abs(time_gap)},
              "method": {"primary": "Data-only assignment correction, conditional on recorded 0000",
                         "diagnostic": "Joint data/syndrome inversion, conditional on inferred true 0000; a different ensemble",
                         "uncertainty": "95% sampling intervals, fixed saved calibration; excludes calibration uncertainty, correlated readout, basis-rotation errors and inter-job drift",
                         "no_new_experiments": True, "active_recovery": False, "extra_codespace_filter": False},
              "script_sha256": file_hash(Path(__file__)),
              "helper_script_sha256": file_hash(ROOT/"scripts/mitigate_ibm_b2ed_readout.py")}
    for directory, run in ((bdir, before), (xdir, after)):
        write_json(directory/"analysis_extended.json", run)
    return result


def report(result):
    lines = ["# Five-qubit B2_ED: no DD versus XY4", "",
             "Target: [[5,1,3]] encoded positive tetrahedral state. Both jobs used ibm_aachen, 64 circuits and 2,048 shots/circuit (131,072 shots/job). Accept recorded syndrome 0000. No active recovery.", "",
             "| Run | Raw fidelity (95% sampling CI) | Data-corrected fidelity (95% sampling CI) | Dedicated acceptance (Wilson 95% CI) |", "|---|---:|---:|---:|"]
    fmt = lambda x, key='normal_95_interval': f"{x['estimate']:.3%} [{x[key][0]:.3%}, {x[key][1]:.3%}]"
    for r in result["runs"]:
        a = r["primary_acceptance"]
        lines.append(f"| {r['variant']} | {fmt(r['fidelity'][MODES[0]])} | {fmt(r['fidelity'][MODES[1]])} | {a['acceptance']:.3%} [{a['acceptance_wilson_95_interval'][0]:.3%}, {a['acceptance_wilson_95_interval'][1]:.3%}] |")
    d = result["comparison"]["fidelity"][MODES[1]]
    lines += ["", f"Data-corrected XY4 minus no DD: {100*d['estimate']:+.3f} percentage points; normal 95% interval {np.round(100*np.array(d['normal_95_interval']), 3).tolist()}, independent-bootstrap interval {np.round(100*np.array(d['bootstrap_95_interval']), 3).tolist()}.", "",
              "Readout correction uses the archived per-qubit directional assignment probabilities. Data-only correction preserves the recorded-0000 accepted ensemble; it changes the estimator, not the prepared state. No new readout-calibration job was run.", "",
              f"Submission order: {result['matching']['submission_order']}; separation {result['matching']['submission_separation_minutes']:.2f} minutes. Both share the same saved calibration, mapping, schedule duration, and Pauli settings. Separate sequential jobs do not isolate DD from time-dependent hardware variation.", "",
              "Uncertainty includes finite-shot sampling only. Calibration uncertainty, correlated assignment errors, basis-rotation errors and drift are not included. Negative quasiprobabilities are retained; no clipping or extra codespace filtering is applied.", "", "## Detailed diagnostics", "",
              "| Quantity | No DD | XY4 |", "|---|---:|---:|"]
    for name, section, field in [("All-outcome raw target score", "groups", "all"), ("Rejected raw target score", "groups", "rejected")]:
        lines.append(f"| {name} | " + " | ".join(f"{r[section][field]['raw_fidelity']:.3%}" for r in result['runs']) + " |")
    for name, section, field in [("Raw codespace score", "codespace_score", MODES[0]), ("Data-corrected codespace score", "codespace_score", MODES[1]), ("Joint-corrected fidelity (different ensemble)", "fidelity", MODES[2])]:
        lines.append(f"| {name} | " + " | ".join(fmt(r[section][field]) for r in result['runs']) + " |")
    lines += ["", "## Provenance and checks", "", "All 128 Runtime PUBs reproduce the archived shot histograms. Raw fidelity and standard error match the retrieval estimator and an independent bit-count implementation. Assignment-corrected observable scores agree with independent full-distribution inversion. Bootstrap: 10,000 multinomial replicates per job; independent seeds between jobs, paired estimators within each job.", ""]
    lines += [f"- {r['variant']}: `{r['job_id']}`; [extended analysis](../{Path(r['bundle']).name}/analysis_extended.json)." for r in result['runs']]
    lines += ["", "Reproduce offline: `python scripts/analyze_ibm_five_b2ed.py`. Material status: ANALYZED (archived hardware; independently checked postprocessing)."]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    result = compare(ROOT/"results/hardware_b2ed/five_pilot_20261014", ROOT/"results/hardware_b2ed/five_xy4_20261014", args.samples)
    output = ROOT/"results/hardware_b2ed/five_comparison_20261014"
    output.mkdir(exist_ok=True)
    write_json(output/"analysis.json", result)
    (output/"report.md").write_text(report(result))
    print(json.dumps({"comparison": result["comparison"], "matching": result["matching"]}, indent=2))
