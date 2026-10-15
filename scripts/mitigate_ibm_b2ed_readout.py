#!/usr/bin/env python3
"""Offline assignment mitigation of the saved B2_ED jobs. No IBM calls.

Primary estimand: data-readout-corrected fidelity conditional on RECORDED 000.
Diagnostic estimand: data+syndrome correction conditional on latent TRUE 000.
The latter changes the selected ensemble and is not the delivered-state fidelity.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from qiskit.quantum_info import SparsePauliOp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logical_magic.codes import CODES
from logical_magic.ibm_b2ed import file_hash, load_bundle, write_json
from logical_magic.noisy_fidelity import accepted_score, parity_weights

MODES = ("raw_recorded_zero", "data_corrected_recorded_zero", "joint_corrected_true_zero")
LABELS = ("Raw; recorded 000", "Data corrected; recorded 000 (primary)",
          "Data + syndrome corrected; true 000 (diagnostic)")
Z95 = 1.959963984540054


def tensor_assignment(matrices):
    """Rows=true, columns=recorded; little-endian qubit list."""
    result = np.ones((1, 1))
    for matrix in reversed(matrices):
        result = np.kron(result, matrix)
    return result


def calibration_matrices(calibration, qubits, scale=1.):
    matrices, rows = [], []
    for bit, qubit in enumerate(qubits):
        props = {v["name"]: v for v in calibration["qubits"][qubit]}
        e01 = float(props["prob_meas1_prep0"]["value"])*scale
        e10 = float(props["prob_meas0_prep1"]["value"])*scale
        if min(e01, e10) < 0 or max(e01, e10) > 1 or e01+e10 >= 1:
            raise ValueError("Invalid or noninformative assignment calibration")
        matrix = np.array([[1-e01, e01], [e10, 1-e10]])
        matrices.append(matrix)
        rows.append({"bit": bit, "physical_qubit": qubit, "p_record_1_given_0": e01,
                     "p_record_0_given_1": e10, "matrix_true_recorded": matrix.tolist(),
                     "calibration_dates": [props[k]["date"] for k in ("prob_meas1_prep0", "prob_meas0_prep1")],
                     "condition_number": float(np.linalg.cond(matrix))})
    return matrices, rows


def histogram(counts, width):
    result = np.zeros(2**width, dtype=np.int64)
    for bits, count in counts.items():
        if len(bits) != width or set(bits)-{"0", "1"} or count < 0 or int(count) != count:
            raise ValueError("Invalid archived histogram")
        result[int(bits, 2)] += count
    return result


def score_weights(pauli, data_matrices, syndrome_matrices):
    """Return outcome weights for the numerator and denominator of each estimand."""
    ns = len(syndrome_matrices)
    syn_zero = np.eye(2**ns)[0]
    syn_inverse = np.linalg.solve(tensor_assignment(syndrome_matrices), syn_zero)
    raw = parity_weights(pauli)
    corrected = parity_weights(pauli, data_matrices)
    numerators = np.column_stack([np.kron(raw, syn_zero), np.kron(corrected, syn_zero),
                                  np.kron(corrected, syn_inverse)])
    denominators = np.column_stack([np.tile(syn_zero, len(raw)), np.tile(syn_zero, len(raw)),
                                    np.tile(syn_inverse, len(raw))])
    return numerators, denominators


def ratio_stats(counts, numerator, denominator):
    """Plug-in ratio and delta-method variance, retaining numerator/denominator covariance."""
    n = counts.sum()
    p = counts/n
    d = p @ denominator
    if np.any(d <= 0) or n < 2:
        raise ValueError("Nonpositive conditional probability or insufficient shots")
    mean = (p @ numerator)/d
    influence = (numerator-denominator*mean)/d
    variance = p @ (influence**2)/(n-1)
    return mean, variance, d


def summary(estimate, variance, draws):
    se = float(np.sqrt(variance))
    return {"estimate": float(estimate), "standard_error": se,
            "normal_95_interval": [float(estimate-Z95*se), float(estimate+Z95*se)],
            "bootstrap_95_interval": np.quantile(draws, [.025, .975]).tolist()}


def analyze_bundle(bundle, bootstrap_samples=10000, seed=20261017):
    bundle, manifest, _ = load_bundle(bundle)
    previous = json.loads((bundle/"analysis_extended.json").read_text())
    for name, digest in previous["input_sha256"].items():
        if file_hash(bundle/name) != digest:
            raise ValueError(f"Previously validated input changed: {name}")
    receipt = json.loads((bundle/"submission.json").read_text())
    if receipt["manifest_sha256"] != file_hash(bundle/"manifest.json"):
        raise ValueError("Submitted manifest changed")
    calibration = json.loads((bundle/"calibration.json").read_text())
    counts = json.loads((bundle/"joint_counts.json").read_text())
    if len(counts) != len(manifest["settings"]) or manifest["code"] != "[[4,1,2]]":
        raise ValueError("Unsupported protocol or missing settings")
    data_q = manifest["resources"]["final_data_physical_qubits"]
    syn_q = manifest["resources"]["syndrome_physical_qubits_in_bit_order"]
    data_a, data_rows = calibration_matrices(calibration, data_q)
    syn_a, syn_rows = calibration_matrices(calibration, syn_q)
    data_tensor, syn_tensor = tensor_assignment(data_a), tensor_assignment(syn_a)
    full_tensor = tensor_assignment(syn_a+data_a)
    project = dict((p, float(c.real)) for p, c in SparsePauliOp.from_operator(CODES["four"].projector).to_list())
    f = np.full(3, manifest["identity_coefficient"])
    cp = np.full(3, project["IIII"])
    variance, cp_variance = np.zeros(3), np.zeros(3)
    bootstrap = np.tile(f, (bootstrap_samples, 1))
    bootstrap_cp = np.tile(cp, (bootstrap_samples, 1))
    rng = np.random.default_rng(seed)
    terms, checks, sensitivity = [], [], {}
    scales = (0., .5, 1., 1.5)
    for scale in scales:
        sensitivity[scale] = np.full(3, manifest["identity_coefficient"])
    for setting, saved in zip(manifest["settings"][1:], counts[1:]):
        c = histogram(saved, 7)
        if c.sum() != manifest["shots_per_circuit"]:
            raise ValueError("Unexpected shot count")
        pauli = setting["pauli"]
        u, v = score_weights(pauli, data_a, syn_a)
        mean, var, denom = ratio_stats(c, u, v)
        # Match the previous raw estimator's exact conditional sample variance.
        for mode, a in enumerate((None, data_a)):
            independent = accepted_score(saved, 3, parity_weights(pauli, a))
            np.testing.assert_allclose(mean[mode], independent["expectation"], atol=1e-12)
            var[mode] = independent["variance_of_mean"]
        draws = rng.multinomial(int(c.sum()), c/c.sum(), size=bootstrap_samples)
        draw_denoms = draws @ v
        if np.any(draw_denoms <= 0):
            raise ValueError("Unstable bootstrap denominator; do not drop replicates")
        draw_means = (draws @ u)/draw_denoms
        coefficient, pc = setting["coefficient"], project.get(pauli, 0.)
        f += coefficient*mean
        variance += coefficient**2*var
        cp += pc*mean
        cp_variance += pc**2*var
        bootstrap += coefficient*draw_means
        bootstrap_cp += pc*draw_means

        # Independent full-distribution solves verify tensor orientation, mapping,
        # latent-syndrome conditioning, and weighted-observable implementation.
        observed = c/c.sum()
        corrected_data = np.linalg.solve(data_tensor.T, observed.reshape(16, 8))[:, 0]
        corrected_joint = np.linalg.solve(full_tensor.T, observed).reshape(16, 8)
        parity = parity_weights(pauli)
        direct = [observed.reshape(16, 8)[:, 0] @ parity/denom[0],
                  corrected_data @ parity/corrected_data.sum(),
                  corrected_joint[:, 0] @ parity/corrected_joint[:, 0].sum()]
        error = float(np.max(np.abs(mean-direct)))
        if error > 1e-11:
            raise AssertionError("Distribution solve and observable solve disagree")
        checks.append(error)
        terms.append({"pauli": pauli, "coefficient": coefficient,
                      "expectations": dict(zip(MODES, map(float, mean))),
                      "standard_errors": dict(zip(MODES, map(float, np.sqrt(var)))),
                      "conditional_probability": dict(zip(MODES, map(float, denom))),
                      "minimum_joint_quasiprobability": float(corrected_joint.min()),
                      "joint_negative_mass": float(-np.minimum(corrected_joint, 0).sum()),
                      "minimum_bootstrap_joint_denominator": float(draw_denoms[:, 2].min()/c.sum())})
        for scale in scales:
            da, _ = calibration_matrices(calibration, data_q, scale)
            sa, _ = calibration_matrices(calibration, syn_q, scale)
            su, sv = score_weights(pauli, da, sa)
            sensitivity[scale] += coefficient*ratio_stats(c, su, sv)[0]

    np.testing.assert_allclose(f[0], previous["groups"]["accepted"]["raw_fidelity"], atol=1e-12)
    np.testing.assert_allclose(np.sqrt(variance[0]), previous["groups"]["accepted"]["standard_error"], atol=1e-12)
    acceptance = histogram(counts[0], 3)
    if acceptance.sum() != manifest["shots_per_circuit"]:
        raise ValueError("Unexpected acceptance-circuit shots")
    weights = np.column_stack([np.eye(8)[0], np.linalg.solve(syn_tensor, np.eye(8)[0])])
    p = acceptance/acceptance.sum()
    a = p @ weights
    av = np.maximum(0., p @ weights**2-a**2)/(acceptance.sum()-1)
    adraws = rng.multinomial(int(acceptance.sum()), p, size=bootstrap_samples) @ weights/acceptance.sum()
    result = {
        "bundle": str(bundle), "job_id": receipt["job_id"],
        "variant": manifest["resources"].get("dd", {}).get("sequence", "no DD"),
        "input_sha256": {**previous["input_sha256"], "analysis_extended.json": file_hash(bundle/"analysis_extended.json"),
                         "submission.json": file_hash(bundle/"submission.json")},
        "calibration": {"data": data_rows, "syndrome": syn_rows, "matrix_orientation": "rows=true, columns=recorded",
                        "data_tensor_condition_number": float(np.linalg.cond(data_tensor)),
                        "joint_tensor_condition_number": float(np.linalg.cond(full_tensor))},
        "bootstrap_samples": bootstrap_samples, "seed": seed,
        "fidelity": {mode: summary(f[i], variance[i], bootstrap[:, i]) for i, mode in enumerate(MODES)},
        "codespace_score": {mode: summary(cp[i], cp_variance[i], bootstrap_cp[:, i]) for i, mode in enumerate(MODES)},
        "dedicated_acceptance": {name: summary(a[i], av[i], adraws[:, i]) for i, name in enumerate(("recorded_zero", "true_zero_diagnostic"))},
        "paired_mitigation_shift": {"estimate": float(f[1]-f[0]),
                                    "bootstrap_95_interval": np.quantile(bootstrap[:, 1]-bootstrap[:, 0], [.025, .975]).tolist()},
        "paired_syndrome_estimand_shift": {"estimate": float(f[2]-f[1]),
                                          "bootstrap_95_interval": np.quantile(bootstrap[:, 2]-bootstrap[:, 1], [.025, .975]).tolist()},
        "assignment_rate_sensitivity": [{"rate_multiplier": s, "fidelity": dict(zip(MODES, map(float, values)))}
                                        for s, values in sensitivity.items()],
        "per_pauli": terms,
        "validation": {"previous_raw_fidelity_and_se_reproduced": True,
                       "max_distribution_vs_observable_solve_error": max(checks),
                       "minimum_bootstrap_joint_denominator": min(t["minimum_bootstrap_joint_denominator"] for t in terms),
                       "quasiprobabilities_clipped": False, "fidelity_clipped": False},
    }
    return result, bootstrap


def report(output):
    before, after = output["runs"]
    lines = ["## Material Passport", "", "- Origin Skill: academic-research-suite / experiment-agent",
             "- Origin Mode: validate", f"- Origin Date: {output['created_at_utc']}",
             "- Verification Status: ANALYZED (archived hardware; postprocessing independently checked)",
             "- Version Label: ibm_b2ed_readout_mitigation_v1", "", "# Readout mitigation: no DD and XY4", "",
             "The primary correction estimates the fidelity of the actual recorded-000 accepted ensemble. "
             "It inverts only final data-bit assignment errors. It does not alter acceptance or repair the prepared quantum state.", "",
             "| Fidelity estimate | No DD | XY4 | XY4 minus no DD (percentage points; 95% sampling interval) |",
             "|---|---:|---:|---:|"]
    for mode, label in zip(MODES, LABELS):
        b, a = before["fidelity"][mode], after["fidelity"][mode]
        d = output["comparison"][mode]
        lines.append(f"| {label} | {b['estimate']:.3%} | {a['estimate']:.3%} | {100*d['estimate']:+.3f} "
                     f"[{100*d['normal_95_interval'][0]:+.3f}, {100*d['normal_95_interval'][1]:+.3f}] |")
    lines += ["", "## Primary estimates and uncertainty", "",
              "| Job | Data-corrected fidelity | 95% sampling interval | Paired change from raw (points) |",
              "|---|---:|---:|---:|"]
    for r in output["runs"]:
        f = r["fidelity"][MODES[1]]
        delta = r["paired_mitigation_shift"]
        lines.append(f"| {r['variant']} (`{r['job_id']}`) | {f['estimate']:.3%} | "
                     f"[{f['normal_95_interval'][0]:.3%}, {f['normal_95_interval'][1]:.3%}] | "
                     f"{100*delta['estimate']:+.3f} (paired bootstrap [{100*delta['bootstrap_95_interval'][0]:+.3f}, "
                     f"{100*delta['bootstrap_95_interval'][1]:+.3f}]) |")
    lines += ["", "All intervals condition on the saved assignment matrices being exact and applicable. "
              "They include finite experimental shot uncertainty only, not calibration uncertainty, drift, "
              "basis-rotation errors, leakage or correlated readout. Each job has 23 Pauli settings plus one "
              "dedicated acceptance circuit, each with 2,048 shots. The 10,000-replicate bootstrap resamples "
              "each full joint histogram, preserving syndrome/data correlations and pairing raw/corrected estimates. "
              "Jobs use distinct random streams. Intervals are pointwise and exploratory.", "",
              "The XY4 and no-DD jobs ran about 24.5 minutes apart. Both use the same saved calibration, "
              "mapping and circuit durations. A between-job difference does not isolate a causal DD effect from drift.", "",
              "## What correcting the syndrome changes", "",
              "Full joint inversion uses every recorded syndrome to estimate the distribution conditioned on "
              "latent (pre-assignment) syndrome 000. This is an additional diagnostic, not the fidelity of the "
              "actually retained ensemble and not a way to identify which individual shots were misread. "
              "It does not undo syndrome-circuit gate faults or guarantee codespace membership.", "",
              "| Job | Actual recorded-000 acceptance | Estimated true-000 probability (diagnostic) |",
              "|---|---:|---:|"]
    for r in output["runs"]:
        a = r["dedicated_acceptance"]
        lines.append(f"| {r['variant']} | {a['recorded_zero']['estimate']:.3%} | {a['true_zero_diagnostic']['estimate']:.3%} |")
    lines += ["", "## Method and calibration", "",
              "For qubit q the row-stochastic matrix is `A_q[true, recorded] = "
              "[[1-e01,e01],[e10,1-e10]]`. The observable weights solve `A_q w = [1,-1]`. "
              "Tensor-product weights are applied to the four measured data bits using their final physical mapping. "
              "For syndrome correction, weights solve `A_s w = indicator(000)`; a ratio of weighted joint "
              "moments gives each conditional Pauli expectation. The target-projector expansion is unchanged. "
              "The ratio variance retains numerator/denominator covariance. No negative quasiprobabilities or "
              "estimates are clipped. A full probability-distribution solve independently checks each setting.", "",
              "This is the assignment-matrix model described in "
              "[IBM's readout-mitigation tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/readout-error-mitigation-sampler). "
              "We use a small tensor-product solve; no M3 package or new dependencies were needed.", "",
              "The backend calibration fields are `prob_meas1_prep0` and `prob_meas0_prep1`, measured around "
              "2026-10-14 19:34 Pacific (about 48 and 72 minutes before the two runs). These were not dedicated "
              "calibration circuits embedded in these jobs. Their sampling counts and covariance are unavailable, "
              "so calibration uncertainty cannot be included honestly. The independent classical channel model "
              "also assumes syndrome misassignment is uncorrelated with the subsequent data dynamics.", "",
              "| Role / bit | Physical qubit | P(record 1 given 0) | P(record 0 given 1) |",
              "|---|---:|---:|---:|"]
    for role in ("data", "syndrome"):
        for r in before["calibration"][role]:
            lines.append(f"| {role} {r['bit']} | {r['physical_qubit']} | {r['p_record_1_given_0']:.3%} | {r['p_record_0_given_1']:.3%} |")
    lines += ["", "## Sensitivity to calibration strength", "",
              "Multiply all directional assignment-error rates by the factor below. This is a model sensitivity "
              "check, not a calibration confidence interval. The same factor is used for both jobs.", "",
              "| Rate multiplier | Data-corrected no DD | Data-corrected XY4 | Difference (points) |",
              "|---|---:|---:|---:|"]
    for b, a in zip(before["assignment_rate_sensitivity"], after["assignment_rate_sensitivity"]):
        x, y = b["fidelity"][MODES[1]], a["fidelity"][MODES[1]]
        lines.append(f"| {b['rate_multiplier']:.1f} | {x:.3%} | {y:.3%} | {100*(y-x):+.3f} |")
    lines += ["", "This processing can reduce measurement bias if the model holds; a higher number alone does "
              "not validate the correction or demonstrate improved state preparation. Final data readout is "
              "corrected statistically, while basis-gate and preparation errors remain.", "",
              "## Validation and statistical interpretation", "",
              "Raw data hashes were checked against the prior validated archive. The original fidelity and SE "
              "are reproduced. Independent full-matrix and weighted-observable solves agree to numerical precision. "
              "Synthetic asymmetric-channel tests verify recovery and the distinction between recorded and true "
              "syndrome conditioning. Identity-channel controls reproduce the raw estimator. "
              "All bootstrap denominators remain positive; diagnostics are saved in JSON.", "",
              "Statistical fallacy scan: 11/11 considered. Simpson: per-setting terms retained; ecological: "
              "no individual-fault inference; Berkson/collider: recorded and latent selected ensembles separated; "
              "base rates/survivorship: actual acceptance reported and all syndrome outcomes retained; "
              "regression to mean: no causal improvement claim from sequential runs; look-elsewhere: all 23 "
              "preselected terms included; forking paths: correction and sensitivity analysis labeled exploratory; "
              "causation/reverse causality: postprocessing changes estimates, not quantum states. "
              "Overall confidence: CAUTION because calibration applicability and correlated errors are unvalidated.", "",
              "## Reproduce", "", "```bash", "conda run -n quantum-jax python -B scripts/mitigate_ibm_b2ed_readout.py", "```", "",
              "Outputs are separate from the original raw reports. No hardware jobs were submitted or "
              "packages modified. JSON includes input hashes, per-Pauli estimates, codespace scores, matrix "
              "condition numbers, negative-mass diagnostics and bootstrap results.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-dd", type=Path, default=ROOT/"results/hardware_b2ed/four_pilot_20261014")
    parser.add_argument("--xy4", type=Path, default=ROOT/"results/hardware_b2ed/four_xy4_20261014")
    parser.add_argument("--output", type=Path, default=ROOT/"results/hardware_b2ed/readout_mitigation_20261014")
    args = parser.parse_args()
    runs, boot = [], []
    for i, bundle in enumerate((args.no_dd, args.xy4)):
        result, draws = analyze_bundle(bundle, seed=20261017+i)
        runs.append(result)
        boot.append(draws)
        print(f"{result['variant']}: " + ", ".join(f"{k}={result['fidelity'][k]['estimate']:.6f}" for k in MODES), flush=True)
    if runs[0]["input_sha256"]["calibration.json"] != runs[1]["input_sha256"]["calibration.json"]:
        raise ValueError("Expected the identical archived calibration for this comparison")
    comparisons = {}
    for i, mode in enumerate(MODES):
        b, a = (r["fidelity"][mode] for r in runs)
        comparisons[mode] = summary(a["estimate"]-b["estimate"], a["standard_error"]**2+b["standard_error"]**2,
                                    boot[1][:, i]-boot[0][:, i])
    output = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "runs": runs, "comparison": comparisons,
              "script_sha256": file_hash(Path(__file__)), "bootstrap_samples": 10000,
              "scope": "Offline correction using archived calibration; no new hardware or calibration jobs",
              "uncertainty": "Experimental shot uncertainty only, fixed calibration, independent shots/jobs"}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/"readout_mitigation.json", output)
    (args.output/"readout_mitigation.md").write_text(report(output))
    with (args.output/"readout_mitigation_terms.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "pauli", "coefficient", *MODES])
        for r in runs:
            for term in r["per_pauli"]:
                writer.writerow([r["variant"], term["pauli"], term["coefficient"], *term["expectations"].values()])
    print(json.dumps(comparisons, indent=2))
    print(f"Saved: {args.output/'readout_mitigation.md'}", flush=True)


if __name__ == "__main__":
    main()
