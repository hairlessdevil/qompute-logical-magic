"""Frozen B2_ED hardware bundles; submission and retrieval are separate actions."""

from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, qpy
from qiskit.providers import BackendV2, Options
from qiskit.quantum_info import Pauli, Statevector
from qiskit.transpiler import Target
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

from .aachen_noise import DEFAULT_PATCH, DEFAULT_SEED, schedule_preparation
from .circuits import build_preparation
from .codes import CODES
from .noisy_fidelity import accepted_score, measurement_circuit, parity_weights
from .simulation import wilson_interval

INSTANCE = "qzh-eu"
BACKEND = "ibm_aachen"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


class LivePatch(BackendV2):
    """Use the existing fixed nine-qubit patch, with the supplied live target."""

    def __init__(self, backend):
        self.physical_qubits = DEFAULT_PATCH
        source = backend.target
        self._target = Target(
            num_qubits=len(self.physical_qubits), dt=source.dt,
            granularity=source.granularity, min_length=source.min_length,
            pulse_alignment=source.pulse_alignment, acquire_alignment=source.acquire_alignment,
            qubit_properties=deepcopy([source.qubit_properties[q] for q in self.physical_qubits]),
        )
        remap = {physical: local for local, physical in enumerate(self.physical_qubits)}
        # Keep the baseline's ordinary measure instruction, including its full
        # readout wait. Do not silently substitute the faster measure_2 protocol.
        for name in ("rz", "sx", "x", "cz", "measure", "delay"):
            properties = {
                tuple(remap[q] for q in wires): deepcopy(prop)
                for wires, prop in source[name].items()
                if wires is not None and all(q in remap for q in wires)
            }
            self._target.add_instruction(source.operation_from_name(name), properties, name=name)
        super().__init__(name="live_aachen_fixed_patch")

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options()

    def run(self, *args, **kwargs):
        raise NotImplementedError("Compilation-only adapter; submit the frozen bundle separately")


def embed_physical(local, patch, backend):
    """Apply the local-to-device mapping once, without re-transpiling the prefix."""
    physical = QuantumCircuit(QuantumRegister(backend.num_qubits, "q"), *local.cregs, name=local.name)
    physical.compose(local, qubits=patch.physical_qubits, clbits=physical.clbits, inplace=True)
    physical.metadata = deepcopy(local.metadata)
    return physical


def ideal_validation(circuits, scheduled, settings):
    """Exact ideal checks of routed wires, acceptance, and every analysis basis.

    Deferring ancilla measurements is valid for this ideal-only validation:
    all subsequent gates act only on data. Hardware circuits retain readout.
    """
    errors = []
    acceptances = []
    for circuit, setting in zip(circuits, settings):
        unitary = QuantumCircuit(circuit.num_qubits)
        unitary.global_phase = circuit.global_phase
        for inst in circuit.data:
            if inst.operation.name not in {"measure", "barrier", "delay"}:
                unitary.append(inst.operation, [circuit.find_bit(q).index for q in inst.qubits])
        state = Statevector.from_instruction(unitary)
        probabilities = state.probabilities([*scheduled.syndrome_wires, *scheduled.data_wires])
        by_data_syndrome = probabilities.reshape(2**scheduled.preparation.n_data, -1)
        accepted_data = by_data_syndrome[:, 0]
        acceptance = float(accepted_data.sum())
        acceptances.append(acceptance)
        if setting["kind"] == "pauli":
            measured = float(accepted_data @ parity_weights(setting["pauli"]) / acceptance)
            target = scheduled.preparation.target
            expected = float(np.vdot(target, Pauli(setting["pauli"]).to_matrix() @ target).real)
            errors.append(abs(measured - expected))
    max_error = max([*errors, *(abs(p - 1) for p in acceptances)])
    if max_error > 1e-10:
        raise ValueError(f"Ideal verification failed: maximum absolute error {max_error}")
    return {"method": "exact local statevector, noise disabled", "max_absolute_error": max_error,
            "acceptance": acceptances[0], "settings_verified": len(settings)}


def prepare_bundle(backend, output, shots=2048, *, live=True):
    """Compile and validate only. No Sampler is constructed and no job is sent."""
    if shots < 2:
        raise ValueError("At least two shots per setting are required")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new bundle directory: {output}")
    patch = LivePatch(backend)
    prep = build_preparation(CODES["four"], "B2_ED")
    scheduled = schedule_preparation(prep, patch, seed=DEFAULT_SEED)
    settings = [{"kind": "acceptance", "pauli": None, "coefficient": None}]
    identity = 0.0
    for label, coefficient in prep.fidelity_observable.to_list():
        if label == "IIII":
            identity += float(coefficient.real)
        else:
            settings.append({"kind": "pauli", "pauli": label, "coefficient": float(coefficient.real)})
    local = [scheduled.circuit]
    for setting in settings[1:]:
        circuit = measurement_circuit(scheduled, patch, setting["pauli"])
        circuit.name = f"four_B2_ED_{setting['pauli']}"
        local.append(circuit)
    validation = ideal_validation(local, scheduled, settings)
    physical = [embed_physical(circuit, patch, backend) for circuit in local]
    validate_isa_circuits(physical, backend.target)
    if live:
        for circuit in physical:
            backend.check_faulty(circuit)
    for index, (setting, circuit) in enumerate(zip(settings, physical)):
        setting.update({"pub_index": index, "circuit_name": circuit.name,
                        "estimated_circuit_seconds": float(circuit.estimate_duration(backend.target))})
    status = backend.status().to_dict()
    config = backend.configuration()
    rep_delay = float(config.default_rep_delay)
    # IBM's approximate single-subjob overhead formula, including initialization.
    reset = max(backend.target["reset"][(q,)].duration for q in patch.physical_qubits)
    estimated_usage = 2 + shots * sum(s["estimated_circuit_seconds"] + reset + rep_delay for s in settings)
    tag = "b2ed-four-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    options = {
        "max_execution_time": 300,
        "environment": {"job_tags": [tag]},
        "execution": {"init_qubits": True, "rep_delay": rep_delay, "meas_type": "classified"},
        "twirling": {"enable_gates": False, "enable_measure": False},
        "dynamical_decoupling": {"enable": False},
    }
    output.mkdir(parents=True)
    with (output / "circuits.qpy").open("wb") as handle:
        qpy.dump(physical, handle)
    with (output / "local_circuits.qpy").open("wb") as handle:
        qpy.dump(local, handle)
    (output / "preparation.txt").write_text(str(scheduled.circuit.draw(output="text", fold=140)))
    write_json(output / "calibration.json", backend.properties().to_dict())
    manifest = {
        "schema_version": 1, "created_at_utc": utc_now(), "instance": INSTANCE, "backend": BACKEND,
        "calibration_source": "live" if live else "offline_fake_backend_DO_NOT_SUBMIT",
        "calibration_date": backend.properties().last_update_date.isoformat(),
        "backend_status_at_preparation": status, "code": "[[4,1,2]]", "strategy": "B2_ED",
        "target": "encoded H state", "stabilizers_in_syndrome_bit_order": list(prep.checks),
        "shots_per_circuit": shots, "number_of_circuits": len(physical), "total_shots": len(physical)*shots,
        "physical_patch": list(patch.physical_qubits), "resources": scheduled.resources,
        "sampler_options": options, "settings": settings, "identity_coefficient": identity,
        "estimated_usage_seconds": estimated_usage,
        "usage_estimate_note": "Approximation including one 2 s subjob overhead; not a bill or queue estimate",
        "ideal_validation": validation, "isa_validation": "passed",
        "environment": {p: version(p) for p in ("qiskit", "qiskit-ibm-runtime", "qiskit-aer", "numpy")},
        "acceptance": "Only recorded syndrome 000; no correction and no extra codespace filtering",
        "analysis": "Raw full-target-projector estimate; analysis/readout errors are not corrected",
        "timing": "Frozen ALAP preparation, ordinary syndrome measure, then data analysis/readout",
        "files": {name: file_hash(output / name) for name in
                  ("circuits.qpy", "local_circuits.qpy", "calibration.json", "preparation.txt")},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_bundle(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["instance"] != INSTANCE or manifest["backend"] != BACKEND:
        raise ValueError("Unsupported bundle or different IBM instance/backend")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or file_hash(path / name) != digest:
            raise ValueError(f"Bundle file changed: {name}")
    with (path / "circuits.qpy").open("rb") as handle:
        circuits = qpy.load(handle)
    if len(circuits) != manifest["number_of_circuits"] or len(circuits) != len(manifest["settings"]):
        raise ValueError("Circuit count does not match manifest")
    return path, manifest, circuits


def submit_bundle(path, approved_manifest_sha256, *, service_factory=QiskitRuntimeService,
                  sampler_factory=SamplerV2):
    """One submission, then only local receipt writing. Never polls or retrieves."""
    path, manifest, circuits = load_bundle(path)
    digest = file_hash(path / "manifest.json")
    if approved_manifest_sha256 != digest:
        raise ValueError("Approval digest does not match the prepared manifest")
    if manifest["calibration_source"] != "live":
        raise ValueError("Offline validation bundles cannot be submitted")
    if (path / "submission_attempt.json").exists():
        raise FileExistsError("Submission already attempted. Inspect the receipt/IBM job list; do not blindly retry.")
    service = service_factory(instance=INSTANCE)
    backend = service.backend(BACKEND)
    if not backend.status().operational:
        raise RuntimeError("ibm_aachen is not operational; no job submitted")
    validate_isa_circuits(circuits, backend.target)
    for circuit in circuits:
        backend.check_faulty(circuit)
    sampler = sampler_factory(mode=backend, options=manifest["sampler_options"])
    # Exclusive persistent marker prevents repeat submission after a timeout or
    # interrupted response. Its presence is NOT evidence that a job was accepted.
    with (path / "submission_attempt.json").open("x") as handle:
        json.dump({"started_at_utc": utc_now(), "manifest_sha256": digest,
                   "job_tag": manifest["sampler_options"]["environment"]["job_tags"][0]}, handle, indent=2)
    job = sampler.run(circuits, shots=manifest["shots_per_circuit"])
    job_id = job.job_id()  # SDK getter; no network request.
    print(f"Submitted IBM job: {job_id}", flush=True)
    receipt = {"job_id": job_id, "submitted_at_utc": utc_now(), "instance": INSTANCE,
               "backend": BACKEND, "bundle": str(path), "manifest_sha256": digest,
               "total_shots": manifest["total_shots"]}
    write_json(path / "submission.json", receipt)
    return receipt


def analyze_counts(manifest, counts_by_setting):
    """Postselect JOINT counts: data MSBs, syndrome LSBs, as in Qiskit counts."""
    if len(counts_by_setting) != len(manifest["settings"]):
        raise ValueError("Result PUB count differs from manifest")
    rows = []
    estimate, variance, radius = manifest["identity_coefficient"], 0.0, 0.0
    enough = True
    for setting, counts in zip(manifest["settings"], counts_by_setting):
        total = sum(counts.values())
        accepted = sum(count for bits, count in counts.items() if int(bits.replace(" ", ""), 2) & 7 == 0)
        row = {**setting, "shots": total, "accepted_shots": accepted,
               "acceptance": accepted/total if total else None,
               "acceptance_wilson_95_interval": list(wilson_interval(accepted, total)) if total else None}
        if setting["kind"] == "pauli":
            if accepted < 2:
                enough = False
                row["raw"] = None
            else:
                score = accepted_score(counts, 3, parity_weights(setting["pauli"]))
                row["raw"] = score
                coefficient = setting["coefficient"]
                estimate += coefficient*score["expectation"]
                variance += coefficient**2*score["variance_of_mean"]
                radius += abs(coefficient)*2*np.sqrt(np.log(2*(len(counts_by_setting)-1)/.05)/(2*accepted))
        rows.append(row)
    standard_error = float(np.sqrt(variance))
    return {
        "acceptance_only_circuit": rows[0], "terms": rows[1:],
        "raw_fidelity_estimate": float(estimate) if enough else None,
        "standard_error": standard_error if enough else None,
        "normal_approx_95_interval": [estimate-1.959964*standard_error, estimate+1.959964*standard_error] if enough else None,
        "conservative_95_interval": [estimate-radius, estimate+radius] if enough else None,
        "interpretation": "Full target projector from recorded-000 shots. Raw measurement estimate, not an exact state fidelity. "
                          "Intervals describe finite-shot uncertainty only; basis/readout bias and drift remain. "
                          "No readout mitigation, active recovery, or extra codespace filtering. Estimates are not clipped.",
        "sufficient_accepted_shots": enough,
    }


def joint_runtime_counts(pub):
    """Concatenate per-shot displayed strings, independent of SDK join ordering."""
    syndrome = pub.data.syndrome
    data = pub.data.quality_readout
    if syndrome.num_bits != 3 or data.num_bits != 4 or syndrome.shape != () or data.shape != ():
        raise ValueError("Expected three syndrome bits and four data bits in a scalar PUB")
    # Do not multiply marginal counts: their shot correlations are essential.
    return dict(Counter(d + s for d, s in zip(data.get_bitstrings(), syndrome.get_bitstrings(), strict=True)))


def retrieve_bundle(path, *, service_factory=QiskitRuntimeService):
    """Check once; download only a completed job. Never submit or wait in a loop."""
    path, manifest, _ = load_bundle(path)
    receipt = json.loads((path / "submission.json").read_text())
    if receipt["manifest_sha256"] != file_hash(path / "manifest.json"):
        raise ValueError("Manifest changed since submission")
    service = service_factory(instance=INSTANCE)
    job = service.job(receipt["job_id"])
    status = job.status()
    status = getattr(status, "name", str(status)).upper()
    if status != "DONE":
        advice = "Inspect the job on IBM Quantum." if status in {"ERROR", "CANCELLED"} else "Run this script later."
        print(f"Job {receipt['job_id']}: {status}. No results downloaded. {advice}")
        return {"job_id": receipt["job_id"], "status": status}
    result = job.result(timeout=30)
    # Archive the full Runtime result before analysis, including per-shot data.
    from qiskit_ibm_runtime import RuntimeEncoder
    (path / "runtime_result.json").write_text(json.dumps(result, cls=RuntimeEncoder))
    if len(result) != len(manifest["settings"]):
        raise ValueError("Unexpected number of PUB results; raw result was saved")
    counts_by_setting = []
    for pub, setting in zip(result, manifest["settings"]):
        if setting["kind"] == "acceptance":
            counts = pub.data.syndrome.get_counts()
        else:
            counts = joint_runtime_counts(pub)
        counts_by_setting.append(counts)
    write_json(path / "joint_counts.json", counts_by_setting)
    summary = {"job_id": receipt["job_id"], "retrieved_at_utc": utc_now(),
               **analyze_counts(manifest, counts_by_setting)}
    write_json(path / "analysis.json", summary)
    return summary
