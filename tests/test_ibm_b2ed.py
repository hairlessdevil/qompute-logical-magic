import json
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult
from qiskit_ibm_runtime.fake_provider import FakeAachen

from logical_magic.ibm_b2ed import (
    analyze_counts, file_hash, joint_runtime_counts, load_bundle, prepare_bundle, retrieve_bundle, submit_bundle, write_json,
)


@pytest.fixture(scope="module", params=("four", "five"))
def prepared(tmp_path_factory, request):
    directory = tmp_path_factory.mktemp("ibm-b2ed") / "prepared"
    manifest = prepare_bundle(FakeAachen(), directory, shots=2048, live=False, code_name=request.param)
    return directory, manifest


def test_full_projector_compilation_and_ideal_validation(prepared):
    directory, manifest = prepared
    expected_circuits = 24 if manifest["code_name"] == "four" else 64
    assert manifest["number_of_circuits"] == expected_circuits
    assert manifest["total_shots"] == expected_circuits*2048
    assert manifest["n_syndrome"] == manifest["n_data"]-1
    assert manifest["identity_coefficient"] == pytest.approx(1/2**manifest["n_data"])
    assert manifest["target"] == ("encoded H state" if manifest["code_name"] == "four" else "encoded positive tetrahedral state")
    assert manifest["ideal_validation"]["max_absolute_error"] < 1e-10
    _, _, physical = load_bundle(directory)
    with (directory / "local_circuits.qpy").open("rb") as handle:
        local = qpy.load(handle)
    prefix = local[0]
    for circuit, device in zip(local, physical):
        assert list(circuit.data[:len(prefix.data)]) == list(prefix.data)
        assert device.num_qubits == 156
        # Check the actual hardware embedding, including measurement bit order.
        restored = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        reverse = {physical: local for local, physical in enumerate(manifest["physical_patch"])}
        for inst in device.data:
            restored.append(inst.operation,
                            [reverse[device.find_bit(q).index] for q in inst.qubits],
                            [device.find_bit(c).index for c in inst.clbits])
        restored.global_phase = device.global_phase
        assert restored == circuit


def test_changed_circuit_bundle_rejected(prepared, tmp_path):
    directory = copytree(prepared[0], tmp_path / "bundle")
    with (directory / "circuits.qpy").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Bundle file changed"):
        load_bundle(directory)


def test_missing_approval_or_offline_bundle_never_contacts_ibm(prepared):
    directory, _ = prepared
    service = Mock(side_effect=AssertionError("Must not contact IBM"))
    with pytest.raises(ValueError, match="Approval digest"):
        submit_bundle(directory, "wrong", service_factory=service)
    with pytest.raises(ValueError, match="Offline validation"):
        submit_bundle(directory, file_hash(directory / "manifest.json"), service_factory=service)
    service.assert_not_called()


def fake_live_bundle(prepared, tmp_path):
    directory = copytree(prepared[0], tmp_path / "bundle")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["calibration_source"] = "live"  # Only mock service factories are used below.
    write_json(directory / "manifest.json", manifest)
    return directory, manifest


def test_submission_exits_without_status_or_result_and_blocks_duplicate(prepared, tmp_path):
    directory, manifest = fake_live_bundle(prepared, tmp_path)
    backend = Mock(target=FakeAachen().target)
    backend.status.return_value = SimpleNamespace(operational=True)
    job = Mock()
    job.job_id.return_value = "test-job-id"
    job.result.side_effect = AssertionError("Submission must not retrieve")
    job.status.side_effect = AssertionError("Submission must not poll")
    service = Mock()
    service.return_value.backend.return_value = backend
    sampler = Mock()
    sampler.return_value.run.return_value = job
    digest = file_hash(directory / "manifest.json")
    receipt = submit_bundle(directory, digest, service_factory=service, sampler_factory=sampler)
    service.assert_called_once_with(instance="qzh-eu")
    service.return_value.backend.assert_called_once_with("ibm_aachen")
    sampler.return_value.run.assert_called_once()
    assert sampler.return_value.run.call_args.kwargs == {"shots": 2048}
    assert len(sampler.return_value.run.call_args.args[0]) == manifest["number_of_circuits"]
    assert receipt["job_id"] == "test-job-id"
    assert [call[0] for call in job.mock_calls] == ["job_id"]
    with pytest.raises(FileExistsError, match="already attempted"):
        submit_bundle(directory, digest, service_factory=service, sampler_factory=sampler)
    assert sampler.return_value.run.call_count == 1


def test_uncertain_submission_cannot_be_retried_automatically(prepared, tmp_path):
    directory, _ = fake_live_bundle(prepared, tmp_path)
    backend = Mock(target=FakeAachen().target)
    backend.status.return_value = SimpleNamespace(operational=True)
    service = Mock()
    service.return_value.backend.return_value = backend
    sampler = Mock()
    sampler.return_value.run.side_effect = TimeoutError("Unknown server outcome")
    digest = file_hash(directory / "manifest.json")
    with pytest.raises(TimeoutError):
        submit_bundle(directory, digest, service_factory=service, sampler_factory=sampler)
    assert not (directory / "submission.json").exists()
    with pytest.raises(FileExistsError):
        submit_bundle(directory, digest, service_factory=service, sampler_factory=sampler)


def test_joint_runtime_register_order_and_postselection():
    pub = SamplerPubResult(DataBin(
        syndrome=BitArray.from_samples(["000", "000", "001", "111"], num_bits=3),
        quality_readout=BitArray.from_samples(["0000", "0001", "0010", "0011"], num_bits=4),
    ))
    joint = joint_runtime_counts(pub)
    assert joint == {"0000000": 1, "0001000": 1, "0010001": 1, "0011111": 1}
    manifest = {"identity_coefficient": .5, "settings": [
        {"kind": "acceptance"}, {"kind": "pauli", "pauli": "IIIZ", "coefficient": .5}]}
    result = analyze_counts(manifest, [{"000": 2, "001": 2}, joint])
    assert result["terms"][0]["accepted_shots"] == 2
    assert result["terms"][0]["raw"]["expectation"] == 0
    assert result["raw_fidelity_estimate"] == .5
    result = analyze_counts(manifest, [{"001": 4}, {"0000001": 4}])
    assert result["raw_fidelity_estimate"] is None
    assert not result["sufficient_accepted_shots"]


def test_five_qubit_register_order_rejects_fourth_syndrome_bit():
    pub = SamplerPubResult(DataBin(
        syndrome=BitArray.from_samples(["0000", "0000", "1000", "0010"], num_bits=4),
        quality_readout=BitArray.from_samples(["00000", "10000", "00001", "11111"], num_bits=5),
    ))
    joint = joint_runtime_counts(pub, n_syndrome=4, n_data=5)
    assert joint == {"000000000": 1, "100000000": 1, "000011000": 1, "111110010": 1}
    manifest = {"n_data": 5, "n_syndrome": 4, "identity_coefficient": .5, "settings": [
        {"kind": "acceptance"}, {"kind": "pauli", "pauli": "ZIIII", "coefficient": .5}]}
    result = analyze_counts(manifest, [pub.data.syndrome.get_counts(), joint])
    assert result["acceptance_only_circuit"]["accepted_shots"] == 2
    assert result["terms"][0]["raw"]["expectation"] == 0
    assert result["raw_fidelity_estimate"] == .5
    assert "recorded-0000" in result["interpretation"]
    with pytest.raises(ValueError, match="Expected 3 syndrome bits"):
        joint_runtime_counts(pub)


def test_retrieval_does_not_wait_for_pending_job(prepared, tmp_path):
    directory = copytree(prepared[0], tmp_path / "bundle")
    write_json(directory / "submission.json", {"job_id": "test-job",
                                               "manifest_sha256": file_hash(directory / "manifest.json")})
    service = Mock()
    job = service.return_value.job.return_value
    job.status.return_value = "QUEUED"
    job.result.side_effect = AssertionError("Must not wait for queued jobs")
    result = retrieve_bundle(directory, service_factory=service)
    assert result["status"] == "QUEUED"
    job.status.assert_called_once()
    job.result.assert_not_called()


def test_completed_result_archived_and_analyzed(prepared, tmp_path):
    directory, manifest = prepared
    directory = copytree(directory, tmp_path / "bundle")
    write_json(directory / "submission.json", {"job_id": "test-job",
                                               "manifest_sha256": file_hash(directory / "manifest.json")})
    pubs = []
    n_data, n_syndrome = manifest["n_data"], manifest["n_syndrome"]
    for setting in manifest["settings"]:
        # The highest syndrome bit alone must reject a shot for either code.
        data = {"syndrome": BitArray.from_samples(
            ["0"*n_syndrome]*3 + ["1"+"0"*(n_syndrome-1)], num_bits=n_syndrome)}
        if setting["kind"] == "pauli":
            data["quality_readout"] = BitArray.from_samples([format(i, f"0{n_data}b") for i in range(4)], num_bits=n_data)
        pubs.append(SamplerPubResult(DataBin(**data)))
    service = Mock()
    job = service.return_value.job.return_value
    job.status.return_value = "DONE"
    job.result.return_value = PrimitiveResult(pubs)
    summary = retrieve_bundle(directory, service_factory=service)
    job.result.assert_called_once_with(timeout=30)
    assert summary["acceptance_only_circuit"]["acceptance"] == .75
    assert np.isfinite(summary["raw_fidelity_estimate"])
    assert (directory / "runtime_result.json").is_file()
    assert (directory / "joint_counts.json").is_file()
    assert (directory / "analysis.json").is_file()
