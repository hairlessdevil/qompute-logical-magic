import json

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime.fake_provider import FakeAachen

from logical_magic.ibm_b2ed import file_hash, load_bundle, prepare_bundle
from logical_magic.ibm_dd import prepare_xy4_bundle


def test_native_xy4_has_exact_identity_with_compensated_phase():
    circuit = QuantumCircuit(1)
    for axis in "XYXY":
        if axis == "Y":
            circuit.rz(-np.pi/2, 0)
        circuit.x(0)
        if axis == "Y":
            circuit.rz(np.pi/2, 0)
    circuit.global_phase += np.pi
    assert np.allclose(Operator(circuit).data, np.eye(2), atol=1e-14)


@pytest.mark.parametrize("code_name", ("four", "five"))
def test_frozen_bundle_xy4_preserves_times_mapping_and_ideal_state(tmp_path, code_name):
    backend = FakeAachen()
    source, output = tmp_path/"no_dd", tmp_path/"xy4"
    old = prepare_bundle(backend, source, live=False, code_name=code_name)
    old_hash = file_hash(source/"circuits.qpy")
    new = prepare_xy4_bundle(source, output, backend, live=False)
    assert file_hash(source/"circuits.qpy") == old_hash
    expected_circuits = 24 if code_name == "four" else 64
    assert new["total_shots"] == expected_circuits*2048
    assert new["sampler_options"]["dynamical_decoupling"]["enable"] is False
    assert new["ideal_validation"]["max_absolute_error"] < 1e-10
    assert new["resources"]["dd"]["original_gate_timestamps_preserved"]
    assert new["resources"]["duration_seconds"] == old["resources"]["duration_seconds"]
    assert new["resources"]["dd"]["added_native_x_pulses"] >= 4*old["n_data"]
    _, _, circuits = load_bundle(output)
    assert len(circuits) == expected_circuits
    assert set(circuits[0].count_ops()) <= {"rz", "sx", "x", "cz", "measure", "barrier", "delay"}
    for a, b in zip(old["settings"], new["settings"]):
        assert a["estimated_circuit_seconds"] == b["estimated_circuit_seconds"]
    with (output/"local_circuits.qpy").open("rb") as handle:
        local = qpy.load(handle)
    for circuit in local[1:]:
        assert list(circuit.data[:len(local[0].data)]) == list(local[0].data)
    assert not (output/"submission_attempt.json").exists()
    with pytest.raises(FileExistsError):
        prepare_xy4_bundle(source, output, backend, live=False)
