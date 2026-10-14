import numpy as np
import pytest
from qiskit.quantum_info import Operator, Pauli

from logical_magic import CODES, build_preparation
from logical_magic.aachen_noise import AachenPatch, schedule_preparation, simulate_scheduled
from logical_magic.noisy_fidelity import (
    accepted_score, analysis_suffix, estimate_noisy_fidelity,
    expected_measurement_score, measurement_circuit, parity_weights,
)


@pytest.fixture(scope="module")
def backend():
    return AachenPatch()


def test_assignment_inverse_is_unbiased_for_asymmetric_errors():
    matrices = [np.array([[.9, .1], [.2, .8]]), np.array([[.85, .15], [.07, .93]])]
    weights = parity_weights("XZ", matrices)
    assignment = np.kron(matrices[1], matrices[0])
    assert np.allclose(assignment @ weights, [1, -1, -1, 1])
    # I factors ignore that wire, even though it is physically read out.
    assert np.allclose(assignment @ parity_weights("IY", matrices), [1, -1, 1, -1])


def test_conditioning_keeps_recorded_zero_and_uses_data_bit_order():
    # data[1]data[0] syndrome[0]; discarded samples must not affect estimates.
    counts = {"00 0": 30, "01 0": 10, "10 0": 20, "11 0": 40, "00 1": 900}
    r = accepted_score(counts, 1, parity_weights("IZ"))
    assert r["accepted_shots"] == 100
    assert np.isclose(r["expectation"], 0)
    assert np.isclose(r["variance_of_mean"], 1/99)
    assert np.isclose(accepted_score(counts, 1, parity_weights("ZI"))["expectation"], -.2)


def test_basis_rotation_observables_and_analysis_idle(backend):
    # Verify Y sign independently of the target-state projector expansion.
    scheduled = schedule_preparation(build_preparation(CODES["five"], "B0"), backend)
    for label in "XYZ":
        suffix, duration = analysis_suffix(scheduled, backend, label)
        one = suffix.copy_empty_like()
        for inst in suffix.data:
            if inst.operation.name != "delay":
                one.append(inst)
        u = Operator(one).data
        z = Pauli("I"*8+"Z").to_matrix()
        expected = Pauli("I"*8+label).to_matrix()
        assert np.allclose(u.conj().T @ z @ u, expected)
        assert np.isclose(duration, 32e-9 if label in "XY" else 0, atol=1e-13)
    four = schedule_preparation(build_preparation(CODES["four"]), backend)
    suffix, duration = analysis_suffix(four, backend, "IIIX")
    assert suffix.count_ops()["delay"] == 3
    assert np.isclose(duration, 32e-9, atol=1e-13)


def test_analysis_preserves_preparation_prefix_and_has_no_saved_state(backend):
    scheduled = schedule_preparation(build_preparation(CODES["four"]), backend)
    circuit = measurement_circuit(scheduled, backend, "IYZX")
    assert list(circuit.data[:len(scheduled.circuit.data)]) == list(scheduled.circuit.data)
    assert not any(inst.operation.name.startswith("save_") for inst in circuit.data)
    assert circuit.num_clbits == 7


def test_exact_score_diagnostic_keeps_syndrome_assignment():
    # Correlated data/syndrome: 60% |0,0>, 40% |1,1> in data,syndrome order.
    probabilities = np.array([.6, 0, 0, .4])
    syn = [np.array([[.8, .2], [.3, .7]])]
    data = [np.array([[.9, .1], [.2, .8]])]
    corrected = expected_measurement_score(probabilities, 1, syn, data, parity_weights("Z", data))
    assert np.isclose(corrected, .6)  # accepted state diag(.8,.2), not pure |0>


@pytest.mark.parametrize("name,strategy", [("five", "B0"), ("four", "B2_ED"), ("five", "B2_BK")])
def test_ideal_count_estimator_and_all_projector_terms(backend, name, strategy):
    scheduled = schedule_preparation(build_preparation(CODES[name], strategy), backend)
    result = estimate_noisy_fidelity(scheduled, backend, shots=2048, seed=83, ideal=True)
    r = result["raw"]
    assert np.isclose(r["infinite_shot_measurement_score_diagnostic"], 1)
    assert abs(r["fidelity_estimate"]-1) < 6*r["standard_error"] + 1e-12
    assert np.isclose(result["data_readout_mitigated"]["fidelity_estimate"], r["fidelity_estimate"])


def test_noisy_estimator_and_readout_ablation(backend):
    scheduled = schedule_preparation(build_preparation(CODES["four"]), backend)
    full, _ = simulate_scheduled(scheduled, backend, shots=64)
    no_readout, _ = simulate_scheduled(scheduled, backend, mode="no_readout", shots=64)
    assert np.allclose(full["true_syndrome_probabilities"], no_readout["true_syndrome_probabilities"])
    assert np.isclose(full["unconditional_fidelity"], no_readout["unconditional_fidelity"])
    assert full["acceptance"] < no_readout["acceptance"]
    result = estimate_noisy_fidelity(scheduled, backend, shots=8192, seed=83)
    for name in ["raw", "data_readout_mitigated"]:
        r = result[name]
        assert abs(r["fidelity_estimate"]-r["infinite_shot_measurement_score_diagnostic"]) < 6*r["standard_error"]
    raw = result["raw"]["infinite_shot_measurement_score_diagnostic"]
    corrected = result["data_readout_mitigated"]["infinite_shot_measurement_score_diagnostic"]
    assert abs(corrected-full["fidelity"]) < .002
    assert abs(raw-full["fidelity"]) > .05
