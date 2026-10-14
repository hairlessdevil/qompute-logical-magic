import numpy as np
import pytest

from logical_magic import CODES
from logical_magic.error_correction import (
    correction_circuit, corrects_codespace, recovery_table,
    residual_logical_operator, single_pauli_experiment, single_pauli_faults,
    syndrome_candidates,
)


@pytest.mark.parametrize("name", ["four", "five"])
def test_syndrome_decoder_detection_and_correction(name):
    result = single_pauli_experiment(CODES[name], shots=32, seed=73)
    assert result["detected_errors"] == result["total_errors"]
    assert result["corrected_errors"] == (15 if name == "five" else 6)
    assert result["nonzero_syndromes"] == (15 if name == "five" else 5)
    assert np.isclose(result["rows"][0]["fidelity_after"], 1)
    for row in result["rows"][1:]:
        assert row["syndrome_counts"] == {row["syndrome"]: 32}
        assert np.isclose(row["fidelity_before"], 0)
        assert np.isclose(row["codespace_after"], 1)
        expected = 1 if row["corrects_arbitrary_logical_state"] else 0.5
        assert np.isclose(row["fidelity_after"], expected)


def test_five_qubit_recovery_preserves_arbitrary_complex_logical_states():
    code = CODES["five"]
    table = recovery_table(code)
    rng = np.random.default_rng(83)
    for fault in single_pauli_faults(code):
        logical = residual_logical_operator(code, fault["label"], table[code.syndrome(fault["label"])])
        assert corrects_codespace(logical)
        for _ in range(4):
            psi = rng.normal(size=2) + 1j*rng.normal(size=2)
            psi /= np.linalg.norm(psi)
            assert np.isclose(abs(np.vdot(psi, logical @ psi))**2, 1)


def test_four_qubit_collision_is_logically_harmful_not_just_degeneracy():
    code = CODES["four"]
    # X0 and X1 have the same syndrome but differ by logical X.
    assert code.syndrome("IIIX") == code.syndrome("IIXI")
    logical = residual_logical_operator(code, "IIXI", "IIIX")
    assert np.allclose(logical, [[0, 1], [1, 0]])
    assert not corrects_codespace(logical)
    # Z0 and Z1 also share a syndrome, but differ by a stabilizer (benign).
    assert corrects_codespace(residual_logical_operator(code, "IIZI", "IIIZ"))
    # Every nonzero four-qubit syndrome has at least one harmful ambiguity.
    for syndrome, labels in syndrome_candidates(code).items():
        if int(syndrome, 2):
            assert any(not corrects_codespace(residual_logical_operator(code, e, labels[0])) for e in labels)


def test_recovery_circuit_measures_only_ancillas_and_keeps_nonzero_syndromes():
    code = CODES["five"]
    qc = correction_circuit(code, "IIIIY")
    measurements = [inst for inst in qc.data if inst.operation.name == "measure"]
    assert len(measurements) == 4
    assert all(qc.find_bit(inst.qubits[0]).index >= code.n for inst in measurements)
    conditions = [inst.operation.condition for inst in qc.data if inst.operation.name == "if_else"]
    assert {condition[1] for condition in conditions} == set(range(1, 16))
    with pytest.raises(ValueError):
        correction_circuit(code, "IIIXX")
