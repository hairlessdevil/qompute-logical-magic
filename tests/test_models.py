import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Pauli, Statevector
from qiskit_aer import AerSimulator

from logical_magic import CODES, build_preparation, exact_result, magic_state
from logical_magic.circuits import check_block
from logical_magic.simulation import branch_result, circuit_fault_audit, estimate_fidelity, storage_fault_audit


@pytest.mark.parametrize("code", CODES.values(), ids=CODES.keys())
def test_code_algebra_and_distance(code):
    p, b = code.projector, code.basis
    assert np.allclose(p @ p, p)
    assert np.isclose(np.trace(p), 2)
    assert np.allclose(b.conj().T @ b, np.eye(2))
    assert np.allclose(b @ b.conj().T, p)
    for s in code.stabilizers:
        assert np.allclose(Pauli(s).to_matrix() @ b, b)
    for label, logical in [(code.logical_x, "X"), (code.logical_z, "Z")]:
        assert np.allclose(b.conj().T @ Pauli(label).to_matrix() @ b, Pauli(logical).to_matrix())
    assert np.allclose(b.conj().T @ code.logical_y @ b, Pauli("Y").to_matrix())
    assert code.computed_distance() == code.distance


@pytest.mark.parametrize("code", CODES.values(), ids=CODES.keys())
def test_encoder_arbitrary_complex_states_and_decode(code):
    rng = np.random.default_rng(41)
    for _ in range(10):
        physical = rng.normal(size=2) + 1j*rng.normal(size=2)
        physical /= np.linalg.norm(physical)
        initial = np.zeros(2**code.n, complex)
        initial[:2] = physical
        encoded = Statevector(initial).evolve(code.encoder())
        assert np.isclose(abs(np.vdot(code.basis @ physical, encoded.data))**2, 1)
        decoded = encoded.evolve(code.encoder().inverse())
        assert np.allclose(decoded.data, initial)


@pytest.mark.parametrize("code", CODES.values(), ids=CODES.keys())
@pytest.mark.parametrize("strategy", ["B0", "B1", "B2_ED"])
def test_ideal_preparation(code, strategy):
    prep = build_preparation(code, strategy)
    result = exact_result(prep)
    assert np.isclose(result["acceptance"], 1)
    assert np.isclose(result["fidelity"], 1)
    assert np.isclose(result["codespace_population"], 1)
    # Preparation measurements may only act on ancillas, keeping data reusable.
    for inst in prep.measured().data:
        if inst.operation.name == "measure":
            assert prep.measured().find_bit(inst.qubits[0]).index >= prep.n_data


@pytest.mark.parametrize("code", CODES.values(), ids=CODES.keys())
def test_all_syndrome_branches_match_independent_projectors(code):
    rng = np.random.default_rng(42)
    data = rng.normal(size=2**code.n) + 1j*rng.normal(size=2**code.n)
    data /= np.linalg.norm(data)
    m = len(code.stabilizers)
    initial = np.zeros(2**(code.n+m), complex)
    initial[:2**code.n] = data
    qc = QuantumCircuit(code.n+m)
    for i, s in enumerate(code.stabilizers):
        qc.compose(check_block(code.n, s), [*range(code.n), code.n+i], inplace=True)
    actual = Statevector(initial).evolve(qc).data.reshape(2**m, 2**code.n)
    ident = np.eye(2**code.n)
    for syndrome in range(2**m):
        predicted = data.copy()
        for i, s in enumerate(code.stabilizers):
            predicted = (ident + (-1)**((syndrome >> i) & 1)*Pauli(s).to_matrix()) @ predicted / 2
        assert np.allclose(actual[syndrome], predicted)


def test_y_factor_in_syndrome_check():
    # Current generators use X/Z, but the generic circuit also promises Y.
    qc = check_block(1, "Y")
    for sign in [1, -1]:
        initial = np.array([1, sign*1j, 0, 0]) / np.sqrt(2)
        out = Statevector(initial).evolve(qc).data.reshape(2, 2)
        assert np.isclose(np.linalg.norm(out[0 if sign == 1 else 1]), 1)


@pytest.mark.parametrize("code", CODES.values(), ids=CODES.keys())
def test_all_single_storage_paulis_rejected(code):
    rows = storage_fault_audit(build_preparation(code))
    assert len(rows) == 3*code.n
    assert all(r["acceptance"] == 0 for r in rows)
    assert all(r["predicted_syndrome"] == r["observed_syndrome"] for r in rows)


def test_bk_acceptance_and_pole_convention():
    code = CODES["five"]
    prep = build_preparation(code, "B2_BK")
    result = exact_result(prep)
    assert np.isclose(result["acceptance"], 1/6)
    assert np.isclose(result["fidelity"], 1)
    # Independent product-state projection fixes the easy-to-miss pole reversal.
    minus_product = magic_state("tetra", True)
    for _ in range(4):
        minus_product = np.kron(minus_product, magic_state("tetra", True))
    projected = code.projector @ minus_product
    assert np.isclose(np.linalg.norm(projected)**2, 1/6)
    assert np.isclose(abs(np.vdot(code.target, projected))**2, 1/6)


def test_bk_rejects_one_orthogonal_magic_input_with_perfect_checks():
    code = CODES["five"]
    for bad in range(5):
        product_state = np.array([1], complex)
        for q in reversed(range(5)):
            product_state = np.kron(product_state, magic_state("tetra", antipodal=q != bad))
        assert np.linalg.norm(code.projector @ product_state) < 1e-10


@pytest.mark.parametrize("code", CODES.values(), ids=CODES.keys())
def test_fidelity_observable_includes_leakage(code):
    op = code.fidelity_observable.to_matrix()
    assert np.allclose(op, np.outer(code.target, code.target.conj()))
    leaked = Statevector(code.target).evolve(Pauli("I"*(code.n-1)+"X")).data
    assert np.isclose(np.vdot(leaked, code.projector @ leaked), 0)
    assert np.isclose(np.vdot(leaked, op @ leaked), 0)
    # An orthogonal logical state still passes every code stabilizer.
    wrong = code.basis @ magic_state(code.magic, antipodal=True)
    assert np.isclose(np.vdot(wrong, code.projector @ wrong), 1)
    assert np.isclose(np.vdot(wrong, op @ wrong), 0)


def test_circuit_audit_exposes_undetected_preparation_fault():
    prep = build_preparation(CODES["four"])
    rows = circuit_fault_audit(prep)
    failures = [r for r in rows if r["stage"] == "magic_input" and r["pauli"] == "Y"]
    assert len(failures) == 1
    assert np.isclose(failures[0]["acceptance"], 1)
    assert np.isclose(failures[0]["fidelity"], 0)
    assert any(r["category"] == "accepted_harmful" and r["stage"].startswith("check_") for r in rows)


@pytest.mark.parametrize("code", CODES.values(), ids=CODES.keys())
def test_aer_matches_exact_statevector(code):
    prep = build_preparation(code)
    qc = prep.unitary.copy()
    qc.save_statevector()
    backend = AerSimulator(method="statevector", max_parallel_threads=1)
    out = backend.run(transpile(qc, backend, optimization_level=0), seed_simulator=10).result().get_statevector()
    ref = Statevector.from_instruction(prep.unitary)
    assert np.isclose(abs(np.vdot(out.data, ref.data))**2, 1)


@pytest.mark.parametrize("name,strategy", [("four", "B2_ED"), ("five", "B2_BK")])
def test_finite_shot_fidelity_same_acceptance(name, strategy):
    result = estimate_fidelity(build_preparation(CODES[name], strategy), shots=4096, seed=13)
    assert abs(result["fidelity_estimate"]-1) < 0.06
    assert result["fidelity_conservative_95_interval"][0] <= 1 <= result["fidelity_conservative_95_interval"][1]
    expected = 1 if strategy == "B2_ED" else 1/6
    assert abs(result["acceptance_estimate"]-expected) < .025


def test_partial_and_reordered_checks():
    code = CODES["four"]
    for checks in [(), ("XXXX",), tuple(reversed(code.stabilizers))]:
        result = exact_result(build_preparation(code, checks=checks))
        assert np.isclose(result["fidelity"], 1)
        assert np.isclose(result["acceptance"], 1)
    with pytest.raises(ValueError):
        build_preparation(code, checks=("XXXX", "XXXX"))
    with pytest.raises(ValueError):
        build_preparation(code, "B2_BK")
