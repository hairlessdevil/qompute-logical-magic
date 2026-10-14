"""Single data-qubit Pauli storage errors and syndrome-only recovery.

The injected error is never passed to the decoder. Recovery uses a fixed
syndrome table. For ambiguous syndromes we choose the first representative
(qubit order, then X/Y/Z) to demonstrate the limits of the four-qubit code.
"""

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, Statevector

from .codes import Code
from .circuits import build_preparation, check_block


def single_pauli_faults(code: Code) -> list[dict]:
    faults = [{"qubit": None, "pauli": "I", "label": "I" * code.n}]
    for q in range(code.n):
        for p in "XYZ":
            label = list("I" * code.n)
            label[-1-q] = p
            faults.append({"qubit": q, "pauli": p, "label": "".join(label)})
    return faults


def syndrome_candidates(code: Code) -> dict[str, list[str]]:
    groups = {}
    for fault in single_pauli_faults(code):
        groups.setdefault(code.syndrome(fault["label"]), []).append(fault["label"])
    return groups


def recovery_table(code: Code) -> dict[str, str]:
    """Fixed decoder, independent of the actual injected fault or logical state."""
    return {syndrome: candidates[0] for syndrome, candidates in syndrome_candidates(code).items()}


def residual_logical_operator(code: Code, error: str, recovery: str) -> np.ndarray:
    return code.basis.conj().T @ Pauli(recovery).to_matrix() @ Pauli(error).to_matrix() @ code.basis


def corrects_codespace(logical: np.ndarray) -> bool:
    phase = np.trace(logical) / 2
    return bool(np.isclose(abs(phase), 1) and np.allclose(logical, phase*np.eye(2)))


def storage_error_circuit(code: Code, error: str) -> QuantumCircuit:
    """Encode, inject the error, then coherently extract all stabilizers."""
    if error not in {fault["label"] for fault in single_pauli_faults(code)}:
        raise ValueError("Expected identity or one single-qubit X/Y/Z of the code's length")
    qc = QuantumCircuit(code.n + len(code.stabilizers), name=f"{code.name}_{error}")
    qc.compose(build_preparation(code, "B1").unitary, range(code.n), inplace=True)
    for q, p in enumerate(reversed(error)):
        if p != "I":
            getattr(qc, p.lower())(q)
    for i, stabilizer in enumerate(code.stabilizers):
        qc.compose(check_block(code.n, stabilizer), [*range(code.n), code.n+i], inplace=True)
    return qc


def correction_circuit(code: Code, error: str) -> QuantumCircuit:
    """Reusable data output with measured-syndrome classical feed-forward.

    No postselection: every syndrome is kept. Checks and recovery gates are
    ideal. Only the injected storage error is faulty. Unlisted syndromes can
    arise outside this single-Pauli model and receive no correction here.
    """
    qc = storage_error_circuit(code, error)
    syn = ClassicalRegister(len(code.stabilizers), "syndrome")
    qc.add_register(syn)
    qc.measure(range(code.n, qc.num_qubits), syn)
    # This table is rebuilt from the code, NOT from `error`.
    for syndrome, recovery in recovery_table(code).items():
        if set(recovery) == {"I"}:
            continue
        with qc.if_test((syn, int(syndrome, 2))):
            for q, p in enumerate(reversed(recovery)):
                if p != "I":
                    getattr(qc, p.lower())(q)
    return qc


def single_pauli_experiment(code: Code, shots: int = 256, seed: int = 20261014) -> dict:
    """Run each error and a no-error control through the same syndrome decoder."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    from qiskit_aer import AerSimulator

    faults = single_pauli_faults(code)
    table = recovery_table(code)
    candidates = syndrome_candidates(code)
    circuits = []
    for fault in faults:
        qc = correction_circuit(code, fault["label"])
        # Reduced data density matrix, averaged over ALL outcomes/shots.
        qc.save_density_matrix(qubits=list(range(code.n)), label="recovered_data")
        circuits.append(qc)
    backend = AerSimulator(method="statevector", max_parallel_threads=1)
    compiled = transpile(circuits, backend, optimization_level=0, seed_transpiler=seed)
    result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    rows = []
    for i, fault in enumerate(faults):
        syndrome = code.syndrome(fault["label"])
        recovery = table[syndrome]
        # Independently evaluate the syndrome probabilities of the gate circuit.
        state = Statevector.from_instruction(storage_error_circuit(code, fault["label"]))
        branches = state.data.reshape(2**len(code.stabilizers), 2**code.n)
        probabilities = np.sum(abs(branches)**2, axis=1)
        before = Statevector(code.target).evolve(Pauli(fault["label"])).data
        after = Statevector(before).evolve(Pauli(recovery)).data
        rho = np.asarray(result.data(i)["recovered_data"])
        logical = residual_logical_operator(code, fault["label"], recovery)
        expected_rho = np.outer(after, after.conj())
        if not np.allclose(rho, expected_rho, atol=1e-10):
            raise AssertionError("Dynamic recovery circuit disagrees with independent Pauli algebra")
        counts = result.get_counts(i)
        if counts != {syndrome: shots}:
            raise AssertionError(f"Expected deterministic syndrome {syndrome}, got {counts}")
        rows.append({
            **fault, "syndrome": syndrome, "syndrome_counts": counts,
            "syndrome_probability": float(probabilities[int(syndrome, 2)]),
            "detection_probability": float(1-probabilities[0]),
            "candidate_errors": candidates[syndrome], "recovery": recovery,
            "fidelity_before": float(abs(np.vdot(code.target, before))**2),
            "fidelity_after": float(np.vdot(code.target, rho @ code.target).real),
            "codespace_after": float(np.trace(code.projector @ rho).real),
            "corrects_arbitrary_logical_state": corrects_codespace(logical),
            "residual_logical_real": logical.real.tolist(),
            "residual_logical_imag": logical.imag.tolist(),
        })
    errors = rows[1:]
    return {
        "code": code.name, "shots_per_error": shots, "seed": seed,
        "total_shots": len(rows)*shots, "recovery_uses_injected_error": False,
        "postselection": False, "assumptions": "One data Pauli after encoding; perfect checks and correction",
        "decoder": "First syndrome representative in qubit order, then X/Y/Z; identity for no error",
        "nonzero_syndromes": len(candidates)-1,
        "detected_errors": int(sum(np.isclose(row["detection_probability"], 1) for row in errors)),
        "corrected_errors": sum(row["corrects_arbitrary_logical_state"] for row in errors),
        "total_errors": len(errors), "rows": rows,
    }
