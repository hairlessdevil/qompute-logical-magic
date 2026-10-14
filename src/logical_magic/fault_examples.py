"""A small, fixed set of fault-location examples, not an exhaustive sweep."""

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, Statevector

from .circuits import build_preparation
from .codes import Code
from .error_correction import recovery_table


def representative_cases(code: Code) -> list[dict]:
    prep = build_preparation(code)
    qc = prep.unitary
    checks_start = next(i for i, stage in enumerate(prep.stages) if stage.startswith("check_"))
    magic_end = max(i for i, stage in enumerate(prep.stages) if stage == "magic_input")
    first_cx = next(i for i, inst in enumerate(qc.data)
                    if prep.stages[i] == "encoding" and inst.operation.name == "cx")
    target = qc.find_bit(qc.data[first_cx].qubits[1]).index
    first_check_couplings = [i for i, inst in enumerate(qc.data)
                             if prep.stages[i].startswith("check_0_") and len(inst.qubits) == 2]
    cases = [
        {"name": "control", "description": "No error", "events": 0},
        {"name": "storage_X0", "description": "X on q0 after encoding (previous test)",
         "location": checks_start-1, "qubit": 0, "pauli": "X", "events": 1},
        {"name": "input_Y0", "description": "Y on q0 after magic preparation, before encoder",
         "location": magic_end, "qubit": 0, "pauli": "Y", "events": 1},
        {"name": "encoding_Z_target", "description": f"Z on q{target} after the first encoder CX",
         "location": first_cx, "qubit": target, "pauli": "Z", "events": 1},
        {"name": "ancilla_hook_X", "description": f"X on ancilla q{code.n} after check 0's second coupling",
         "location": first_check_couplings[1], "qubit": code.n, "pauli": "X", "events": 1},
        {"name": "late_X0", "description": "X on q0 after extraction, before ancilla readout",
         "location": len(qc.data)-1, "qubit": 0, "pauli": "X", "events": 1},
        {"name": "readout_flip", "description": "Flip recorded syndrome bit 0; no data fault",
         "readout_bit": 0, "events": 1},
        {"name": "faulty_recovery", "description": "Storage X0, then Z after the executed recovery gate",
         "location": checks_start-1, "qubit": 0, "pauli": "X", "recovery_fault": "Z", "events": 2},
    ]
    for case in cases:
        if "location" in case:
            inst = qc.data[case["location"]]
            case["after_operation"] = inst.operation.name
            case["operation_qubits"] = [qc.find_bit(q).index for q in inst.qubits]
    return cases


def example_unitary(code: Code, case: dict) -> QuantumCircuit:
    source = build_preparation(code).unitary
    qc = QuantumCircuit(source.num_qubits, name=f"{code.name}_{case['name']}")
    qc.global_phase = source.global_phase
    for i, inst in enumerate(source.data):
        qc.append(inst.operation, [source.find_bit(q).index for q in inst.qubits])
        if case.get("location") == i:
            getattr(qc, case["pauli"].lower())(case["qubit"])
    return qc


def example_recovery_circuit(code: Code, case: dict) -> QuantumCircuit:
    qc = example_unitary(code, case)
    syn = ClassicalRegister(len(code.stabilizers), "syndrome")
    qc.add_register(syn)
    if "readout_bit" in case:
        # For a terminal Z measurement, swapping the ancilla's 0/1 labels
        # reproduces a classical recorded-bit flip, including data conditioning.
        qc.x(code.n + case["readout_bit"])
    qc.measure(range(code.n, qc.num_qubits), syn)
    for syndrome, recovery in recovery_table(code).items():
        if set(recovery) == {"I"}:
            continue
        with qc.if_test((syn, int(syndrome, 2))):
            for q, p in enumerate(reversed(recovery)):
                if p != "I":
                    getattr(qc, p.lower())(q)
                    if "recovery_fault" in case:
                        getattr(qc, case["recovery_fault"].lower())(q)
    return qc


def ideal_cleanup_observable(code: Code) -> np.ndarray:
    """Fidelity AFTER a hypothetical extra ideal syndrome/recovery round.

    This is a diagnostic observable of the actual output, not its delivered
    fidelity. Use the same fixed decoder, including its four-qubit limitations.
    """
    ident = np.eye(2**code.n, dtype=complex)
    target = np.outer(code.target, code.target.conj())
    table = recovery_table(code)
    out = np.zeros_like(ident)
    m = len(code.stabilizers)
    for syndrome in range(2**m):
        projector = ident.copy()
        for bit, label in enumerate(code.stabilizers):
            projector = projector @ ((ident + (-1)**((syndrome >> bit) & 1)*Pauli(label).to_matrix()) / 2)
        recovery = Pauli(table.get(format(syndrome, f"0{m}b"), "I"*code.n)).to_matrix()
        out += projector @ recovery.conj().T @ target @ recovery @ projector
    return out


def exact_example(code: Code, case: dict) -> tuple[dict, np.ndarray]:
    """Sum all recorded-syndrome branches; never discard a nonzero syndrome."""
    state = Statevector.from_instruction(example_unitary(code, case))
    m = len(code.stabilizers)
    branches = state.data.reshape(2**m, 2**code.n)
    table = recovery_table(code)
    mask = (1 << case["readout_bit"]) if "readout_bit" in case else 0
    rho = np.zeros((2**code.n, 2**code.n), complex)
    recorded = {}
    for true_syndrome, vector in enumerate(branches):
        probability = float(np.vdot(vector, vector).real)
        if probability < 1e-12:
            continue
        syndrome = format(true_syndrome ^ mask, f"0{m}b")
        recorded[syndrome] = probability
        recovery = table.get(syndrome, "I"*code.n)
        corrected = Statevector(vector).evolve(Pauli(recovery))
        if "recovery_fault" in case:
            for q, p in enumerate(reversed(recovery)):
                if p != "I":
                    corrected = corrected.evolve(Pauli(case["recovery_fault"]), qargs=[q])
        rho += np.outer(corrected.data, corrected.data.conj())
    result = {
        **case, "syndrome_probabilities": recorded,
        "zero_syndrome_probability": recorded.get("0"*m, 0.0),
        "fidelity_after_recovery": float(np.vdot(code.target, rho @ code.target).real),
        "codespace_after_recovery": float(np.trace(code.projector @ rho).real),
        "fidelity_after_hypothetical_ideal_round": float(np.trace(ideal_cleanup_observable(code) @ rho).real),
    }
    return result, rho


def run_representative_examples(code: Code, shots: int = 256, seed: int = 20261014) -> dict:
    from qiskit_aer import AerSimulator

    cases = representative_cases(code)
    circuits = []
    for case in cases:
        qc = example_recovery_circuit(code, case)
        qc.save_density_matrix(qubits=list(range(code.n)), label="data_out")
        circuits.append(qc)
    backend = AerSimulator(method="statevector", max_parallel_threads=1)
    result = backend.run(transpile(circuits, backend, optimization_level=0, seed_transpiler=seed),
                         shots=shots, seed_simulator=seed).result()
    rows = []
    for i, case in enumerate(cases):
        row, rho = exact_example(code, case)
        actual = np.asarray(result.data(i)["data_out"])
        if not np.allclose(actual, rho, atol=1e-10):
            raise AssertionError("Representative dynamic circuit disagrees with exact branch calculation")
        counts = result.get_counts(i)
        # These selected Pauli examples have deterministic ideal syndromes.
        if len(row["syndrome_probabilities"]) != 1 or set(counts) != set(row["syndrome_probabilities"]):
            raise AssertionError("Unexpected syndrome in representative example")
        rows.append({**row, "syndrome_counts": counts})
    return {"code": code.name, "shots_per_case": shots, "seed": seed, "rows": rows,
            "scope": "Selected examples only; no fault rates, no idle timing model, no exhaustive guarantee",
            "recovery_fault_example": "One storage error plus one recovery-gate fault: two events, separately labeled"}
