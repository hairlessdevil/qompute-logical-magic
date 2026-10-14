"""Exact branch states, finite-shot fidelity estimation, and circuit fault audit."""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, Statevector

from .circuits import Preparation

TOL = 1e-10


def branch_result(prep: Preparation, state: Statevector, syndrome: int = 0) -> dict:
    """Condition ONLY on physically recorded ancilla bits, never on data codespace."""
    branches = state.data.reshape(2**prep.n_checks, 2**prep.n_data)
    probabilities = np.sum(abs(branches) ** 2, axis=1)
    probability = float(probabilities[syndrome])
    if probability < TOL:
        return {"acceptance": 0.0, "fidelity": None, "codespace_population": None,
                "logical_bloch": None, "attempts_per_accept": None}
    accepted = branches[syndrome] / np.sqrt(probability)
    fidelity = float(abs(np.vdot(prep.target, accepted)) ** 2)
    if prep.strategy == "B0":
        codespace = 1.0
        ops = [Pauli(p).to_matrix() for p in "XYZ"]
    else:
        codespace = float(np.vdot(accepted, prep.code.projector @ accepted).real)
        ops = [Pauli(prep.code.logical_x).to_matrix(), prep.code.logical_y,
               Pauli(prep.code.logical_z).to_matrix()]
    return {"acceptance": probability, "fidelity": fidelity,
            "codespace_population": codespace,
            "logical_bloch": [float(np.vdot(accepted, op @ accepted).real) for op in ops],
            "attempts_per_accept": 1 / probability}


def exact_result(prep: Preparation) -> dict:
    state = Statevector.from_instruction(prep.unitary)
    out = branch_result(prep, state)
    branches = state.data.reshape(2**prep.n_checks, 2**prep.n_data)
    probs = np.sum(abs(branches) ** 2, axis=1)
    out["syndrome_probabilities"] = {
        format(i, f"0{prep.n_checks}b") if prep.n_checks else "": float(p)
        for i, p in enumerate(probs) if p > TOL
    }
    out["resources"] = prep.resources()
    return out


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    den = 1 + z*z / trials
    center = (p + z*z / (2*trials)) / den
    half = z * np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / den
    return [max(0.0, float(center-half)), min(1.0, float(center+half))]


def estimate_fidelity(prep: Preparation, shots: int = 4096, seed: int = 20261014) -> dict:
    """Measure every nonidentity term of |target><target| on fresh copies.

    The linear estimate is NOT clipped to [0,1]. The reported standard error is
    a conditional plug-in estimate, not an exact coverage guarantee. A separate
    Hoeffding/union bound supplies a conservative simultaneous 95% interval,
    including when sampled Pauli outcomes happen to be deterministic.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    from qiskit_aer import AerSimulator

    terms = prep.fidelity_observable.to_list()
    identity = "I" * prep.n_data
    offset = float(sum(c.real for p, c in terms if p == identity))
    measured_terms = [(p, float(c.real)) for p, c in terms if p != identity]
    circuits = [prep.measure_pauli(p) for p, _ in measured_terms]
    # A dedicated acceptance circuit avoids tying the acceptance estimate to
    # whichever Pauli setting happens to be listed first.
    if prep.n_checks:
        circuits.append(prep.measured())
    backend = AerSimulator(method="statevector", max_parallel_threads=1)
    compiled = transpile(circuits, backend, optimization_level=0, seed_transpiler=seed)
    result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    estimate, variance, radius = offset, 0.0, 0.0
    rows = []
    mask_syn = (1 << prep.n_checks) - 1
    for i, (label, coefficient) in enumerate(measured_terms):
        counts = result.get_counts(i)
        data_mask = sum(1 << q for q, p in enumerate(reversed(label)) if p != "I")
        accepted, signed = 0, 0
        for key, count in counts.items():
            bits = int(key.replace(" ", ""), 2)
            if bits & mask_syn:
                continue
            parity = ((bits >> prep.n_checks) & data_mask).bit_count() % 2
            accepted += count
            signed += (1 - 2*parity) * count
        if not accepted:
            raise RuntimeError(f"No accepted shots for {label}; increase the shot budget")
        mean = signed / accepted
        estimate += coefficient * mean
        variance += coefficient**2 * max(0.0, 1 - mean**2) / accepted
        # Each +/-1 sample has range 2. Union bound across measured settings.
        eps = np.sqrt(2 * np.log(2 * len(measured_terms) / 0.05) / accepted)
        radius += abs(coefficient) * eps
        rows.append({"pauli": label, "coefficient": coefficient, "accepted": accepted,
                     "expectation": mean, "counts": counts})
    if prep.n_checks:
        counts = result.get_counts(len(measured_terms))
        n_accept = sum(v for k, v in counts.items() if int(k.replace(" ", ""), 2) == 0)
        acceptance_counts = counts
        acceptance_ci = wilson_interval(n_accept, shots)
    else:
        n_accept, acceptance_counts, acceptance_ci = shots, {}, [1.0, 1.0]
    return {"fidelity_estimate": estimate, "standard_error_plugin": float(np.sqrt(variance)),
            "fidelity_conservative_95_interval": [max(0.0, estimate-radius), min(1.0, estimate+radius)],
            "acceptance_estimate": n_accept/shots, "acceptance_wilson_95_interval": acceptance_ci,
            "acceptance_counts": acceptance_counts, "shots_per_setting": shots,
            "measurement_settings": len(measured_terms), "total_shots": shots*len(circuits),
            "seed": seed, "identity_coefficient": offset, "terms": rows}


def circuit_fault_audit(prep: Preparation) -> list[dict]:
    """Enumerate faults on the ACTUAL unitary circuit and classical readout.

    Model: one nonidentity Pauli after each 1q/2q gate (all 15 two-qubit
    Paulis), X/Y/Z on initial |0> qubits, or one syndrome-bit flip. There
    are no idle, coherent-angle, leakage, relaxation, or simultaneous faults.
    Enumeration fractions are NOT physical error probabilities.
    """
    qc = prep.unitary
    prefixes = [Statevector.from_int(0, 2**qc.num_qubits)]
    for inst in qc.data:
        qargs = [qc.find_bit(q).index for q in inst.qubits]
        prefixes.append(prefixes[-1].evolve(inst.operation, qargs=qargs))
    rows = []

    def record(stage, location, gate, qubits, pauli, state, syndrome=0):
        result = branch_result(prep, state, syndrome)
        if result["acceptance"] < TOL:
            category = "rejected"
        elif result["fidelity"] < 1 - TOL:
            category = "accepted_harmful"
        else:
            category = "accepted_target"
        rows.append({"stage": stage, "location": location, "gate": gate,
                     "qubits": ",".join(map(str, qubits)), "pauli": pauli,
                     "category": category, **result})

    for q in range(qc.num_qubits):
        for p in "XYZ":
            state = prefixes[0].evolve(Pauli(p), qargs=[q]).evolve(qc)
            record("initialization", -1, "initial_zero", [q], p, state)
    for index, inst in enumerate(qc.data):
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if len(qubits) not in (1, 2):
            raise ValueError("Audit requires decomposed 1q/2q circuits")
        suffix = QuantumCircuit(qc.num_qubits)
        for later in qc.data[index+1:]:
            suffix.append(later.operation, [qc.find_bit(q).index for q in later.qubits])
        for chars in product("IXYZ", repeat=len(qubits)):
            p = "".join(chars)
            if set(p) == {"I"}:
                continue
            state = prefixes[index+1].evolve(Pauli(p), qargs=qubits).evolve(suffix)
            record(prep.stages[index], index, inst.operation.name, qubits, p, state)
    for bit in range(prep.n_checks):
        # Recorded all-zero syndrome with bit i flipped means true syndrome 2^i.
        record("readout", len(qc.data)+bit, "classical_bit_flip", [prep.n_data+bit],
               "bit_flip", prefixes[-1], syndrome=1 << bit)
    return rows


def storage_fault_audit(prep: Preparation) -> list[dict]:
    """Pauli faults inserted after encoding, BEFORE ideal syndrome extraction."""
    if prep.strategy != "B2_ED":
        raise ValueError("Use B2_ED for a storage-error syndrome audit")
    first_check = next(i for i, s in enumerate(prep.stages) if s.startswith("check_"))
    prefix, suffix = QuantumCircuit(prep.unitary.num_qubits), QuantumCircuit(prep.unitary.num_qubits)
    for i, inst in enumerate(prep.unitary.data):
        dest = prefix if i < first_check else suffix
        dest.append(inst.operation, [prep.unitary.find_bit(q).index for q in inst.qubits])
    state = Statevector.from_instruction(prefix)
    rows = []
    for q in range(prep.n_data):
        for p in "XYZ":
            label = list("I"*prep.n_data)
            label[-q-1] = p
            label = "".join(label)
            final = state.evolve(Pauli(p), qargs=[q]).evolve(suffix)
            probs = np.sum(abs(final.data.reshape(2**prep.n_checks, 2**prep.n_data))**2, axis=1)
            rows.append({"qubit": q, "pauli": p, "data_pauli": label,
                         "predicted_syndrome": prep.code.syndrome(label),
                         "observed_syndrome": format(int(np.argmax(probs)), f"0{prep.n_checks}b"),
                         **branch_result(prep, final)})
    return rows
