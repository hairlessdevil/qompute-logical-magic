"""Count-based target-projector estimation on copies of an Aachen output.

Preparation routing and every preparation delay are frozen. Basis rotations
are appended at the original delivery boundary, followed by destructive data
readout. Only recorded-zero syndrome shots enter the estimator.
"""

from functools import lru_cache

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .aachen_noise import DEFAULT_SEED, ScheduledPreparation


@lru_cache(maxsize=4)
def _native_rotation(pauli):
    one = QuantumCircuit(1)
    if pauli == "Y":
        one.sdg(0)
    if pauli in "XY":
        one.h(0)
    return transpile(one, basis_gates=["rz", "sx", "x", "cz"], optimization_level=1)


def analysis_suffix(scheduled, backend, label):
    """Local native basis rotations, padded to a common data-readout start.

    H measures X, Sdg then H measures Y. Virtual Rz takes zero time. Z/I wires
    wait while other data wires rotate. No routing or preparation recompilation.
    """
    n = scheduled.preparation.n_data
    if len(label) != n or any(p not in "IXYZ" for p in label):
        raise ValueError("Expected a data-register Pauli label")
    suffix = QuantumCircuit(scheduled.circuit.num_qubits)
    ticks = {}
    for i, p in enumerate(reversed(label)):
        wire = scheduled.data_wires[i]
        native = _native_rotation(p)
        suffix.compose(native, [wire], inplace=True)
        ticks[wire] = sum(round(backend.target[inst.operation.name][(wire,)].duration / backend.dt)
                          for inst in native.data)
    max_ticks = max(ticks.values())
    for wire, elapsed in ticks.items():
        if max_ticks > elapsed:
            suffix.delay(max_ticks-elapsed, wire, unit="dt")
    return suffix, max_ticks * backend.dt


def measurement_circuit(scheduled: ScheduledPreparation, backend, label,
                        defer_syndrome=False, save_probabilities=False):
    """Build a hardware-form circuit or an equivalent efficient Aer circuit.

    Hardware form retains terminal syndrome measurements in the preparation
    prefix. Aer form defers those measurements across the data-only analysis
    suffix, which is equivalent for this independent assignment/channel model.
    The original data readout waits stay in both forms. Deferral does not add
    quantum evolution to syndrome wires after their original measurement time.
    """
    if save_probabilities and not defer_syndrome:
        raise ValueError("Unconditional probability diagnostic requires deferred syndrome")
    qc = scheduled.circuit.copy_empty_like()
    syndrome_measurements = []
    for inst in scheduled.circuit.data:
        if defer_syndrome and inst.operation.name == "measure":
            syndrome_measurements.append(inst)
        else:
            qc.append(inst)
    qc.barrier()
    suffix, duration = analysis_suffix(scheduled, backend, label)
    qc.compose(suffix, inplace=True)
    qc.barrier(*scheduled.data_wires)
    data = ClassicalRegister(scheduled.preparation.n_data, "quality_readout")
    qc.add_register(data)
    if save_probabilities:
        qc.save_probabilities([*scheduled.syndrome_wires, *scheduled.data_wires], label="before_assignment")
    for inst in syndrome_measurements:
        qc.append(inst)
    qc.measure(scheduled.data_wires, data)
    qc.metadata = {"analysis_pauli": label, "analysis_basis_duration_seconds": duration,
                   "syndrome_measurement_deferred": defer_syndrome}
    return qc


def parity_weights(label, assignment_matrices=None):
    """A[true, recorded] @ weights = parity; invert data assignment only."""
    n = len(label)
    values = np.ones(2**n)
    indices = np.arange(2**n)
    for q, p in enumerate(reversed(label)):
        if p == "I":
            continue
        weights = np.array([1., -1.])
        if assignment_matrices is not None:
            weights = np.linalg.solve(assignment_matrices[q], weights)
        values *= weights[(indices >> q) & 1]
    return values


def accepted_score(counts, n_syndrome, weights):
    """Mean, variance of mean, and sample count from recorded-zero shots."""
    total, first, second = 0, 0., 0.
    mask = (1 << n_syndrome)-1
    for key, count in counts.items():
        value = int(key.replace(" ", ""), 2)
        if value & mask:
            continue
        score = weights[value >> n_syndrome]
        total += count
        first += count*score
        second += count*score**2
    if total < 2:
        raise ValueError("Need at least two accepted shots per setting; increase the budget")
    mean = first/total
    variance = max(0., second/total-mean**2)/(total-1)
    return {"expectation": float(mean), "variance_of_mean": float(variance), "accepted_shots": total}


def expected_measurement_score(probabilities, n_syndrome, syndrome_assignment,
                               data_assignment, weights):
    """Simulator diagnostic, never used to calculate the finite-shot estimate."""
    by_data_syn = np.asarray(probabilities).reshape(2**len(data_assignment), 2**n_syndrome)
    accept_weights = np.ones(2**n_syndrome)
    for s in range(len(accept_weights)):
        for i, matrix in enumerate(syndrome_assignment):
            accept_weights[s] *= matrix[(s >> i) & 1, 0]
    true_data = by_data_syn @ accept_weights
    # Tensor product uses q[n-1] ... q0 order.
    assignment = np.array([[1.]])
    for matrix in reversed(data_assignment):
        assignment = np.kron(assignment, matrix)
    recorded_data = true_data @ assignment
    return float(recorded_data @ weights / true_data.sum())


def summarize_terms(terms, identity_coefficient, field, confidence=.95):
    estimate = identity_coefficient
    variance, radius, infinite = 0., 0., identity_coefficient
    for term in terms:
        c, r = term["coefficient"], term[field]
        estimate += c*r["expectation"]
        variance += c*c*r["variance_of_mean"]
        # Conditional on each accepted sample count, scores are independent
        # bounded samples. Union bound across all settings gives 95% coverage
        # of the measurement-score mean, not unknown systematic error.
        radius += abs(c)*r["score_range"]*np.sqrt(
            np.log(2*len(terms)/(1-confidence))/(2*r["accepted_shots"]))
        infinite += c*r["infinite_shot_measurement_expectation"]
    se = float(np.sqrt(variance))
    return {
        "fidelity_estimate": float(estimate), "standard_error": se,
        "normal_approx_95_interval": [float(estimate-1.95996398454*se), float(estimate+1.95996398454*se)],
        "conservative_95_interval_for_measurement_score": [float(estimate-radius), float(estimate+radius)],
        "infinite_shot_measurement_score_diagnostic": float(infinite),
    }


def estimate_noisy_fidelity(scheduled, backend, shots=8192, seed=DEFAULT_SEED,
                            ideal=False, circuit_callback=None):
    """Estimate the full physical target overlap from independent Pauli settings.

    Returns raw and data-readout-mitigated scores with sampling uncertainty.
    Mitigation uses exact assignment matrices from the chosen snapshot; their
    calibration uncertainty is not included. Basis-gate/delay noise remains a
    systematic bias. Syndrome bits are never mitigated: the accepted ensemble
    stays the one selected by the actual recorded syndrome.
    """
    if shots < 2:
        raise ValueError("At least two shots per setting are required")
    prep = scheduled.preparation
    terms = prep.fidelity_observable.to_list()
    identity = "I" * prep.n_data
    offset = float(sum(c.real for p, c in terms if p == identity))
    paulis = [(p, float(c.real)) for p, c in terms if p != identity]
    matrices = backend.readout_matrices()
    data_assignment = [np.eye(2) if ideal else matrices[q] for q in scheduled.data_wires]
    syndrome_assignment = [np.eye(2) if ideal else matrices[q] for q in scheduled.syndrome_wires]
    simulator = AerSimulator(method="density_matrix", max_parallel_threads=1,
                             noise_model=None if ideal else backend.noise_model())
    rows = []
    for i, (label, coefficient) in enumerate(paulis):
        qc = measurement_circuit(scheduled, backend, label, defer_syndrome=True, save_probabilities=True)
        if circuit_callback:
            circuit_callback(label, measurement_circuit(scheduled, backend, label))
        result = simulator.run(qc, shots=shots, seed_simulator=seed+i).result(timeout=600)
        if not result.success:
            raise RuntimeError(result.status)
        counts = result.get_counts()
        probabilities = result.data(0)["before_assignment"]
        term = {"pauli": label, "coefficient": coefficient, "counts": counts,
                "seed_simulator": seed+i, "analysis_basis_duration_seconds": qc.metadata["analysis_basis_duration_seconds"]}
        for key, assignment in [("raw", None), ("data_readout_mitigated", data_assignment)]:
            weights = parity_weights(label, assignment)
            term[key] = accepted_score(counts, prep.n_checks, weights)
            term[key]["score_range"] = float(np.ptp(weights))
            term[key]["infinite_shot_measurement_expectation"] = expected_measurement_score(
                probabilities, prep.n_checks, syndrome_assignment, data_assignment, weights)
        rows.append(term)
    return {
        "shots_per_setting": shots, "measurement_settings": len(paulis),
        "total_attempted_shots": shots*len(paulis),
        "total_accepted_shots": sum(t["raw"]["accepted_shots"] for t in rows),
        "seed": seed, "ideal_control": ideal, "identity_coefficient": offset,
        "data_physical_qubits": [backend.physical_qubits[q] for q in scheduled.data_wires],
        "data_assignment_matrices": [m.tolist() for m in data_assignment],
        "syndrome_assignment_matrices": [m.tolist() for m in syndrome_assignment],
        "mitigation": "data assignment only, exact snapshot matrices; no syndrome mitigation or basis-noise inversion",
        "raw": summarize_terms(rows, offset, "raw"),
        "data_readout_mitigated": summarize_terms(rows, offset, "data_readout_mitigated"),
        "terms": rows,
    }
