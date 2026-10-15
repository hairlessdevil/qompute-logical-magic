"""Disable idle channels in a time window without changing scheduled delays."""

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit_aer.noise.device import basic_device_gate_errors, basic_device_readout_errors

from .aachen_noise import recorded_syndrome_state, state_metrics, syndrome_blocks
from .simulation import wilson_interval


def gate_and_readout_noise_model(backend):
    """Same calibrated gate/readout channels, with delay channels added explicitly.

    These are the public device helpers used by NoiseModel.from_backend. Gate
    duration relaxation remains enabled. No automatic Delay noise pass is added.
    """
    noise = NoiseModel(basis_gates=backend.operation_names)
    for qubits, error in basic_device_readout_errors(target=backend.target):
        noise.add_readout_error(error, qubits)
    for name, qubits, error in basic_device_gate_errors(
        target=backend.target, gate_error=True, thermal_relaxation=True, temperature=0,
    ):
        noise.add_quantum_error(error, name, qubits)
    return noise


def delay_ablation_snapshot(scheduled, backend, disabled_window=None):
    """Retain every Delay; add only its relaxation outside [start_dt, end_dt).

    Delays crossing a boundary receive the channel for their retained duration.
    The stationary relaxation channels compose as a semigroup, and no operation
    on that wire intervenes within a Delay. It is therefore equivalent to apply
    that retained channel at the end of the unchanged Delay instruction.
    All physical wires are included, so routing does not change the scope.
    """
    if disabled_window is not None:
        left, right = disabled_window
        if not (0 <= left <= right):
            raise ValueError("Expected a nonnegative ordered window in dt")
    source = scheduled.circuit
    starts = source.op_start_times
    if len(starts) != len(source.data):
        raise ValueError("Circuit must have scheduling metadata")
    qc = source.copy_empty_like()
    measurements, audit = [], []
    for index, (start, inst) in enumerate(zip(starts, source.data)):
        if inst.operation.name == "measure":
            measurements.append(inst)
            continue
        qc.append(inst)
        if inst.operation.name != "delay":
            continue
        if inst.operation.unit != "dt":
            raise ValueError("Expected Delay durations in dt")
        duration = int(inst.operation.duration)
        removed = 0 if disabled_window is None else max(0, min(start+duration, right)-max(start, left))
        retained = duration - removed
        wire = source.find_bit(inst.qubits[0]).index
        audit.append({"instruction_index": index, "local_wire": wire,
                      "physical_qubit": backend.physical_qubits[wire], "start_dt": int(start),
                      "duration_dt": duration, "noise_disabled_dt": int(removed),
                      "noise_retained_dt": int(retained)})
        if retained:
            prop = backend.target.qubit_properties[wire]
            error = thermal_relaxation_error(prop.t1, min(prop.t2, 2*prop.t1),
                                             retained * backend.dt, excited_state_population=0)
            qc.append(error.to_instruction(), inst.qubits)
    qc.save_density_matrix(label="before_readout")
    for inst in measurements:
        qc.append(inst)
    # Simulator channels and deferred terminal measurements are the only changes.
    original = [i for i in source.data if i.operation.name != "measure"] + measurements
    kept = [i for i in qc.data if i.operation.name not in {"quantum_channel", "save_density_matrix"}]
    assert kept == original
    return qc, audit


def simulate_delay_ablation(scheduled, backend, disabled_window=None, shots=8192, seed=20261014):
    qc, audit = delay_ablation_snapshot(scheduled, backend, disabled_window)
    sim = AerSimulator(method="density_matrix", max_parallel_threads=1,
                       noise_model=gate_and_readout_noise_model(backend))
    result = sim.run(qc, shots=shots, seed_simulator=seed).result(timeout=600)
    if not result.success:
        raise RuntimeError(result.status)
    blocks = syndrome_blocks(np.asarray(result.data(0)["before_readout"]),
                             scheduled.data_wires, scheduled.syndrome_wires)
    assignment = backend.readout_matrices()
    accepted, probabilities = recorded_syndrome_state(blocks, [assignment[q] for q in scheduled.syndrome_wires])
    counts = result.get_counts()
    n_checks = scheduled.preparation.n_checks
    accepted_shots = counts.get("0" * n_checks, 0)
    assert np.isclose(np.trace(accepted), 1, atol=1e-10, rtol=0)
    assert np.linalg.eigvalsh(accepted).min() > -1e-10
    metrics = {**probabilities, **state_metrics(accepted, scheduled.preparation),
               "unconditional_fidelity": state_metrics(blocks.sum(axis=0), scheduled.preparation)["fidelity"],
               "shots": shots, "seed_simulator": seed, "syndrome_counts": counts,
               "acceptance_sampled": accepted_shots / shots,
               "acceptance_wilson_95_interval": list(wilson_interval(accepted_shots, shots)),
               "disabled_window_dt": list(disabled_window) if disabled_window is not None else None,
               "delay_audit": audit}
    return metrics, accepted
