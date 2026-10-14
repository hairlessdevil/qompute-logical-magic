"""Local Aachen calibration model, scheduled delays, and recorded-syndrome states.

No IBM service connection, hardware execution, or manually injected faults.
"""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.providers import BackendV2, Options
from qiskit.quantum_info import Pauli
from qiskit.transpiler import Target
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.device.models import basic_device_readout_errors
from qiskit_ibm_runtime.fake_provider import FakeAachen

from .circuits import Preparation
from .simulation import wilson_interval


# Fixed connected breadth-first patch rooted at physical qubit 3. No calibration
# or fidelity optimization is used to select it. The same target serves every row.
DEFAULT_PATCH = (3, 2, 4, 16, 1, 5, 23, 0, 6)
DEFAULT_SEED = 20261014


class AachenPatch(BackendV2):
    """Reduced BackendV2 retaining actual Aachen instruction/qubit properties."""

    def __init__(self, physical_qubits=DEFAULT_PATCH):
        self.snapshot = FakeAachen()
        self.physical_qubits = tuple(physical_qubits)
        if len(set(self.physical_qubits)) != len(self.physical_qubits):
            raise ValueError("Patch must not contain duplicate physical qubits")
        source = self.snapshot.target
        self._target = Target(
            description="Fixed connected patch of the packaged FakeAachen snapshot",
            num_qubits=len(self.physical_qubits), dt=source.dt,
            granularity=source.granularity, min_length=source.min_length,
            pulse_alignment=source.pulse_alignment,
            acquire_alignment=source.acquire_alignment,
            qubit_properties=deepcopy([source.qubit_properties[q] for q in self.physical_qubits]),
        )
        remap = {p: i for i, p in enumerate(self.physical_qubits)}
        for name in sorted(source.operation_names):
            if name == "if_else":
                continue  # These circuits use terminal postselection, no feed-forward.
            properties = {
                tuple(remap[q] for q in qs): deepcopy(prop)
                for qs, prop in source[name].items()
                if qs is not None and all(q in remap for q in qs)
            }
            if properties:
                self._target.add_instruction(source.operation_from_name(name), properties, name=name)
        super().__init__(name="fake_aachen_fixed_patch")

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @classmethod
    def _default_options(cls):
        return Options()

    def run(self, run_input, **options):
        raise NotImplementedError("This target is for local transpilation only; use AerSimulator")

    def noise_model(self, readout_error=True):
        # Aer 0.17.2 also attaches a Delay-only RelaxationNoisePass here. Do not
        # attach another one: that would apply idle relaxation twice.
        return NoiseModel.from_backend(self, gate_error=True, readout_error=readout_error,
                                       thermal_relaxation=True, temperature=0)

    def readout_matrices(self):
        matrices = {q: np.eye(2) for q in range(self.num_qubits)}
        for qs, error in basic_device_readout_errors(target=self.target):
            matrices[qs[0]] = np.asarray(error.probabilities)
        return matrices

    def provenance(self):
        folder = Path(self.snapshot.dirname)
        files = [folder / self.snapshot.conf_filename, folder / self.snapshot.props_filename]
        return {
            "device": "ibm_aachen", "provider": "qiskit_ibm_runtime.fake_provider.FakeAachen",
            "calibration_date": self.snapshot.properties().last_update_date.isoformat(),
            "calibration_kind": "packaged snapshot, not a current live calibration",
            "snapshot_files": {p.name: {"sha256": sha256(p.read_bytes()).hexdigest(),
                                         "source_path": str(p)} for p in files},
            "physical_patch": list(self.physical_qubits), "dt_seconds": self.dt,
            "coupling_edges_local": [list(e) for e in self.coupling_map.get_edges()],
            "temperature_mK": 0,
            "readout_model": "independent symmetric assignment errors from BackendV2 Target",
            "qubits": [
                {"local": i, "physical": p, "t1_seconds": prop.t1,
                 "t2_seconds": prop.t2, "effective_t2_seconds": min(prop.t2, 2*prop.t1),
                 "readout_error": self.target["measure"][(i,)].error,
                 "readout_duration_seconds": self.target["measure"][(i,)].duration}
                for i, (p, prop) in enumerate(zip(self.physical_qubits, self.target.qubit_properties))
            ],
        }


@dataclass
class ScheduledPreparation:
    preparation: Preparation
    circuit: QuantumCircuit
    data_wires: tuple[int, ...]
    syndrome_wires: tuple[int, ...]
    resources: dict


def schedule_preparation(prep: Preparation, backend: AachenPatch,
                         seed=DEFAULT_SEED) -> ScheduledPreparation:
    qc = prep.unitary.copy()
    # Establish a common delivery boundary. In B2, data wait until ALL terminal
    # syndrome readouts finish; PadDelay makes this wait explicit on each wire.
    qc.barrier()
    if prep.n_checks:
        qc.add_register(ClassicalRegister(prep.n_checks, "syndrome"))
        qc.measure(range(prep.n_data, prep.n_data + prep.n_checks), range(prep.n_checks))
    compiled = transpile(
        qc, target=backend.target, initial_layout=list(range(qc.num_qubits)),
        optimization_level=1, seed_transpiler=seed, scheduling_method="alap",
    )
    final = compiled.layout.final_index_layout(filter_ancillas=True)
    data = tuple(final[:prep.n_data])
    syndrome = [None] * prep.n_checks
    idle_ticks = np.zeros(backend.num_qubits, dtype=int)
    end_ticks = np.zeros(backend.num_qubits, dtype=int)
    readout_starts = []
    readout_ends = []
    for start, inst in zip(compiled.op_start_times, compiled.data):
        name = inst.operation.name
        wires = tuple(compiled.find_bit(q).index for q in inst.qubits)
        if name == "delay":
            assert inst.operation.unit == "dt"
            duration = inst.operation.duration
            idle_ticks[wires[0]] += duration
        elif name == "barrier":
            duration = 0
        else:
            duration = round(backend.target[name][wires].duration / backend.dt)
        for q in wires:
            end_ticks[q] = max(end_ticks[q], start + duration)
        if name == "measure":
            syndrome[compiled.find_bit(inst.clbits[0]).index] = wires[0]
            readout_starts.append(int(start))
            readout_ends.append(int(start + duration))
    duration_ticks = int(max(end_ticks))
    if prep.n_checks:
        # Exact conditioning below requires terminal simultaneous readout. Fail
        # explicitly if a future backend/compiler produces another protocol.
        assert len(set(readout_starts)) == len(set(readout_ends)) == 1
        for q in data:
            covering = [inst.operation.duration for start, inst in zip(compiled.op_start_times, compiled.data)
                        if inst.operation.name == "delay"
                        and compiled.find_bit(inst.qubits[0]).index == q
                        and start <= readout_starts[0]
                        and start + inst.operation.duration >= readout_ends[0]]
            assert covering, "Missing data idle interval during syndrome readout"
    resources = {
        "duration_seconds": duration_ticks * backend.dt,
        "syndrome_readout_wait_seconds": (readout_ends[0]-readout_starts[0])*backend.dt if prep.n_checks else 0,
        "delay_count": int(compiled.count_ops().get("delay", 0)),
        "idle_seconds_by_local_wire": (idle_ticks * backend.dt).tolist(),
        "operation_counts": {k: int(v) for k, v in compiled.count_ops().items()},
        "two_qubit_gates": int(compiled.count_ops().get("cz", 0)),
        "initial_logical_to_physical": list(backend.physical_qubits[:qc.num_qubits]),
        "final_data_local_wires": list(data),
        "final_data_physical_qubits": [backend.physical_qubits[q] for q in data],
        "syndrome_local_wires_in_bit_order": syndrome,
        "syndrome_physical_qubits_in_bit_order": [backend.physical_qubits[q] for q in syndrome],
        "seed_transpiler": seed, "optimization_level": 1, "scheduling_method": "alap",
    }
    return ScheduledPreparation(prep, compiled, data, tuple(syndrome), resources)


def snapshot_circuit(scheduled: ScheduledPreparation, include_delays=True):
    """Save the state with data idles included, immediately before terminal readout.

    The model puts assignment error (no quantum channel) on measurement. Data
    delays commute with the terminal ancilla measurements, so those measurements
    can be placed after the save without changing the classical-quantum output.
    """
    qc = scheduled.circuit.copy_empty_like()
    measurements = []
    for inst in scheduled.circuit.data:
        if inst.operation.name == "measure":
            measurements.append(inst)
        elif include_delays or inst.operation.name != "delay":
            qc.append(inst)
    qc.save_density_matrix(label="before_readout")
    for inst in measurements:
        qc.append(inst)
    return qc


def syndrome_blocks(rho, data_wires, syndrome_wires):
    """Trace spectators and keep diagonal ancilla blocks in logical data order."""
    n = int(np.log2(len(rho)))
    rest = [q for q in range(n) if q not in (*data_wires, *syndrome_wires)]
    order = [*data_wires, *syndrome_wires, *rest]
    if sorted(order) != list(range(n)):
        raise ValueError("Data and syndrome wires must be distinct valid indices")
    indices = np.arange(2**n)
    permutation = sum(((indices >> i) & 1) << q for i, q in enumerate(order))
    d, s, e = 2**len(data_wires), 2**len(syndrome_wires), 2**len(rest)
    ordered = np.asarray(rho)[np.ix_(permutation, permutation)].reshape(e, s, d, e, s, d)
    return np.einsum("esiesj->sij", ordered)


def recorded_syndrome_state(blocks, assignment_matrices):
    """Exact state conditioned on recorded zero; assignment[true, recorded]."""
    n = len(assignment_matrices)
    assignment = np.ones((2**n, 2**n))
    for true in range(2**n):
        for recorded in range(2**n):
            for i, matrix in enumerate(assignment_matrices):
                assignment[true, recorded] *= matrix[(true >> i) & 1, (recorded >> i) & 1]
    true_probs = np.trace(blocks, axis1=1, axis2=2).real
    recorded_probs = true_probs @ assignment
    accepted = np.einsum("s,sij->ij", assignment[:, 0], blocks)
    acceptance = float(recorded_probs[0])
    if acceptance <= 0:
        raise ValueError("Zero acceptance: conditional state is undefined")
    return accepted / acceptance, {
        "acceptance": acceptance,
        "true_syndrome_probabilities": true_probs.tolist(),
        "recorded_syndrome_probabilities": recorded_probs.tolist(),
        "accepted_fraction_from_nonzero_true_syndrome": float(
            (acceptance - true_probs[0]*assignment[0, 0]) / acceptance),
    }


def state_metrics(rho, prep):
    fidelity = float(np.vdot(prep.target, rho @ prep.target).real)
    projected = rho if prep.strategy == "B0" else prep.code.basis.conj().T @ rho @ prep.code.basis
    codespace = float(np.trace(projected).real)
    return {
        "fidelity": fidelity, "codespace_population": codespace,
        "fidelity_conditioned_on_codespace_diagnostic": fidelity / codespace,
        "bloch_conditioned_on_codespace_diagnostic": [
            float(np.trace(projected @ Pauli(p).to_matrix()).real / codespace) for p in "XYZ"
        ],
    }


def simulate_scheduled(scheduled: ScheduledPreparation, backend: AachenPatch,
                       mode="full", shots=8192, seed=DEFAULT_SEED):
    """Exact channel evolution plus finite-shot recorded-syndrome sampling.

    full: calibrated gates/readout + all scheduled delay relaxation.
    no_delay: identical gate sequence/noise/readout; omit Delay instructions only.
    no_readout: same gate and delay channels; ideal syndrome assignment only.
    ideal: same routed native circuit with all noise disabled.
    """
    if mode not in {"full", "no_delay", "no_readout", "ideal"}:
        raise ValueError(f"Unknown simulation mode {mode}")
    if shots < 1:
        raise ValueError("shots must be positive")
    simulator = AerSimulator(method="density_matrix", max_parallel_threads=1,
                             noise_model=None if mode == "ideal" else backend.noise_model(readout_error=mode != "no_readout"))
    qc = snapshot_circuit(scheduled, include_delays=mode != "no_delay")
    result = simulator.run(qc, shots=shots, seed_simulator=seed).result(timeout=600)
    if not result.success:
        raise RuntimeError(result.status)
    full_rho = np.asarray(result.data(0)["before_readout"])
    blocks = syndrome_blocks(full_rho, scheduled.data_wires, scheduled.syndrome_wires)
    matrices = backend.readout_matrices()
    assignment = [np.eye(2) if mode in {"ideal", "no_readout"} else matrices[q] for q in scheduled.syndrome_wires]
    accepted, probabilities = recorded_syndrome_state(blocks, assignment)
    m = scheduled.preparation.n_checks
    counts = result.get_counts() if m else {"": shots}
    accepted_shots = counts.get("0" * m, 0)
    metrics = {
        "mode": mode, **probabilities, **state_metrics(accepted, scheduled.preparation),
        "unconditional_fidelity": state_metrics(blocks.sum(axis=0), scheduled.preparation)["fidelity"],
        "shots": shots, "seed_simulator": seed,
        "syndrome_counts": counts, "acceptance_sampled": accepted_shots/shots,
        "acceptance_wilson_95_interval": list(wilson_interval(accepted_shots, shots)),
        "aer_method": result.results[0].metadata["method"],
    }
    return metrics, accepted


def validate_idle_relaxation(backend: AachenPatch, duration_seconds=50e-6):
    """Analytic T1/T2 check: clean initial states followed ONLY by a delay.

    These are noise-model diagnostics, not inserted faults in the experiment.
    """
    ticks = round(duration_seconds / backend.dt)
    duration = ticks * backend.dt
    circuits = []
    for rho in [np.array([[0, 0], [0, 1]]), np.ones((2, 2))/2]:
        qc = QuantumCircuit(1)
        qc.set_density_matrix(rho)
        qc.delay(ticks, 0, unit="dt")
        qc.save_density_matrix()
        circuits.append(qc)
    sim = AerSimulator(method="density_matrix", max_parallel_threads=1, noise_model=backend.noise_model())
    result = sim.run(circuits, shots=1).result(timeout=60)
    one, plus = [np.asarray(result.data(i)["density_matrix"]) for i in range(2)]
    prop = backend.target.qubit_properties[0]
    actual = [float(one[1, 1].real), float(2*plus[0, 1].real)]
    expected = [float(np.exp(-duration/prop.t1)), float(np.exp(-duration/min(prop.t2, 2*prop.t1)))]
    return {"physical_qubit": backend.physical_qubits[0], "delay_seconds": duration,
            "observed_excited_population_and_x": actual,
            "expected_exp_minus_t_over_t1_t2": expected,
            "max_absolute_error": float(np.max(np.abs(np.array(actual)-expected)))}
