from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from logical_magic import CODES, build_preparation
from logical_magic.aachen_noise import AachenPatch, schedule_preparation, simulate_scheduled
from logical_magic.delay_ablation import (
    delay_ablation_snapshot, gate_and_readout_noise_model, simulate_delay_ablation,
)


@pytest.fixture(scope="module")
def backend():
    return AachenPatch()


def test_partial_window_retains_analytic_population_and_coherence_decay(backend):
    ticks = 12500  # 50 microseconds; remove only the middle 20 microseconds.
    circuits = []
    for initial in [np.array([[0, 0], [0, 1]]), np.ones((2, 2)) / 2]:
        qc = QuantumCircuit(1)
        qc.set_density_matrix(initial)
        qc.delay(ticks, 0, unit="dt")
        # Only this analytic fixture has manually set timestamps; experiment
        # timestamps come from Qiskit's ALAP scheduling analysis.
        qc._op_start_times = [0, 0]
        circuit, audit = delay_ablation_snapshot(SimpleNamespace(circuit=qc), backend, (2500, 7500))
        assert audit[0]["noise_disabled_dt"] == 5000
        assert circuit.data[1].operation.duration == ticks
        circuits.append(circuit)
    result = AerSimulator(method="density_matrix", noise_model=gate_and_readout_noise_model(backend)).run(circuits).result()
    one, plus = [np.asarray(result.data(i)["before_readout"]) for i in range(2)]
    prop = backend.target.qubit_properties[0]
    retained = 7500 * backend.dt
    assert np.isclose(one[1, 1].real, np.exp(-retained / prop.t1), atol=1e-12, rtol=0)
    assert np.isclose(2*plus[0, 1].real, np.exp(-retained / min(prop.t2, 2*prop.t1)), atol=1e-12, rtol=0)


def test_explicit_delay_channels_match_existing_full_and_no_delay_modes(backend):
    scheduled = schedule_preparation(build_preparation(CODES["four"], "B2_ED"), backend)
    end = round(scheduled.resources["duration_seconds"] / backend.dt)
    for mode, window in [("full", None), ("no_delay", (0, end))]:
        old, old_rho = simulate_scheduled(scheduled, backend, mode=mode, shots=64)
        new, new_rho = simulate_delay_ablation(scheduled, backend, disabled_window=window, shots=64)
        assert np.allclose(new_rho, old_rho, atol=1e-11, rtol=0)
        assert np.allclose(new["recorded_syndrome_probabilities"], old["recorded_syndrome_probabilities"], atol=1e-11, rtol=0)
    # Original duration and every Delay instruction remain available unchanged.
    assert scheduled.resources["duration_seconds"] > 0


def test_preparation_delays_are_not_removed_by_later_window(backend):
    scheduled = schedule_preparation(build_preparation(CODES["four"], "B2_ED"), backend)
    end = round(scheduled.resources["duration_seconds"] / backend.dt)
    readout = next(t for t, i in zip(scheduled.circuit.op_start_times, scheduled.circuit.data)
                   if i.operation.name == "measure")
    full, _ = simulate_delay_ablation(scheduled, backend, shots=64)
    ablated, _ = simulate_delay_ablation(scheduled, backend, (readout, end), shots=64)
    assert np.allclose(full["recorded_syndrome_probabilities"], ablated["recorded_syndrome_probabilities"], atol=1e-11, rtol=0)
    assert ablated["codespace_population"] > full["codespace_population"]
    for delay in ablated["delay_audit"]:
        if delay["start_dt"] + delay["duration_dt"] <= readout:
            assert delay["noise_disabled_dt"] == 0
