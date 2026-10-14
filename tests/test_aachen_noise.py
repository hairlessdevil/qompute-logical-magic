import numpy as np
import pytest

from logical_magic import CODES, build_preparation
from logical_magic.aachen_noise import (
    AachenPatch, recorded_syndrome_state, schedule_preparation,
    simulate_scheduled, syndrome_blocks, validate_idle_relaxation,
)


@pytest.fixture(scope="module")
def backend():
    return AachenPatch()


def test_patch_preserves_calibration_and_connectivity(backend):
    for name in backend.operation_names:
        for wires, prop in backend.target[name].items():
            original = backend.snapshot.target[name][tuple(backend.physical_qubits[q] for q in wires)]
            if prop is None:
                assert original is None
            else:
                assert prop.duration == original.duration
                assert prop.error == original.error
    assert backend.coupling_map.is_connected()


def test_idle_noise_is_present_once_and_has_correct_time_units(backend):
    # Both populations and coherence must match exp(-t/T), detecting omitted,
    # duplicated, or incorrectly scaled delay relaxation.
    check = validate_idle_relaxation(backend)
    assert check["max_absolute_error"] < 1e-12
    assert max(check["observed_excited_population_and_x"]) < .9


def test_state_extraction_respects_routing_and_traces_spectators():
    rng = np.random.default_rng(52)
    psi = rng.normal(size=32) + 1j*rng.normal(size=32)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    blocks = syndrome_blocks(rho, [3, 0], [2, 4])
    # Independent explicit basis-index calculation, including spectator q1.
    expected = np.zeros((4, 4, 4), complex)
    for s in range(4):
        for i in range(4):
            for j in range(4):
                for e in range(2):
                    idx = lambda d: ((d & 1) << 3) + ((d >> 1) & 1) + (e << 1) + ((s & 1) << 2) + ((s >> 1) << 4)
                    expected[s, i, j] += rho[idx(i), idx(j)]
    assert np.allclose(blocks, expected)
    assert np.isclose(np.trace(blocks.sum(axis=0)), 1)


def test_acceptance_uses_recorded_syndrome_with_asymmetric_assignment():
    blocks = np.array([[[.6, 0], [0, 0]], [[0, 0], [0, .4]]])
    assignment = np.array([[.8, .2], [.3, .7]])
    rho, metrics = recorded_syndrome_state(blocks, [assignment])
    assert np.isclose(metrics["acceptance"], .6)
    assert np.allclose(rho, [[.8, 0], [0, .2]])
    assert np.allclose(metrics["recorded_syndrome_probabilities"], [.6, .4])
    assert np.isclose(metrics["accepted_fraction_from_nonzero_true_syndrome"], .2)


@pytest.mark.parametrize("name,strategy", [
    ("four", "B0"), ("four", "B1"), ("four", "B2_ED"),
    ("five", "B0"), ("five", "B1"), ("five", "B2_ED"), ("five", "B2_BK"),
])
def test_routed_scheduled_circuit_preserves_ideal_state_and_waits(backend, name, strategy):
    scheduled = schedule_preparation(build_preparation(CODES[name], strategy), backend)
    result, rho = simulate_scheduled(scheduled, backend, mode="ideal", shots=64)
    assert np.isclose(result["fidelity"], 1)
    assert np.isclose(result["acceptance"], 1/6 if strategy == "B2_BK" else 1)
    assert np.isclose(np.trace(rho), 1)
    if strategy.startswith("B2"):
        assert np.isclose(scheduled.resources["syndrome_readout_wait_seconds"], 2.584e-6, atol=1e-12)
        for q in scheduled.data_wires:
            assert scheduled.resources["idle_seconds_by_local_wire"][q] >= 2.584e-6


def test_full_noise_vs_idle_ablation_and_aer_recorded_counts(backend):
    scheduled = schedule_preparation(build_preparation(CODES["four"], "B2_ED"), backend)
    full, rho = simulate_scheduled(scheduled, backend, shots=65536, seed=83)
    without, _ = simulate_scheduled(scheduled, backend, mode="no_delay", shots=64)
    assert np.linalg.eigvalsh(rho).min() > -1e-10
    assert np.isclose(np.trace(rho), 1)
    assert full["fidelity"] < without["fidelity"] - .01
    m = scheduled.preparation.n_checks
    for i, probability in enumerate(full["recorded_syndrome_probabilities"]):
        observed = full["syndrome_counts"].get(format(i, f"0{m}b"), 0) / full["shots"]
        stderr = np.sqrt(probability * (1-probability) / full["shots"])
        assert abs(observed-probability) < 6*stderr + 2/full["shots"]
