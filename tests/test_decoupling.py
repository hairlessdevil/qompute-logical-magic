import numpy as np
import pytest

from logical_magic import CODES, build_preparation
from logical_magic.aachen_noise import AachenPatch, schedule_preparation, simulate_scheduled
from logical_magic.decoupling import insert_dd, markovian_echo_sanity


@pytest.fixture(scope="module")
def backend():
    return AachenPatch()


@pytest.mark.parametrize("n", [2, 4, 8])
def test_dd_preserves_schedule_and_ideal_logical_output(backend, n):
    scheduled = schedule_preparation(build_preparation(CODES["four"], "B2_ED"), backend)
    dd = insert_dd(scheduled, backend, n)
    assert dd.resources["dd"]["added_pulse_count"] >= n*4
    assert dd.resources["duration_seconds"] == scheduled.resources["duration_seconds"]
    ideal, _ = simulate_scheduled(dd, backend, mode="ideal", shots=32)
    assert np.isclose(ideal["acceptance"], 1, atol=1e-12, rtol=0)
    assert np.isclose(ideal["fidelity"], 1, atol=1e-12, rtol=0)


def test_echo_distinguishes_memoryless_decay_from_constant_detuning(backend):
    result = markovian_echo_sanity(backend)
    values = [r["markovian_x_coherence"] for r in result["rows"]]
    assert max(values) - min(values) < 1e-12
    assert values[0] < 1
    assert result["rows"][-1]["constant_detuning_toy_x_coherence"] > .999999999
