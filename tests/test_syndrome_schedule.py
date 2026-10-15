import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit_aer import AerSimulator

from logical_magic import CODES, build_preparation
from logical_magic.aachen_noise import AachenPatch, ScheduledPreparation, syndrome_blocks
from logical_magic.syndrome_schedule import (
    discard_measured_ancilla_idles, retime_syndrome_readout, simulate_early_readout,
)


@pytest.fixture(scope="module")
def backend():
    return AachenPatch()


@pytest.mark.parametrize("method", ["fixed", "asap", "alap"])
def test_native_schedule_preserves_ideal_preparation_and_never_lengthens_delivery(backend, method):
    folder = Path(__file__).resolve().parents[1]/"results/noisy_aachen"
    row = next(r for r in json.loads((folder/"noisy_results.json").read_text())["rows"]
               if r["code"] == "four" and r["strategy"] == "B2_ED")
    res = row["resources"]
    with (folder/"circuits/four_B2_ED.qpy").open("rb") as stream:
        circuit = qpy.load(stream)[0]
    scheduled = ScheduledPreparation(build_preparation(CODES["four"], "B2_ED"), circuit,
                                     tuple(res["final_data_local_wires"]),
                                     tuple(res["syndrome_local_wires_in_bit_order"]), res)
    changed = retime_syndrome_readout(scheduled, backend, method)
    metrics, _ = simulate_early_readout(changed, backend, mode="ideal", shots=1)
    assert np.isclose(metrics["fidelity"], 1, atol=1e-12, rtol=0)
    assert np.isclose(metrics["acceptance"], 1, atol=1e-12, rtol=0)
    assert changed.resources["duration_seconds"] <= res["duration_seconds"]+1e-13
    assert changed.resources["two_qubit_gates"] == res["two_qubit_gates"]


def test_ancilla_decay_after_readout_does_not_rewrite_recorded_syndrome(backend):
    # Bell branches: true syndrome 0 => data |0>; syndrome 1 => decaying |1>.
    # If the already measured ancilla's long idle were mistakenly included in
    # deferred evolution, it would feed the syndrome-1 branch into syndrome 0.
    bell = np.array([1, 0, 0, 1])/np.sqrt(2)
    qc = QuantumCircuit(2, 1)
    qc.set_density_matrix(np.outer(bell, bell))
    qc.measure(1, 0)
    qc.delay(12500, 0, unit="dt")  # 50 us data relaxation must remain.
    qc.delay(250000, 1, unit="dt")  # 1 ms after ancilla readout is irrelevant.
    clean = discard_measured_ancilla_idles(qc)
    deferred = clean.copy_empty_like()
    for inst in clean.data:
        if inst.operation.name != "measure":
            deferred.append(inst)
    deferred.save_density_matrix()
    sim = AerSimulator(method="density_matrix", max_parallel_threads=1,
                       noise_model=backend.noise_model(readout_error=False))
    rho = np.asarray(sim.run(deferred, shots=1).result().data(0)["density_matrix"])
    blocks = syndrome_blocks(rho, (0,), (1,))
    survival = np.exp(-50e-6/backend.target.qubit_properties[0].t1)
    assert np.allclose(blocks[0], np.diag([.5, 0]), atol=1e-12, rtol=0)
    assert np.allclose(blocks[1], np.diag([.5*(1-survival), .5*survival]), atol=1e-12, rtol=0)


def test_ancilla_reuse_is_rejected():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.x(0)
    with pytest.raises(ValueError, match="no later operations"):
        discard_measured_ancilla_idles(qc)
