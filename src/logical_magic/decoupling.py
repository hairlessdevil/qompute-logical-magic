"""Insert calibrated native DD pulses without resynthesizing the saved circuit."""

from collections import Counter
from dataclasses import replace

import numpy as np
from qiskit.circuit.library import XGate
from qiskit.quantum_info import DensityMatrix, Operator, Pauli
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit_aer.noise import thermal_relaxation_error


def instruction_ticks(circuit, inst, backend):
    if inst.operation.name == "delay":
        if inst.operation.unit != "dt":
            raise ValueError("Expected dt delays")
        return int(inst.operation.duration)
    if inst.operation.name == "barrier":
        return 0
    wires = tuple(circuit.find_bit(q).index for q in inst.qubits)
    return round(backend.target[inst.operation.name][wires].duration / backend.dt)


def timed_instructions(circuit, backend):
    return [{"name": inst.operation.name, "start_dt": int(t),
             "duration_dt": instruction_ticks(circuit, inst, backend),
             "wires": tuple(circuit.find_bit(q).index for q in inst.qubits),
             "clbits": tuple(circuit.find_bit(b).index for b in inst.clbits),
             "params": tuple(float(p) for p in inst.operation.params),
             "label": inst.operation.label}
            for t, inst in zip(circuit.op_start_times, circuit.data)]


def operation_signature(row):
    return (row["name"], row["start_dt"], row["duration_dt"], row["wires"],
            row["clbits"], row["params"], row["label"])


def insert_dd(scheduled, backend, pulses_per_idle=2):
    """Balanced even-X sequence in each sufficiently long, noninitial idle.

    Uses Qiskit's padding pass only. Original native instructions and their
    start times must be unchanged. Added X gates retain their ordinary names
    so the full Aachen noise model applies their calibrated gate channels.
    """
    if pulses_per_idle not in {2, 4, 8}:
        raise ValueError("Use 2, 4, or 8 X pulses per eligible idle")
    before = PassManager([ALAPScheduleAnalysis(target=backend.target)]).run(scheduled.circuit)
    after = PassManager([
        ALAPScheduleAnalysis(target=backend.target),
        PadDynamicalDecoupling(target=backend.target,
                              dd_sequence=[XGate() for _ in range(pulses_per_idle)],
                              skip_reset_qubits=True, extra_slack_distribution="middle"),
    ]).run(scheduled.circuit)
    old, new = timed_instructions(before, backend), timed_instructions(after, backend)
    old_ops = Counter(operation_signature(r) for r in old if r["name"] != "delay")
    new_ops = Counter(operation_signature(r) for r in new if r["name"] != "delay")
    if old_ops - new_ops:
        raise AssertionError("DD moved, removed or modified an original operation")
    added = new_ops - old_ops
    pulses = []
    for sig, count in added.items():
        name, start, duration, wires, clbits, params, label = sig
        assert name == "x" and len(wires) == 1 and not clbits and not params and label is None
        assert count == 1
        assert start % backend.target.pulse_alignment == 0
        assert any(r["name"] == "delay" and r["wires"] == wires
                   and r["start_dt"] <= start and start+duration <= r["start_dt"]+r["duration_dt"] for r in old)
        pulses.append({"local_wire": wires[0], "physical_qubit": backend.physical_qubits[wires[0]],
                       "start_dt": start, "duration_dt": duration})
    end = lambda rows: max(r["start_dt"]+r["duration_dt"] for r in rows)
    assert end(old) == end(new)
    assert np.isclose(np.exp(1j*(float(before.global_phase)-float(after.global_phase))), 1)
    idle = [sum(r["duration_dt"] for r in new if r["name"] == "delay" and r["wires"] == (q,))
            for q in range(after.num_qubits)]
    original_idle = sum(r["duration_dt"] for r in old if r["name"] == "delay")
    assert original_idle == sum(idle) + sum(p["duration_dt"] for p in pulses)
    windows = []
    for r in old:
        if r["name"] != "delay":
            continue
        found = sorted([p for p in pulses if p["local_wire"] == r["wires"][0]
                        and r["start_dt"] <= p["start_dt"] < r["start_dt"]+r["duration_dt"]],
                       key=lambda p: p["start_dt"])
        if found:
            assert len(found) == pulses_per_idle
            windows.append({"local_wire": r["wires"][0], "start_dt": r["start_dt"],
                            "duration_dt": r["duration_dt"], "pulse_starts_dt": [p["start_dt"] for p in found]})
    # The readout wait is the main long idle of interest and must receive DD.
    readout_start = next(r["start_dt"] for r in old if r["name"] == "measure")
    for q in scheduled.data_wires:
        found = [p for p in pulses if p["local_wire"] == q and p["start_dt"] >= readout_start]
        assert len(found) == pulses_per_idle
    resources = dict(scheduled.resources)
    resources.update({"duration_seconds": end(new)*backend.dt,
                      "delay_count": int(after.count_ops().get("delay", 0)),
                      "idle_seconds_by_local_wire": [t*backend.dt for t in idle],
                      "operation_counts": {k: int(v) for k, v in after.count_ops().items()},
                      "dd": {"sequence": "X"*pulses_per_idle, "pulses_per_eligible_idle": pulses_per_idle,
                             "skip_reset_qubits": True, "added_pulses": pulses,
                             "added_pulse_count": len(pulses), "protected_idle_windows": windows,
                             "protected_idle_count": len(windows), "original_gate_timestamps_preserved": True,
                             "original_total_duration_preserved": True}})
    return replace(scheduled, circuit=after, resources=resources)


def markovian_echo_sanity(backend, duration_seconds=2.584e-6, local_wire=0):
    """Ideal instantaneous pulses cannot undo this channel's transverse decay.

    A constant Z rotation is a separate toy control only. It is never added to
    the Aachen model and has no fitted device parameter.
    """
    prop = backend.target.qubit_properties[local_wire]
    t1, t2 = prop.t1, min(prop.t2, 2*prop.t1)
    plus = DensityMatrix(np.ones((2, 2))/2)
    x = Operator(Pauli("X"))
    rows = []
    for n in [0, 2, 4, 8]:
        fractions = [1.] if n == 0 else [1/(2*n)] + [1/n]*(n-1) + [1/(2*n)]
        rho = plus
        coherent = plus
        # Total unrefocused phase pi/2: a dimensionless illustration, not a
        # claim about Aachen's noise spectrum or detuning strength.
        for index, fraction in enumerate(fractions):
            channel = thermal_relaxation_error(t1, t2, fraction*duration_seconds).to_quantumchannel()
            rho = rho.evolve(channel)
            angle = np.pi/2*fraction
            coherent = coherent.evolve(Operator(np.diag([np.exp(-.5j*angle), np.exp(.5j*angle)])))
            if index < n:
                rho = rho.evolve(x)
                coherent = coherent.evolve(x)
        transverse = float(rho.expectation_value(Pauli("X")).real)
        expected = float(np.exp(-duration_seconds/t2))
        assert np.isclose(transverse, expected, atol=1e-12, rtol=0)
        static_x = float(coherent.expectation_value(Pauli("X")).real)
        assert np.isclose(static_x, 0 if n == 0 else 1, atol=1e-12, rtol=0)
        rows.append({"ideal_instantaneous_x_pulses": n, "markovian_x_coherence": transverse,
                     "constant_detuning_toy_x_coherence": static_x})
    return {"physical_qubit": backend.physical_qubits[local_wire], "duration_seconds": duration_seconds,
            "expected_markovian_x_coherence": float(np.exp(-duration_seconds/t2)), "rows": rows,
            "toy_control_not_in_aachen_model": True}
