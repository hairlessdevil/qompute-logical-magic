"""Retiming frozen native circuits with independent, early ancilla readout.

This models overlap allowed by per-qubit dependencies and the local Target.
It does not certify IBM controller/readout-group constraints or crosstalk.
"""

from dataclasses import replace

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, ASAPScheduleAnalysis

from .aachen_noise import simulate_scheduled
from .decoupling import instruction_ticks, timed_instructions


def retime_syndrome_readout(scheduled, backend, method="fixed"):
    """Keep native gates, physical mappings and per-wire order; retime only.

    fixed: original gate timestamps, read each ancilla at its last gate's end.
    asap: compact gates by their per-wire dependencies, then read early.
    alap: take asap readout times as deadlines and move gates as late as possible.
    """
    if method not in {"fixed", "asap", "alap"}:
        raise ValueError("Expected fixed, asap or alap")
    original = PassManager([ALAPScheduleAnalysis(target=backend.target)]).run(scheduled.circuit)
    gates, measurements = [], []
    for index, (start, inst) in enumerate(zip(original.op_start_times, original.data)):
        name = inst.operation.name
        if name in {"barrier", "delay"}:
            continue
        wires = tuple(original.find_bit(q).index for q in inst.qubits)
        entry = {"index": index, "start": int(start), "inst": inst,
                 "wires": wires, "duration": instruction_ticks(original, inst, backend)}
        (measurements if name == "measure" else gates).append(entry)
    assert len(measurements) == scheduled.preparation.n_checks
    assert {r["wires"][0] for r in measurements} == set(scheduled.syndrome_wires)
    pulse_alignment = backend.target.pulse_alignment or 1
    acquire_alignment = backend.target.acquire_alignment or 1
    ceil = lambda t, a: ((t+a-1)//a)*a
    ends = [0]*original.num_qubits
    for event in gates:
        start = event["start"] if method == "fixed" else ceil(max(ends[q] for q in event["wires"]), pulse_alignment)
        assert start % pulse_alignment == 0
        assert all(start >= ends[q] for q in event["wires"])
        event["start"] = start
        for q in event["wires"]:
            ends[q] = start+event["duration"]
    for event in measurements:
        event["start"] = ceil(ends[event["wires"][0]], acquire_alignment)
    delivery = max([*ends, *(r["start"]+r["duration"] for r in measurements)])
    if method == "alap":
        deadlines = [delivery]*original.num_qubits
        for event in measurements:
            deadlines[event["wires"][0]] = event["start"]
        for event in reversed(gates):
            start = min(deadlines[q] for q in event["wires"])-event["duration"]
            start = (start//pulse_alignment)*pulse_alignment
            assert start >= event["start"]  # ASAP witnesses feasibility.
            event["start"] = start
            for q in event["wires"]:
                deadlines[q] = start

    # Materialize actual durations, including data storage through delivery.
    qc = original.copy_empty_like()
    qc.global_phase = original.global_phase
    ends = [0]*qc.num_qubits
    for event in sorted([*gates, *measurements], key=lambda r: (r["start"], r["index"])):
        start = event["start"]
        for q in event["wires"]:
            assert start >= ends[q]
            if start > ends[q]:
                qc.delay(start-ends[q], q, unit="dt")
            ends[q] = start+event["duration"]
        qc.append(event["inst"])
    for q, end in enumerate(ends):
        if end < delivery:
            qc.delay(delivery-end, q, unit="dt")
    qc.barrier(label="delivery")
    qc = PassManager([ASAPScheduleAnalysis(target=backend.target)]).run(qc)
    qc.duration = delivery
    qc.unit = "dt"
    old_rows, new_rows = timed_instructions(original, backend), timed_instructions(qc, backend)
    signature = lambda r: (r["name"], r["wires"], r["clbits"], r["params"], r["label"])
    for q in range(qc.num_qubits):
        old_seq = [signature(r) for r in old_rows if q in r["wires"] and r["name"] not in {"delay", "barrier"}]
        new_seq = [signature(r) for r in new_rows if q in r["wires"] and r["name"] not in {"delay", "barrier"}]
        assert old_seq == new_seq, "Changed native operations or their per-wire order"
    if method == "fixed":
        for q in range(qc.num_qubits):
            old_times = [r["start_dt"] for r in old_rows if q in r["wires"] and r["name"] not in {"delay", "barrier", "measure"}]
            new_times = [r["start_dt"] for r in new_rows if q in r["wires"] and r["name"] not in {"delay", "barrier", "measure"}]
            assert old_times == new_times
    assert max(r["start_dt"]+r["duration_dt"] for r in new_rows) == delivery
    assert delivery*backend.dt <= scheduled.resources["duration_seconds"]+1e-13
    idle = [sum(r["duration_dt"] for r in new_rows if r["name"] == "delay" and r["wires"] == (q,))
            for q in range(qc.num_qubits)]
    readouts = [r for r in new_rows if r["name"] == "measure"]
    coherent_end = max(r["start_dt"]+r["duration_dt"] for r in new_rows
                       if r["name"] not in {"delay", "barrier", "measure"})
    resources = dict(scheduled.resources)
    resources.update({"duration_seconds": delivery*backend.dt,
                      "idle_seconds_by_local_wire": [t*backend.dt for t in idle],
                      "delay_count": qc.count_ops().get("delay", 0),
                      "operation_counts": dict(qc.count_ops()),
                      "scheduling_method": "early_readout_"+method,
                      "syndrome_readout_wait_seconds": None,
                      "retiming": {"method": method, "native_gates_and_per_wire_order_preserved": True,
                                   "original_gate_timestamps_preserved": method == "fixed",
                                   "delivery_us": delivery*backend.dt*1e6,
                                   "last_gate_end_us": coherent_end*backend.dt*1e6,
                                   "final_gate_to_delivery_us": (delivery-coherent_end)*backend.dt*1e6,
                                   "readouts": readouts, "instructions": new_rows}})
    return replace(scheduled, circuit=qc, resources=resources)


def discard_measured_ancilla_idles(circuit):
    """Discard evolution of already measured, never reused wires.

    Required before deferring measurement to calculate exact syndrome blocks:
    later T1 on an unused ancilla cannot change the *stored classical* outcome.
    Data and all pre-readout idles are retained. Any gate reuse raises an error.
    """
    qc = circuit.copy_empty_like()
    measured = set()
    for inst in circuit.data:
        wires = {circuit.find_bit(q).index for q in inst.qubits}
        if inst.operation.name == "barrier":
            continue
        if inst.operation.name == "delay" and wires <= measured:
            continue
        if wires & measured:
            raise ValueError("Measurement deferral requires no later operations on a measured ancilla")
        qc.append(inst)
        if inst.operation.name == "measure":
            measured.update(wires)
    return qc


def simulate_early_readout(scheduled, backend, mode="full", shots=8192, seed=20261014):
    """Exact local-channel simulation at the schedule's actual delivery time."""
    effective = replace(scheduled, circuit=discard_measured_ancilla_idles(scheduled.circuit))
    return simulate_scheduled(effective, backend, mode=mode, shots=shots, seed=seed)
