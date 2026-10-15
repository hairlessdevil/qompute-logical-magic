"""Explicit native XY4 inside the frozen B2_ED preparation's long idle windows."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit_ibm_runtime.models import BackendProperties
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

from .aachen_noise import ScheduledPreparation
from .circuits import build_preparation
from .codes import CODES
from .decoupling import operation_signature, timed_instructions
from .ibm_b2ed import LivePatch, embed_physical, file_hash, ideal_validation, load_bundle, utc_now, write_json


def insert_xy4(scheduled, backend, minimum_idle_seconds=500e-9):
    """Replace eligible delays by gap-X-gap-Y-gap-X-gap-Y-gap, without routing.

    Y = Rz(pi/2) X Rz(-pi/2) in matrix order, implemented with two virtual
    Z rotations and one native X pulse. XYXY is -I; compensate its global phase.
    Free-delay fractions are [1/8, 1/4, 1/4, 1/4, 1/8], rounded to alignment.
    """
    original = PassManager(ALAPScheduleAnalysis(target=backend.target)).run(scheduled.circuit)
    old = timed_instructions(original, backend)
    modified = original.copy_empty_like()
    alignment = int(backend.target.pulse_alignment)
    active = set()
    windows = []
    fractions = np.array([1, 2, 2, 2, 1])/8
    for inst, row in zip(original.data, old):
        q = row["wires"][0] if len(row["wires"]) == 1 else None
        eligible = row["name"] == "delay" and q in active and row["duration_dt"]*backend.dt >= minimum_idle_seconds-1e-15
        if eligible:
            pulse = round(backend.target["x"][(q,)].duration/backend.dt)
            if backend.target["rz"][(q,)].duration != 0:
                raise ValueError("Native Y decomposition requires zero-duration Rz")
            free = row["duration_dt"]-4*pulse
            if free < 8*alignment:
                eligible = False
            elif pulse % alignment or row["start_dt"] % alignment or free % alignment:
                raise ValueError("XY4 window cannot preserve native pulse alignment")
        if eligible:
            gaps = (np.floor(fractions*free/alignment).astype(int)*alignment).tolist()
            gaps[2] += free-sum(gaps)
            starts, elapsed = [], row["start_dt"]
            for i, axis in enumerate("XYXY"):
                modified.delay(gaps[i], q, unit="dt")
                elapsed += gaps[i]
                starts.append(elapsed)
                if axis == "Y":
                    modified.rz(-np.pi/2, q)
                modified.x(q)
                if axis == "Y":
                    modified.rz(np.pi/2, q)
                elapsed += pulse
            modified.delay(gaps[-1], q, unit="dt")
            modified.global_phase += np.pi
            windows.append({"local_wire": q, "physical_qubit": backend.physical_qubits[q],
                            "start_dt": row["start_dt"], "duration_dt": row["duration_dt"],
                            "pulse_starts_dt": starts, "pulse_duration_dt": pulse, "gap_lengths_dt": gaps})
        else:
            modified.append(inst)
        if row["name"] in {"measure", "reset"}:
            active.difference_update(row["wires"])
        elif row["name"] not in {"barrier", "delay"}:
            active.update(row["wires"])
    after = PassManager(ALAPScheduleAnalysis(target=backend.target)).run(modified)
    new = timed_instructions(after, backend)
    original_ops = Counter(operation_signature(r) for r in old if r["name"] != "delay")
    new_ops = Counter(operation_signature(r) for r in new if r["name"] != "delay")
    if original_ops-new_ops:
        raise AssertionError("XY4 moved or modified an original gate/measurement")
    added = new_ops-original_ops
    if any(sig[0] not in {"x", "rz"} for sig in added):
        raise AssertionError("Unexpected added instruction")
    if sum(n for sig, n in added.items() if sig[0] == "x") != 4*len(windows):
        raise AssertionError("DD pulse count mismatch")
    end = lambda rows: max(r["start_dt"]+r["duration_dt"] for r in rows)
    if end(old) != end(new):
        raise AssertionError("XY4 changed the total duration")
    readout_start = next(r["start_dt"] for r in old if r["name"] == "measure")
    for q in scheduled.data_wires:
        if not any(w["local_wire"] == q and w["start_dt"] <= readout_start
                   and w["start_dt"]+w["duration_dt"] >= end(old) for w in windows):
            raise AssertionError("A data wire's syndrome-readout wait did not receive XY4")
    resources = deepcopy(scheduled.resources)
    resources.update({"operation_counts": dict(after.count_ops()),
                      "delay_count": int(after.count_ops().get("delay", 0)),
                      "idle_seconds_by_local_wire": [sum(r["duration_dt"] for r in new
                           if r["name"] == "delay" and r["wires"] == (q,))*backend.dt for q in range(after.num_qubits)],
                      "dd": {"sequence": "XY4", "axes_in_time_order": "X Y X Y",
                             "implementation": "Explicit native X pulses with virtual Rz frames for Y; Runtime DD disabled",
                             "minimum_idle_seconds": minimum_idle_seconds,
                             "free_delay_fractions": fractions.tolist(), "skip_initial_and_post_measurement_idles": True,
                             "protected_idle_count": len(windows), "added_native_x_pulses": 4*len(windows),
                             "added_virtual_rz_gates": 4*len(windows), "windows": windows,
                             "original_gate_timestamps_preserved": True, "original_total_duration_preserved": True}})
    return replace(scheduled, circuit=after, resources=resources)


def validate_saved_durations(circuits, original_calibration, backend, dt):
    """Do not silently alter the baseline schedule after a timing-calibration change."""
    if not np.isclose(backend.dt, dt, rtol=0, atol=1e-15):
        raise ValueError("Device dt changed since the original run")
    props = BackendProperties.from_dict(original_calibration)
    checked = set()
    for circuit in circuits:
        for inst in circuit.data:
            name = inst.operation.name
            wires = tuple(circuit.find_bit(q).index for q in inst.qubits)
            if name in {"delay", "barrier"} or (name, wires) in checked:
                continue
            checked.add((name, wires))
            old = 0. if name == "rz" else props.readout_length(wires[0]) if name == "measure" else props.gate_length(name, wires)
            current = backend.target[name][wires].duration
            if current is None or not np.isclose(current, old, atol=1e-15, rtol=0):
                raise ValueError(f"Duration changed for {name}{wires}; explicit rescheduling decision needed")


def prepare_xy4_bundle(source, output, backend, *, live=True, minimum_idle_seconds=500e-9):
    """Prepare a new reviewable bundle from an unchanged prior hardware bundle."""
    source, original, source_physical = load_bundle(source)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new directory: {output}")
    if "dd" in original["resources"]:
        raise ValueError("Expected the original no-DD bundle")
    saved_calibration = json.loads((source/"calibration.json").read_text())
    validate_saved_durations(source_physical, saved_calibration, backend, 4e-9)
    patch = LivePatch(backend)
    with (source/"local_circuits.qpy").open("rb") as handle:
        source_local = qpy.load(handle)
    prep = build_preparation(CODES["four"], "B2_ED")
    resources = original["resources"]
    scheduled = ScheduledPreparation(prep, source_local[0], tuple(resources["final_data_local_wires"]),
                                     tuple(resources["syndrome_local_wires_in_bit_order"]), resources)
    dd = insert_xy4(scheduled, patch, minimum_idle_seconds)
    local = [dd.circuit]
    prefix_length = len(source_local[0].data)
    for baseline in source_local[1:]:
        if list(baseline.data[:prefix_length]) != list(source_local[0].data):
            raise ValueError("Source measurement circuits do not share their frozen preparation")
        circuit = baseline.copy_empty_like()
        circuit.global_phase += float(dd.circuit.global_phase)-float(source_local[0].global_phase)
        for inst in dd.circuit.data:
            circuit.append(inst)
        for inst in baseline.data[prefix_length:]:
            circuit.append(inst)
        circuit.metadata = {**(baseline.metadata or {}), "preparation_dd": "XY4"}
        local.append(circuit)
    settings = deepcopy(original["settings"])
    validation = ideal_validation(local, dd, settings)
    physical = [embed_physical(circuit, patch, backend) for circuit in local]
    validate_isa_circuits(physical, backend.target)
    for circuit, setting in zip(physical, settings):
        if live:
            backend.check_faulty(circuit)
        duration = float(circuit.estimate_duration(backend.target))
        if not np.isclose(duration, setting["estimated_circuit_seconds"], rtol=0, atol=1e-15):
            raise AssertionError("A full measurement circuit changed duration")
    manifest = deepcopy(original)
    manifest.update({"created_at_utc": utc_now(), "calibration_date": backend.properties().last_update_date.isoformat(),
                     "calibration_source": "live" if live else "offline_fake_backend_DO_NOT_SUBMIT",
                     "backend_status_at_preparation": backend.status().to_dict(),
                     "resources": dd.resources, "ideal_validation": validation, "settings": settings,
                     "dt_seconds": backend.dt,
                     "timing": "Original frozen schedule; explicit XY4 in noninitial idles >= 500 ns; analysis suffix unchanged",
                     "source_no_dd_bundle": str(source), "source_manifest_sha256": file_hash(source/"manifest.json"),
                     "source_circuits_sha256": file_hash(source/"circuits.qpy"),
                     "comparison_limit": "Separate jobs; changes between runs can include calibration drift"})
    manifest["sampler_options"]["environment"]["job_tags"] = [
        "b2ed-four-xy4-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")]
    manifest["sampler_options"]["dynamical_decoupling"] = {"enable": False}
    output.mkdir(parents=True)
    for name, circuits in (("circuits.qpy", physical), ("local_circuits.qpy", local)):
        with (output/name).open("wb") as handle:
            qpy.dump(circuits, handle)
    write_json(output/"calibration.json", backend.properties().to_dict())
    (output/"preparation.txt").write_text(str(dd.circuit.draw(output="text", fold=140)))
    manifest["files"] = {name: file_hash(output/name) for name in original["files"]}
    write_json(output/"manifest.json", manifest)
    return manifest
