"""Portable preparation circuits with fresh ancillas and terminal measurements."""

from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .codes import Code, magic_angles, magic_state


@dataclass
class Preparation:
    code: Code
    strategy: str
    unitary: QuantumCircuit
    stages: tuple[str, ...]
    checks: tuple[str, ...]
    n_data: int

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    @property
    def target(self) -> np.ndarray:
        return magic_state(self.code.magic) if self.strategy == "B0" else self.code.target

    @property
    def fidelity_observable(self) -> SparsePauliOp:
        if self.strategy != "B0":
            return self.code.fidelity_observable
        psi = self.target
        return SparsePauliOp.from_operator(np.outer(psi, psi.conj())).simplify()

    def measured(self) -> QuantumCircuit:
        """Only ancillas are measured: the accepted data block remains reusable."""
        qc = self.unitary.copy()
        if self.n_checks:
            syn = ClassicalRegister(self.n_checks, "syndrome")
            qc.add_register(syn)
            qc.measure(range(self.n_data, qc.num_qubits), syn)
        return qc

    def measure_pauli(self, label: str) -> QuantumCircuit:
        """Destructive evaluation on separate copies, with the same acceptance rule."""
        if len(label) != self.n_data or any(p not in "IXYZ" for p in label):
            raise ValueError("Expected an unsigned data-register Pauli string")
        qc = self.unitary.copy()
        if self.n_checks:
            qc.add_register(ClassicalRegister(self.n_checks, "syndrome"))
        obs = ClassicalRegister(self.n_data, "data_readout")
        qc.add_register(obs)
        for q, p in enumerate(reversed(label)):
            if p == "X":
                qc.h(q)
            elif p == "Y":
                qc.sdg(q)
                qc.h(q)
        # All readout occurs at the end. Ancilla and data measurements commute.
        if self.n_checks:
            qc.measure(range(self.n_data, qc.num_qubits), range(self.n_checks))
        qc.measure(range(self.n_data), obs)
        return qc

    def resources(self) -> dict:
        qc = self.measured()
        two = sum(len(inst.qubits) == 2 for inst in self.unitary.data)
        return {
            "data_qubits": self.n_data,
            "ancilla_qubits": self.n_checks,
            "total_qubits": qc.num_qubits,
            "two_qubit_gates": two,
            "depth": qc.depth(),
            "operation_counts": dict(qc.count_ops()),
            "duration_seconds": None,
            "connectivity": "all-to-all, before hardware transpilation",
            "magic_inputs": self.code.n if self.strategy == "B2_BK" else 1,
        }


def append_magic(qc: QuantumCircuit, qubit: int, kind: str, antipodal: bool = False) -> None:
    theta, phi = magic_angles(kind, antipodal)
    qc.ry(theta, qubit)
    if abs(phi) > 1e-14:
        qc.rz(phi, qubit)


def check_block(n_data: int, label: str) -> QuantumCircuit:
    """Hadamard-test measurement of positive Pauli S using one fresh ancilla.

    Each nonidentity factor uses one entangling gate. These are unflagged
    checks: an ancilla fault can spread to multiple data qubits.
    """
    if len(label) != n_data or any(p not in "IXYZ" for p in label):
        raise ValueError("Expected a positive, unsigned Pauli of length n_data")
    qc = QuantumCircuit(n_data + 1)
    anc = n_data
    qc.h(anc)
    for q, p in enumerate(reversed(label)):
        if p == "X":
            qc.cx(anc, q)
        elif p == "Y":
            qc.cy(anc, q)
        elif p == "Z":
            qc.cz(anc, q)
    qc.h(anc)
    return qc


def build_preparation(code: Code, strategy: str = "B2_ED", checks=None) -> Preparation:
    """B0 bare; B1 encoded; B2_ED syndrome screened; B2_BK five-copy projection.

    B2_ED is deliberately not called fault-tolerant protected preparation.
    B2_BK is the encoded-output Bravyi--Kitaev projection with ideal checks.
    """
    if strategy not in {"B0", "B1", "B2_ED", "B2_BK"}:
        raise ValueError(f"Unknown strategy {strategy!r}")
    if strategy == "B2_BK" and code.name != "five":
        raise ValueError("The BK five-copy primitive requires the five-qubit code")
    if checks is not None and strategy in {"B0", "B1"}:
        raise ValueError("Checks are only supported by screened strategies")
    selected = tuple(code.stabilizers if checks is None else checks) if strategy.startswith("B2") else ()
    if len(set(selected)) != len(selected) or any(s not in code.stabilizers for s in selected):
        raise ValueError("Checks must be a nonrepeating subset/order of the code generators")
    if strategy == "B2_BK" and set(selected) != set(code.stabilizers):
        raise ValueError("The BK primitive requires all four stabilizers")
    n = 1 if strategy == "B0" else code.n
    qc = QuantumCircuit(n + len(selected), name=f"{code.name}_{strategy}")
    stages = []

    def record(stage: str, before: int) -> None:
        stages.extend([stage] * (len(qc.data) - before))

    if strategy == "B2_BK":
        # Projection reverses the tetrahedral pole. Start with five |T_->
        # inputs so the encoded output is the SAME |T_+>_L as B1/B2_ED.
        for q in range(n):
            before = len(qc.data)
            append_magic(qc, q, "tetra", antipodal=True)
            record("magic_input", before)
    else:
        append_magic(qc, 0, code.magic)
        record("magic_input", 0)
        if strategy != "B0":
            before = len(qc.data)
            qc.compose(code.encoder(), range(n), inplace=True)
            record("encoding", before)
    for i, label in enumerate(selected):
        before = len(qc.data)
        qc.compose(check_block(n, label), [*range(n), n + i], inplace=True)
        record(f"check_{i}_{label}", before)
    return Preparation(code, strategy, qc, tuple(stages), selected, n)
