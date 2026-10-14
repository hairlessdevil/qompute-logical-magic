"""Independent algebraic references and gate-level encoders.

All strings use Qiskit's |q[n-1] ... q[0]> ordering. Logical input is q[0].
Reference states are defined by stabilizers, not by executing the encoder.
"""

from dataclasses import dataclass
from functools import cached_property
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli, SparsePauliOp


def magic_angles(kind: str, antipodal: bool = False) -> tuple[float, float]:
    """Ry(theta) then Rz(phi); tetra is NOT the equatorial pi/8-gate state."""
    if kind == "H":
        theta, phi = np.pi / 4, 0.0
    elif kind == "tetra":
        theta, phi = np.arccos(1 / np.sqrt(3)), np.pi / 4
    else:
        raise ValueError(f"Unknown magic state {kind!r}")
    return (np.pi - theta, phi + np.pi) if antipodal else (theta, phi)


def magic_state(kind: str, antipodal: bool = False) -> np.ndarray:
    theta, phi = magic_angles(kind, antipodal)
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


@dataclass(frozen=True)
class Code:
    name: str
    n: int
    distance: int
    stabilizers: tuple[str, ...]
    logical_x: str
    logical_z: str
    magic: str

    @cached_property
    def projector(self) -> np.ndarray:
        ident = np.eye(2**self.n, dtype=complex)
        out = ident.copy()
        for label in self.stabilizers:
            out = out @ ((ident + Pauli(label).to_matrix()) / 2)
        return out

    @cached_property
    def basis(self) -> np.ndarray:
        """Columns |0_L>, |1_L>; relative phase fixed by X_L."""
        seed = np.eye(2**self.n, dtype=complex)[:, 0]
        zero = self.projector @ seed
        zero = (zero + Pauli(self.logical_z).to_matrix() @ zero) / 2
        zero /= np.linalg.norm(zero)
        one = Pauli(self.logical_x).to_matrix() @ zero
        return np.column_stack([zero, one])

    @cached_property
    def logical_y(self) -> np.ndarray:
        return 1j * Pauli(self.logical_x).to_matrix() @ Pauli(self.logical_z).to_matrix()

    @cached_property
    def target(self) -> np.ndarray:
        return self.basis @ magic_state(self.magic)

    @cached_property
    def fidelity_observable(self) -> SparsePauliOp:
        """Full physical target projector, including the codespace projector."""
        return SparsePauliOp.from_operator(np.outer(self.target, self.target.conj())).simplify()

    def encoder(self) -> QuantumCircuit:
        if self.name == "four":
            circuit = QuantumCircuit(4, name="encode_412")
            circuit.cx(0, 1)
            circuit.h(2)
            circuit.cx(2, 3)
            circuit.cx(2, 0)
            circuit.cx(2, 1)
            return circuit
        if self.name == "five":
            # Full symplectic frame: X0 -> XXXXX, Z0 -> ZZZZZ.
            # The other input Zs map to the four positive stabilizers.
            frame = {
                "stabilizer": [self.logical_z, *self.stabilizers],
                "destabilizer": [self.logical_x, "IIIYY", "IIYZX", "IZZXX", "IXXXX"],
            }
            circuit = Clifford.from_dict(frame).to_circuit()
            circuit.name = "encode_513"
            return circuit
        raise ValueError(f"No encoder for {self.name}")

    def syndrome(self, label: str) -> str:
        """Display s[m-1]...s[0]; 1 means the corresponding check is -1."""
        p = Pauli(label)
        return "".join(str(int(p.anticommutes(Pauli(s)))) for s in reversed(self.stabilizers))

    def computed_distance(self) -> int:
        for weight in range(1, self.n + 1):
            for chars in product("IXYZ", repeat=self.n):
                if sum(p != "I" for p in chars) != weight:
                    continue
                p = Pauli("".join(chars))
                if all(p.commutes(Pauli(s)) for s in self.stabilizers):
                    logical = self.basis.conj().T @ p.to_matrix() @ self.basis
                    if not np.allclose(logical, np.trace(logical) * np.eye(2) / 2):
                        return weight
        raise ValueError("No nontrivial logical operator found")


CODES = {
    "four": Code("four", 4, 2, ("XXXX", "IIZZ", "ZZII"), "IIXX", "IZIZ", "H"),
    "five": Code("five", 5, 3, ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"), "XXXXX", "ZZZZZ", "tetra"),
}
