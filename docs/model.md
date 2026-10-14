# Mathematical model and validation boundary

## Conventions

Qubits are displayed as |q[n−1] … q[0]⟩. Pauli strings follow the same ordering.
The input logical qubit is q0; the remaining data and syndrome qubits start in |0⟩.
The +1 result of a stabilizer measurement is recorded as 0. Acceptance means all
recorded syndrome bits are 0. Fidelity is the **squared** overlap with the pure
encoded target, F = ⟨ψL|ρaccepted|ψL⟩.

## Four-qubit model

The codewords are

\[
|0_L\rangle=(|0000\rangle+|1111\rangle)/\sqrt2,\qquad
|1_L\rangle=(|0011\rangle+|1100\rangle)/\sqrt2.
\]

Choose generators `XXXX`, `IIZZ`, `ZZII`, logical X = `IIXX`, and logical
Z = `IZIZ`. Logical Y is defined as i X_L Z_L, including its sign. The target is

\[
|H_L\rangle=\cos(\pi/8)|0_L\rangle+\sin(\pi/8)|1_L\rangle.
\]

The encoder applies CX(0,1), H(2), CX(2,3), CX(2,0), CX(2,1). Four CX gates
encode any a|0⟩+b|1⟩ on q0 into a|0L⟩+b|1L⟩. Input preparation is Ry(π/4).

The codewords are from [Leung et al., *Approximate quantum error correction can
lead to better codes*, Eqs. 9–10](https://arxiv.org/html/quant-ph/9704002v1).
The stabilizer/logical conventions and compact encoder here are verified
algebraically and numerically. The paper's amplitude-damping recovery is **not**
implemented in this stage; its performance cannot be attributed to simple
stabilizer postselection.

## Five-qubit model

Choose generators `XZZXI`, `IXZZX`, `XIXZZ`, `ZXIXZ`; logical X = `XXXXX`,
logical Z = `ZZZZZ`. Define the code projector

\[
P=\prod_j (I+S_j)/2.
\]

The independent logical reference is |0L⟩ ∝ P(I+ZL)|00000⟩ and |1L⟩ = XL|0L⟩.
The ordinary encoder is synthesized from a full Clifford tableau with
Z0 → ZL, X0 → XL, and Z1,…,Z4 → the four generators. Its remaining destabilizers
are `IIIYY`, `IIYZX`, `IZZXX`, `IXXXX`. Tests validate the complete action on
arbitrary complex inputs, not only |0L⟩. The synthesis is not gate-optimal.

The positive tetrahedral target has Bloch vector (1,1,1)/√3:

\[
|T_+\rangle=\cos(\theta/2)|0\rangle+
e^{i\pi/4}\sin(\theta/2)|1\rangle,\quad
\theta=\arccos(1/\sqrt3).
\]

Use Ry(θ), Rz(π/4), ignoring the resulting global phase. Its antipode T− uses
angles π−θ and 5π/4. The antipodal states are orthogonal.

For B2_BK, prepare |T−⟩ on each of five data qubits and measure all four
stabilizers without decoding. The accepted state is |T+⟩L with probability 1/6.
The sign matters: five positive inputs instead produce the negative logical
pole. This follows from [Bravyi–Kitaev, Sec. V, Eqs. 15–22](https://arxiv.org/html/quant-ph/0403025v2)
and is independently checked by direct matrix projection. A single input replaced
by its orthogonal tetrahedral state is rejected when the checks are perfect.
This ideal input-error property does not protect against arbitrary faults in
the check circuit. The original [five-qubit code](https://arxiv.org/abs/quant-ph/9602019)
also corrects arbitrary single-qubit storage errors. Notebook Section 6 now
tests this with a separate syndrome-controlled recovery experiment, described below.

## Syndrome extraction and delivered state

For each positive Pauli generator S, prepare a fresh ancilla in |+⟩, apply
controlled-S as a product of controlled single-qubit Pauli gates, then apply H
and measure the ancilla in Z. On any data vector v, the output branch s is

\[
v_s=\prod_j\frac{I+(-1)^{s_j}S_j}{2}v.
\]

Checks use distinct ancillas. Readout can be deferred to the end because the
ancilla is not reused or acted on again. This equivalence is an ideal-circuit
statement; measurement timing must be included when adding realistic noise.

The exact simulator evolves the **whole gate circuit**, slices its ancilla-zero
branch, and normalizes by that branch's probability. It does not perform a
second codespace projection. In the fault audit, data can leave the codespace
after a check and still be accepted. Such leakage remains in the delivered
state and lowers its reported physical target fidelity.

## Fidelity evaluation

Within the codespace, logical Bloch-vector overlap determines fidelity. For a
state that may leak outside the codespace, the appropriate observable is

\[
|\psi_L\rangle\langle\psi_L|=
\frac12P(I+r_xX_L+r_yY_L+r_zZ_L).
\]

Using only (1+r·⟨σL⟩)/2 without P is generally incorrect when leakage is present.
We expand the full target projector into physical Pauli terms, measure each
nonidentity term on fresh copies of the same preparation, and condition only on
the same syndrome-acceptance rule. This uses 23 nonidentity terms for H_L and 63
for the tetrahedral logical target, versus 4^n−1 terms for full state tomography.
It evaluates actual accepted-state fidelity; including P in the observable does
not physically project the output or improve its state.

Terms are sampled independently using local Aer statevector simulation. The
linear estimator is the identity coefficient plus the coefficient-weighted
sample means. Its plug-in variance is Σ c_j²(1−mean_j²)/Naccepted,j. We also record
a conservative Hoeffding/union-bound 95% interval; this avoids interpreting zero
empirical variance as zero uncertainty. Acceptance has a separate count sample
and Wilson 95% interval. These intervals concern shot statistics only; no SPAM
or hardware systematic error is included. Samples are not used to optimize a
protocol. No significance claim is made from ideal ties.

## Fault audit

Two distinct questions are tested:

1. Storage faults: insert each of the 3n single-qubit Paulis after encoding and
   before ideal extraction. Every one is detected for both codes.
2. Circuit faults: insert a Pauli after each gate, enumerating X/Y/Z after a 1q
   gate and all 15 nonidentity two-qubit Paulis after a 2q gate. Also enumerate
   initial-qubit Pauli faults and individual syndrome-bit flips.

In the CSV, `qubits` lists the gate operands in Qiskit order, and the **rightmost**
character of the local `pauli` label acts on `qubits[0]`. `location` is the
zero-based index in the unmeasured unitary circuit; −1 is initialization.
`accepted_harmful` means positive acceptance and conditional fidelity below 1,
and may include partial acceptance. No weighting is assigned to these cases.

The audit is exhaustive only for this stated discrete fault set. It excludes
idle faults, leakage outside the qubit Hilbert space, coherent rotation errors,
relaxation, correlated events across gates, and hardware-transpiled fault
locations. It is not a formal fault-tolerance proof. The existence of accepted
harmful single faults is sufficient to rule out a claim that this circuit
detects every such fault.

A concrete counterexample is Y just after Ry(π/4) in the four-qubit input
preparation. Encoding turns it into logical Y. All checks pass, but the output
is orthogonal to H_L. Unflagged ancilla faults can also propagate to multiple
data qubits. Additional state verification or a different protected protocol
would be needed before calling this fully protected preparation.

## Single-Pauli detection and active correction

Notebook Section 6 and `error_correction.py` inject X, Y, and Z on each data
qubit after ordinary encoding. Ancillas measure all stabilizers, and a fixed
lookup table selects a conditional Pauli recovery using `if_test`. No branch is
discarded, and the decoder never receives the injected Pauli or its location.
The experiment includes an identity/no-error control for each code.

All 15 five-qubit errors have distinct nonzero syndromes and are corrected with
fidelity 1. The four-qubit code detects all 12 errors, but they occupy only five
nonzero syndromes. For example, X0 and X1 share syndrome 010 and differ by
logical X. Z0 and Z1 instead differ by a stabilizer and are equivalent, while
Z0 and Z2 differ by logical Z. The calculation checks residual action on the
entire codespace to distinguish harmless degeneracy from logical ambiguity.

The illustrative four-qubit decoder chooses the first representative in qubit
order, then X/Y/Z. It corrects 6 of the 12 listed errors. The other six return
to the codespace but have H-state fidelity 1/2. This unweighted enumeration is
not a physical success rate or an optimal recovery claim. The five-qubit
decoder preserves any logical state, checked through its residual logical
operator and arbitrary complex input states.

Aer runs the actual measurement/conditional-gate circuits and saves the reduced
data density matrix averaged over all shots. There is no postselection. The
reported recovered-state fidelity is a simulator state diagnostic, not a
finite-shot tomography estimate. In this ideal experiment, each injected
Pauli produces a deterministic syndrome; 256-shot syndrome counts are also
saved. Preparation, syndrome extraction, and recovery are ideal throughout.

## Representative circuit locations

Section 7 complements the restricted storage test with a small fixed set of
faults during magic input preparation, encoding, and syndrome extraction,
plus a late data error and a wrong recorded syndrome bit. These are individual
examples, not another exhaustive sweep or a calibrated noise model.

The selected five-qubit encoder and ancilla faults each produce a codespace
state with target fidelity 1/3 after the fixed syndrome decoder's correction.
A further ideal correction round does not repair these examples. The late data
fault and false-syndrome correction instead leave residual errors repaired by
that ideal round. The latter diagnostic is explicitly separate from delivered
fidelity: residual correctable errors must not be conflated with logical damage
when discussing fault tolerance.

Each selected circuit is run with Aer measurement and conditional gates and
compared with an exact branch sum. One separately labeled recovery-gate example
combines an input storage error with a fault after the executed recovery gate:
two error events, not a single fault on an otherwise ideal preparation.
The ideal zero-syndrome path does not execute a nonidentity recovery gate.

For background on why faults in operations and syndrome measurements require
more than a code-distance check, see [IBM's error-propagation lesson](https://quantum.cloud.ibm.com/learning/en/courses/foundations-of-quantum-error-correction/fault-tolerant-quantum-computing/controlling-error-propagation).

## Implementation sources

The circuit API and simulator behavior were checked against the installed
packages and [Qiskit Clifford documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Clifford)
and [AerSimulator documentation](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html).
Tests compare Aer statevectors with Qiskit's independent quantum-information
evolution and compare extraction circuits with algebraic projectors.

No source supports a claim that these ideal results improve noisy preparation;
that remains the next experiment.
