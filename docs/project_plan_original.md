# Project Plan: Noise-Adaptive Logical Magic-State Preparation on IBM Quantum Hardware

**Project:** Qompute 2026 — Track 3: Can We Trust a Noisy Quantum Computer?

**Status:** Research and implementation plan — protocols and experimental benefits not yet validated.

## 1. Project objective

We aim to prepare a high-fidelity encoded magic state on an IBM quantum computer and investigate whether hardware-aware error detection and suppression can improve its quality compared with conventional preparation methods.

Our central hypothesis is that **the most comprehensive error-detection protocol is not necessarily the most effective under realistic hardware noise**. Additional verification gates and measurements can themselves introduce errors. By tailoring the verification procedure to the noise characteristics and hardware constraints, we may achieve a better tradeoff between logical fidelity, preparation success probability, and resource consumption.

The intended contribution is a hardware-aware verification and preparation strategy, rather than a new quantum error-correcting code.

The primary objective is to improve the fidelity of a physically prepared, reusable logical magic state. Improvements arising only from classical error mitigation will be evaluated separately.

## 2. Phase I — Select the quantum code and target magic state

We will initially investigate two candidates.

| Candidate | Code | Target state | Motivation |
|---|---|---|---|
| A | $[[4,1,2]]$ | $H$-type magic state | Minimal size; amplitude-damping-adapted code |
| B | $[[5,1,3]]$ | Tetrahedral magic state | Stronger error protection; useful transversal Clifford symmetry |

For each candidate, we will:

1. Establish the logical-state definition, stabilizers, and relevant logical operators.
2. Construct an ordinary encoding circuit as an unprotected baseline.
3. Identify an existing or implementable error-detected preparation protocol, including any necessary logical-state verification.
4. Determine the approximate physical resources required: data qubits, ancillas, gates, measurements, and circuit duration.
5. Check that the proposed verification can detect the intended faults, including errors introduced during preparation and verification.

**Key distinction:** Stabilizer checks alone do not necessarily verify that the correct magic state has been prepared. We must examine both codespace errors and undetectable logical-state errors.

**Decision point:** Select one candidate as the primary experimental system based on protection capability, circuit complexity, and IBM hardware compatibility.

We will not assume that Candidate B is preferable simply because it has a larger code distance, or that Candidate A is preferable because it uses fewer qubits.

If neither candidate supports a practical protected-preparation protocol, we will narrow the project to a clearly labeled error-detection demonstration rather than claim fault-tolerant preparation.

## 3. Phase II — Establish the ideal and noisy baselines

For the selected code–state pair, we will establish four preparation strategies.

| Strategy | Description | Purpose |
|---|---|---|
| B0. Bare physical preparation | Prepare the corresponding physical magic state without encoding | Establish the fundamental physical-state baseline |
| B1. Unprotected logical preparation | Prepare the physical magic state and encode it directly | Measure errors introduced by ordinary encoding |
| B2. Protected logical preparation | Implement a reference error-detected preparation protocol with its standard verification checks | Establish the conventional protected baseline |
| B3. Optimized protected preparation | Apply our proposed hardware-aware error-suppression strategy to B2 | Evaluate the project's contribution |

B3 will be developed in Phase III. B0–B2 must be established first.

### 3.1 Ideal simulation

Use noiseless Qiskit simulation to verify:

- Each preparation circuit produces the intended quantum state.
- The encoding maps physical states to the correct logical states.
- The protected protocol has the expected successful outcomes and acceptance probability.
- The verification measurements preserve the intended output state.
- The chosen fidelity-measurement procedure gives the correct ideal result.

A separate fault-injection check will determine which single physical faults are detected and which can propagate into logical errors.

**Deliverable:** Validated ideal circuits and exact reference states.

### 3.2 Noisy simulation

Construct a noise model containing:

1. Single-qubit and two-qubit depolarizing errors.
2. Thermal relaxation during gates, idling, and relevant waiting periods.
3. Asymmetric measurement/readout errors.

Start with individually enabled noise mechanisms to identify their respective contributions. Then combine them into a realistic backend-derived noise model.

All strategies will be simulated using consistent assumptions and comparable hardware constraints.

At this stage, we should distinguish two numerical approaches:

- Exact noisy-state simulation to calculate the underlying fidelity and acceptance probability.
- Finite-shot simulation to reproduce actual experimental statistics and uncertainty.

The reference protected protocol B2 will use its standard checks, without additional suppression or optimization.

**Deliverable:** A complete baseline comparison, including the impact of encoding and verification overhead.

## 4. Phase III — Develop the error-suppression strategy

This is the main development phase.

### 4.1 Core idea: hardware-aware verification optimization

Instead of applying the same verification circuit regardless of the noise environment, we will explore how the choice and implementation of error checks affect overall performance.

Three optimization dimensions are proposed:

**A. Verification selection**

Investigate whether some checks provide substantially more error-detection benefit than others relative to their implementation cost.

Compare full verification, partial verification, and selected additional checks or flags.

Partial verification must be clearly distinguished from full single-fault detection if it sacrifices a protection guarantee.

**B. Verification ordering and conditional execution**

Investigate whether rearranging the checks or terminating a failed preparation early reduces accumulated noise or average resource consumption.

Where supported, use intermediate measurement outcomes to avoid unnecessary operations.

**C. Hardware-aware circuit implementation**

Optimize the mapping and scheduling of the verification circuit to account for:

- Qubit-specific gate and readout errors.
- Device connectivity and routing overhead.
- Gate durations and idle-time decoherence.
- Measurement and classical-control latency.

Dynamical decoupling may be evaluated as an additional option where idle-time noise is relevant, but it will not be assumed beneficial.

### 4.2 Optimization procedure

We will initially explore a small, structured set of verification strategies rather than attempt an unrestricted search.

For each strategy:

1. Construct the modified circuit.
2. Confirm its ideal operation and characterize its error-detection capability.
3. Evaluate its performance using the selected noise model.
4. Measure the fidelity improvement, acceptance probability, and added hardware cost.
5. Compare it directly against the unmodified protected baseline B2.

Select the most promising strategies using a common resource-aware performance criterion.

The primary comparison must remain B3 versus B2; outperforming only unprotected encoding is not sufficient to establish the value of our optimization.

**Deliverable:** An optimized verification strategy supported by a quantitative comparison with conventional verification.

## 5. Phase IV — Verification and performance evaluation

We will evaluate the quality of the preparation along three dimensions.

### 5.1 Logical-state quality

The central measure is the fidelity of the actual accepted quantum state with respect to the ideal encoded magic state.

In simulation, calculate this directly from the resulting quantum state.

For hardware, develop an experimentally accessible fidelity-estimation protocol using logical observables and appropriate code-space checks, without requiring full physical-state tomography.

All measurements used to reconstruct the accepted state's fidelity must correspond to the same preparation and acceptance procedure.

Virtual codespace projection and other postprocessing estimates may be useful diagnostics, but they must not be presented as evidence that a better physical state was delivered.

### 5.2 Success probability and resource cost

Report:

- Conditional logical-state fidelity.
- Probability of accepting a prepared state.
- Number of attempts required per accepted state.
- Physical qubit count, including ancillas.
- Entangling-gate count and circuit duration.
- Total measurement shots and calibration overhead.

Conditional fidelity must always be interpreted alongside acceptance probability. A protocol that achieves high fidelity by rejecting nearly every attempt may not be the most efficient solution.

### 5.3 Statistical reliability and robustness

Evaluate performance using finite-shot simulations and uncertainty estimates.

Test whether the optimized strategy remains effective under variations in noise strength and backend calibration parameters.

Separate the noise conditions used for optimization from those used for final validation where practical.

Classical readout-error mitigation may be used for measurement analysis, but it must be applied fairly across baselines and kept distinct from physical state improvement.

**Deliverable:** A comparison establishing where the proposed strategy helps, where it fails, and what the improvements cost.

## 6. Phase V — IBM quantum hardware demonstration

After selecting and validating the primary protocol, implement a small demonstration on a real IBM backend.

### 6.1 Hardware preparation

- Select an available backend with suitable connectivity and calibration characteristics.
- Identify a favorable connected region for the required data and ancilla qubits.
- Transpile every circuit to the backend's native gate set.
- Review the final gate counts, depths, and measurement/control requirements.
- Repeat noisy simulation using the transpiled circuits and current backend calibration data.

The hardware implementation should remain as close as possible to the validated simulation.

### 6.2 Hardware experiments

The preferred experimental comparison consists of:

1. Bare physical magic-state preparation (B0).
2. Unprotected logical preparation (B1).
3. Conventional protected preparation (B2).
4. Optimized protected preparation (B3).

Use comparable measurement procedures and a predefined allocation of the hardware measurement budget.

If backend access or circuit complexity is restrictive, prioritize **B1, B2, and B3**, while retaining the physical-state baseline in simulation and including it on hardware if feasible.

Record raw measurement counts, acceptance outcomes, relevant calibration information, and transpiled circuit statistics.

### 6.3 Hardware analysis

Compare the hardware results with both the ideal predictions and noisy simulations.

Evaluate:

- Whether the expected error-detection benefit appears experimentally.
- Whether the optimized strategy improves upon conventional verification.
- Whether the improvement is statistically distinguishable.
- Whether disagreement with simulation can be explained by plausible unmodeled noise or calibration limitations.

**Deliverable:** An experimentally supported comparison, with conclusions limited to the tested hardware and operating conditions.

## 7. Phase VI — Results and scientific interpretation

The final analysis should address four questions.

**Q1. Is logical encoding beneficial?**

Does conventional encoding improve or degrade preparation compared with the bare physical state?

**Q2. How much does protected preparation help?**

Does the reference verification protocol improve the accepted state's fidelity, and at what cost in rejected attempts and extra operations?

**Q3. Can hardware-aware optimization improve on conventional verification?**

Does our optimized protocol provide a better fidelity–acceptance–cost tradeoff than the standard protected baseline?

**Q4. Which hardware errors determine the outcome?**

Are the results primarily limited by two-qubit gate errors, thermal relaxation, readout errors, or the verification circuit itself?

We should explicitly report negative results. If additional error checks introduce more noise than they detect, that is a meaningful result about the limits of small-scale error protection.

## 8. Success criteria and project scope

A successful project must at least establish valid baseline circuits, a correctly modeled noise environment, and a fair comparison of error-detection strategies.

The stronger research objective is to demonstrate a **measurable improvement over conventional protected preparation**, after including verification overhead and discarded shots.

A desirable result is an optimized protocol that either:

- Produces higher-fidelity accepted states at comparable resource cost, or
- Achieves comparable fidelity with a higher acceptance rate or lower total preparation cost.

The numerical thresholds, statistical confidence requirements, and total resource budgets should be fixed before final testing.

### Minimum viable project

For the hackathon, prioritize:

1. One selected code–state pair.
2. Ideal and realistic noisy simulation.
3. Unprotected and conventionally protected baselines.
4. One clearly defined verification optimization.
5. Finite-shot evaluation and a small IBM hardware demonstration, if the required circuits are supported.

### Extended investigation

If the minimum project is successful, extend to:

- The second code–state candidate.
- More sophisticated conditional verification.
- Multiple backend noise profiles.
- More systematic optimization of verification circuits.
- Comparison with additional hardware-aware suppression methods.

The second candidate should not delay completion of the primary experiment.

## 9. Final deliverables

The project should produce:

1. **Qiskit implementation:** Reproducible circuits for state preparation, verification, noisy simulation, and hardware experiments.
2. **Baseline analysis:** Ideal and noisy results for bare, unprotected, and protected preparation.
3. **Optimization results:** A comparison of verification strategies under identical noise and resource assumptions.
4. **Hardware demonstration:** Experimental results for the selected circuits, with raw data and uncertainty estimates.
5. **Scientific report and presentation:** Explanation of the physical task, error mechanisms, optimization concept, quantitative results, overhead, and limitations.

### Intended scientific message

The objective is not to claim a new quantum error-correcting code or a fundamentally new magic-state preparation method.

Instead, the project investigates whether **noise-aware design of error-detection circuits can make small-scale logical magic-state preparation more reliable and resource-efficient on real superconducting quantum hardware**.

The principal result should be an evidence-based understanding of when additional quantum error protection is worthwhile—and when its overhead makes performance worse.
