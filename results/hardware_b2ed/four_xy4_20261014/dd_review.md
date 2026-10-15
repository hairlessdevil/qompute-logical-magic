# Proposed B2_ED hardware run with explicit XY4

**Prepared and validated; not submitted.**

- Instance/backend: `qzh-eu` / `ibm_aachen`.
- Four-qubit encoded H-state, same recorded-`000` acceptance rule.
- 24 circuits × 2,048 shots = **49,152 shots**, one SamplerV2 job.
- Source: the exact no-DD circuits used in job `db84dj8331ic73fu5jfg`.

## Sequence choice

Use standard **XY4**, with axes `Xπ, Yπ, Xπ, Yπ`. It is a useful first test
for a stored arbitrary state when the refocusable noise axes are not known.
It is not an optimized sequence for this device and does not reverse T1 decay,
Markovian decoherence, gate faults, or measurement assignment errors.
Identically timed DD on coupled qubits also need not refocus their mutual ZZ
interaction; this first run does not implement staggered decoupling.

The explicit ideal sequence is identity up to a compensated global phase.
For each eligible idle, the free-delay fractions are
`1/8, 1/4, 1/4, 1/4, 1/8`, with discrete-time rounding placed in the middle.
The X pulse is native; Y uses the same native X pulse bracketed by virtual
`Rz(-π/2)` and `Rz(+π/2)` gates in circuit time order.

## Actual inserted pulses

- Protect nine noninitial idle windows of at least 500 ns.
- Add **36 physical π pulses and 36 zero-duration Rz gates** per preparation.
- All four output-data wires receive XY4 during the **2.180 µs** syndrome-readout wait.
- Each π pulse lasts 32 ns, so four occupy 128 ns of an existing idle.
- Preserve all original native gate and measurement start times and the total
  **5.020 µs** preparation duration. Basis rotations and final data readout are unchanged.
- Preserve the physical qubit mapping, all 39 CZ gates, original shot budget,
  and measurement settings. Short idles and initial ground-state idles are skipped.

Automatic Runtime DD is **off** because it is incompatible with dynamic
circuits. The submitted circuits themselves contain the added native pulses;
this is not a request for server-side DD insertion.

## Validation and interpretation

Both new offline tests passed. All 24 prepared settings pass native-instruction
validation against the live backend. The ideal statevector check has maximum
absolute error `1.56e-15`. Original gate durations still match the saved source
calibration. Original gate/measurement timestamps and complete circuit durations
are verified unchanged locally.

The first job is the no-DD reference. Because the two variants run as separate
jobs, any observed difference can also include drift. No fresh no-DD reference
or measurement-calibration circuits are included in this 49,152-shot budget.
The existing independent-noise simulation does not contain the slow coherent
noise that DD is intended to refocus, so its DD result cannot establish the
expected hardware benefit.

The submission script will print and save the job ID, then exit without polling
or retrieving results. Retrieval remains a separate later script.

Manifest SHA256:
`025ff88c6d932f457da8e6f4bb411be7695d7ca1306e8148e9288e4a3c14710b`

References: [IBM DD guidance](https://quantum.cloud.ibm.com/learning/en/courses/tools-for-quantum-advantage/error-suppression),
[Sampler feature compatibility](https://quantum.cloud.ibm.com/docs/en/guides/sampler-options).
