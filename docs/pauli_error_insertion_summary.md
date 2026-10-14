# Pauli-error insertion: implementation status and results

## Which baselines are implemented?

The implementation contains more than B2:

- **B0:** bare physical magic-state preparation.
- **B1:** ordinary logical encoding.
- **B2_ED:** encoding followed by stabilizer checks, for both codes.
- **B2_BK:** five-copy tetrahedral-state projection, for the five-qubit code.
- **B3:** not implemented.

The recent Pauli-insertion sections use the **B2_ED encoder/check circuit**, with an additional active-correction experiment. They do not test B2_BK. B2_ED is a stabilizer-screening baseline, not a validated fault-tolerant magic-state preparation protocol.

## Acceptance versus active correction

**B2_ED accepts only zero syndrome.** The active-correction experiment instead keeps every syndrome and applies a correction.

The earlier fidelity comparison reported results after active correction. To answer what happens to an *accepted* output, the table below applies the B2_ED zero-syndrome acceptance rule.

## Were all Pauli errors detected?

**No—not when faults can occur throughout the circuit.**

After encoding and before otherwise perfect checks, all single data-qubit X/Y/Z errors were detected:

- **Four-qubit code:** 12/12 errors detected.
- **Five-qubit code:** 15/15 errors detected.

At other insertion points, the representative results were:

| Error insertion | Four-qubit B2_ED | Five-qubit B2_ED |
|---|---|---|
| X/Y/Z after encoding, before checks | Detected → reject | Detected → reject |
| Y on input before encoding | **Accepted:** logical Y_L, fidelity **0** | **Accepted:** logical Y_L, fidelity **1/3** |
| Selected Z fault during encoding | **Accepted:** logical Z_L, fidelity **1/2** | Detected → reject |
| Selected ancilla X during a check | **Accepted:** logical X_L, fidelity **1/2** | Detected → reject |
| X on data after extraction, before readout | **Accepted:** outside codespace, fidelity **0** | **Accepted:** outside codespace, fidelity **0** |
| Recorded syndrome-bit flip | False rejection of a correct state | False rejection of a correct state |

These accept/reject outcomes are deterministic for the selected ideal examples. Fidelity is the squared overlap with the intended encoded magic state. The selected encoder and ancilla examples do not establish detection behavior for every possible insertion location.

## How do accepted faults affect the logical state?

### 1. Undetected logical errors

The state remains inside the codespace, so every stabilizer still returns +1. However, a logical Pauli changes the encoded magic state:

- **Z_L** reverses the relative sign of the logical amplitudes.
- **X_L** swaps the logical amplitudes.
- **Y_L** combines a logical bit flip and phase flip, up to an overall phase.

For the four-qubit H-state target, the selected logical Y_L error produces an orthogonal state, giving fidelity 0. The selected logical X_L and Z_L errors each give fidelity 1/2.

For the five-qubit tetrahedral target, the logical Y_L error gives fidelity 1/3.

Another perfect stabilizer-check round cannot detect these logical changes: the incorrect state still satisfies all code stabilizers.

### 2. Residual physical errors introduced too late to be checked

The late X example leaves the codespace after syndrome information has already been extracted. The recorded syndrome remains zero, so the preparation is accepted even though its physical target fidelity is zero.

For this example, another ideal syndrome-extraction and correction round restores fidelity 1 for both codes. This is recoverable residual damage, unlike the undetected logical errors above. The additional ideal round is a diagnostic; it is not part of the delivered state in the original circuit.

## Interpretation

**Zero syndrome does not guarantee that the intended logical magic state was delivered.** It can miss a logical error, or fail to reflect an error introduced after the relevant check.

The five-qubit encoder and ancilla examples with fidelity 1/3 in the earlier comparison were evaluated **after active correction**. Their syndromes were nonzero, so **B2_ED postselection would reject those cases**; they would not contaminate its accepted output in these examples.

The separately labeled recovery-gate example combines a storage error with a recovery-gate fault—two error events—and is not included in the single-fault table above.

## Local evidence

- [Executed notebook, Sections 6–7](../notebooks/01_ideal_magic_states.ipynb)
- [Single-Pauli storage-error results](../results/single_pauli_errors.json)
- [Representative circuit-location results](../results/representative_fault_locations.json)
- [Baseline ideal results](../results/ideal/ideal_results.json)
