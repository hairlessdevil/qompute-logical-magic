# Syndrome readout error and finite-shot state quality

## What was included in the original noisy result?

Yes: `NoiseModel.from_backend(..., readout_error=True)` was enabled. The existing fidelity used an exact density matrix conditioned on the **recorded** all-zero syndrome. Ancilla assignment error changes which data blocks are accepted; it is not an extra quantum gate acting on the data. False rejections reduce yield, and false acceptances mix in blocks associated with nonzero true ancilla outcomes. A true zero outcome itself does not guarantee an undamaged logical state.

The original noisy notebook sampled acceptance only. Its quoted fidelities were simulator diagnostics, not finite-shot measurements. There were no destructive data measurements in those preparation circuits, so data readout error did not enter those fidelities.

## Isolating syndrome assignment error

Here only readout assignment is disabled. Routing, gate noise, every delay, and the 2.584 µs syndrome measurement wait stay fixed. This ablation does not remove measurement latency.

| Protocol | Acceptance without → with readout error | Fidelity without → with readout error | Nonzero true syndrome among accepted |
|---|---:|---:|---:|
| four/B2_ED | 91.703% → 90.505% | 94.334% → 94.309% | 0.0309% |
| five/B2_ED | 75.679% → 74.422% | 88.725% → 88.682% | 0.0627% |
| five/B2_BK | 14.864% → 14.707% | 86.017% → 85.452% | 0.6781% |

In these circuits, false rejection accounts for most of the readout-induced yield loss. The BK accepted ensemble is more affected by false acceptance. B0/B1 have no syndrome measurement, so this ablation does not change them.

## Finite-shot target fidelity

Expand the **full physical target projector** into Pauli strings:

$$|\psi\rangle\langle\psi|=\sum_P c_P P,\qquad \widehat F=c_I+\sum_{P\ne I}c_P\,\overline{p}_{P,\mathrm{recorded\ syndrome}=0}.$$

For every nonidentity term, prepare fresh copies, retain the recorded-zero shots, and measure the relevant data-qubit parity. Use native decompositions of H for X and S† followed by H for Y; Z needs no rotation. Each saved preparation schedule is frozen, so analysis rotations start at its original delivery boundary. The model includes their gate noise, padding delays, and data readout error. These destructive measurements consume analysis copies. The fidelity operator includes codespace loss; measuring just logical X/Y/Z without the codespace projector would be insufficient.

| Protocol | Exact preparation fidelity | Raw finite-shot estimate | Data-readout-mitigated estimate | Approx. 95% statistical interval (mitigated) | Settings × shots |
|---|---:|---:|---:|---:|---:|
| four/B0 | 0.999792 | 0.993214 | 0.996365 | [0.988556, 1.004173] | 2 × 8,192 |
| four/B1 | 0.985589 | 0.840238 | 0.984672 | [0.980214, 0.989129] | 23 × 8,192 |
| four/B2_ED | 0.943088 | 0.809648 | 0.941372 | [0.936544, 0.946201] | 23 × 8,192 |
| five/B0 | 0.999784 | 0.991297 | 0.994436 | [0.985462, 1.003410] | 3 × 8,192 |
| five/B1 | 0.908950 | 0.770784 | 0.908311 | [0.904752, 0.911870] | 63 × 8,192 |
| five/B2_ED | 0.886818 | 0.747499 | 0.885097 | [0.880877, 0.889317] | 63 × 8,192 |
| five/B2_BK | 0.854524 | 0.720743 | 0.853481 | [0.843772, 0.863191] | 63 × 8,192 |

Total attempted quality-measurement shots: **1,966,080**; shots per setting are attempts, not accepted shots. B2_BK's low yield reduces its effective sample size. Identity terms require no shots. The JSON records every count, accepted sample size, coefficient, seed, and raw/mitigated score.

## What readout mitigation does and does not correct

Only **data readout** is corrected, using the known assignment matrices of the simulation snapshot. For one qubit with A[true, recorded], solve A w = (1, −1); multiply the corresponding per-qubit weights for a Pauli parity. For symmetric error e this is division by 1−2e for each nonidentity factor. Syndrome bits are not mitigated, so the estimated state is still the actually accepted output. These calculations use counts alone; saved simulator probabilities are used only to diagnose systematic bias.

The statistical intervals use independent settings and the sample variance of their weighted parity scores. Normal intervals are approximate; conservative Hoeffding/union-bound intervals are also saved. Estimates/intervals are not clipped to [0,1]. Both describe sampling uncertainty around the measurement-score mean. They do not cover calibration error, drift, correlated assignment, or residual basis-rotation/delay bias. The assignment matrices are treated as exact here; hardware use would need calibration and its uncertainty.

| Protocol | Infinite-shot raw score | Infinite-shot mitigated score | Residual mitigated bias relative to delivery fidelity |
|---|---:|---:|---:|
| four/B0 | 0.996602 | 0.999774 | -0.000018 |
| four/B1 | 0.840401 | 0.984940 | -0.000649 |
| four/B2_ED | 0.810400 | 0.942383 | -0.000705 |
| five/B0 | 0.996603 | 0.999775 | -0.000009 |
| five/B1 | 0.770542 | 0.908027 | -0.000923 |
| five/B2_ED | 0.747860 | 0.885548 | -0.001270 |
| five/B2_BK | 0.720857 | 0.853304 | -0.001221 |

The large raw bias in some rows comes from data assignment, especially the 8.67% readout error on physical qubit 2 when it carries a data qubit. That error did not affect the earlier ancilla-only acceptance measurement when qubit 2 was unmeasured.

## Execution and reproduction

The saved `quality_circuits/*.qpy` contain hardware-form measurement circuits without simulator state saves. For efficient Aer sampling, syndrome measurements are mathematically deferred across the later data-only rotations. No extra evolution is applied to the deferred ancillas. This equivalence relies on this model's independent channels; the original data waits and the physically recorded acceptance rule are preserved.

```bash
conda run -n quantum-jax python scripts/analyze_readout.py --shots 8192
```

No packages are installed or upgraded. No manual faults are inserted. No hardware job is submitted.

Reference: [Aer NoiseModel and readout_error option](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html).
