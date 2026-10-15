# Why accepted states can lie outside the code subspace

An ideal complete stabilizer measurement, with no intervening or subsequent noise, projects the all-zero branch into the code subspace. The implemented verification circuit has noisy gates, time-separated checks, and a finite readout wait. Its recorded zero syndrome therefore does not guarantee codespace membership at delivery.

After the last data–ancilla interaction, an ancilla retains information about the earlier state. Later data relaxation cannot update that syndrome record. In particular, the data wait 2.584 microseconds during terminal syndrome readout. Noise during this interval can change the delivered state without changing the syndrome probabilities.

## Controlled comparisons

Each entry is the percentage of the accepted density matrix outside the code subspace. The controls independently omit the named channel while retaining all other noise. Removing assignment errors retains the full readout duration. Removing data relaxation during readout keeps the readout errors, earlier delays, gates and routing; it is a channel ablation, not a proposed shorter hardware schedule.

| Protocol | Full model | No syndrome assignment errors | No data relaxation during readout wait | No delay relaxation anywhere |
|---|---:|---:|---:|---:|
| four/B2_ED | 5.4659% | 5.4412% | 2.1373% | 1.8064% |
| five/B2_ED | 10.5075% | 10.4644% | 6.0773% | 4.3371% |
| five/B2_BK | 14.3191% | 13.7551% | 10.0895% | 8.4459% |

Syndrome assignment errors explain only a small part of the observed outside-code population. Data relaxation during readout explains a substantial part; remaining errors include faults in the extraction circuit and earlier idle evolution. These conditional-state differences are not additive event probabilities.

The data-readout-wait ablation leaves acceptance unchanged to numerical precision for every B2 circuit. This directly separates late data damage from classical syndrome misclassification. The remaining outside-code population with all delay relaxation disabled shows that noisy extraction gates also prevent an ideal final codespace projection.

Outside-code population here means departure from the computational code subspace. It is distinct from a logical error within that subspace, and from physical leakage into higher transmon levels (not modeled).

Raw metrics and ablation metadata: [delivery_wait_ablation.json](delivery_wait_ablation.json). The original baseline results and finite-shot estimates are unchanged.

Reference: [IBM stabilizer-code definition](https://quantum.cloud.ibm.com/learning/en/courses/foundations-of-quantum-error-correction/stabilizer-formalism/stabilizer-codes).
