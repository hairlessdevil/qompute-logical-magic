# How much bad population does syndrome filtering reject?

Here bad means **outside the code subspace at the original output delivery time**, after all gates and delays, including the syndrome-readout wait. This does not classify logical errors inside the code subspace. All probabilities include syndrome assignment error. The diagnostic replays the exact saved circuits without recompilation and verifies acceptance and fidelities against the saved results.

For unnormalized accepted/rejected data states R_A and R_R and codespace projector P, the joint populations are Tr(P R_A), Tr((I-P) R_A), Tr(P R_R), and Tr((I-P) R_R). These correspond to an ideal final codespace test on data, jointly with the recorded syndrome. No such extra test or postselection is applied in the actual preparation.

## Joint populations: percentages of all attempts

| Protocol | Accepted, inside code | Accepted, outside code | Rejected, inside code | Rejected, outside code |
|---|---:|---:|---:|---:|
| four/B2_ED | 85.558% | 4.947% | 2.255% | 7.239% |
| five/B2_ED | 66.602% | 7.820% | 3.691% | 21.887% |
| five/B2_BK | 12.601% | 2.106% | 1.552% | 83.741% |

## Filtering efficiency at delivery

| Protocol | Total rejected | Outside-code population without selection | Fraction of that population rejected | Outside-code fraction among accepted | Original rejected shots |
|---|---:|---:|---:|---:|---:|
| four/B2_ED | 9.495% | 12.186% | 59.406% | 5.466% | 827/8192 |
| five/B2_ED | 25.578% | 29.706% | 73.676% | 10.507% | 2129/8192 |
| five/B2_BK | 85.293% | 85.847% | 97.547% | 14.319% | 6951/8192 |

All exact populations refer to the same completed circuit, with or without selecting its accepted branch. They are not estimates of how many faults were present before extraction, nor a comparison to omitting the extraction gates. Some noise occurs during extraction or after the relevant check. Density-matrix populations do not give a unique good/bad label to each sampled shot.

B2_BK rejects 5/6 of attempts even ideally: its large rejection includes the intended projection of product inputs, not just removal of noise-induced errors. Inside-code does not necessarily mean the correct logical state, and rejected inside-code population is not all attributable to readout errors.

Reproduce: `conda run -n quantum-jax python scripts/analyze_filtering.py`.
