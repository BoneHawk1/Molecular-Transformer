# Hybrid evaluation summary

Means over molecules. *noise floor* = second baseline seed vs baseline, at the same frame spacing.

## test molecules (7): acetamide, acetamide_ethyl, anisole, ethanethiol, pyridine, tetrahydrofuran, toluene

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xtb_reference | 1.002 | 1.015 | 1.079 | 0.005 | 0.009 | 0.012 | 0.018 | 0.814 | 0.977 | 1.000 | 10.999 |
| zero_k40 | 0.066 | 0.056 | 0.071 | 0.080 | 0.046 | 0.073 | 0.062 | 0.820 | 14.000 | 1.000 | 1.563 |
| flow_k40 | 1.603 | 1.216 | 1.656 | 0.189 | 0.188 | 0.238 | 0.457 | 3.074 | 8.996 | 0.898 | 3.574 |
| flow_k40_erescale | 1.197 | 0.850 | 0.678 | 0.251 | 0.159 | 0.208 | 0.442 | 1.568 | 10.202 | 0.924 | 5.079 |
| flow_k40_guarded | 1.119 | 0.804 | 0.737 | 0.253 | 0.153 | 0.187 | 0.424 | 1.170 | 9.336 | 0.814 | 8.591 |
| *noise_floor* | 1.003 | 1.055 | 0.965 | 0.083 | 0.024 | 0.057 | 0.038 | 0.973 | – | – | – |

## all molecules (7): acetamide, acetamide_ethyl, anisole, ethanethiol, pyridine, tetrahydrofuran, toluene

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xtb_reference | 1.002 | 1.015 | 1.079 | 0.005 | 0.009 | 0.012 | 0.018 | 0.814 | 0.977 | 1.000 | 10.999 |
| zero_k40 | 0.066 | 0.056 | 0.071 | 0.080 | 0.046 | 0.073 | 0.062 | 0.820 | 14.000 | 1.000 | 1.563 |
| flow_k40 | 1.603 | 1.216 | 1.656 | 0.189 | 0.188 | 0.238 | 0.457 | 3.074 | 8.996 | 0.898 | 3.574 |
| flow_k40_erescale | 1.197 | 0.850 | 0.678 | 0.251 | 0.159 | 0.208 | 0.442 | 1.568 | 10.202 | 0.924 | 5.079 |
| flow_k40_guarded | 1.119 | 0.804 | 0.737 | 0.253 | 0.153 | 0.187 | 0.424 | 1.170 | 9.336 | 0.814 | 8.591 |
| *noise_floor* | 1.003 | 1.055 | 0.965 | 0.083 | 0.024 | 0.057 | 0.038 | 0.973 | – | – | – |

Metric notes: **lag-RMSD ratio** – 1 = moves as far per step as MD; **torsion-step ratio** – 1 = MD; **torsion-flip-rate ratio** – 1 = MD; **torsion JS** – 0 = identical distribution; **<KE> ratio** – 1 = correct temperature
