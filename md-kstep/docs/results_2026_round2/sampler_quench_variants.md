# Hybrid evaluation summary

Means over molecules. *noise floor* = second baseline seed vs baseline, at the same frame spacing.

## all molecules (4): CC(=O)Oc1ccccc1C(=O)O, CC(C)CO, CC(C)NCCO, CCN(CC)CCO

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ode_full | 1.071 | 1.155 | 0.941 | 0.142 | 0.027 | 0.029 | 0.121 | 0.992 | 6.032 | 0.743 | 0.248 |
| sde1_full | 1.048 | 1.079 | 0.908 | 0.139 | 0.034 | 0.033 | 0.142 | 0.988 | 4.950 | 0.755 | 0.260 |
| ode_thermal | 1.127 | 1.204 | 0.936 | 0.140 | 0.018 | 0.015 | 0.094 | 0.994 | 6.898 | 0.741 | 0.216 |
| sde1_thermal | 1.067 | 1.118 | 0.924 | 0.166 | 0.022 | 0.018 | 0.116 | 0.995 | 5.805 | 0.771 | 0.406 |
| sde3_thermal | 1.030 | 1.015 | 0.928 | 0.110 | 0.012 | 0.012 | 0.061 | 0.996 | 0.999 | 0.001 | 0.260 |
| sde3_noquench | 1.015 | 1.004 | 0.926 | 0.096 | 0.012 | 0.012 | 0.064 | 0.995 | 1.000 | 0.000 | 0.261 |
| *noise_floor* | 1.012 | 1.009 | 1.020 | 0.120 | 0.012 | 0.012 | 0.039 | 0.996 | – | – | – |

Metric notes: **lag-RMSD ratio** – 1 = moves as far per step as MD; **torsion-step ratio** – 1 = MD; **torsion-flip-rate ratio** – 1 = MD; **torsion JS** – 0 = identical distribution; **<KE> ratio** – 1 = correct temperature
