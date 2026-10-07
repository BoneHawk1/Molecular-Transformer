# Hybrid evaluation summary

Means over molecules. *noise floor* = second baseline seed vs baseline, at the same frame spacing.

## test molecules (3): CC(=O)Oc1ccccc1C(=O)O, c1ccc(cc1)C(=O)N, c1ccccc1O

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k4 | 0.518 | 0.755 | 0.952 | 0.003 | 0.006 | 0.010 | 0.016 | 0.994 | 21.000 | 1.000 | 0.007 |
| zero_k4_quench | 0.117 | 0.177 | 0.780 | 0.114 | 0.379 | 0.417 | 0.332 | 0.994 | 7.000 | 1.000 | 0.122 |
| legacy_transformer | 0.277 | 0.349 | 0.875 | 0.076 | 0.073 | 0.081 | 0.137 | 0.724 | 18.826 | 0.994 | 0.124 |
| mean_k4 | 0.278 | 0.286 | 0.624 | 0.130 | 0.238 | 0.258 | 0.283 | 0.983 | 4.738 | 0.905 | 0.111 |
| flow_k4 | 0.860 | 0.803 | 0.868 | 0.044 | 0.055 | 0.130 | 0.237 | 1.002 | 5.040 | 0.740 | 0.120 |
| flow_k8 | 0.799 | 0.848 | 0.937 | 0.025 | 0.033 | 0.075 | 0.107 | 0.984 | 5.860 | 0.670 | 0.066 |
| *noise_floor* | 0.993 | 1.000 | 1.006 | 0.003 | 0.006 | 0.006 | 0.013 | 1.000 | – | – | – |

## val molecules (1): CC(C)NCCO

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k4 | 0.345 | 0.541 | 0.594 | 0.175 | 0.007 | 0.009 | 0.057 | 1.008 | 21.000 | 1.000 | 0.007 |
| zero_k4_quench | 0.138 | 0.184 | 0.401 | 0.504 | 0.321 | 0.333 | 0.347 | 1.008 | 7.000 | 1.000 | 0.092 |
| legacy_transformer | 0.203 | 0.270 | 0.373 | 0.384 | 0.092 | 0.072 | 0.352 | 0.789 | 15.880 | 0.984 | 0.143 |
| mean_k4 | 0.177 | 0.213 | 0.314 | 0.462 | 0.341 | 0.316 | 0.394 | 1.008 | 6.950 | 0.999 | 0.086 |
| flow_k4 | 0.806 | 0.961 | 0.776 | 0.171 | 0.042 | 0.081 | 0.093 | 1.008 | 6.801 | 0.995 | 0.090 |
| flow_k8 | 1.055 | 1.257 | 0.918 | 0.189 | 0.028 | 0.031 | 0.113 | 0.993 | 6.125 | 0.926 | 0.049 |
| *noise_floor* | 1.028 | 1.018 | 1.031 | 0.044 | 0.006 | 0.006 | 0.035 | 1.006 | – | – | – |

## train molecules (8): C1=CC(=O)NC(=O)N1, CC(C)C(=O)O, CC(C)CO, CC1=CC(=O)NC(=O)N1, CCN(CC)CCO, CCOC(=O)N, c1ccc2c(c1)OCO2, c1ccncc1

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k4 | 0.645 | 0.754 | 0.825 | 0.188 | 0.008 | 0.009 | 0.058 | 1.003 | 21.000 | 1.000 | 0.007 |
| zero_k4_quench | 0.171 | 0.212 | 0.688 | 0.345 | 0.361 | 0.373 | 0.382 | 1.003 | 7.000 | 1.000 | 0.086 |
| legacy_transformer | 0.351 | 0.338 | 0.684 | 0.228 | 0.116 | 0.082 | 0.193 | 0.743 | 18.280 | 0.992 | 0.127 |
| mean_k4 | 0.300 | 0.306 | 0.683 | 0.233 | 0.268 | 0.273 | 0.325 | 0.997 | 5.193 | 0.926 | 0.084 |
| flow_k4 | 0.858 | 0.876 | 0.914 | 0.076 | 0.042 | 0.113 | 0.134 | 1.003 | 6.878 | 0.997 | 0.085 |
| flow_k8 | 0.981 | 1.021 | 0.949 | 0.097 | 0.038 | 0.065 | 0.137 | 0.996 | 9.589 | 0.989 | 0.043 |
| *noise_floor* | 0.998 | 1.000 | 1.002 | 0.087 | 0.006 | 0.006 | 0.028 | 1.000 | – | – | – |

## all molecules (12): C1=CC(=O)NC(=O)N1, CC(=O)Oc1ccccc1C(=O)O, CC(C)C(=O)O, CC(C)CO, CC(C)NCCO, CC1=CC(=O)NC(=O)N1, CCN(CC)CCO, CCOC(=O)N, c1ccc(cc1)C(=O)N, c1ccc2c(c1)OCO2, c1ccccc1O, c1ccncc1

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k4 | 0.589 | 0.737 | 0.838 | 0.141 | 0.007 | 0.009 | 0.048 | 1.001 | 21.000 | 1.000 | 0.007 |
| zero_k4_quench | 0.155 | 0.201 | 0.687 | 0.301 | 0.362 | 0.380 | 0.367 | 1.001 | 7.000 | 1.000 | 0.095 |
| legacy_transformer | 0.321 | 0.335 | 0.706 | 0.203 | 0.103 | 0.081 | 0.192 | 0.742 | 18.216 | 0.992 | 0.128 |
| mean_k4 | 0.284 | 0.293 | 0.637 | 0.226 | 0.267 | 0.273 | 0.320 | 0.995 | 5.225 | 0.927 | 0.091 |
| flow_k4 | 0.854 | 0.865 | 0.891 | 0.076 | 0.045 | 0.114 | 0.156 | 1.003 | 6.412 | 0.933 | 0.094 |
| flow_k8 | 0.941 | 0.997 | 0.943 | 0.086 | 0.036 | 0.065 | 0.127 | 0.993 | 8.368 | 0.904 | 0.049 |
| *noise_floor* | 0.999 | 1.001 | 1.006 | 0.062 | 0.006 | 0.006 | 0.025 | 1.001 | – | – | – |

Metric notes: **lag-RMSD ratio** – 1 = moves as far per step as MD; **torsion-step ratio** – 1 = MD; **torsion-flip-rate ratio** – 1 = MD; **torsion JS** – 0 = identical distribution; **<KE> ratio** – 1 = correct temperature
