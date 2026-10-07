# Hybrid evaluation summary

Ratio columns are medians over molecules (ratios blow up for molecules where MD barely moves); other columns are means. *noise floor* = second baseline seed vs baseline, at the same frame spacing.

## test molecules (3): CC(=O)Oc1ccccc1C(=O)O, c1ccc(cc1)C(=O)N, c1ccccc1O

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k4 | 0.431 | 0.904 | 0.473 | 0.003 | 0.006 | 0.010 | 0.016 | 0.995 | 21.000 | 1.000 | 0.007 |
| zero_k4_quench | 0.103 | 0.210 | 0.000 | 0.114 | 0.379 | 0.417 | 0.332 | 0.995 | 7.000 | 1.000 | 0.122 |
| legacy_transformer | 0.208 | 0.374 | 0.041 | 0.076 | 0.073 | 0.081 | 0.137 | 0.682 | 18.826 | 0.994 | 0.124 |
| mean_k4 | 0.326 | 0.354 | 0.065 | 0.130 | 0.238 | 0.258 | 0.283 | 0.980 | 4.738 | 0.905 | 0.111 |
| flow_k4 | 0.724 | 0.794 | 0.736 | 0.044 | 0.055 | 0.130 | 0.237 | 0.995 | 5.040 | 0.740 | 0.120 |
| flow_k8 | 0.709 | 0.829 | 0.758 | 0.025 | 0.033 | 0.075 | 0.107 | 0.984 | 5.860 | 0.670 | 0.066 |
| *noise_floor* | 0.979 | 0.999 | 0.939 | 0.003 | 0.006 | 0.006 | 0.013 | 0.994 | – | – | – |

## val molecules (1): CC(C)NCCO

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k4 | 0.345 | 0.541 | 0.072 | 0.175 | 0.007 | 0.009 | 0.057 | 1.008 | 21.000 | 1.000 | 0.007 |
| zero_k4_quench | 0.138 | 0.184 | 0.014 | 0.504 | 0.321 | 0.333 | 0.347 | 1.008 | 7.000 | 1.000 | 0.092 |
| legacy_transformer | 0.203 | 0.270 | 0.113 | 0.384 | 0.092 | 0.072 | 0.352 | 0.789 | 15.880 | 0.984 | 0.143 |
| mean_k4 | 0.177 | 0.213 | 0.002 | 0.462 | 0.341 | 0.316 | 0.394 | 1.008 | 6.950 | 0.999 | 0.086 |
| flow_k4 | 0.806 | 0.961 | 1.387 | 0.171 | 0.042 | 0.081 | 0.093 | 1.008 | 6.801 | 0.995 | 0.090 |
| flow_k8 | 1.055 | 1.257 | 2.364 | 0.189 | 0.028 | 0.031 | 0.113 | 0.993 | 6.125 | 0.926 | 0.049 |
| *noise_floor* | 1.028 | 1.018 | 1.144 | 0.044 | 0.006 | 0.006 | 0.035 | 1.006 | – | – | – |

## train molecules (8): C1=CC(=O)NC(=O)N1, CC(C)C(=O)O, CC(C)CO, CC1=CC(=O)NC(=O)N1, CCN(CC)CCO, CCOC(=O)N, c1ccc2c(c1)OCO2, c1ccncc1

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k4 | 0.625 | 0.722 | 0.533 | 0.188 | 0.008 | 0.009 | 0.058 | 1.004 | 21.000 | 1.000 | 0.007 |
| zero_k4_quench | 0.174 | 0.205 | 0.000 | 0.345 | 0.361 | 0.373 | 0.382 | 1.004 | 7.000 | 1.000 | 0.086 |
| legacy_transformer | 0.336 | 0.302 | 0.094 | 0.228 | 0.116 | 0.082 | 0.193 | 0.775 | 18.280 | 0.992 | 0.127 |
| mean_k4 | 0.333 | 0.308 | 0.033 | 0.233 | 0.268 | 0.273 | 0.325 | 0.995 | 5.193 | 0.926 | 0.084 |
| flow_k4 | 0.899 | 0.911 | 0.890 | 0.076 | 0.042 | 0.113 | 0.134 | 1.005 | 6.878 | 0.997 | 0.085 |
| flow_k8 | 0.997 | 1.000 | 1.208 | 0.097 | 0.038 | 0.065 | 0.137 | 0.997 | 9.589 | 0.989 | 0.043 |
| *noise_floor* | 0.991 | 0.999 | 1.041 | 0.087 | 0.006 | 0.006 | 0.028 | 0.999 | – | – | – |

## all molecules (12): C1=CC(=O)NC(=O)N1, CC(=O)Oc1ccccc1C(=O)O, CC(C)C(=O)O, CC(C)CO, CC(C)NCCO, CC1=CC(=O)NC(=O)N1, CCN(CC)CCO, CCOC(=O)N, c1ccc(cc1)C(=O)N, c1ccc2c(c1)OCO2, c1ccccc1O, c1ccncc1

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k4 | 0.552 | 0.722 | 0.473 | 0.141 | 0.007 | 0.009 | 0.048 | 1.002 | 21.000 | 1.000 | 0.007 |
| zero_k4_quench | 0.167 | 0.205 | 0.000 | 0.301 | 0.362 | 0.380 | 0.367 | 1.002 | 7.000 | 1.000 | 0.095 |
| legacy_transformer | 0.275 | 0.302 | 0.068 | 0.203 | 0.103 | 0.081 | 0.192 | 0.763 | 18.216 | 0.992 | 0.128 |
| mean_k4 | 0.326 | 0.308 | 0.033 | 0.226 | 0.267 | 0.273 | 0.320 | 0.994 | 5.225 | 0.927 | 0.091 |
| flow_k4 | 0.847 | 0.911 | 0.890 | 0.076 | 0.045 | 0.114 | 0.156 | 1.005 | 6.412 | 0.933 | 0.094 |
| flow_k8 | 0.979 | 0.959 | 1.190 | 0.086 | 0.036 | 0.065 | 0.127 | 0.993 | 8.368 | 0.904 | 0.049 |
| *noise_floor* | 0.991 | 1.000 | 0.985 | 0.062 | 0.006 | 0.006 | 0.025 | 0.999 | – | – | – |

Metric notes: **lag-RMSD ratio** – 1 = moves as far per step as MD; **torsion-step ratio** – 1 = MD; **torsion-flip-rate ratio** – 1 = MD; **torsion JS** – 0 = identical distribution; **<KE> ratio** – 1 = correct temperature
