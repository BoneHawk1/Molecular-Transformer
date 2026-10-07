# Hybrid evaluation summary

Ratio columns are medians over molecules (ratios blow up for molecules where MD barely moves); other columns are means. *noise floor* = second baseline seed vs baseline, at the same frame spacing.

## test molecules (15): C1CCOC1, CC(=O)NC(C)C(=O)NC, CC(=O)Oc1ccccc1C(=O)O, CC(C)C(N)C(=O)O, CCCCCO, COc1ccccc1, Cc1ccccc1, Cc1ncc[nH]1, Clc1ccccc1, OC1CCCCC1, OCCO, OCc1ccccc1, c1ccc(cc1)C(=O)N, c1ccc2[nH]ccc2c1, c1ccccc1O

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k8 | 0.663 | 0.844 | 0.475 | 0.049 | 0.024 | 0.026 | 0.077 | 0.996 | 18.589 | 1.000 | 0.060 |
| flow_k8_r1 | 0.967 | 0.970 | 0.921 | 0.094 | 0.033 | 0.048 | 0.176 | 0.998 | 7.060 | 0.866 | 0.158 |
| flow_k8_v2 | 1.057 | 1.017 | 1.152 | 0.116 | 0.021 | 0.043 | 0.132 | 0.996 | 11.162 | 0.988 | 0.125 |
| flow_k8_v2_large | 1.060 | 1.051 | 1.056 | 0.113 | 0.022 | 0.042 | 0.120 | 0.997 | 11.701 | 0.981 | 0.109 |
| flow_k12_v2 | 1.115 | 1.034 | 1.240 | 0.112 | 0.028 | 0.039 | 0.121 | 0.994 | 8.488 | 0.940 | 0.097 |
| *noise_floor* | 0.997 | 1.003 | 0.972 | 0.016 | 0.012 | 0.012 | 0.026 | 1.004 | – | – | – |

## val molecules (9): C1CCCCC1, CC(C)CN, CC(C)Cc1ccc(C(C)C(=O)O)cc1, CC(C)NCCO, CCCCO, CN(C)C, CN(C)C=O, Cc1ccc(O)cc1, c1ccc2ccccc2c1

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k8 | 0.860 | 0.941 | 0.436 | 0.139 | 0.023 | 0.024 | 0.088 | 0.992 | 18.635 | 1.000 | 0.025 |
| flow_k8_r1 | 1.028 | 1.172 | 2.846 | 0.131 | 0.025 | 0.030 | 0.153 | 0.994 | 8.257 | 0.861 | 0.220 |
| flow_k8_v2 | 1.161 | 1.213 | 2.955 | 0.116 | 0.018 | 0.028 | 0.125 | 0.993 | 10.549 | 0.941 | 0.228 |
| flow_k8_v2_large | 0.980 | 1.130 | 1.513 | 0.111 | 0.018 | 0.029 | 0.105 | 0.993 | 10.957 | 0.943 | 0.225 |
| flow_k12_v2 | 1.359 | 1.794 | 3.811 | 0.137 | 0.024 | 0.027 | 0.103 | 0.992 | 7.871 | 0.857 | 0.166 |
| *noise_floor* | 0.999 | 1.000 | 0.975 | 0.080 | 0.012 | 0.012 | 0.054 | 1.003 | – | – | – |

## all molecules (24): C1CCCCC1, C1CCOC1, CC(=O)NC(C)C(=O)NC, CC(=O)Oc1ccccc1C(=O)O, CC(C)C(N)C(=O)O, CC(C)CN, CC(C)Cc1ccc(C(C)C(=O)O)cc1, CC(C)NCCO, CCCCCO, CCCCO, CN(C)C, CN(C)C=O, COc1ccccc1, Cc1ccc(O)cc1, Cc1ccccc1, Cc1ncc[nH]1, Clc1ccccc1, OC1CCCCC1, OCCO, OCc1ccccc1, c1ccc(cc1)C(=O)N, c1ccc2[nH]ccc2c1, c1ccc2ccccc2c1, c1ccccc1O

| condition | lag-RMSD ratio | torsion-step ratio | torsion-flip-rate ratio | torsion JS | angle JS | bond JS | pair-dist L1 | <KE> ratio | force-call savings | accepted | wall s / ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_k8 | 0.713 | 0.846 | 0.458 | 0.083 | 0.024 | 0.025 | 0.081 | 0.995 | 18.606 | 1.000 | 0.047 |
| flow_k8_r1 | 0.977 | 0.987 | 1.021 | 0.108 | 0.030 | 0.041 | 0.167 | 0.996 | 7.509 | 0.864 | 0.181 |
| flow_k8_v2 | 1.125 | 1.058 | 1.673 | 0.116 | 0.020 | 0.038 | 0.129 | 0.996 | 10.932 | 0.971 | 0.164 |
| flow_k8_v2_large | 1.029 | 1.076 | 1.431 | 0.112 | 0.020 | 0.038 | 0.115 | 0.996 | 11.422 | 0.967 | 0.152 |
| flow_k12_v2 | 1.142 | 1.115 | 1.890 | 0.121 | 0.027 | 0.034 | 0.115 | 0.993 | 8.257 | 0.909 | 0.123 |
| *noise_floor* | 0.998 | 1.001 | 0.974 | 0.040 | 0.012 | 0.012 | 0.036 | 1.004 | – | – | – |

Metric notes: **lag-RMSD ratio** – 1 = moves as far per step as MD; **torsion-step ratio** – 1 = MD; **torsion-flip-rate ratio** – 1 = MD; **torsion JS** – 0 = identical distribution; **<KE> ratio** – 1 = correct temperature
