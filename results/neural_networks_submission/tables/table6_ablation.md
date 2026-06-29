<!-- benchmark = permuted_mnist; moderate=40% high=80% symmetric noise. -->

| ablation_arm | description | acc@40% | acc@80% | separation@80% | inversion_rate@80% |
| --- | --- | --- | --- | --- | --- |
| er | no gate / random replay admission | 0.722±0.091 | 0.228±0.017 |  |  |
| gate_conf | confidence gate only (label-free) | 0.723±0.095 | 0.232±0.009 | 0.022±0.006 | 0.000±0.000 |
| gate_loss | loss/error gate only (supervised) | 0.802±0.083 | 0.334±0.015 | 0.108±0.006 | 0.000±0.000 |
| gate_predstab | prediction-stability gate only (label-free) | 0.751±0.061 | 0.209±0.002 | -0.001±0.001 | 0.933±0.094 |
| gate_coteach | agreement / co-teaching gate only | 0.805±0.014 | 0.335±0.008 | 0.144±0.003 | 0.000±0.000 |
| ewc | consolidation only (EWC; PNN-anchor family, no replay gate) | 0.789±0.035 | 0.327±0.005 |  |  |
| ewc_loss | consolidation + reliability gate (gated Fisher) | 0.790±0.038 | 0.265±0.020 |  |  |
| oracle | oracle clean-sample selector (upper bound) | 0.818±0.011 |  |  |  |