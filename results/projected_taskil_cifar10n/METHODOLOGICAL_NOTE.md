# Mandatory methodological note

Projected task-IL CIFAR-10N uses the fixed clean CIFAR-10 class pairs
`(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`, and `(8,9)`, matching native
Split-CIFAR-10. For each example, the CIFAR-10N correctness mask is retained
exactly. Correct annotations keep the clean local label; incorrect annotations
are projected to `1 - clean_local`.

The projection preserves the exact error incidence and per-task noise rate.
Buffer purity, gate--correctness separation, and inversion are computed against
that retained correctness mask, and clean test labels remain unchanged. The
projection does not preserve the external human confusion target when it lies
outside the binary task. The existing class-incremental CIFAR-10N experiment
uses the original global human labels and therefore retains the full confusion
structure.

The no-training sanity check passes for every seed at the run budget. On the
full label sets, aggregate noise is approximately `0.091` versus the published
`0.090`, worst noise is approximately `0.402`, and the fractions of incorrect
human targets outside the corresponding binary task are `88.1%` and `87.4%`.

Two differences from the earlier native Split-CIFAR-10 runs must accompany any
comparison:

1. CIFAR-10N corruption is projected into the binary task-local label space.
2. Every bridge seed uses `max_train_per_task = 2500`; earlier native
   Split-CIFAR-10 runs use 1,500 examples for seeds 0--2 and 2,500 for seeds
   3--4.

All 50 bridge runs completed: two label variants, five methods, and five seeds.
The term **projected task-IL CIFAR-10N** is used throughout these artifacts.
