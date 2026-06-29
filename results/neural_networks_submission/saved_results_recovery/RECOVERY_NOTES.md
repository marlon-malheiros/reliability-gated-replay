# Saved-results recovery note

- No training was launched by `recover_saved_manuscript_results.py`.
- Main-table source collection: `nn_submission_raw_logs` (mapped in
  `evidence/README.md`).
- Reviewer-control source collection: `reviewer_controls_raw_logs` (mapped in
  `evidence/README.md`).
- Requested control families found: buffer_sensitivity, oracle_thinned, random_matched, threshold_sensitivity.
- Requested control families not found and not rerun: none.
- Protocol metadata in the saved five-seed main files: seed 0: [1500], seed 1: [1500], seed 2: [1500], seed 3: [2500], seed 4: [2500].
- The aborted correction-run directory `results/neural_networks_submission/corrected_2500/` was excluded.
- At Split-CIFAR-10 symmetric 60%, clean-only admission cannot reach the matched target because the clean fraction is lower than the target admission rate; the saved oracle-thinned run therefore saturates at admitting all available clean samples.
