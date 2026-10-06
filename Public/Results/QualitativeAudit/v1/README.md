# Public qualitative-audit artifacts

The full local audit uses the original third-party source text so that source-synthetic pairs can
be inspected directly.  This public release intentionally omits that source-text column rather
than assuming redistribution rights for the underlying datasets.

`audit_pairs_public.csv` and `representative_examples_public.csv` provide `cell_id`, `row_id`,
`source_row_index`, and `source_text_hash` so that the exact source can be reconstructed after the
user obtains the original dataset and runs the public preprocessing pipeline.  Synthetic text is
included because it was generated as part of this study.

The deterministic selection procedure itself is available in
`Sources/Public/BinaryMatchedSizeExperiment/run_qualitative_audit.py`.
