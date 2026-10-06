# Public reproducibility materials

This directory is a curated redistribution-safe package assembled from the project `Data/` and `Results/` trees for the revision reproducibility release.

## Included

- Experiment manifests for the five sentiment datasets and five offensive-language robustness datasets.
- Exact repetition membership and ordering (`repetitions.jsonl`).
- Experiment-bank row identifiers, source-row indices, hashes, labels, and class positions. The third-party text column is removed.
- Stage-A generation plans.
- Sanitized valid synthetic outputs, linked by `global_generation_index` and `real_row_id`.
- Main development, selection, weighting, adaptive-policy, and held-out test numeric result files.
- Completed generator, embedding, classifier, and task sensitivity result files, including the XLM-R classifier sensitivity analysis.
- Suite-level robustness summaries that combine the fixed reference, generator, representation, classifier, and task arms.
- Complete candidate-count sensitivity outputs used for the revision analysis, including feasibility, repeated-measures, planned pairwise, and per-cell unfiltered results.
- A redistribution-safe qualitative-audit release containing the deterministic 50-pair selection, synthetic texts, source-row identifiers, source-text hashes, and descriptive observations while omitting third-party source text.
- Generation count/validation summaries and prompt templates for the main experiment.
- Prompt templates used for the alternative-generator and offensive-language robustness generation arms.

## Intentionally excluded

- `Data/Raw/` and `Data/Prepared/` real-text files. Original datasets have differing redistribution terms and should be obtained from their original providers.
- The `text` column from every real-data experiment bank.
- Original third-party source text from the qualitative-audit artifacts. The public audit files instead provide `row_id`, `source_row_index`, and `source_text_hash` for local reconstruction after obtaining the source datasets.
- Embedding arrays, embedding manifests, model caches, SQLite status databases, raw model responses, failure logs, and archived failed runs.
- Low-level generation endpoint/runtime metadata not needed to identify the published synthetic examples.
- Lock files that refer to omitted private/intermediate files. This package has its own SHA-256 manifest.

## Real-data reconstruction

Retrieve each original dataset from its cited source and place it under the paths expected by the public preprocessing script. Then run:

```bash
py -3 Sources/Public/Data/prepare_datasets.py
py -3 Sources/Public/Data/reconstruct_public_experiment_data.py
```

`reconstruct_public_experiment_data.py` recomputes the stable row identifiers and SHA-256 text hashes from the locally prepared source data, verifies them against the released `experiment_bank_index.csv` files, and restores the exact source text for the 1,400-row experiment bank of each primary sentiment dataset. It also materializes the exact Base membership and ordered non-overlapping additional-real pool for every paired repetition. The released `ratio_prefixes.csv` reconstruction output specifies how much of that ordered pool belongs to each Matched All-real condition. To materialize every Base + additional-real condition explicitly, run the reconstruction script with `--expand-ratios`.

The original source text is therefore reconstructed only on the user's machine from datasets obtained from their original providers; it is never distributed in the public package itself.

The synthetic exports contain the generated text and the identifiers required to link it to the experiment bank without redistributing the third-party source text. The downstream selection-audit files identify selected synthetic `global_generation_index` values where applicable.

The qualitative-audit release follows the same principle. `Results/QualitativeAudit/v1/` omits the original source-text column but retains the deterministic audit membership, source locators/hashes, generated text, and representative qualitative observations. Users with lawful access to the original datasets can reconstruct the exact source side locally and verify it against the supplied SHA-256 hash.
