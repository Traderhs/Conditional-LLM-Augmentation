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

Retrieve each original dataset from its cited source, run the repository's public preprocessing code, and match rows using `_source_row`, `_text_hash`, and `row_id` in `experiment_bank_index.csv`. `repetitions.jsonl` then identifies the Base rows and the ordered additional-real pool used to construct Matched All-real conditions.

The synthetic exports contain the generated text and the identifiers required to link it to the experiment bank without redistributing the third-party source text. The downstream selection-audit files identify selected synthetic `global_generation_index` values where applicable.

The qualitative-audit release follows the same principle. `Results/QualitativeAudit/v1/` omits the original source-text column but retains the deterministic audit membership, source locators/hashes, generated text, and representative qualitative observations. Users with lawful access to the original datasets can reconstruct the exact source side locally and verify it against the supplied SHA-256 hash.
