#!/usr/bin/env python3
"""Build the deterministic qualitative synthetic-data audit used in revision reporting.

This script does not perform human rating and does not read downstream performance files.
It only selects a fixed, class-balanced set of source/synthetic pairs from the existing
experiment banks and valid candidate pools, then writes reproducible audit artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


AUDIT_VERSION = "qualitative-audit-v1"
PAIRS_PER_DATASET = 10
PAIRS_PER_LABEL = 5
CANDIDATE_INDEX = 0

DATASETS = [
    ("en_binary_sst2", "English"),
    ("ko_binary_nsmc", "Korean"),
    ("bn_binary_cinexdrama", "Bengali"),
    ("ha_binary_hausa_movie_review", "Hausa"),
    ("ml_binary_dravidian_codemix", "Malayalam"),
]

# Representatives are chosen from the deterministic 50-pair audit set to make the
# manuscript examples traceable to concrete rows. They are not used to select the
# 50-pair audit set itself.
REPRESENTATIVE_ROWS = {
    "en_binary_sst2": "c42d8a077ea9260f0e72ab8e",
    "ko_binary_nsmc": "c0600632979a1f6d1cd9f8d1",
    "bn_binary_cinexdrama": "b53e0079c2897601ec4c9a6a",
    "ha_binary_hausa_movie_review": "bc3ef9f80c65eb04bc02e7a8",
    "ml_binary_dravidian_codemix": "c60ffe24cfcfdece22b68636",
}

REPRESENTATIVE_OBSERVATIONS = {
    "en_binary_sst2": (
        "Negative polarity is preserved; the very short source is expanded into a complete "
        "sentence with event context not explicit in the seed."
    ),
    "ko_binary_nsmc": (
        "Positive evaluation is preserved; the short colloquial source is expanded from liking "
        "the song to explicitly praising its melody and atmosphere."
    ),
    "bn_binary_cinexdrama": (
        "Positive affect and the reaction of being moved to tears are closely preserved in a "
        "more standardized sentence."
    ),
    "ha_binary_hausa_movie_review": (
        "Negative stance is preserved, but a specific conditional narrative complaint is "
        "generalized into a broader negative evaluation."
    ),
    "ml_binary_dravidian_codemix": (
        "The negative evaluation of the film-related behavior is preserved; a Romanized Malayalam "
        "source is normalized into Malayalam script, while the synthetic text adds a stronger "
        "directive not stated explicitly in the source."
    ),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def selection_hash(cell_id: str, row_id: str) -> str:
    return hashlib.sha256(f"{cell_id}|{row_id}".encode("utf-8")).hexdigest()


def read_experiment_bank(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_id = row["row_id"]
            if row_id in rows:
                raise RuntimeError(f"Duplicate row_id in experiment bank: {row_id}")
            rows[row_id] = row
    return rows


def read_candidate_zero(valid_dir: Path) -> Dict[str, dict]:
    candidates: Dict[str, dict] = {}
    parts = sorted(valid_dir.glob("part_*.jsonl"))
    if not parts:
        raise FileNotFoundError(f"No valid candidate parts found under {valid_dir}")

    for part in parts:
        with part.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                if int(payload["candidate_index"]) != CANDIDATE_INDEX:
                    continue
                row_id = payload["real_row_id"]
                prior = candidates.get(row_id)
                if prior is not None and prior != payload:
                    raise RuntimeError(
                        f"Conflicting candidate-index-0 records for {row_id} in {part}:{line_number}"
                    )
                candidates[row_id] = payload
    return candidates


def word_count(text: str) -> int:
    return len(text.split())


def select_rows(cell_id: str, bank: Dict[str, dict], candidates: Dict[str, dict]) -> List[dict]:
    labels = sorted({str(row["label"]).strip() for row in bank.values()})
    if labels != ["0", "1"]:
        raise RuntimeError(f"Expected binary labels 0/1 for {cell_id}; found {labels}")

    selected: List[dict] = []
    for label in labels:
        eligible = [
            row_id
            for row_id, row in bank.items()
            if str(row["label"]).strip() == label and row_id in candidates
        ]
        eligible.sort(key=lambda row_id: selection_hash(cell_id, row_id))
        if len(eligible) < PAIRS_PER_LABEL:
            raise RuntimeError(
                f"Only {len(eligible)} candidate-index-0 rows for {cell_id} label {label}; "
                f"need {PAIRS_PER_LABEL}"
            )

        for rank, row_id in enumerate(eligible[:PAIRS_PER_LABEL], start=1):
            source = bank[row_id]
            synthetic = candidates[row_id]
            source_text = source["text"]
            synthetic_text = synthetic["generated_text"]
            source_words = word_count(source_text)
            synthetic_words = word_count(synthetic_text)
            selected.append(
                {
                    "audit_version": AUDIT_VERSION,
                    "cell_id": cell_id,
                    "row_id": row_id,
                    "label": label,
                    "inherited_label": synthetic.get("inherited_label", ""),
                    "candidate_index": int(synthetic["candidate_index"]),
                    "generation_seed": synthetic.get("generation_seed", ""),
                    "within_label_rank": rank,
                    "selection_hash": selection_hash(cell_id, row_id),
                    "source_text": source_text,
                    "synthetic_text": synthetic_text,
                    "source_word_count": source_words,
                    "synthetic_word_count": synthetic_words,
                    "synthetic_to_source_word_ratio": (
                        round(synthetic_words / source_words, 6) if source_words else ""
                    ),
                }
            )
    return selected


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_findings(path: Path, representatives: List[dict]) -> None:
    by_cell = {row["cell_id"]: row for row in representatives}
    lines = [
        "# Qualitative Synthetic-Data Audit Findings",
        "",
        "This file records descriptive observations from the deterministic 50-pair audit. "
        "It is not a formal multilingual human-rating study and contains no human quality scores.",
        "",
        "## Representative examples",
        "",
    ]
    for cell_id, display_name in DATASETS:
        row = by_cell[cell_id]
        lines.extend(
            [
                f"### {display_name}",
                "",
                f"- Row ID: `{row['row_id']}`",
                f"- Label: `{row['label']}` / inherited label: `{row['inherited_label']}`",
                f"- Source: {row['source_text']}",
                f"- Synthetic: {row['synthetic_text']}",
                f"- Observation: {row['qualitative_observation']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Cross-dataset interpretation",
            "",
            "- Bengali provides a representative case of close preservation of affective content; "
            "this is qualitatively consistent with the retained augmentation policy and its modest "
            "held-out improvement.",
            "- Hausa shows a representative semantic generalization from a specific narrative "
            "complaint to a broader evaluation, and Malayalam shows normalization of Romanized/"
            "code-mixed input into Malayalam script. Both are plausible generated-to-target "
            "distribution-mismatch mechanisms and are qualitatively consistent with augmentation "
            "not being retained for those datasets.",
            "- These mechanisms do not explain all outcomes: Korean also contains close sentiment-"
            "preserving transformations despite Base selection, while English retains augmentation "
            "despite examples with semantic expansion.",
            "- Therefore the audit supports only a partial interpretation: candidate-level quality "
            "differences may contribute to the observed heterogeneity, but they are neither necessary "
            "nor sufficient to explain downstream augmentation utility.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = repository_root()
    output_dir = root / "Results" / "BinaryMatchedSizeExperiment" / "QualitativeAudit" / "v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    display_names = dict(DATASETS)
    for cell_id, _ in DATASETS:
        bank_path = root / "Data" / "ExperimentManifests" / "v1" / cell_id / "experiment_bank.csv"
        valid_dir = (
            root
            / "Results"
            / "BinaryMatchedSizeExperiment"
            / "Generation"
            / "valid_outputs"
            / cell_id
        )
        bank = read_experiment_bank(bank_path)
        candidates = read_candidate_zero(valid_dir)
        selected = select_rows(cell_id, bank, candidates)
        for row in selected:
            row["dataset"] = display_names[cell_id]
        all_rows.extend(selected)

    if len(all_rows) != len(DATASETS) * PAIRS_PER_DATASET:
        raise RuntimeError(f"Expected 50 audit rows; found {len(all_rows)}")

    counts = Counter((row["cell_id"], row["label"]) for row in all_rows)
    for cell_id, _ in DATASETS:
        for label in ("0", "1"):
            if counts[(cell_id, label)] != PAIRS_PER_LABEL:
                raise RuntimeError(
                    f"Unexpected count for {cell_id} label {label}: {counts[(cell_id, label)]}"
                )

    audit_fields = [
        "audit_version",
        "dataset",
        "cell_id",
        "row_id",
        "label",
        "inherited_label",
        "candidate_index",
        "generation_seed",
        "within_label_rank",
        "selection_hash",
        "source_text",
        "synthetic_text",
        "source_word_count",
        "synthetic_word_count",
        "synthetic_to_source_word_ratio",
    ]
    write_csv(output_dir / "audit_pairs.csv", all_rows, audit_fields)

    representatives: List[dict] = []
    for cell_id, display_name in DATASETS:
        row_id = REPRESENTATIVE_ROWS[cell_id]
        matches = [row for row in all_rows if row["cell_id"] == cell_id and row["row_id"] == row_id]
        if len(matches) != 1:
            raise RuntimeError(
                f"Representative {row_id} for {cell_id} is not uniquely present in the 50-pair audit"
            )
        row = dict(matches[0])
        row["dataset"] = display_name
        row["qualitative_observation"] = REPRESENTATIVE_OBSERVATIONS[cell_id]
        representatives.append(row)

    representative_fields = audit_fields + ["qualitative_observation"]
    write_csv(output_dir / "representative_examples.csv", representatives, representative_fields)

    summary = {
        "audit_version": AUDIT_VERSION,
        "formal_human_evaluation": False,
        "human_quality_scores_present": False,
        "inter_rater_agreement_present": False,
        "selection_uses_downstream_performance": False,
        "selection_algorithm": (
            "For each dataset and label, retain valid candidate_index=0 rows, sort source row IDs by "
            "SHA-256(cell_id + '|' + row_id), and select the first five."
        ),
        "candidate_index": CANDIDATE_INDEX,
        "pairs_total": len(all_rows),
        "pairs_per_dataset": PAIRS_PER_DATASET,
        "pairs_per_label_per_dataset": PAIRS_PER_LABEL,
        "datasets": [
            {
                "cell_id": cell_id,
                "display_name": display_name,
                "pairs": sum(1 for row in all_rows if row["cell_id"] == cell_id),
                "negative": counts[(cell_id, "0")],
                "positive": counts[(cell_id, "1")],
                "representative_row_id": REPRESENTATIVE_ROWS[cell_id],
            }
            for cell_id, display_name in DATASETS
        ],
        "outputs": [
            "audit_pairs.csv",
            "representative_examples.csv",
            "audit_summary.json",
            "qualitative_findings.md",
        ],
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_findings(output_dir / "qualitative_findings.md", representatives)

    print(f"Wrote {len(all_rows)} audit pairs to {output_dir}")
    for cell_id, display_name in DATASETS:
        print(
            f"  {display_name}: n=10 "
            f"(negative={counts[(cell_id, '0')]}, positive={counts[(cell_id, '1')]}), "
            f"representative={REPRESENTATIVE_ROWS[cell_id]}"
        )


if __name__ == "__main__":
    main()
