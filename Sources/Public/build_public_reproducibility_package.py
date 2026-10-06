#!/usr/bin/env python3
"""Assemble the redistribution-safe reproducibility package under ``Public/``.

The research tree intentionally keeps third-party source text and large model artifacts
outside version control.  This exporter copies numeric/manuscript-facing results that are
safe to redistribute and builds a sanitized qualitative-audit release in which original
dataset text is replaced by stable source-row identifiers and text hashes.

Run from any working directory with::

    py -3 Sources/Public/build_public_reproducibility_package.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "Public"
RESULTS = ROOT / "Results" / "BinaryMatchedSizeExperiment"


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_named_files(source_dir: Path, destination_dir: Path, names: list[str]) -> None:
    for name in names:
        copy_file(source_dir / name, destination_dir / name)


def copy_tree_text(source_dir: Path, destination_dir: Path, pattern: str = "*") -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    for source in sorted(source_dir.glob(pattern)):
        if source.is_file():
            copy_file(source, destination_dir / source.name)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def public_bank_index(cell_id: str) -> dict[str, dict[str, str]]:
    path = PUBLIC / "Data" / "Sentiment" / cell_id / "experiment_bank_index.csv"
    rows = read_csv(path)
    index = {row["row_id"]: row for row in rows}
    if len(index) != len(rows):
        raise RuntimeError(f"Duplicate row_id in public experiment-bank index: {path}")
    return index


def sanitize_audit_rows(path: Path, include_observation: bool) -> list[dict[str, object]]:
    source_rows = read_csv(path)
    index_by_cell: dict[str, dict[str, dict[str, str]]] = {}
    sanitized: list[dict[str, object]] = []

    for row in source_rows:
        cell_id = row["cell_id"]
        bank_index = index_by_cell.setdefault(cell_id, public_bank_index(cell_id))
        bank_row = bank_index.get(row["row_id"])
        if bank_row is None:
            raise RuntimeError(
                f"Audit row {row['row_id']} is absent from the public bank index for {cell_id}"
            )

        public_row: dict[str, object] = {
            "audit_version": row["audit_version"],
            "dataset": row["dataset"],
            "cell_id": cell_id,
            "row_id": row["row_id"],
            "source_row_index": bank_row["_source_row"],
            "source_text_hash": bank_row["_text_hash"],
            "label": row["label"],
            "inherited_label": row["inherited_label"],
            "candidate_index": row["candidate_index"],
            "generation_seed": row["generation_seed"],
            "within_label_rank": row["within_label_rank"],
            "selection_hash": row["selection_hash"],
            "synthetic_text": row["synthetic_text"],
            "source_word_count": row["source_word_count"],
            "synthetic_word_count": row["synthetic_word_count"],
            "synthetic_to_source_word_ratio": row["synthetic_to_source_word_ratio"],
        }
        if include_observation:
            public_row["qualitative_observation"] = row["qualitative_observation"]
        sanitized.append(public_row)

    return sanitized


def export_qualitative_audit() -> None:
    source_dir = RESULTS / "QualitativeAudit" / "v1"
    destination_dir = PUBLIC / "Results" / "QualitativeAudit" / "v1"
    destination_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = sanitize_audit_rows(source_dir / "audit_pairs.csv", include_observation=False)
    representative_rows = sanitize_audit_rows(
        source_dir / "representative_examples.csv", include_observation=True
    )

    fields = [
        "audit_version",
        "dataset",
        "cell_id",
        "row_id",
        "source_row_index",
        "source_text_hash",
        "label",
        "inherited_label",
        "candidate_index",
        "generation_seed",
        "within_label_rank",
        "selection_hash",
        "synthetic_text",
        "source_word_count",
        "synthetic_word_count",
        "synthetic_to_source_word_ratio",
    ]
    write_csv(destination_dir / "audit_pairs_public.csv", audit_rows, fields)
    write_csv(
        destination_dir / "representative_examples_public.csv",
        representative_rows,
        fields + ["qualitative_observation"],
    )

    summary = json.loads((source_dir / "audit_summary.json").read_text(encoding="utf-8"))
    summary["source_text_redistributed"] = False
    summary["source_reconstruction"] = (
        "Obtain the original dataset from its provider, run the public preprocessing code, "
        "and identify the source using cell_id, row_id, source_row_index, and source_text_hash."
    )
    summary["public_outputs"] = [
        "audit_pairs_public.csv",
        "representative_examples_public.csv",
        "audit_summary_public.json",
        "qualitative_findings_public.md",
        "README.md",
    ]
    summary.pop("outputs", None)
    (destination_dir / "audit_summary_public.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    by_cell = {row["cell_id"]: row for row in representative_rows}
    findings = [
        "# Qualitative Synthetic-Data Audit Findings (public release)",
        "",
        "This redistribution-safe file records the descriptive observations from the deterministic "
        "50-pair audit without redistributing third-party source text.",
        "",
        "The original source text can be reconstructed locally from the cited dataset using the "
        "row identifier, source-row index, and SHA-256 text hash supplied in the CSV files.",
        "",
        "## Representative examples",
        "",
    ]
    display_order = [
        ("en_binary_sst2", "English"),
        ("ko_binary_nsmc", "Korean"),
        ("bn_binary_cinexdrama", "Bengali"),
        ("ha_binary_hausa_movie_review", "Hausa"),
        ("ml_binary_dravidian_codemix", "Malayalam"),
    ]
    for cell_id, display_name in display_order:
        row = by_cell[cell_id]
        findings.extend(
            [
                f"### {display_name}",
                "",
                f"- Row ID: `{row['row_id']}`",
                f"- Source-row index: `{row['source_row_index']}`",
                f"- Source-text SHA-256: `{row['source_text_hash']}`",
                f"- Label: `{row['label']}` / inherited label: `{row['inherited_label']}`",
                "- Source text: omitted from the public package; retrieve it from the original dataset.",
                f"- Synthetic: {row['synthetic_text']}",
                f"- Observation: {row['qualitative_observation']}",
                "",
            ]
        )

    findings.extend(
        [
            "## Interpretation boundary",
            "",
            "The audit is descriptive rather than a formal multilingual human-rating study. It "
            "contains no numerical human quality scores or inter-rater agreement statistics, and "
            "the selected examples were not chosen using downstream performance.",
            "",
        ]
    )
    (destination_dir / "qualitative_findings_public.md").write_text(
        "\n".join(findings), encoding="utf-8"
    )

    readme = """# Public qualitative-audit artifacts

The full local audit uses the original third-party source text so that source-synthetic pairs can
be inspected directly.  This public release intentionally omits that source-text column rather
than assuming redistribution rights for the underlying datasets.

`audit_pairs_public.csv` and `representative_examples_public.csv` provide `cell_id`, `row_id`,
`source_row_index`, and `source_text_hash` so that the exact source can be reconstructed after the
user obtains the original dataset and runs the public preprocessing pipeline.  Synthetic text is
included because it was generated as part of this study.

The deterministic selection procedure itself is available in
`Sources/Public/BinaryMatchedSizeExperiment/run_qualitative_audit.py`.
"""
    (destination_dir / "README.md").write_text(readme, encoding="utf-8")


def export_classifier_robustness() -> None:
    source_dir = RESULTS / "Robustness" / "v1" / "Classifier" / "XLMRBase"
    destination_dir = PUBLIC / "Results" / "Robustness" / "Classifier" / "XLMRBase"
    copy_named_files(
        source_dir,
        destination_dir,
        ["paired_summary.csv", "repeat_results.csv", "report.json"],
    )


def export_robustness_summary() -> None:
    source_dir = RESULTS / "Robustness" / "v1"
    destination_dir = PUBLIC / "Results" / "Robustness"
    copy_named_files(
        source_dir,
        destination_dir,
        ["robustness_summary.csv", "robustness_summary.json"],
    )


def export_robustness_prompts() -> None:
    prompt_pairs = [
        (
            RESULTS / "Robustness" / "v1" / "Generator" / "Qwen38" / "Generation" / "prompts",
            PUBLIC
            / "Results"
            / "Robustness"
            / "Generator"
            / "Qwen38"
            / "Generation"
            / "Prompts",
        ),
        (
            RESULTS
            / "Robustness"
            / "v1"
            / "Task"
            / "OffensiveLanguage"
            / "Generation"
            / "prompts",
            PUBLIC
            / "Results"
            / "Robustness"
            / "Task"
            / "OffensiveLanguage"
            / "Generation"
            / "Prompts",
        ),
    ]
    for source_dir, destination_dir in prompt_pairs:
        copy_tree_text(source_dir, destination_dir, "*.txt")


def export_candidate_count_sensitivity() -> None:
    source_dir = RESULTS / "CandidateCountSensitivity" / "v3"
    destination_dir = PUBLIC / "Results" / "CandidateCountSensitivity" / "v3"
    excluded = {"CANDIDATE_COUNT_SENSITIVITY_LOCK.json", "candidate_count_status.sqlite3"}
    for source in sorted(source_dir.iterdir()):
        if source.is_file() and source.name not in excluded:
            copy_file(source, destination_dir / source.name)


def regenerate_package_manifest() -> None:
    manifest_path = PUBLIC / "PACKAGE_MANIFEST.sha256"
    entries: list[str] = []
    for path in sorted(PUBLIC.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(PUBLIC).as_posix()
        entries.append(f"{digest}  {relative}")
    manifest_path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def validate_public_qualitative_audit() -> None:
    audit_dir = PUBLIC / "Results" / "QualitativeAudit" / "v1"
    forbidden = "source_text,"
    for name in ("audit_pairs_public.csv", "representative_examples_public.csv"):
        header = (audit_dir / name).read_text(encoding="utf-8-sig").splitlines()[0]
        if forbidden in header or header.endswith("source_text"):
            raise RuntimeError(f"Third-party source text column leaked into {name}: {header}")

    rows = read_csv(audit_dir / "audit_pairs_public.csv")
    if len(rows) != 50:
        raise RuntimeError(f"Expected 50 public qualitative-audit rows; found {len(rows)}")


def main() -> None:
    export_classifier_robustness()
    export_robustness_summary()
    export_robustness_prompts()
    export_candidate_count_sensitivity()
    export_qualitative_audit()
    validate_public_qualitative_audit()
    regenerate_package_manifest()
    print(f"Updated public reproducibility package: {PUBLIC}")


if __name__ == "__main__":
    main()
