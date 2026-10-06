from __future__ import annotations

"""Reconstruct the exact real-data experiment membership from the public release.

The public package deliberately omits third-party source text.  After a user obtains
the original datasets and runs ``prepare_datasets.py``, this script joins the prepared
training rows back to the released text-free experiment-bank indices, verifies every
row against its released SHA-256 hash and stable row identifier, and materializes the
exact Base and additional-real membership used in each paired repetition.

By default the script reconstructs the five sentiment datasets used in the primary
experiment.  The released repetition records are sufficient to reconstruct every
Matched All-real(r) condition: a condition consists of the repetition's Base rows plus
the first ``n_add`` rows of its ordered additional-real master pool.  ``--expand-ratios``
can materialize those condition memberships explicitly.
"""

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREPARED_ROOT = PROJECT_ROOT / "Data" / "Prepared"
DEFAULT_PUBLIC_DATA_ROOT = PROJECT_ROOT / "Public" / "Data" / "Sentiment"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Data" / "ReconstructedPublic" / "Sentiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the exact real-data experiment bank and paired Base/additional-real "
            "membership from the redistribution-safe public package."
        )
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=DEFAULT_PREPARED_ROOT,
        help=(
            "Prepared dataset root created by Sources/Public/Data/prepare_datasets.py "
            "(default: Data/Prepared)."
        ),
    )
    parser.add_argument(
        "--public-data-root",
        type=Path,
        default=DEFAULT_PUBLIC_DATA_ROOT,
        help="Released text-free data package (default: Public/Data/Sentiment).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination for reconstructed files.",
    )
    parser.add_argument(
        "--cell-id",
        action="append",
        dest="cell_ids",
        help="Reconstruct only this cell. May be supplied more than once.",
    )
    parser.add_argument(
        "--expand-ratios",
        action="store_true",
        help=(
            "Also write a long-form matched_all_real_membership.csv containing the exact "
            "Base + additional-real rows for every repetition and augmentation ratio."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing per-cell reconstruction directory.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def canonical_value(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return unicodedata.normalize("NFC", str(value)).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prepared_train(cell_id: str, prepared_root: Path) -> pd.DataFrame:
    path = prepared_root / cell_id / "train.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared train split not found: {path}. Run prepare_datasets.py first."
        )

    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    frame = frame.copy()
    frame["text"] = frame["text"].map(normalize_text)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame.insert(0, "_source_row", np.arange(len(frame), dtype=np.int64))
    frame["_text_hash"] = frame["text"].map(sha256_text)

    source_columns = [column for column in frame.columns if not column.startswith("_")]

    def compute_row_id(row: pd.Series) -> str:
        values = [canonical_value(row[column]) for column in sorted(source_columns)]
        values.append(str(int(row["_source_row"])))
        payload = f"{cell_id}\x1ftrain\x1f" + "\x1f".join(values)
        return sha256_text(payload)[:24]

    frame["row_id"] = frame.apply(compute_row_id, axis=1)
    if frame["row_id"].duplicated().any():
        raise RuntimeError(f"{cell_id}: duplicate reconstructed row_id values")
    return frame


def load_public_bank_index(cell_dir: Path) -> pd.DataFrame:
    path = cell_dir / "experiment_bank_index.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "row_id",
        "_source_row",
        "_text_hash",
        "label",
        "bank_class_position",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    if frame["row_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate row_id values")
    return frame


def reconstruct_bank(
    cell_id: str,
    prepared: pd.DataFrame,
    public_index: pd.DataFrame,
) -> pd.DataFrame:
    prepared_by_row_id = prepared.set_index("row_id", drop=False)
    reconstructed: list[dict[str, Any]] = []

    for public_row in public_index.to_dict(orient="records"):
        row_id = str(public_row["row_id"])
        if row_id not in prepared_by_row_id.index:
            raise RuntimeError(
                f"{cell_id}: released row_id {row_id} was not reproduced from prepared train.csv"
            )
        source = prepared_by_row_id.loc[row_id]
        if isinstance(source, pd.DataFrame):
            raise RuntimeError(f"{cell_id}: row_id {row_id} is not unique in prepared data")

        expected_source_row = int(public_row["_source_row"])
        expected_label = int(public_row["label"])
        expected_hash = str(public_row["_text_hash"])
        actual_source_row = int(source["_source_row"])
        actual_label = int(source["label"])
        actual_hash = str(source["_text_hash"])

        if actual_source_row != expected_source_row:
            raise RuntimeError(
                f"{cell_id}/{row_id}: source-row mismatch "
                f"({actual_source_row} != {expected_source_row})"
            )
        if actual_label != expected_label:
            raise RuntimeError(
                f"{cell_id}/{row_id}: label mismatch ({actual_label} != {expected_label})"
            )
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{cell_id}/{row_id}: text-hash mismatch ({actual_hash} != {expected_hash})"
            )

        reconstructed.append(
            {
                "row_id": row_id,
                "_source_row": actual_source_row,
                "_text_hash": actual_hash,
                "text": source["text"],
                "label": actual_label,
                "bank_class_position": int(public_row["bank_class_position"]),
            }
        )

    result = pd.DataFrame(reconstructed)
    if len(result) != len(public_index):
        raise AssertionError("reconstructed experiment-bank row count changed unexpectedly")
    return result


def read_repetitions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    if not records:
        raise ValueError(f"{path}: no repetition records")
    return records


def ordered_rows(
    ids: list[str],
    bank_by_id: dict[str, dict[str, Any]],
    context: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, row_id in enumerate(ids):
        if row_id in seen:
            raise RuntimeError(f"{context}: duplicate row_id {row_id}")
        seen.add(row_id)
        if row_id not in bank_by_id:
            raise RuntimeError(f"{context}: row_id {row_id} not present in reconstructed bank")
        row = dict(bank_by_id[row_id])
        row["membership_order"] = position
        output.append(row)
    return output


def reconstruct_repetition_membership(
    cell_id: str,
    bank: pd.DataFrame,
    repetitions: list[dict[str, Any]],
    expand_ratios: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    bank_by_id = {row["row_id"]: row for row in bank.to_dict(orient="records")}
    all_bank_ids = set(bank_by_id)
    base_long: list[dict[str, Any]] = []
    additional_long: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    matched_long: list[dict[str, Any]] = []

    for record in repetitions:
        if record.get("cell_id") != cell_id:
            raise RuntimeError(
                f"{cell_id}: repetition record has unexpected cell_id={record.get('cell_id')}"
            )
        repeat_seed = int(record["repeat_seed"])
        base_ids = [str(value) for value in record["base_row_ids"]]
        additional_ids = [str(value) for value in record["additional_real_master_order"]]
        if set(base_ids) & set(additional_ids):
            raise RuntimeError(f"{cell_id}/{repeat_seed}: Base/additional-real overlap")
        if set(base_ids) | set(additional_ids) != all_bank_ids:
            raise RuntimeError(f"{cell_id}/{repeat_seed}: repetition does not partition the bank")

        base_rows = ordered_rows(base_ids, bank_by_id, f"{cell_id}/{repeat_seed}/Base")
        additional_rows = ordered_rows(
            additional_ids,
            bank_by_id,
            f"{cell_id}/{repeat_seed}/additional-real",
        )
        for row in base_rows:
            base_long.append({"repeat_seed": repeat_seed, **row})
        for row in additional_rows:
            additional_long.append({"repeat_seed": repeat_seed, **row})

        for ratio_spec in record["ratios"]:
            ratio = float(ratio_spec["ratio"])
            n_add = int(ratio_spec["additional_real_prefix_n"])
            expected_total = int(ratio_spec["total_train_n"])
            if len(base_rows) + n_add != expected_total:
                raise RuntimeError(
                    f"{cell_id}/{repeat_seed}/r={ratio}: total size mismatch"
                )
            ratio_rows.append(
                {
                    "cell_id": cell_id,
                    "repeat_seed": repeat_seed,
                    "ratio": ratio,
                    "base_n": len(base_rows),
                    "additional_real_prefix_n": n_add,
                    "matched_all_real_total_n": expected_total,
                    "synthetic_target_n": int(ratio_spec["synthetic_target_n"]),
                    "base_label_vector": json.dumps(
                        record["base_label_vector"], sort_keys=True, separators=(",", ":")
                    ),
                    "additional_label_vector": json.dumps(
                        ratio_spec["additional_label_vector"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )

            if expand_ratios:
                for row in base_rows:
                    matched_long.append(
                        {
                            "repeat_seed": repeat_seed,
                            "ratio": ratio,
                            "member_role": "Base",
                            **row,
                        }
                    )
                for row in additional_rows[:n_add]:
                    matched_long.append(
                        {
                            "repeat_seed": repeat_seed,
                            "ratio": ratio,
                            "member_role": "Additional real",
                            **row,
                        }
                    )

    return (
        pd.DataFrame(base_long),
        pd.DataFrame(additional_long),
        pd.DataFrame(ratio_rows),
        pd.DataFrame(matched_long) if expand_ratios else None,
    )


def reconstruct_cell(
    cell_id: str,
    prepared_root: Path,
    public_data_root: Path,
    output_root: Path,
    expand_ratios: bool,
    force: bool,
) -> dict[str, Any]:
    source_dir = public_data_root / cell_id
    destination = output_root / cell_id
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"Output already exists: {destination}. Use --force to replace it."
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    prepared = load_prepared_train(cell_id, prepared_root)
    public_index = load_public_bank_index(source_dir)
    bank = reconstruct_bank(cell_id, prepared, public_index)
    repetitions = read_repetitions(source_dir / "repetitions.jsonl")
    base, additional, ratios, matched = reconstruct_repetition_membership(
        cell_id, bank, repetitions, expand_ratios
    )

    bank.to_csv(destination / "experiment_bank_reconstructed.csv", index=False, encoding="utf-8-sig")
    base.to_csv(destination / "base_membership.csv", index=False, encoding="utf-8-sig")
    additional.to_csv(
        destination / "additional_real_master_order.csv", index=False, encoding="utf-8-sig"
    )
    ratios.to_csv(destination / "ratio_prefixes.csv", index=False, encoding="utf-8-sig")
    if matched is not None:
        matched.to_csv(
            destination / "matched_all_real_membership.csv",
            index=False,
            encoding="utf-8-sig",
        )

    report = {
        "cell_id": cell_id,
        "verified_against_public_index": True,
        "experiment_bank_rows": int(len(bank)),
        "repetitions": int(len(repetitions)),
        "base_membership_rows": int(len(base)),
        "additional_real_membership_rows": int(len(additional)),
        "ratio_specs": int(len(ratios)),
        "expanded_matched_all_real_rows": int(len(matched)) if matched is not None else None,
        "source_text_redistributed_by_public_package": False,
        "source_text_reconstructed_from_user_supplied_original_data": True,
        "files": [
            "experiment_bank_reconstructed.csv",
            "base_membership.csv",
            "additional_real_master_order.csv",
            "ratio_prefixes.csv",
            *(["matched_all_real_membership.csv"] if matched is not None else []),
        ],
    }
    (destination / "reconstruction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def discover_cells(public_data_root: Path) -> list[str]:
    if not public_data_root.is_dir():
        raise FileNotFoundError(public_data_root)
    cells = sorted(
        path.name
        for path in public_data_root.iterdir()
        if path.is_dir()
        and (path / "experiment_bank_index.csv").is_file()
        and (path / "repetitions.jsonl").is_file()
    )
    if not cells:
        raise RuntimeError(f"No reconstructable cell directories found under {public_data_root}")
    return cells


def main() -> None:
    args = parse_args()
    prepared_root = args.prepared_root.expanduser().resolve()
    public_data_root = args.public_data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    available = discover_cells(public_data_root)
    cell_ids = args.cell_ids or available
    unknown = sorted(set(cell_ids) - set(available))
    if unknown:
        raise SystemExit(
            f"Requested cell(s) not present in the public package: {', '.join(unknown)}"
        )

    reports: list[dict[str, Any]] = []
    for cell_id in cell_ids:
        report = reconstruct_cell(
            cell_id=cell_id,
            prepared_root=prepared_root,
            public_data_root=public_data_root,
            output_root=output_root,
            expand_ratios=args.expand_ratios,
            force=args.force,
        )
        reports.append(report)
        print(
            f"{cell_id}: verified {report['experiment_bank_rows']} bank rows; "
            f"{report['repetitions']} repetitions"
        )

    summary = {
        "public_data_root": str(public_data_root),
        "prepared_root": str(prepared_root),
        "output_root": str(output_root),
        "cells": reports,
    }
    (output_root / "reconstruction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Reconstruction complete: {output_root}")


if __name__ == "__main__":
    main()
