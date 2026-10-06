"""Shared utilities for the compact Reviewer-3 robustness experiments.

The primary sentiment experiment is immutable input to this module.  The
robustness protocol deliberately fixes r=1, w=1 and disables similarity
filtering instead of repeating the main combinatorial model-selection grid.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROBUSTNESS_VERSION = "r3-targeted-robustness-v1"
BASE_N = 280
FIXED_RATIO = 1.0
FIXED_SYNTHETIC_WEIGHT = 1.0
CANDIDATES_PER_SEED = 3
EXPECTED_REPEATS = 50
SELECTION_VERSION = "r3-parent-round-robin-sha256-v1"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    prepared_root: Path
    manifest_root: Path
    cells: tuple[str, ...]
    languages: Mapping[str, str]
    label_names: Mapping[int, str]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def task_spec(name: str) -> TaskSpec:
    root = repository_root()
    if name == "sentiment":
        cells = (
            "en_binary_sst2",
            "ko_binary_nsmc",
            "bn_binary_cinexdrama",
            "ha_binary_hausa_movie_review",
            "ml_binary_dravidian_codemix",
        )
        return TaskSpec(
            name="sentiment",
            prepared_root=root / "Data" / "Prepared",
            manifest_root=root / "Data" / "ExperimentManifests" / "v1",
            cells=cells,
            languages={
                "en_binary_sst2": "en",
                "ko_binary_nsmc": "ko",
                "bn_binary_cinexdrama": "bn",
                "ha_binary_hausa_movie_review": "ha",
                "ml_binary_dravidian_codemix": "ml",
            },
            label_names={0: "negative", 1: "positive"},
        )
    if name == "offensive":
        cells = (
            "en_binary_olid",
            "ko_binary_kold",
            "bn_binary_tb_olid",
            "ha_binary_hausahate",
            "ml_binary_dravidian_offensive",
        )
        return TaskSpec(
            name="offensive",
            prepared_root=root / "Data" / "Prepared" / "OffensiveLanguage",
            manifest_root=root / "Data" / "ExperimentManifests" / "OffensiveLanguage" / "v1",
            cells=cells,
            languages={
                "en_binary_olid": "en",
                "ko_binary_kold": "ko",
                "bn_binary_tb_olid": "bn",
                "ha_binary_hausahate": "ha",
                "ml_binary_dravidian_offensive": "ml",
            },
            label_names={0: "non-offensive", 1: "offensive"},
        )
    raise ValueError(f"unknown task: {name}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def verify_hash_lock(root: Path, lock_name: str) -> dict[str, Any]:
    """Verify a standard lock without mutating its directory."""
    lock_path = root / lock_name
    if not lock_path.is_file():
        raise RuntimeError(f"missing lock: {lock_path}")
    lock = read_json(lock_path)
    if not isinstance(lock, dict):
        raise RuntimeError(f"invalid lock object: {lock_path}")
    status = str(lock.get("status", "")).strip().lower()
    if status in {"failed", "running", "incomplete", "partial"}:
        raise RuntimeError(f"lock is not terminal/usable: {lock_path} (status={status})")
    files = lock.get("files")
    if isinstance(files, dict):
        mismatches: list[str] = []
        for relative, expected in files.items():
            path = root / str(relative)
            if not path.is_file() or sha256_file(path) != str(expected):
                mismatches.append(str(relative))
        if mismatches:
            raise RuntimeError(f"lock verification failed for {lock_path}: {mismatches[:10]}")
    return lock


PRIMARY_LOCKS: tuple[tuple[str, str], ...] = (
    (".", "BINARY_MATCHED_SIZE_EXPERIMENT_LOCK.json"),
    ("Generation", "GENERATION_LOCK.json"),
    ("Embeddings/Qwen3", "QWEN_EMBEDDING_LOCK.json"),
    ("Embeddings/BGE_M3", "BGE_EMBEDDING_LOCK.json"),
    ("Embeddings/Qwen3Eval", "QWEN_EVAL_EMBEDDING_LOCK.json"),
    ("Downstream/DevelopmentGrid", "DEVELOPMENT_GRID_LOCK.json"),
    ("Downstream/SyntheticWeightGrid", "SYNTHETIC_WEIGHT_GRID_LOCK.json"),
    ("Downstream/AdaptivePolicy", "BINARY_ADAPTIVE_PROTOCOL_LOCK.json"),
    ("Downstream/AdaptivePolicy/DecisionBoundaryDiagnostic", "ADAPTIVE_SELECTION_LOCK.json"),
    ("Downstream/AdaptivePolicy/OneShotTest", "PRETEST_THRESHOLD_LOCK.json"),
    ("Downstream/AdaptivePolicy/OneShotTest", "ADAPTIVE_ONE_SHOT_TEST_LOCK.json"),
    (
        "Downstream/AdaptivePolicy/OneShotTest/FinalThreeConditionFriedmanV1",
        "FRIEDMAN_ANALYSIS_LOCK.json",
    ),
)


def primary_lock_snapshot(primary_root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for relative_root, lock_name in PRIMARY_LOCKS:
        root = (primary_root / relative_root).resolve()
        verify_hash_lock(root, lock_name)
        lock_path = root / lock_name
        snapshot[lock_path.relative_to(primary_root.resolve()).as_posix()] = sha256_file(lock_path)
    return snapshot


def assert_primary_unchanged(primary_root: Path, before: Mapping[str, str]) -> None:
    after = primary_lock_snapshot(primary_root)
    if dict(before) != after:
        raise RuntimeError("primary sentiment lock fingerprint changed during robustness execution")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_bank(spec: TaskSpec, cell_id: str) -> dict[str, dict[str, Any]]:
    path = spec.manifest_root / cell_id / "experiment_bank.csv"
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row_id = str(raw["row_id"])
            label_value = float(str(raw["label"]).strip())
            if not math.isfinite(label_value) or not label_value.is_integer():
                raise RuntimeError(f"{path}: invalid label for row {row_id}: {raw['label']!r}")
            rows[row_id] = {
                "row_id": row_id,
                "text": str(raw["text"]),
                "label": int(label_value),
            }
    if len(rows) != 1400:
        raise RuntimeError(f"{cell_id}: expected 1400 bank rows, found {len(rows)}")
    return rows


def load_repetitions(spec: TaskSpec, cell_id: str) -> list[dict[str, Any]]:
    rows = read_jsonl(spec.manifest_root / cell_id / "repetitions.jsonl")
    if len(rows) != EXPECTED_REPEATS:
        raise RuntimeError(f"{cell_id}: expected {EXPECTED_REPEATS} repetitions, found {len(rows)}")
    return rows


def ratio_row(repeat: Mapping[str, Any], ratio: float = FIXED_RATIO) -> dict[str, Any]:
    rows = repeat.get("ratios")
    if isinstance(rows, dict):
        for value in rows.values():
            if math.isclose(float(value["ratio"]), ratio, rel_tol=0.0, abs_tol=1e-12):
                return dict(value)
    elif isinstance(rows, list):
        for value in rows:
            if math.isclose(float(value["ratio"]), ratio, rel_tol=0.0, abs_tol=1e-12):
                return dict(value)
    raise RuntimeError(f"repeat {repeat.get('repeat_seed')}: missing ratio {ratio}")


def load_valid_generation(
    generation_root: Path,
    cells: Sequence[str],
    candidate_limit: int | None = CANDIDATES_PER_SEED,
) -> dict[str, dict[int, dict[str, Any]]]:
    verify_hash_lock(generation_root, "GENERATION_LOCK.json")
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for cell_id in cells:
        rows: dict[int, dict[str, Any]] = {}
        folder = generation_root / "valid_outputs" / cell_id
        for path in sorted(folder.glob("part_*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    candidate_index = int(row["candidate_index"])
                    if candidate_limit is not None and candidate_index >= candidate_limit:
                        continue
                    index = int(row["global_generation_index"])
                    if index in rows:
                        raise RuntimeError(f"{cell_id}: duplicate generation index {index}")
                    rows[index] = row
        result[cell_id] = rows
    return result


def candidate_order_hash(task: str, cell_id: str, repeat_seed: int, label: int, index: int) -> str:
    payload = "|".join(
        (SELECTION_VERSION, task, cell_id, str(repeat_seed), str(label), str(index))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_fixed_synthetic(
    spec: TaskSpec,
    cell_id: str,
    repeat: Mapping[str, Any],
    records: Mapping[int, Mapping[str, Any]],
) -> list[int]:
    base_order = [str(value) for value in repeat["base_row_ids"]]
    base_set = set(base_order)
    target = {
        int(label): int(count)
        for label, count in ratio_row(repeat)["synthetic_target_label_vector"].items()
    }
    grouped: dict[int, dict[str, list[int]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }
    for index, row in records.items():
        parent = str(row["real_row_id"])
        if parent not in base_set:
            continue
        label = int(row["label_int"])
        grouped[label][parent].append(int(index))

    selected_by_label: dict[int, list[int]] = {0: [], 1: []}
    repeat_seed = int(repeat["repeat_seed"])
    for label in (0, 1):
        queues: dict[str, list[int]] = {}
        for parent in base_order:
            values = grouped[label].get(parent, [])
            values = sorted(
                values,
                key=lambda index: (
                    candidate_order_hash(spec.name, cell_id, repeat_seed, label, index),
                    index,
                ),
            )
            if values:
                queues[parent] = values
        offsets = {parent: 0 for parent in queues}
        order: list[int] = []
        while True:
            added = 0
            for parent in base_order:
                values = queues.get(parent)
                if values is None:
                    continue
                offset = offsets[parent]
                if offset < len(values):
                    order.append(values[offset])
                    offsets[parent] = offset + 1
                    added += 1
            if added == 0:
                break
        if len(order) < target[label]:
            raise RuntimeError(
                f"{cell_id}/{repeat_seed}: insufficient label-{label} synthetic candidates "
                f"({len(order)} < {target[label]})"
            )
        selected_by_label[label] = order[: target[label]]
    selected = selected_by_label[0] + selected_by_label[1]
    if len(selected) != BASE_N:
        raise RuntimeError(f"{cell_id}/{repeat_seed}: expected {BASE_N} synthetic rows")
    if len(set(selected)) != len(selected):
        raise RuntimeError(f"{cell_id}/{repeat_seed}: duplicate synthetic selection")
    return selected


def fixed_condition_rows(
    spec: TaskSpec,
    cell_id: str,
    repeat: Mapping[str, Any],
    bank: Mapping[str, Mapping[str, Any]],
    synthetic_records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    base_ids = [str(value) for value in repeat["base_row_ids"]]
    if len(base_ids) != BASE_N:
        raise RuntimeError(f"{cell_id}: invalid Base size")
    additional_ids = [str(value) for value in repeat["additional_real_master_order"][:BASE_N]]
    selected = select_fixed_synthetic(spec, cell_id, repeat, synthetic_records)
    base_labels = [int(bank[row_id]["label"]) for row_id in base_ids]
    additional_labels = [int(bank[row_id]["label"]) for row_id in additional_ids]
    synthetic_labels = [int(synthetic_records[index]["label_int"]) for index in selected]
    if Counter(additional_labels) != Counter(synthetic_labels):
        raise RuntimeError(f"{cell_id}/{repeat['repeat_seed']}: Hybrid/Matched label vector mismatch")
    return {
        "base_ids": base_ids,
        "matched_ids": base_ids + additional_ids,
        "selected_synthetic_indices": selected,
        "base_labels": base_labels,
        "matched_labels": base_labels + additional_labels,
        "hybrid_labels": base_labels + synthetic_labels,
    }


def load_split(spec: TaskSpec, cell_id: str, split: str) -> list[dict[str, Any]]:
    path = spec.prepared_root / cell_id / f"{split}.csv"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise RuntimeError(f"{path}: missing text/label columns")
        for index, raw in enumerate(reader):
            label_value = float(str(raw["label"]).strip())
            if not math.isfinite(label_value) or not label_value.is_integer():
                raise RuntimeError(f"{path}: non-binary label at row {index}: {raw['label']!r}")
            rows.append({
                "split_row_index": index,
                "text": str(raw["text"]),
                "label": int(label_value),
            })
    return rows


def paired_bootstrap_ci(values: Sequence[float], n_resamples: int = 10_000, seed: int = 20260713) -> tuple[float, float, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, 1000):
        stop = min(start + 1000, n_resamples)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(array.mean()), float(lower), float(upper)


def lock_file_map(root: Path, excluded: Iterable[str] = ()) -> dict[str, str]:
    excluded_set = set(excluded)
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded_set:
            continue
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result

