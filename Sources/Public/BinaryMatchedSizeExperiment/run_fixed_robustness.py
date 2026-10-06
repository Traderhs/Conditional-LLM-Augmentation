"""Run the fixed r=1, w=1 Base/Hybrid/Matched robustness comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from robustness_common import (  # noqa: E402
    BASE_N,
    EXPECTED_REPEATS,
    FIXED_RATIO,
    FIXED_SYNTHETIC_WEIGHT,
    ROBUSTNESS_VERSION,
    fixed_condition_rows,
    load_bank,
    load_repetitions,
    load_split,
    load_valid_generation,
    lock_file_map,
    paired_bootstrap_ci,
    sha256_file,
    task_spec,
    verify_hash_lock,
    write_csv,
    write_json,
)


LOCK_NAME = "FIXED_ROBUSTNESS_LOCK.json"
RESULT_FIELDS = (
    "arm",
    "task",
    "cell_id",
    "repeat_seed",
    "condition",
    "train_n",
    "macro_f1",
    "auroc",
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def artifact_lock_hash(root: Path) -> str:
    for name in (
        "ROBUSTNESS_EMBEDDING_LOCK.json",
        "QWEN_EMBEDDING_LOCK.json",
        "BGE_EMBEDDING_LOCK.json",
        "QWEN_EVAL_EMBEDDING_LOCK.json",
    ):
        path = root / name
        if path.is_file():
            verify_hash_lock(root, name)
            return sha256_file(path)
    raise RuntimeError(f"no supported embedding lock found in {root}")


def input_fingerprints(
    spec: Any,
    generation_root: Path,
    real_root: Path,
    synthetic_root: Path,
    test_root: Path,
) -> dict[str, str]:
    verify_hash_lock(generation_root, "GENERATION_LOCK.json")
    return {
        "generation": sha256_file(generation_root / "GENERATION_LOCK.json"),
        "real_embedding": artifact_lock_hash(real_root),
        "synthetic_embedding": artifact_lock_hash(synthetic_root),
        "test_embedding": artifact_lock_hash(test_root),
        "manifest": sha256_file(spec.manifest_root / "MANIFEST_LOCK.json"),
    }


def load_train_embedding(root: Path, cell_id: str) -> dict[str, Any]:
    import numpy as np

    robust_lock = root / "ROBUSTNESS_EMBEDDING_LOCK.json"
    if robust_lock.is_file():
        verify_hash_lock(root, robust_lock.name)
    else:
        for lock_name in ("QWEN_EMBEDDING_LOCK.json", "BGE_EMBEDDING_LOCK.json"):
            if (root / lock_name).is_file():
                verify_hash_lock(root, lock_name)
                break
    rows = read_manifest(root / f"{cell_id}_manifest.csv")
    array = np.load(root / f"{cell_id}.npy", mmap_mode="r")
    real: dict[str, int] = {}
    synthetic: dict[int, int] = {}
    for row in rows:
        index = int(row["embedding_row_index"])
        record_type = str(row.get("record_type") or "")
        if record_type == "real":
            real[str(row["real_row_id"])] = index
        elif record_type == "synthetic":
            synthetic[int(row["global_generation_index"])] = index
    return {"array": array, "real": real, "synthetic": synthetic}


def load_test_embedding(root: Path, cell_id: str) -> tuple[Any, Any]:
    import numpy as np

    robust_lock = root / "ROBUSTNESS_EMBEDDING_LOCK.json"
    if robust_lock.is_file():
        verify_hash_lock(root, robust_lock.name)
        rows = read_manifest(root / f"{cell_id}_manifest.csv")
        array = np.load(root / f"{cell_id}.npy", mmap_mode="r")
        ordered = sorted(rows, key=lambda row: int(row["split_row_index"]))
        indices = np.asarray([int(row["embedding_row_index"]) for row in ordered], dtype=np.int64)
        labels = np.asarray([int(row["label"]) for row in ordered], dtype=np.int64)
        return np.asarray(array[indices], dtype=np.float32), labels

    # Primary Qwen eval layout.
    lock_path = root / "QWEN_EVAL_EMBEDDING_LOCK.json"
    if lock_path.is_file():
        verify_hash_lock(root, lock_path.name)
    manifest = root / f"{cell_id}_test_manifest.csv"
    array_path = root / f"{cell_id}_test.npy"
    rows = read_manifest(manifest)
    array = np.load(array_path, mmap_mode="r")
    ordered = sorted(rows, key=lambda row: int(row["eval_row_index"]))
    labels = np.asarray([int(row["label_int"]) for row in ordered], dtype=np.int64)
    return np.asarray(array, dtype=np.float32), labels


def fit_metrics(x_train: Any, y_train: Any, x_test: Any, y_test: Any, seed: int) -> tuple[float, float]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-4,
        fit_intercept=True,
        class_weight=None,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    macro_f1 = float(f1_score(y_test, pred, labels=[0, 1], average="macro", zero_division=0))
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(x_test)[:, positive_column]
    auroc = float(roc_auc_score(y_test, probabilities))
    return macro_f1, auroc


def condition_vectors(
    condition: str,
    fixed: Mapping[str, Any],
    real_embeddings: Mapping[str, Any],
    synthetic_embeddings: Mapping[str, Any],
    synthetic_records: Mapping[int, Mapping[str, Any]],
    bank: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, Any]:
    import numpy as np

    real_array = real_embeddings["array"]
    synth_array = synthetic_embeddings["array"]
    if condition == "Base":
        real_ids = list(fixed["base_ids"])
        vectors = [np.asarray(real_array[real_embeddings["real"][row_id]], dtype=np.float32) for row_id in real_ids]
        labels = list(fixed["base_labels"])
    elif condition == "Matched All-real":
        real_ids = list(fixed["matched_ids"])
        vectors = [np.asarray(real_array[real_embeddings["real"][row_id]], dtype=np.float32) for row_id in real_ids]
        labels = list(fixed["matched_labels"])
    elif condition == "Hybrid":
        real_ids = list(fixed["base_ids"])
        vectors = [np.asarray(real_array[real_embeddings["real"][row_id]], dtype=np.float32) for row_id in real_ids]
        for generation_index in fixed["selected_synthetic_indices"]:
            embedding_index = synthetic_embeddings["synthetic"].get(int(generation_index))
            if embedding_index is None:
                raise RuntimeError(f"missing synthetic embedding {generation_index}")
            vectors.append(np.asarray(synth_array[embedding_index], dtype=np.float32))
        labels = list(fixed["hybrid_labels"])
    else:
        raise ValueError(condition)
    if len(vectors) != len(labels):
        raise RuntimeError(f"{condition}: vector/label count mismatch")
    return np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def build_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    cells = sorted({str(row["cell_id"]) for row in rows})
    for cell_id in cells:
        cell_rows = [row for row in rows if row["cell_id"] == cell_id]
        by_key = {
            (int(row["repeat_seed"]), str(row["condition"])): row
            for row in cell_rows
        }
        for condition in ("Hybrid", "Matched All-real"):
            for metric in ("macro_f1", "auroc"):
                deltas = [
                    float(by_key[(seed, condition)][metric]) - float(by_key[(seed, "Base")][metric])
                    for seed in range(1000, 1050)
                ]
                mean, lower, upper = paired_bootstrap_ci(deltas)
                summary.append(
                    {
                        "cell_id": cell_id,
                        "comparison": f"{condition} - Base",
                        "metric": metric,
                        "mean_difference": mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )
        for metric in ("macro_f1", "auroc"):
            deltas = [
                float(by_key[(seed, "Matched All-real")][metric])
                - float(by_key[(seed, "Hybrid")][metric])
                for seed in range(1000, 1050)
            ]
            mean, lower, upper = paired_bootstrap_ci(deltas)
            summary.append(
                {
                    "cell_id": cell_id,
                    "comparison": "Matched All-real - Hybrid",
                    "metric": metric,
                    "mean_difference": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return summary


def validate_existing(root: Path, arm: str, task: str, current_inputs: Mapping[str, str]) -> None:
    lock = verify_hash_lock(root, LOCK_NAME)
    config = lock.get("configuration") or {}
    if config.get("arm") != arm or config.get("task") != task:
        raise RuntimeError("existing fixed-robustness lock belongs to a different arm/task")
    if config.get("robustness_version") != ROBUSTNESS_VERSION:
        raise RuntimeError("existing fixed-robustness lock has a different protocol version")
    if config.get("input_lock_sha256") != dict(current_inputs):
        raise RuntimeError("existing fixed-robustness lock no longer matches its input artifacts")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--task", choices=("sentiment", "offensive"), required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--real-embedding-root", type=Path, required=True)
    parser.add_argument("--synthetic-embedding-root", type=Path, required=True)
    parser.add_argument("--test-embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = task_spec(str(args.task))
    generation_root = args.generation_root.resolve()
    real_root = args.real_embedding_root.resolve()
    synthetic_root = args.synthetic_embedding_root.resolve()
    test_root = args.test_embedding_root.resolve()
    output_root = args.output_root.resolve()
    plan = {
        "arm": args.arm,
        "task": spec.name,
        "robustness_version": ROBUSTNESS_VERSION,
        "ratio": FIXED_RATIO,
        "synthetic_weight": FIXED_SYNTHETIC_WEIGHT,
        "repetitions": EXPECTED_REPEATS,
        "planned_fits": len(spec.cells) * 50 * 3,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run and not (generation_root / "GENERATION_LOCK.json").is_file():
        print("selection feasibility deferred until compact generation is complete")
        return 0
    synthetic_records = load_valid_generation(generation_root, spec.cells)
    if args.dry_run:
        # Selection feasibility is intentionally checked even in dry-run mode.
        for cell_id in spec.cells:
            bank = load_bank(spec, cell_id)
            for repeat in load_repetitions(spec, cell_id):
                fixed_condition_rows(spec, cell_id, repeat, bank, synthetic_records[cell_id])
        return 0
    lock_path = output_root / LOCK_NAME
    current_inputs = input_fingerprints(spec, generation_root, real_root, synthetic_root, test_root)
    if lock_path.is_file():
        validate_existing(output_root, str(args.arm), spec.name, current_inputs)
        print(f"SKIP fixed robustness: verified completed lock at {lock_path}")
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"non-empty robustness result root without lock; refusing overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for cell_id in spec.cells:
        bank = load_bank(spec, cell_id)
        repetitions = load_repetitions(spec, cell_id)
        real_embeddings = load_train_embedding(real_root, cell_id)
        synthetic_embeddings = (
            real_embeddings if synthetic_root == real_root else load_train_embedding(synthetic_root, cell_id)
        )
        x_test, y_test = load_test_embedding(test_root, cell_id)
        for repeat in repetitions:
            repeat_seed = int(repeat["repeat_seed"])
            fixed = fixed_condition_rows(spec, cell_id, repeat, bank, synthetic_records[cell_id])
            for condition in ("Base", "Hybrid", "Matched All-real"):
                x_train, y_train = condition_vectors(
                    condition,
                    fixed,
                    real_embeddings,
                    synthetic_embeddings,
                    synthetic_records[cell_id],
                    bank,
                )
                macro_f1, auroc = fit_metrics(x_train, y_train, x_test, y_test, repeat_seed)
                rows.append(
                    {
                        "arm": str(args.arm),
                        "task": spec.name,
                        "cell_id": cell_id,
                        "repeat_seed": repeat_seed,
                        "condition": condition,
                        "train_n": len(y_train),
                        "macro_f1": macro_f1,
                        "auroc": auroc,
                    }
                )
    expected = len(spec.cells) * 50 * 3
    if len(rows) != expected:
        raise RuntimeError(f"fixed robustness produced {len(rows)} rows, expected {expected}")
    summary = build_summary(rows)
    write_csv(output_root / "repeat_results.csv", rows, RESULT_FIELDS)
    write_csv(
        output_root / "paired_summary.csv",
        summary,
        ("cell_id", "comparison", "metric", "mean_difference", "ci_lower", "ci_upper"),
    )
    configuration = {
        **plan,
        "classifier": "sklearn.linear_model.LogisticRegression",
        "classifier_C": 1.0,
        "evaluation_split": "test",
        "selection": "unfiltered parent-round-robin; three candidates per seed maximum",
        "input_lock_sha256": current_inputs,
    }
    write_json(output_root / "report.json", {"configuration": configuration, "summary": summary})
    write_json(
        lock_path,
        {
            "lock_version": "r3-fixed-robustness-v1",
            "status": "completed",
            "configuration": configuration,
            "files": lock_file_map(output_root, excluded={LOCK_NAME}),
        },
    )
    validate_existing(output_root, str(args.arm), spec.name, current_inputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

