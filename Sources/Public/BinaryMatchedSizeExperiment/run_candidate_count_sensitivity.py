"""Reviewer-1 candidate-count sensitivity without post-hoc policy retuning.

The primary experiment generated 20 independent requests per real seed. This
analysis treats the number of available candidates, K, as an explicit
experimental variable over every integer K=1,...,20. For K, only generation
records with candidate_index < K are visible.

The sensitivity analysis preserves the frozen primary study. It does not
regenerate text, reselect the final augmentation policy, or access the test
split. Instead it evaluates candidate availability, feasibility for all 17
original similarity-selection conditions and all 11 augmentation ratios, and
unfiltered Hybrid development performance for every feasible K/ratio/dataset/
repetition cell under the original 50 paired repetitions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

import run_binary_downstream_experiment as development  # noqa: E402
from robustness_common import primary_lock_snapshot, sha256_file  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_ROOT = PROJECT_ROOT / "Results" / "BinaryMatchedSizeExperiment"
MANIFEST_ROOT = PROJECT_ROOT / "Data" / "ExperimentManifests" / "v1"
PREPARED_ROOT = PROJECT_ROOT / "Data" / "Prepared"
OUTPUT_ROOT = PRIMARY_ROOT / "CandidateCountSensitivity" / "v3"
DATABASE_PATH = OUTPUT_ROOT / "candidate_count_status.sqlite3"
LOCK_NAME = "CANDIDATE_COUNT_SENSITIVITY_LOCK.json"
LOCK_VERSION = "candidate-count-sensitivity-v3"
CANDIDATE_COUNTS = tuple(range(1, 21))
REFERENCE_COUNT = 20
BOOTSTRAP_REPS = 10_000
STATISTICS_VERSION = "candidate-count-statistics-v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_completed_lock() -> None:
    path = OUTPUT_ROOT / LOCK_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    lock = json.loads(path.read_text(encoding="utf-8-sig"))
    if lock.get("status") != "completed":
        raise RuntimeError(f"incomplete sensitivity lock: {path}")
    for relative, expected in (lock.get("files") or {}).items():
        target = OUTPUT_ROOT / str(relative)
        if not target.is_file():
            raise FileNotFoundError(target)
        actual = sha256_file(target)
        if actual != str(expected):
            raise RuntimeError(
                f"candidate-count artifact hash mismatch: {target}: "
                f"expected={expected}, actual={actual}"
            )


def candidate_index_maps(
    valid_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for cell_id, rows in valid_by_cell.items():
        mapping: dict[int, int] = {}
        for row in rows:
            global_index = int(row["global_generation_index"])
            candidate_index = int(row["candidate_index"])
            if candidate_index not in range(REFERENCE_COUNT):
                raise RuntimeError(
                    f"{cell_id}/{global_index}: candidate_index={candidate_index} outside 0..19"
                )
            if global_index in mapping:
                raise RuntimeError(f"{cell_id}: duplicate generation index {global_index}")
            mapping[global_index] = candidate_index
        result[cell_id] = mapping
    return result


def filter_data(
    data: Mapping[str, Mapping[str, Any]],
    index_maps: Mapping[str, Mapping[int, int]],
    candidate_count: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for cell_id, cell in data.items():
        current = dict(cell)
        records = {
            int(global_index): record
            for global_index, record in cell["synthetic_records"].items()
            if int(index_maps[cell_id][int(global_index)]) < candidate_count
        }
        current["synthetic_records"] = records
        current["valid_synthetic_count"] = len(records)
        result[cell_id] = current
    return result


def availability_rows(
    valid_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_count in CANDIDATE_COUNTS:
        for cell_id in development.TARGET_CELLS:
            valid = sum(
                int(row["candidate_index"]) < candidate_count
                for row in valid_by_cell[cell_id]
            )
            planned = 1_400 * candidate_count
            rows.append(
                {
                    "candidate_count": candidate_count,
                    "cell_id": cell_id,
                    "planned_requests": planned,
                    "valid_candidates": valid,
                    "invalid_candidates": planned - valid,
                    "valid_rate": valid / planned,
                }
            )
    return rows


def similarity_feasibility_rows(
    repetitions: Mapping[str, Sequence[Mapping[str, Any]]],
    full_data: Mapping[str, Mapping[str, Any]],
    index_maps: Mapping[str, Mapping[int, int]],
) -> list[dict[str, Any]]:
    """Evaluate K x similarity-band x ratio feasibility without model fitting."""
    conditions = development.SIMILARITY_CONDITIONS
    aggregate: dict[tuple[int, str, str, float], dict[str, float]] = defaultdict(
        lambda: {
            "planned": 0.0,
            "feasible": 0.0,
            "eligible_total_sum": 0.0,
            "eligible_label_0_sum": 0.0,
            "eligible_label_1_sum": 0.0,
        }
    )
    for cell_id in development.TARGET_CELLS:
        records = list(full_data[cell_id]["synthetic_records"].values())
        index_map = index_maps[cell_id]
        for repeat in repetitions[cell_id]:
            base_set = set(str(value) for value in repeat["base_row_ids"])
            hist = np.zeros((len(conditions), 2, REFERENCE_COUNT), dtype=np.int32)
            for record in records:
                if str(record["real_row_id"]) not in base_set:
                    continue
                label = int(record["label_int"])
                candidate_index = int(index_map[int(record["global_generation_index"])])
                cosine = float(record["cosine_similarity"])
                hist[0, label, candidate_index] += 1
                for condition_index, condition in enumerate(conditions[1:], 1):
                    if float(condition["lower"]) <= cosine <= float(condition["upper"]):
                        hist[condition_index, label, candidate_index] += 1
            cumulative = np.cumsum(hist, axis=2)
            for candidate_count in CANDIDATE_COUNTS:
                available = cumulative[:, :, candidate_count - 1]
                for condition_index, condition in enumerate(conditions):
                    a0 = int(available[condition_index, 0])
                    a1 = int(available[condition_index, 1])
                    for ratio in development.RATIOS:
                        ratio_row = repeat["ratios"][development.ratio_key(ratio)]
                        target = {
                            int(label): int(count)
                            for label, count in ratio_row[
                                "synthetic_target_label_vector"
                            ].items()
                        }
                        feasible = a0 >= target[0] and a1 >= target[1]
                        key = (
                            candidate_count,
                            cell_id,
                            str(condition["similarity_condition_id"]),
                            float(ratio),
                        )
                        item = aggregate[key]
                        item["planned"] += 1
                        item["feasible"] += int(feasible)
                        item["eligible_total_sum"] += a0 + a1
                        item["eligible_label_0_sum"] += a0
                        item["eligible_label_1_sum"] += a1
    rows: list[dict[str, Any]] = []
    for key in sorted(
        aggregate,
        key=lambda x: (x[0], x[1], x[2] != "OFF", x[2], x[3]),
    ):
        candidate_count, cell_id, condition_id, ratio = key
        item = aggregate[key]
        planned = int(item["planned"])
        feasible = int(item["feasible"])
        rows.append(
            {
                "candidate_count": candidate_count,
                "cell_id": cell_id,
                "similarity_condition_id": condition_id,
                "ratio": ratio,
                "planned_repetitions": planned,
                "feasible_repetitions": feasible,
                "infeasible_repetitions": planned - feasible,
                "feasible_rate": feasible / planned,
                "mean_eligible_candidates": item["eligible_total_sum"] / planned,
                "mean_eligible_label_0": item["eligible_label_0_sum"] / planned,
                "mean_eligible_label_1": item["eligible_label_1_sum"] / planned,
            }
        )
    return rows


def unfiltered_tasks(
    candidate_count: int,
    repetitions: Mapping[str, Sequence[Mapping[str, Any]]],
    data: Mapping[str, Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    condition = next(
        item
        for item in development.SIMILARITY_CONDITIONS
        if str(item["similarity_condition_id"]) == "OFF"
    )
    for cell_id in development.TARGET_CELLS:
        real_indices = data[cell_id]["real_indices"]
        bank_labels = data[cell_id]["bank_labels"]
        synthetic_records = data[cell_id]["synthetic_records"]
        for repeat in repetitions[cell_id]:
            repeat_seed = int(repeat["repeat_seed"])
            base_ids = [str(value) for value in repeat["base_row_ids"]]
            base_indices = [int(real_indices[row_id]) for row_id in base_ids]
            base_labels = [int(bank_labels[row_id]) for row_id in base_ids]
            orders, eligible_vector = development.synthetic_master_orders(
                cell_id,
                repeat,
                condition,
                synthetic_records,
            )
            for ratio in development.RATIOS:
                ratio_row = repeat["ratios"][development.ratio_key(ratio)]
                selection = development.select_hybrid(
                    cell_id,
                    repeat,
                    ratio_row,
                    condition,
                    orders,
                    eligible_vector,
                    synthetic_records,
                )
                task = {
                    "candidate_count": candidate_count,
                    "condition_key": (
                        f"K{candidate_count:02d}|{cell_id}|{repeat_seed}|"
                        f"{development.ratio_key(ratio)}"
                    ),
                    "cell_id": cell_id,
                    "repeat_seed": repeat_seed,
                    "ratio": float(ratio),
                    "target_n": int(ratio_row["n_add"]),
                    "eligible_n": int(selection["eligible_n"]),
                    "eligible_label_vector": selection["eligible_label_vector"],
                    "feasible": bool(selection["feasible"]),
                    "infeasible_reason": selection["infeasible_reason"],
                    "selection_sha256": str(selection["selection_sha256"]),
                    "selected_n": int(selection["selected_n"]),
                    "selected_parent_count": int(selection["selected_parent_count"]),
                    "run_fit": bool(selection["feasible"]),
                }
                if selection["feasible"]:
                    selected = selection["selected_global_generation_indices"]
                    synthetic_indices = [
                        int(synthetic_records[index]["qwen_embedding_row_index"])
                        for index in selected
                    ]
                    synthetic_labels = [
                        int(synthetic_records[index]["label_int"])
                        for index in selected
                    ]
                    task["train_indices"] = base_indices + synthetic_indices
                    task["train_labels"] = base_labels + synthetic_labels
                yield task


def result_cache_key(task: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(task["cell_id"]),
        int(task["repeat_seed"]),
        float(task["ratio"]),
        str(task["selection_sha256"]),
    )


def create_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            candidate_count INTEGER NOT NULL,
            cell_id TEXT NOT NULL,
            repeat_seed INTEGER NOT NULL,
            ratio REAL NOT NULL,
            target_n INTEGER NOT NULL,
            eligible_n INTEGER NOT NULL,
            eligible_label_vector_json TEXT NOT NULL,
            feasible INTEGER NOT NULL,
            infeasible_reason TEXT,
            selection_sha256 TEXT NOT NULL,
            selected_n INTEGER NOT NULL,
            selected_parent_count INTEGER NOT NULL,
            macro_f1 REAL,
            auroc REAL,
            auroc_reason TEXT,
            status TEXT NOT NULL,
            fit_seconds REAL,
            n_iter INTEGER,
            converged INTEGER,
            convergence_warning TEXT,
            exception_type TEXT,
            exception_message TEXT,
            result_source TEXT NOT NULL,
            completed_at REAL NOT NULL,
            PRIMARY KEY(candidate_count, cell_id, repeat_seed, ratio)
        )
        """
    )
    return connection


def terminal_keys(connection: sqlite3.Connection) -> set[tuple[int, str, int, float]]:
    return {
        (int(k), str(cell), int(seed), float(ratio))
        for k, cell, seed, ratio in connection.execute(
            "SELECT candidate_count,cell_id,repeat_seed,ratio FROM results "
            "WHERE status IN ('completed','completed_with_warning','infeasible')"
        )
    }


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def row_result(row: Mapping[str, Any]) -> dict[str, Any]:
    converged = clean_value(row.get("converged"))
    return {
        "status": str(row.get("status") or "completed"),
        "fit_seconds": 0.0,
        "n_iter": (
            None
            if clean_value(row.get("n_iter")) is None
            else int(float(row["n_iter"]))
        ),
        "converged": (
            None
            if converged is None
            else str(converged).strip().lower() in {"true", "1", "yes"}
        ),
        "convergence_warning": clean_value(row.get("convergence_warning")),
        "exception_type": clean_value(row.get("exception_type")),
        "exception_message": clean_value(row.get("exception_message")),
        "macro_f1": clean_value(row.get("macro_f1")),
        "auroc": clean_value(row.get("auroc")),
        "auroc_reason": clean_value(row.get("auroc_reason")),
    }


def commit_result(
    connection: sqlite3.Connection,
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    result_source: str,
) -> None:
    with connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO results(
                candidate_count,cell_id,repeat_seed,ratio,target_n,eligible_n,
                eligible_label_vector_json,feasible,infeasible_reason,
                selection_sha256,selected_n,selected_parent_count,
                macro_f1,auroc,auroc_reason,status,fit_seconds,n_iter,converged,
                convergence_warning,exception_type,exception_message,
                result_source,completed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                int(task["candidate_count"]),
                str(task["cell_id"]),
                int(task["repeat_seed"]),
                float(task["ratio"]),
                int(task["target_n"]),
                int(task["eligible_n"]),
                json.dumps(
                    task["eligible_label_vector"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                int(bool(task["feasible"])),
                task.get("infeasible_reason"),
                str(task["selection_sha256"]),
                int(task["selected_n"]),
                int(task["selected_parent_count"]),
                result.get("macro_f1"),
                result.get("auroc"),
                result.get("auroc_reason"),
                str(result["status"]),
                result.get("fit_seconds"),
                result.get("n_iter"),
                (
                    None
                    if result.get("converged") is None
                    else int(bool(result["converged"]))
                ),
                result.get("convergence_warning"),
                result.get("exception_type"),
                result.get("exception_message"),
                result_source,
                time.time(),
            ),
        )


def infeasible_result() -> dict[str, Any]:
    return {
        "status": "infeasible",
        "fit_seconds": 0.0,
        "n_iter": None,
        "converged": None,
        "convergence_warning": None,
        "exception_type": None,
        "exception_message": None,
        "macro_f1": None,
        "auroc": None,
        "auroc_reason": None,
    }


def seed_primary_k20(
    connection: sqlite3.Connection,
    cache: dict[tuple[Any, ...], dict[str, Any]],
) -> None:
    root = PRIMARY_ROOT / "Downstream" / "DevelopmentGrid"
    hybrid = pd.read_csv(root / "hybrid_results.csv")
    hybrid = hybrid[hybrid["similarity_condition_id"] == "OFF"].copy()
    paired = pd.read_csv(root / "paired_results.csv")
    paired = paired[paired["similarity_condition_id"] == "OFF"].copy()
    hybrid_map = {
        (str(row.cell_id), int(row.repeat_seed), float(row.ratio)): row._asdict()
        for row in hybrid.itertuples(index=False)
    }
    existing = terminal_keys(connection)
    for row in paired.itertuples(index=False):
        key = (
            REFERENCE_COUNT,
            str(row.cell_id),
            int(row.repeat_seed),
            float(row.ratio),
        )
        hrow = hybrid_map[(key[1], key[2], key[3])]
        feasible = str(row.feasible).strip().lower() == "true"
        task = {
            "candidate_count": REFERENCE_COUNT,
            "cell_id": key[1],
            "repeat_seed": key[2],
            "ratio": key[3],
            "target_n": int(row.n_add),
            "eligible_n": int(row.eligible_synthetic_n),
            "eligible_label_vector": {
                "0": int(row.additional_label_0_n),
                "1": int(row.additional_label_1_n),
            },
            "feasible": feasible,
            "infeasible_reason": (
                None
                if feasible or pd.isna(row.infeasible_reason)
                else str(row.infeasible_reason)
            ),
            "selection_sha256": str(row.selection_sha256),
            "selected_n": int(row.selected_synthetic_n),
            "selected_parent_count": int(row.selected_parent_count),
        }
        if key not in existing:
            result = row_result(hrow) if feasible else infeasible_result()
            commit_result(connection, task, result, "primary_k20_reuse")
        if feasible and str(hrow["status"]) in {"completed", "completed_with_warning"}:
            cache[result_cache_key(task)] = row_result(hrow)


def recover_cache(
    connection: sqlite3.Connection,
    cache: dict[tuple[Any, ...], dict[str, Any]],
) -> None:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT * FROM results WHERE status IN ('completed','completed_with_warning')"
    ).fetchall()
    connection.row_factory = None
    for raw in rows:
        row = dict(raw)
        cache[
            (
                str(row["cell_id"]),
                int(row["repeat_seed"]),
                float(row["ratio"]),
                str(row["selection_sha256"]),
            )
        ] = row_result(row)


def initialize_shared_cell(
    data: Mapping[str, Mapping[str, Any]],
    cell_id: str,
) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    cell = data[cell_id]
    dev = np.load(cell["dev_path"], mmap_mode="r")
    dev_shared = np.ascontiguousarray(dev, dtype=np.float64)
    development._WORKER_CONTEXT = {
        cell_id: {
            "qwen": np.load(cell["qwen_path"], mmap_mode="r"),
            "dev": dev_shared,
            "dev_labels": np.asarray(cell["dev_labels"], dtype=np.int64),
        }
    }


def clear_shared_cell() -> None:
    development._WORKER_CONTEXT = {}
    gc.collect()


def execute_pending(
    connection: sqlite3.Connection,
    data: Mapping[str, Mapping[str, Any]],
    pending_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    cache: dict[tuple[Any, ...], dict[str, Any]],
    *,
    workers: int,
    progress_every: int,
    candidate_count: int,
) -> None:
    completed = 0
    started = time.perf_counter()
    for cell_id in development.TARGET_CELLS:
        tasks = list(pending_by_cell.get(cell_id, ()))
        if not tasks:
            continue
        initialize_shared_cell(data, cell_id)
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                iterator = iter(tasks)
                futures: dict[Any, Mapping[str, Any]] = {}
                exhausted = False
                while not exhausted or futures:
                    while not exhausted and len(futures) < workers * 2:
                        try:
                            task = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                        futures[executor.submit(development.execute_fit, task)] = task
                    if futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            task = futures.pop(future)
                            result = future.result()
                            if result.get("status") == "failed":
                                raise RuntimeError(
                                    f"K={candidate_count} fit failed for "
                                    f"{task['condition_key']}: "
                                    f"{result.get('exception_type')}: "
                                    f"{result.get('exception_message')}"
                                )
                            commit_result(
                                connection,
                                task,
                                result,
                                "new_candidate_count_fit",
                            )
                            cache[result_cache_key(task)] = dict(result)
                            completed += 1
                            if completed % progress_every == 0:
                                elapsed = time.perf_counter() - started
                                print(
                                    f"K={candidate_count} | new fits={completed} | "
                                    f"elapsed_seconds={elapsed:.1f}",
                                    flush=True,
                                )
        finally:
            clear_shared_cell()


def run_unfiltered_sensitivity(
    connection: sqlite3.Connection,
    repetitions: Mapping[str, Sequence[Mapping[str, Any]]],
    full_data: Mapping[str, Mapping[str, Any]],
    index_maps: Mapping[str, Mapping[int, int]],
    *,
    workers: int,
    progress_every: int,
) -> None:
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    seed_primary_k20(connection, cache)
    recover_cache(connection, cache)
    terminal = terminal_keys(connection)
    for candidate_count in range(REFERENCE_COUNT - 1, 0, -1):
        data = filter_data(full_data, index_maps, candidate_count)
        pending_by_cell: dict[str, list[dict[str, Any]]] = {
            cell_id: [] for cell_id in development.TARGET_CELLS
        }
        reused = 0
        infeasible = 0
        skipped = 0
        for task in unfiltered_tasks(candidate_count, repetitions, data):
            db_key = (
                candidate_count,
                str(task["cell_id"]),
                int(task["repeat_seed"]),
                float(task["ratio"]),
            )
            if db_key in terminal:
                skipped += 1
                continue
            if not task["run_fit"]:
                commit_result(
                    connection,
                    task,
                    infeasible_result(),
                    "candidate_count_infeasible",
                )
                terminal.add(db_key)
                infeasible += 1
                continue
            cached = cache.get(result_cache_key(task))
            if cached is not None:
                result = dict(cached)
                result["fit_seconds"] = 0.0
                commit_result(
                    connection,
                    task,
                    result,
                    "exact_training_set_reuse",
                )
                terminal.add(db_key)
                reused += 1
                continue
            pending_by_cell[str(task["cell_id"])].append(dict(task))
        pending = sum(len(rows) for rows in pending_by_cell.values())
        print(
            f"K={candidate_count} plan | skipped={skipped} | infeasible={infeasible} | "
            f"exact_reuse={reused} | new_fits={pending}",
            flush=True,
        )
        execute_pending(
            connection,
            data,
            pending_by_cell,
            cache,
            workers=workers,
            progress_every=progress_every,
            candidate_count=candidate_count,
        )
        terminal = terminal_keys(connection)
        del data
        gc.collect()


def base_metric_map() -> dict[tuple[str, int], dict[str, float]]:
    frame = pd.read_csv(
        PRIMARY_ROOT / "Downstream" / "DevelopmentGrid" / "base_results.csv"
    )
    return {
        (str(row.cell_id), int(row.repeat_seed)): {
            "base_macro_f1": float(row.macro_f1),
            "base_auroc": float(row.auroc),
        }
        for row in frame.itertuples(index=False)
    }


def export_result_rows(
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    base = base_metric_map()
    connection.row_factory = sqlite3.Row
    db_rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM results ORDER BY candidate_count,cell_id,repeat_seed,ratio"
        )
    ]
    connection.row_factory = None
    expected = len(CANDIDATE_COUNTS) * len(development.TARGET_CELLS) * 50 * len(
        development.RATIOS
    )
    if len(db_rows) != expected:
        raise RuntimeError(f"candidate-count result rows {len(db_rows)} != {expected}")
    rows: list[dict[str, Any]] = []
    for row in db_rows:
        anchor = base[(str(row["cell_id"]), int(row["repeat_seed"]))]
        completed = row["status"] in {"completed", "completed_with_warning"}
        macro = row["macro_f1"] if completed else None
        auroc = row["auroc"] if completed else None
        rows.append(
            {
                "candidate_count": int(row["candidate_count"]),
                "cell_id": str(row["cell_id"]),
                "repeat_seed": int(row["repeat_seed"]),
                "ratio": float(row["ratio"]),
                "target_n": int(row["target_n"]),
                "eligible_n": int(row["eligible_n"]),
                "feasible": bool(row["feasible"]),
                "infeasible_reason": row["infeasible_reason"],
                "selection_sha256": str(row["selection_sha256"]),
                "selected_n": int(row["selected_n"]),
                "selected_parent_count": int(row["selected_parent_count"]),
                "base_macro_f1": anchor["base_macro_f1"],
                "hybrid_macro_f1": macro,
                "delta_macro_f1": (
                    None if macro is None else float(macro) - anchor["base_macro_f1"]
                ),
                "base_auroc": anchor["base_auroc"],
                "hybrid_auroc": auroc,
                "delta_auroc": (
                    None if auroc is None else float(auroc) - anchor["base_auroc"]
                ),
                "status": str(row["status"]),
                "result_source": str(row["result_source"]),
            }
        )
    return rows


def ratio_summary_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(results)
    frame = frame[frame["status"].isin(["completed", "completed_with_warning"])].copy()
    rows: list[dict[str, Any]] = []
    for (candidate_count, cell_id, ratio), group in frame.groupby(
        ["candidate_count", "cell_id", "ratio"], sort=True
    ):
        rows.append(
            {
                "candidate_count": int(candidate_count),
                "cell_id": str(cell_id),
                "ratio": float(ratio),
                "completed_repetitions": int(len(group)),
                "mean_hybrid_macro_f1": float(group["hybrid_macro_f1"].mean()),
                "mean_delta_macro_f1": float(group["delta_macro_f1"].mean()),
                "mean_hybrid_auroc": float(group["hybrid_auroc"].mean()),
                "mean_delta_auroc": float(group["delta_auroc"].mean()),
            }
        )
    return rows


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def bootstrap_mean(values: np.ndarray, *key: object) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(stable_seed(LOCK_VERSION, *key))
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPS, values.size))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def k20_comparison_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(results)
    frame = frame[frame["status"].isin(["completed", "completed_with_warning"])].copy()
    reference = frame[frame["candidate_count"] == REFERENCE_COUNT][
        ["cell_id", "repeat_seed", "ratio", "hybrid_macro_f1", "hybrid_auroc"]
    ].rename(
        columns={
            "hybrid_macro_f1": "reference_macro_f1",
            "hybrid_auroc": "reference_auroc",
        }
    )
    rows: list[dict[str, Any]] = []
    for candidate_count in CANDIDATE_COUNTS:
        if candidate_count == REFERENCE_COUNT:
            continue
        current = frame[frame["candidate_count"] == candidate_count]
        merged = current.merge(
            reference,
            on=["cell_id", "repeat_seed", "ratio"],
            how="inner",
            validate="one_to_one",
        )
        for cell_id, cell in merged.groupby("cell_id", sort=True):
            repeat_rows: list[dict[str, float]] = []
            for repeat_seed, repeat in cell.groupby("repeat_seed", sort=True):
                repeat_rows.append(
                    {
                        "repeat_seed": float(repeat_seed),
                        "macro": float(
                            (
                                repeat["hybrid_macro_f1"]
                                - repeat["reference_macro_f1"]
                            ).mean()
                        ),
                        "auroc": float(
                            (
                                repeat["hybrid_auroc"] - repeat["reference_auroc"]
                            ).mean()
                        ),
                        "common_ratios": float(len(repeat)),
                    }
                )
            repeat_frame = pd.DataFrame(repeat_rows)
            macro = bootstrap_mean(
                repeat_frame["macro"].to_numpy(),
                candidate_count,
                cell_id,
                "macro",
            )
            auroc = bootstrap_mean(
                repeat_frame["auroc"].to_numpy(),
                candidate_count,
                cell_id,
                "auroc",
            )
            rows.append(
                {
                    "candidate_count": candidate_count,
                    "cell_id": str(cell_id),
                    "paired_repetitions": int(len(repeat_frame)),
                    "mean_common_ratios_per_repeat": float(
                        repeat_frame["common_ratios"].mean()
                    ),
                    "mean_macro_f1_difference_vs_k20": macro[0],
                    "macro_f1_ci_lower": macro[1],
                    "macro_f1_ci_upper": macro[2],
                    "mean_auroc_difference_vs_k20": auroc[0],
                    "auroc_ci_lower": auroc[1],
                    "auroc_ci_upper": auroc[2],
                }
            )
    return rows


def complete_unfiltered_candidate_counts(
    similarity: Sequence[Mapping[str, Any]],
) -> list[int]:
    frame = pd.DataFrame(similarity)
    unfiltered = frame[frame["similarity_condition_id"] == "OFF"]
    complete: list[int] = []
    for candidate_count in CANDIDATE_COUNTS:
        subset = unfiltered[unfiltered["candidate_count"] == candidate_count]
        if int(subset["infeasible_repetitions"].sum()) == 0:
            complete.append(candidate_count)
    return complete


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Holm adjustment expects a one-dimensional p-value vector")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    m = values.size
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (m - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def candidate_count_statistical_tests(
    results: Sequence[Mapping[str, Any]],
    complete_counts: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Formal repeated-measures tests across complete-grid candidate counts.

    For each dataset and metric, performance is first averaged across all 11
    unfiltered augmentation ratios within each repetition.  The 50 repetition-
    level means are then compared across K=5..20 using a Friedman omnibus test.
    Planned paired comparisons against the original K=20 reference use two-
    sided Wilcoxon signed-rank tests with Holm correction across the 15 K-vs-20
    contrasts within each dataset/metric family.
    """
    frame = pd.DataFrame(results)
    frame = frame[
        frame["status"].isin(["completed", "completed_with_warning"])
        & frame["candidate_count"].isin(list(complete_counts))
    ].copy()
    if REFERENCE_COUNT not in set(complete_counts):
        raise RuntimeError("K=20 must be present in the complete candidate-count range")

    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    metric_columns = (
        ("macro_f1", "hybrid_macro_f1"),
        ("auroc", "hybrid_auroc"),
    )
    expected_counts = list(sorted(int(value) for value in complete_counts))
    for cell_id, cell in frame.groupby("cell_id", sort=True):
        for metric_name, metric_column in metric_columns:
            per_repeat = (
                cell.groupby(["repeat_seed", "candidate_count"], as_index=False)[metric_column]
                .mean()
                .pivot(index="repeat_seed", columns="candidate_count", values=metric_column)
                .sort_index()
            )
            missing_counts = [value for value in expected_counts if value not in per_repeat.columns]
            if missing_counts:
                raise RuntimeError(
                    f"{cell_id}/{metric_name}: missing complete candidate counts {missing_counts}"
                )
            per_repeat = per_repeat[expected_counts]
            if per_repeat.shape[0] != 50 or per_repeat.isna().any().any():
                raise RuntimeError(
                    f"{cell_id}/{metric_name}: expected a complete 50 x {len(expected_counts)} matrix, "
                    f"observed {per_repeat.shape}"
                )
            friedman = friedmanchisquare(
                *[per_repeat[value].to_numpy(dtype=np.float64) for value in expected_counts]
            )
            statistic = float(friedman.statistic)
            p_value = float(friedman.pvalue)
            kendalls_w = statistic / (per_repeat.shape[0] * (len(expected_counts) - 1))
            omnibus_rows.append(
                {
                    "cell_id": str(cell_id),
                    "metric": metric_name,
                    "candidate_count_min": min(expected_counts),
                    "candidate_count_max": max(expected_counts),
                    "candidate_count_levels": len(expected_counts),
                    "paired_repetitions": int(per_repeat.shape[0]),
                    "friedman_chi_square": statistic,
                    "friedman_p_value": p_value,
                    "kendalls_w": float(kendalls_w),
                    "significant_at_0_05": bool(p_value < 0.05),
                }
            )

            raw_rows: list[dict[str, Any]] = []
            reference = per_repeat[REFERENCE_COUNT].to_numpy(dtype=np.float64)
            for candidate_count in expected_counts:
                if candidate_count == REFERENCE_COUNT:
                    continue
                current = per_repeat[candidate_count].to_numpy(dtype=np.float64)
                differences = current - reference
                if np.all(differences == 0.0):
                    statistic_value = 0.0
                    raw_p = 1.0
                else:
                    test = wilcoxon(
                        differences,
                        zero_method="wilcox",
                        correction=False,
                        alternative="two-sided",
                        method="auto",
                    )
                    statistic_value = float(test.statistic)
                    raw_p = float(test.pvalue)
                raw_rows.append(
                    {
                        "cell_id": str(cell_id),
                        "metric": metric_name,
                        "candidate_count": int(candidate_count),
                        "reference_candidate_count": REFERENCE_COUNT,
                        "paired_repetitions": int(per_repeat.shape[0]),
                        "mean_difference_vs_k20": float(differences.mean()),
                        "median_difference_vs_k20": float(np.median(differences)),
                        "wilcoxon_statistic": statistic_value,
                        "raw_p_value": raw_p,
                    }
                )
            adjusted = holm_adjust([row["raw_p_value"] for row in raw_rows])
            for row, adjusted_p in zip(raw_rows, adjusted, strict=True):
                pairwise_rows.append(
                    {
                        **row,
                        "holm_adjusted_p_value": float(adjusted_p),
                        "significant_at_0_05": bool(adjusted_p < 0.05),
                    }
                )
    return omnibus_rows, pairwise_rows


def manuscript_table_summary_rows(
    report: Mapping[str, Any],
    omnibus: Sequence[Mapping[str, Any]],
    pairwise: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    omnibus_frame = pd.DataFrame(omnibus)
    pairwise_frame = pd.DataFrame(pairwise)
    max_abs = report["max_absolute_mean_deviation_from_k20_over_complete_counts"]
    rows: list[dict[str, Any]] = []
    for cell_id in sorted(max_abs):
        entry: dict[str, Any] = {
            "cell_id": cell_id,
            "complete_candidate_count_range": "5-20",
            "max_abs_mean_macro_f1_difference_vs_k20": max_abs[cell_id][
                "max_abs_mean_macro_f1_difference_vs_k20"
            ],
            "max_abs_mean_auroc_difference_vs_k20": max_abs[cell_id][
                "max_abs_mean_auroc_difference_vs_k20"
            ],
        }
        for metric in ("macro_f1", "auroc"):
            omnibus_row = omnibus_frame[
                (omnibus_frame["cell_id"] == cell_id)
                & (omnibus_frame["metric"] == metric)
            ].iloc[0]
            family = pairwise_frame[
                (pairwise_frame["cell_id"] == cell_id)
                & (pairwise_frame["metric"] == metric)
            ].sort_values("candidate_count")
            significant = family[family["significant_at_0_05"]]
            prefix = "macro_f1" if metric == "macro_f1" else "auroc"
            entry[f"{prefix}_friedman_p_value"] = float(omnibus_row["friedman_p_value"])
            entry[f"{prefix}_kendalls_w"] = float(omnibus_row["kendalls_w"])
            entry[f"{prefix}_min_holm_adjusted_p_value"] = float(
                family["holm_adjusted_p_value"].min()
            )
            entry[f"{prefix}_holm_significant_candidate_counts_vs_k20"] = (
                ";".join(str(int(value)) for value in significant["candidate_count"].tolist())
                if not significant.empty
                else "none"
            )
        rows.append(entry)
    return rows


def build_report(
    availability: Sequence[Mapping[str, Any]],
    similarity: Sequence[Mapping[str, Any]],
    ratio_summary: Sequence[Mapping[str, Any]],
    versus_k20: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    similarity_frame = pd.DataFrame(similarity)
    unfiltered = similarity_frame[
        similarity_frame["similarity_condition_id"] == "OFF"
    ]
    complete_unfiltered: list[int] = []
    for candidate_count in CANDIDATE_COUNTS:
        subset = unfiltered[unfiltered["candidate_count"] == candidate_count]
        if int(subset["infeasible_repetitions"].sum()) == 0:
            complete_unfiltered.append(candidate_count)
    versus = pd.DataFrame(versus_k20)
    max_abs: dict[str, dict[str, float]] = {}
    if not versus.empty:
        for cell_id, cell in versus.groupby("cell_id", sort=True):
            full = cell[cell["candidate_count"].isin(complete_unfiltered)]
            max_abs[str(cell_id)] = {
                "max_abs_mean_macro_f1_difference_vs_k20": float(
                    full["mean_macro_f1_difference_vs_k20"].abs().max()
                ),
                "max_abs_mean_auroc_difference_vs_k20": float(
                    full["mean_auroc_difference_vs_k20"].abs().max()
                ),
            }
    return {
        "analysis_version": LOCK_VERSION,
        "candidate_counts": list(CANDIDATE_COUNTS),
        "candidate_definition": "retain valid generation records with candidate_index < K",
        "reference_candidate_count": REFERENCE_COUNT,
        "new_generation_performed": False,
        "test_data_accessed": False,
        "policy_reselected": False,
        "sensitivity_scope": {
            "datasets": len(development.TARGET_CELLS),
            "paired_repetitions": 50,
            "augmentation_ratios": list(development.RATIOS),
            "similarity_conditions": [
                str(item["similarity_condition_id"])
                for item in development.SIMILARITY_CONDITIONS
            ],
            "downstream_performance_condition": "unfiltered Hybrid",
        },
        "candidate_counts_with_complete_unfiltered_ratio_grid": complete_unfiltered,
        "max_absolute_mean_deviation_from_k20_over_complete_counts": max_abs,
        "availability": list(availability),
        "similarity_feasibility": list(similarity),
        "ratio_summary": list(ratio_summary),
        "versus_k20": list(versus_k20),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    versus = pd.DataFrame(report["versus_k20"])
    table_summary = pd.DataFrame(report.get("manuscript_table_summary", []))
    lines = [
        "# Candidate-count sensitivity (Reviewer 1, Concern #5)",
        "",
        "Candidate count K was varied over every integer from 1 through 20 using nested, prespecified generation-request prefixes. No new generation was performed, the frozen final policy was not reselected, and the test split was not accessed.",
        "",
        "All 17 original similarity-selection conditions and all 11 augmentation ratios were checked for feasibility. Downstream performance was rerun for the unfiltered Hybrid condition at every feasible K/ratio/dataset/repetition cell.",
        "",
        "## Complete unfiltered ratio grid",
        "",
        "Candidate counts for which every original unfiltered ratio cell was feasible:",
        "",
        ", ".join(
            str(value)
            for value in report["candidate_counts_with_complete_unfiltered_ratio_grid"]
        ),
        "",
        "## Mean development difference from K=20",
        "",
        "| K | Dataset | ΔMacro-F1 vs K=20 | 95% CI | ΔAUROC vs K=20 | 95% CI |",
        "|---:|---|---:|---|---:|---|",
    ]
    if not versus.empty:
        for row in versus.sort_values(["candidate_count", "cell_id"]).itertuples(
            index=False
        ):
            lines.append(
                f"| {int(row.candidate_count)} | {row.cell_id} | "
                f"{float(row.mean_macro_f1_difference_vs_k20):+.6f} | "
                f"[{float(row.macro_f1_ci_lower):+.6f}, {float(row.macro_f1_ci_upper):+.6f}] | "
                f"{float(row.mean_auroc_difference_vs_k20):+.6f} | "
                f"[{float(row.auroc_ci_lower):+.6f}, {float(row.auroc_ci_upper):+.6f}] |"
            )
    if not table_summary.empty:
        lines.extend(
            [
                "",
                "## Formal candidate-count tests over the complete K=5..20 grid",
                "",
                "Performance was averaged across the 11 unfiltered augmentation ratios within each repetition before testing. Friedman tests compared all 16 candidate-count levels, and planned K-vs-20 comparisons used paired Wilcoxon signed-rank tests with Holm correction across the 15 comparisons within each dataset/metric family.",
                "",
                "| Dataset | max abs ΔMacro-F1 | Friedman p (W) | Holm-significant K vs 20 | max abs ΔAUROC | Friedman p (W) | Holm-significant K vs 20 |",
                "|---|---:|---|---|---:|---|---|",
            ]
        )
        for row in table_summary.itertuples(index=False):
            lines.append(
                f"| {row.cell_id} | "
                f"{float(row.max_abs_mean_macro_f1_difference_vs_k20):.6f} | "
                f"{float(row.macro_f1_friedman_p_value):.6g} ({float(row.macro_f1_kendalls_w):.3f}) | "
                f"{row.macro_f1_holm_significant_candidate_counts_vs_k20} | "
                f"{float(row.max_abs_mean_auroc_difference_vs_k20):.6f} | "
                f"{float(row.auroc_friedman_p_value):.6g} ({float(row.auroc_kendalls_w):.3f}) | "
                f"{row.auroc_holm_significant_candidate_counts_vs_k20} |"
            )
    return "\n".join(lines) + "\n"


def finalize(
    availability: Sequence[Mapping[str, Any]],
    similarity: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ratio_summary = ratio_summary_rows(results)
    versus_k20 = k20_comparison_rows(results)
    report = build_report(availability, similarity, ratio_summary, versus_k20)
    complete_counts = report["candidate_counts_with_complete_unfiltered_ratio_grid"]
    omnibus, pairwise = candidate_count_statistical_tests(results, complete_counts)
    table_summary = manuscript_table_summary_rows(report, omnibus, pairwise)
    report["statistics_version"] = STATISTICS_VERSION
    report["statistical_analysis"] = {
        "unit": "per-repetition mean across all 11 unfiltered augmentation ratios",
        "complete_candidate_counts": list(complete_counts),
        "omnibus_test": "Friedman repeated-measures test",
        "omnibus_effect_size": "Kendall's W",
        "planned_reference_comparison": "two-sided paired Wilcoxon signed-rank test against K=20",
        "multiplicity_control": "Holm correction across the 15 K-vs-20 comparisons within each dataset/metric family",
        "alpha": 0.05,
    }
    report["friedman_omnibus"] = omnibus
    report["wilcoxon_holm_vs_k20"] = pairwise
    report["manuscript_table_summary"] = table_summary
    write_csv(
        OUTPUT_ROOT / "candidate_count_availability.csv",
        availability,
        (
            "candidate_count",
            "cell_id",
            "planned_requests",
            "valid_candidates",
            "invalid_candidates",
            "valid_rate",
        ),
    )
    write_csv(
        OUTPUT_ROOT / "candidate_count_similarity_feasibility.csv",
        similarity,
        (
            "candidate_count",
            "cell_id",
            "similarity_condition_id",
            "ratio",
            "planned_repetitions",
            "feasible_repetitions",
            "infeasible_repetitions",
            "feasible_rate",
            "mean_eligible_candidates",
            "mean_eligible_label_0",
            "mean_eligible_label_1",
        ),
    )
    write_csv(
        OUTPUT_ROOT / "candidate_count_unfiltered_results.csv",
        results,
        (
            "candidate_count",
            "cell_id",
            "repeat_seed",
            "ratio",
            "target_n",
            "eligible_n",
            "feasible",
            "infeasible_reason",
            "selection_sha256",
            "selected_n",
            "selected_parent_count",
            "base_macro_f1",
            "hybrid_macro_f1",
            "delta_macro_f1",
            "base_auroc",
            "hybrid_auroc",
            "delta_auroc",
            "status",
            "result_source",
        ),
    )
    write_csv(
        OUTPUT_ROOT / "candidate_count_ratio_summary.csv",
        ratio_summary,
        (
            "candidate_count",
            "cell_id",
            "ratio",
            "completed_repetitions",
            "mean_hybrid_macro_f1",
            "mean_delta_macro_f1",
            "mean_hybrid_auroc",
            "mean_delta_auroc",
        ),
    )
    write_csv(
        OUTPUT_ROOT / "candidate_count_vs_k20.csv",
        versus_k20,
        (
            "candidate_count",
            "cell_id",
            "paired_repetitions",
            "mean_common_ratios_per_repeat",
            "mean_macro_f1_difference_vs_k20",
            "macro_f1_ci_lower",
            "macro_f1_ci_upper",
            "mean_auroc_difference_vs_k20",
            "auroc_ci_lower",
            "auroc_ci_upper",
        ),
    )
    write_csv(
        OUTPUT_ROOT / "candidate_count_friedman_omnibus.csv",
        omnibus,
        (
            "cell_id",
            "metric",
            "candidate_count_min",
            "candidate_count_max",
            "candidate_count_levels",
            "paired_repetitions",
            "friedman_chi_square",
            "friedman_p_value",
            "kendalls_w",
            "significant_at_0_05",
        ),
    )
    write_csv(
        OUTPUT_ROOT / "candidate_count_wilcoxon_holm_vs_k20.csv",
        pairwise,
        (
            "cell_id",
            "metric",
            "candidate_count",
            "reference_candidate_count",
            "paired_repetitions",
            "mean_difference_vs_k20",
            "median_difference_vs_k20",
            "wilcoxon_statistic",
            "raw_p_value",
            "holm_adjusted_p_value",
            "significant_at_0_05",
        ),
    )
    write_csv(
        OUTPUT_ROOT / "candidate_count_manuscript_table.csv",
        table_summary,
        (
            "cell_id",
            "complete_candidate_count_range",
            "max_abs_mean_macro_f1_difference_vs_k20",
            "macro_f1_friedman_p_value",
            "macro_f1_kendalls_w",
            "macro_f1_min_holm_adjusted_p_value",
            "macro_f1_holm_significant_candidate_counts_vs_k20",
            "max_abs_mean_auroc_difference_vs_k20",
            "auroc_friedman_p_value",
            "auroc_kendalls_w",
            "auroc_min_holm_adjusted_p_value",
            "auroc_holm_significant_candidate_counts_vs_k20",
        ),
    )
    write_json(OUTPUT_ROOT / "candidate_count_report.json", report)
    (OUTPUT_ROOT / "candidate_count_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    files: dict[str, str] = {}
    for path in sorted(OUTPUT_ROOT.iterdir()):
        if not path.is_file() or path.name in {LOCK_NAME, DATABASE_PATH.name}:
            continue
        files[path.name] = sha256_file(path)
    write_json(
        OUTPUT_ROOT / LOCK_NAME,
        {
            "lock_version": LOCK_VERSION,
            "status": "completed",
            "candidate_counts": list(CANDIDATE_COUNTS),
            "new_generation_performed": False,
            "test_data_accessed": False,
            "policy_reselected": False,
            "files": files,
        },
    )
    verify_completed_lock()
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0 or args.progress_every <= 0:
        raise SystemExit("workers and progress-every must be positive")
    if (OUTPUT_ROOT / LOCK_NAME).is_file() and not args.dry_run:
        verify_completed_lock()
        print("SKIP candidate-count sensitivity: verified completed lock")
        return 0
    before = primary_lock_snapshot(PRIMARY_ROOT)
    development.input_lock_hashes(PRIMARY_ROOT)
    _manifest_info, banks, _plan = development.load_and_validate_inputs(
        PREPARED_ROOT, MANIFEST_ROOT
    )
    repetitions = development.validate_repetitions(MANIFEST_ROOT, banks)
    valid_by_cell = development.read_valid_outputs(PRIMARY_ROOT / "Generation")
    index_maps = candidate_index_maps(valid_by_cell)
    full_data = development.validate_and_load_embeddings(
        PRIMARY_ROOT, banks, valid_by_cell
    )
    availability = availability_rows(valid_by_cell)
    similarity = similarity_feasibility_rows(
        repetitions, full_data, index_maps
    )
    similarity_frame = pd.DataFrame(similarity)
    primary_paired = pd.read_csv(
        PRIMARY_ROOT / "Downstream" / "DevelopmentGrid" / "paired_results.csv"
    )
    expected_primary_infeasible = int(
        (~primary_paired["feasible"].astype(str).str.lower().isin({"true", "1"})).sum()
    )
    observed_primary_infeasible = int(
        similarity_frame[
            similarity_frame["candidate_count"] == REFERENCE_COUNT
        ]["infeasible_repetitions"].sum()
    )
    if observed_primary_infeasible != expected_primary_infeasible:
        raise RuntimeError(
            "K=20 feasibility reconstruction does not match the locked primary analysis: "
            f"observed={observed_primary_infeasible}, expected={expected_primary_infeasible}"
        )
    complete_counts = (
        similarity_frame[
            similarity_frame["similarity_condition_id"] == "OFF"
        ]
        .groupby("candidate_count")["infeasible_repetitions"]
        .sum()
    )
    complete_counts = [
        int(index) for index, value in complete_counts.items() if int(value) == 0
    ]
    print(
        json.dumps(
            {
                "candidate_counts": list(CANDIDATE_COUNTS),
                "candidate_counts_with_complete_unfiltered_ratio_grid": complete_counts,
                "new_generation_performed": False,
                "test_data_accessed": False,
                "policy_reselected": False,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    connection = create_database(DATABASE_PATH)
    try:
        run_unfiltered_sensitivity(
            connection,
            repetitions,
            full_data,
            index_maps,
            workers=args.workers,
            progress_every=args.progress_every,
        )
        results = export_result_rows(connection)
    finally:
        connection.close()
    report = finalize(availability, similarity, results)
    after = primary_lock_snapshot(PRIMARY_ROOT)
    if before != after:
        raise RuntimeError(
            "primary locked artifacts changed during candidate-count sensitivity"
        )
    print(markdown_report(report))
    print(f"Candidate-count sensitivity complete: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
