"""Run the full OffensiveLanguage robustness pipeline without manual checkpoints."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RESULT_ROOT = PROJECT_ROOT / "Results" / "OffensiveLanguage"
PREPARED_ROOT = PROJECT_ROOT / "Data" / "Prepared" / "OffensiveLanguage"
MANIFEST_ROOT = PROJECT_ROOT / "Data" / "ExperimentManifests" / "OffensiveLanguage" / "v1"


def stage_path(name: str) -> Path:
    return ROOT / f"{name}.py"


def run_stage(name: str, *args: str) -> None:
    command = [sys.executable, str(stage_path(name)), *args]
    print("\n" + "=" * 80)
    print(f"RUN {name}")
    print("=" * 80)
    subprocess.run(command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all OffensiveLanguage stages in numbered protocol order. "
            "Stops immediately on any failed stage."
        )
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        nargs=3,
        metavar=("API_URL", "MODEL", "CONCURRENCY"),
    )
    parser.add_argument(
        "--qwen-endpoint",
        action="append",
        nargs=3,
        metavar=("API_URL", "MODEL", "CONCURRENCY"),
    )
    parser.add_argument(
        "--bge-endpoint",
        action="append",
        nargs=3,
        metavar=("API_URL", "MODEL", "CONCURRENCY"),
    )
    parser.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--export-shard-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--resume-generation",
        action="store_true",
        help=(
            "Resume an incomplete generation DB only when it was created with this corrected "
            "OffensiveLanguage configuration."
        ),
    )
    return parser.parse_args(argv)


def append_endpoints(
    target: list[str], option: str, values: list[list[str]] | None
) -> None:
    for value in values or []:
        target.extend([option, *value])


def directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def has_augmentation_on(diagnostic_root: Path) -> bool:
    path = diagnostic_root / "adaptive_selection.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "augmentation" not in rows[0]:
        raise RuntimeError(f"invalid adaptive selection file: {path}")
    return any(str(row.get("augmentation") or "").strip().upper() == "ON" for row in rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.timeout_seconds <= 0
        or args.embedding_batch_size <= 0
        or args.export_shard_size <= 0
        or args.workers <= 0
        or args.progress_every <= 0
    ):
        raise SystemExit("all numeric execution options must be positive")

    common_model_args = [
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--embedding-batch-size",
        str(args.embedding_batch_size),
        "--export-shard-size",
        str(args.export_shard_size),
    ]
    if args.api_key:
        common_model_args.extend(["--api-key", args.api_key])
    append_endpoints(common_model_args, "--endpoint", args.endpoint)
    append_endpoints(common_model_args, "--qwen-endpoint", args.qwen_endpoint)
    append_endpoints(common_model_args, "--bge-endpoint", args.bge_endpoint)

    generation_root = RESULT_ROOT / "Generation"
    generation_lock = generation_root / "GENERATION_LOCK.json"
    generation_db = generation_root / "generation_status.sqlite3"
    stage00_args = list(common_model_args)
    if generation_db.is_file() and not generation_lock.is_file():
        if not args.resume_generation:
            raise RuntimeError(
                "Incomplete generation_status.sqlite3 exists. Delete "
                "Results/OffensiveLanguage/Generation before the corrected rerun. Use "
                "--resume-generation only for a DB created by this corrected configuration."
            )
        stage00_args.append("--resume")
    stage00_args.extend([
        "--prepared-root", str(PREPARED_ROOT),
        "--manifest-root", str(MANIFEST_ROOT),
        "--output-root", str(RESULT_ROOT),
    ])
    run_stage("run_binary_matched_size_experiment", *stage00_args)

    eval_root = RESULT_ROOT / "Embeddings" / "Qwen3Eval"
    eval_lock = eval_root / "QWEN_EVAL_EMBEDDING_LOCK.json"
    if not eval_lock.is_file():
        stage08_args = [
            "--prepared-root", str(PREPARED_ROOT),
            "--manifest-root", str(MANIFEST_ROOT),
            "--output-root", str(RESULT_ROOT),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--embedding-batch-size",
            str(args.embedding_batch_size),
        ]
        if args.api_key:
            stage08_args.extend(["--api-key", args.api_key])
        append_endpoints(stage08_args, "--qwen-endpoint", args.qwen_endpoint)
        run_stage("embed_binary_eval_splits", *stage08_args)
    else:
        print("SKIP stage08: QWEN_EVAL_EMBEDDING_LOCK.json exists")

    development_root = RESULT_ROOT / "Downstream" / "DevelopmentGrid"
    development_lock = development_root / "DEVELOPMENT_GRID_LOCK.json"
    if not development_lock.is_file():
        stage04_args = [
            "--manifest-root", str(MANIFEST_ROOT),
            "--output-root", str(RESULT_ROOT),
            "--workers",
            str(args.workers),
            "--progress-every",
            str(args.progress_every),
        ]
        if (development_root / "downstream_status.sqlite3").is_file():
            stage04_args.append("--resume")
        run_stage("run_binary_downstream_experiment", *stage04_args)
    else:
        print("SKIP stage04: DEVELOPMENT_GRID_LOCK.json exists")

    run_stage("summarize_binary_downstream_results", "--output-root", str(RESULT_ROOT))

    weight_root = RESULT_ROOT / "Downstream" / "SyntheticWeightGrid"
    weight_lock = weight_root / "SYNTHETIC_WEIGHT_GRID_LOCK.json"
    if not weight_lock.is_file():
        stage06_args = [
            "--manifest-root", str(MANIFEST_ROOT),
            "--output-root", str(RESULT_ROOT),
            "--workers",
            str(args.workers),
            "--progress-every",
            str(args.progress_every),
        ]
        if (weight_root / "synthetic_weight_status.sqlite3").is_file():
            stage06_args.append("--resume")
        run_stage("run_binary_synthetic_weight_grid", *stage06_args)
    else:
        print("SKIP stage06: SYNTHETIC_WEIGHT_GRID_LOCK.json exists")

    adaptive_root = RESULT_ROOT / "Downstream" / "AdaptivePolicy"
    adaptive_lock = adaptive_root / "BINARY_ADAPTIVE_PROTOCOL_LOCK.json"
    if not adaptive_lock.is_file():
        stage07_args: list[str] = [
            "--weight-grid-root", str(weight_root),
            "--output-root", str(adaptive_root),
        ]
        if directory_has_files(adaptive_root):
            stage07_args.append("--overwrite")
        run_stage("freeze_binary_adaptive_protocol", *stage07_args)
    else:
        print("SKIP stage07: BINARY_ADAPTIVE_PROTOCOL_LOCK.json exists")

    diagnostic_root = adaptive_root / "DecisionBoundaryDiagnostic"
    diagnostic_lock = diagnostic_root / "ADAPTIVE_SELECTION_LOCK.json"
    if not diagnostic_lock.is_file():
        stage09_args: list[str] = [
            "--prepared-root", str(PREPARED_ROOT),
            "--manifest-root", str(MANIFEST_ROOT),
            "--experiment-root", str(RESULT_ROOT),
            "--eval-embedding-root", str(eval_root),
            "--weight-grid-root", str(weight_root),
            "--adaptive-root", str(adaptive_root),
            "--output-root", str(diagnostic_root),
        ]
        if directory_has_files(diagnostic_root):
            stage09_args.append("--overwrite")
        run_stage("run_binary_decision_boundary_diagnostic", *stage09_args)
    else:
        print("SKIP stage09: ADAPTIVE_SELECTION_LOCK.json exists")

    one_shot_root = adaptive_root / "OneShotTest"
    pretest_lock = one_shot_root / "PRETEST_THRESHOLD_LOCK.json"
    final_test_lock = one_shot_root / "ADAPTIVE_ONE_SHOT_TEST_LOCK.json"
    test_access_marker = one_shot_root / "TEST_ACCESS_STARTED.json"

    if not pretest_lock.is_file():
        stage10_common = [
            "--prepared-root", str(PREPARED_ROOT),
            "--manifest-root", str(MANIFEST_ROOT),
            "--experiment-root", str(RESULT_ROOT),
            "--eval-embedding-root", str(eval_root),
            "--weight-grid-root", str(weight_root),
            "--adaptive-diagnostic-root", str(diagnostic_root),
            "--output-root", str(one_shot_root),
        ]
        run_stage("run_binary_adaptive_one_shot_test", *stage10_common, "--prepare-only")
    else:
        print("SKIP stage10 prepare: PRETEST_THRESHOLD_LOCK.json exists")

    if final_test_lock.is_file():
        print("SKIP stage10 evaluate: ADAPTIVE_ONE_SHOT_TEST_LOCK.json exists")
    else:
        if test_access_marker.is_file():
            raise RuntimeError(
                "TEST_ACCESS_STARTED.json exists without a final one-shot lock. "
                "Automatic rerun is intentionally blocked by the frozen protocol."
            )
        stage10_common = [
            "--prepared-root", str(PREPARED_ROOT),
            "--manifest-root", str(MANIFEST_ROOT),
            "--experiment-root", str(RESULT_ROOT),
            "--eval-embedding-root", str(eval_root),
            "--weight-grid-root", str(weight_root),
            "--adaptive-diagnostic-root", str(diagnostic_root),
            "--output-root", str(one_shot_root),
        ]
        run_stage("run_binary_adaptive_one_shot_test", *stage10_common, "--evaluate")

    analysis_root = one_shot_root / "FinalThreeConditionFriedmanV1"
    analysis_lock = analysis_root / "FRIEDMAN_ANALYSIS_LOCK.json"
    if not has_augmentation_on(diagnostic_root):
        print("SKIP stage11: no augmentation-ON dataset; three-condition Friedman is inapplicable")
    elif not analysis_lock.is_file():
        run_stage("analyze_binary_friedman", "--input-root", str(one_shot_root))
    else:
        print("SKIP stage11: FRIEDMAN_ANALYSIS_LOCK.json exists")

    print("\nALL OFFENSIVE LANGUAGE STAGES COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
