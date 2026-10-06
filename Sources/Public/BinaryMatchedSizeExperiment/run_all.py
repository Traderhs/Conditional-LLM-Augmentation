"""Canonical primary + reviewer sensitivities + qualitative audit + robustness runner.

Existing primary results are treated as immutable.  If any primary artifacts
exist, every required primary lock must verify before this runner will proceed;
the primary pipeline is then skipped entirely.  New robustness outputs are
written only below Results/BinaryMatchedSizeExperiment/Robustness/v1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from robustness_common import (  # noqa: E402
    ROBUSTNESS_VERSION,
    assert_primary_unchanged,
    primary_lock_snapshot,
    sha256_file,
    verify_hash_lock,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_ROOT = PROJECT_ROOT / "Results" / "BinaryMatchedSizeExperiment"
ROBUSTNESS_ROOT = PRIMARY_ROOT / "Robustness" / "v1"
OFFENSIVE_PILOT_ROOT = PROJECT_ROOT / "Results" / "OffensiveLanguage" / "Pilot"
SUITE_LOCK = "ROBUSTNESS_SUITE_LOCK.json"
FIGURE_RUNNER = PROJECT_ROOT / "Sources" / "Public" / "Figures" / "run_all.py"
TRANSFORMER_LOCK_VERSION = "r3-transformer-robustness-v2"
TRANSFORMER_FIXED_EPOCHS = 20


def script(name: str) -> Path:
    return _MODULE_ROOT / f"{name}.py"


def run_stage(name: str, *args: str) -> None:
    command = [sys.executable, str(script(name)), *args]
    print("\n" + "=" * 88)
    print("RUN", " ".join(command))
    print("=" * 88, flush=True)
    subprocess.run(command, check=True)


def run_figures() -> None:
    if not FIGURE_RUNNER.is_file():
        raise FileNotFoundError(f"Figure runner not found: {FIGURE_RUNNER}")
    command = [sys.executable, str(FIGURE_RUNNER)]
    print("\n" + "=" * 88)
    print("RUN", " ".join(command))
    print("=" * 88, flush=True)
    subprocess.run(command, check=True)


def append_endpoints(target: list[str], option: str, values: list[list[str]] | None) -> None:
    for value in values or []:
        target.extend([option, *value])


def directory_has_anything(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def has_augmentation_on(diagnostic_root: Path) -> bool:
    path = diagnostic_root / "adaptive_selection.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return any(str(row.get("augmentation") or "").strip().upper() == "ON" for row in rows)


def lock_valid(root: Path, lock_name: str) -> bool:
    lock_path = root / lock_name
    if not lock_path.is_file():
        return False
    verify_hash_lock(root, lock_name)
    return True


def require_current_transformer_lock(args: argparse.Namespace) -> None:
    """Refuse to treat an obsolete XLM-R result as a current completed suite."""
    root = ROBUSTNESS_ROOT / "Classifier" / "XLMRBase"
    lock = verify_hash_lock(root, "TRANSFORMER_ROBUSTNESS_LOCK.json")
    configuration = lock.get("configuration") or {}
    expected = {
        "model_id": str(args.transformer_model),
        "epochs": TRANSFORMER_FIXED_EPOCHS,
        "training_schedule": "fixed_epoch_budget_no_early_stopping",
        "checkpoint_selection": "development_macro_f1_then_auroc",
        "train_batch_size": int(args.transformer_train_batch_size),
        "eval_batch_size": int(args.transformer_eval_batch_size),
        "max_length": int(args.transformer_max_length),
    }
    mismatches = {
        key: (configuration.get(key), value)
        for key, value in expected.items()
        if configuration.get(key) != value
    }
    if lock.get("lock_version") != TRANSFORMER_LOCK_VERSION or mismatches:
        raise RuntimeError(
            "obsolete XLM-R robustness result detected; remove only "
            f"{root} and rerun run_all.py so the fixed 20-epoch protocol is executed. "
            f"lock_version={lock.get('lock_version')!r}, mismatches={mismatches}"
        )


def experiment_lock_valid_or_rebuildable(root: Path) -> bool:
    """Treat a parent experiment lock with missing child locks as rebuildable.

    Hash mismatches on files that still exist remain fatal.  A missing child
    lock means that child stage can be reproduced and the parent lock can then
    be regenerated from the newly verified children.
    """
    lock_path = root / "BINARY_MATCHED_SIZE_EXPERIMENT_LOCK.json"
    if not lock_path.is_file():
        return False
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    files = lock.get("files") or {}
    missing = False
    for relative, expected in files.items():
        path = root / str(relative)
        if not path.is_file():
            missing = True
            continue
        if sha256_file(path) != str(expected):
            raise RuntimeError(
                f"primary experiment lock hash mismatch for existing artifact; refusing overwrite: {path}"
            )
    return not missing


def require_safe_missing_stage(root: Path, *, resumable_file: str | None = None) -> bool:
    """Return whether a missing stage may resume; otherwise require an empty root.

    Completed stages are handled before this function is called.  We never
    delete or overwrite an ambiguous partial stage.  Only stages with an
    explicit resume database may continue from a non-empty directory.
    """
    if not root.exists() or not any(root.iterdir()):
        return False
    if resumable_file is not None and (root / resumable_file).is_file():
        return True
    raise RuntimeError(
        f"stage output exists without a valid completion lock and is not safely resumable; "
        f"refusing overwrite: {root}"
    )


def completion_lock_exists(relative_root: str, lock_name: str) -> bool:
    return (PRIMARY_ROOT / relative_root / lock_name).is_file()


def guard_against_upstream_holes() -> None:
    """Never combine regenerated upstream data with stale completed downstream data."""
    experiment_complete = experiment_lock_valid_or_rebuildable(PRIMARY_ROOT)
    eval_complete = completion_lock_exists("Embeddings/Qwen3Eval", "QWEN_EVAL_EMBEDDING_LOCK.json")
    development_complete = completion_lock_exists("Downstream/DevelopmentGrid", "DEVELOPMENT_GRID_LOCK.json")
    weight_complete = completion_lock_exists("Downstream/SyntheticWeightGrid", "SYNTHETIC_WEIGHT_GRID_LOCK.json")
    adaptive_complete = completion_lock_exists("Downstream/AdaptivePolicy", "BINARY_ADAPTIVE_PROTOCOL_LOCK.json")
    diagnostic_complete = completion_lock_exists(
        "Downstream/AdaptivePolicy/DecisionBoundaryDiagnostic", "ADAPTIVE_SELECTION_LOCK.json"
    )
    pretest_complete = completion_lock_exists(
        "Downstream/AdaptivePolicy/OneShotTest", "PRETEST_THRESHOLD_LOCK.json"
    )
    one_shot_complete = completion_lock_exists(
        "Downstream/AdaptivePolicy/OneShotTest", "ADAPTIVE_ONE_SHOT_TEST_LOCK.json"
    )
    friedman_complete = completion_lock_exists(
        "Downstream/AdaptivePolicy/OneShotTest/FinalThreeConditionFriedmanV1",
        "FRIEDMAN_ANALYSIS_LOCK.json",
    )
    ordered = [
        ("primary generation/embedding bundle", experiment_complete),
        ("evaluation embeddings", eval_complete),
        ("development grid", development_complete),
        ("synthetic-weight grid", weight_complete),
        ("adaptive policy", adaptive_complete),
        ("decision-boundary diagnostic", diagnostic_complete),
        ("pretest threshold freeze", pretest_complete),
        ("one-shot evaluation", one_shot_complete),
        ("Friedman analysis", friedman_complete),
    ]
    seen_missing = False
    missing_name = ""
    for name, complete in ordered:
        if not complete and not seen_missing:
            seen_missing = True
            missing_name = name
            continue
        if seen_missing and complete:
            raise RuntimeError(
                f"unsafe primary-result hole: upstream stage '{missing_name}' is missing while "
                f"downstream stage '{name}' is completed. Refusing to mix regenerated upstream "
                "artifacts with stale downstream results. Preserve the existing results and "
                "reproduce the dependent suffix in a clean result tree instead."
            )


def run_primary_missing(args: argparse.Namespace) -> None:
    """Execute only missing primary stages, in dependency order.

    Valid completed locks are always skipped. Missing stages are recreated;
    resumable SQLite stages continue only when their existing DB contract can
    be checked by the stage itself. Ambiguous partial artifacts abort rather
    than being overwritten.
    """
    guard_against_upstream_holes()
    common = [
        "--timeout-seconds", str(args.timeout_seconds),
        "--embedding-batch-size", str(args.embedding_batch_size),
        "--export-shard-size", str(args.export_shard_size),
    ]
    if args.api_key:
        common.extend(["--api-key", args.api_key])
    append_endpoints(common, "--endpoint", args.gemma_endpoint)
    append_endpoints(common, "--qwen-endpoint", args.qwen_embedding_endpoint)
    append_endpoints(common, "--bge-endpoint", args.bge_embedding_endpoint)

    experiment_lock_valid = experiment_lock_valid_or_rebuildable(PRIMARY_ROOT)
    if experiment_lock_valid:
        print("SKIP primary generation/train embeddings: verified experiment lock")
    else:
        generation_root = PRIMARY_ROOT / "Generation"
        qwen_root = PRIMARY_ROOT / "Embeddings" / "Qwen3"
        bge_root = PRIMARY_ROOT / "Embeddings" / "BGE_M3"
        generation_done = lock_valid(generation_root, "GENERATION_LOCK.json")
        qwen_done = lock_valid(qwen_root, "QWEN_EMBEDDING_LOCK.json")
        bge_done = lock_valid(bge_root, "BGE_EMBEDDING_LOCK.json")
        resume_generation = False
        if not generation_done:
            resume_generation = require_safe_missing_stage(
                generation_root, resumable_file="generation_status.sqlite3"
            )
        if not qwen_done:
            require_safe_missing_stage(qwen_root)
        if not bge_done:
            require_safe_missing_stage(bge_root)
        stage_args = list(common)
        if resume_generation:
            stage_args.append("--resume")
        run_stage("run_binary_matched_size_experiment", *stage_args)
        verify_hash_lock(PRIMARY_ROOT, "BINARY_MATCHED_SIZE_EXPERIMENT_LOCK.json")

    eval_args = [
        "--timeout-seconds", str(args.timeout_seconds),
        "--embedding-batch-size", str(args.embedding_batch_size),
    ]
    if args.api_key:
        eval_args.extend(["--api-key", args.api_key])
    append_endpoints(eval_args, "--qwen-endpoint", args.qwen_embedding_endpoint)
    eval_root = PRIMARY_ROOT / "Embeddings" / "Qwen3Eval"
    if lock_valid(eval_root, "QWEN_EVAL_EMBEDDING_LOCK.json"):
        print("SKIP primary eval embeddings: verified lock")
    else:
        require_safe_missing_stage(eval_root)
        run_stage("embed_binary_eval_splits", *eval_args)
        verify_hash_lock(eval_root, "QWEN_EVAL_EMBEDDING_LOCK.json")

    development_root = PRIMARY_ROOT / "Downstream" / "DevelopmentGrid"
    if lock_valid(development_root, "DEVELOPMENT_GRID_LOCK.json"):
        print("SKIP primary development grid: verified lock")
    else:
        resume = require_safe_missing_stage(
            development_root, resumable_file="downstream_status.sqlite3"
        )
        stage_args = [
            "--manifest-root", str(PROJECT_ROOT / "Data" / "ExperimentManifests" / "v1"),
            "--output-root", str(PRIMARY_ROOT),
            "--workers", str(args.workers),
            "--progress-every", str(args.progress_every),
        ]
        if resume:
            stage_args.append("--resume")
        run_stage("run_binary_downstream_experiment", *stage_args)
        verify_hash_lock(development_root, "DEVELOPMENT_GRID_LOCK.json")

    selection_root = PRIMARY_ROOT / "Downstream" / "DevelopmentSelection"
    if lock_valid(selection_root, "CONDITION_SELECTION_LOCK.json"):
        print("SKIP primary development summary: verified lock")
    else:
        require_safe_missing_stage(selection_root)
        run_stage("summarize_binary_downstream_results", "--output-root", str(PRIMARY_ROOT))
        verify_hash_lock(selection_root, "CONDITION_SELECTION_LOCK.json")

    weight_root = PRIMARY_ROOT / "Downstream" / "SyntheticWeightGrid"
    if lock_valid(weight_root, "SYNTHETIC_WEIGHT_GRID_LOCK.json"):
        print("SKIP primary synthetic-weight grid: verified lock")
    else:
        resume = require_safe_missing_stage(
            weight_root, resumable_file="synthetic_weight_status.sqlite3"
        )
        stage_args = [
            "--manifest-root", str(PROJECT_ROOT / "Data" / "ExperimentManifests" / "v1"),
            "--output-root", str(PRIMARY_ROOT),
            "--workers", str(args.workers),
            "--progress-every", str(args.progress_every),
        ]
        if resume:
            stage_args.append("--resume")
        run_stage("run_binary_synthetic_weight_grid", *stage_args)
        verify_hash_lock(weight_root, "SYNTHETIC_WEIGHT_GRID_LOCK.json")

    weight_root = PRIMARY_ROOT / "Downstream" / "SyntheticWeightGrid"
    adaptive_root = PRIMARY_ROOT / "Downstream" / "AdaptivePolicy"
    if lock_valid(adaptive_root, "BINARY_ADAPTIVE_PROTOCOL_LOCK.json"):
        print("SKIP primary adaptive policy: verified lock")
    else:
        require_safe_missing_stage(adaptive_root)
        run_stage(
            "freeze_binary_adaptive_protocol",
            "--weight-grid-root", str(weight_root),
            "--output-root", str(adaptive_root),
        )
        verify_hash_lock(adaptive_root, "BINARY_ADAPTIVE_PROTOCOL_LOCK.json")

    diagnostic_root = adaptive_root / "DecisionBoundaryDiagnostic"
    if lock_valid(diagnostic_root, "ADAPTIVE_SELECTION_LOCK.json"):
        print("SKIP primary decision-boundary diagnostic: verified lock")
    else:
        require_safe_missing_stage(diagnostic_root)
        run_stage(
            "run_binary_decision_boundary_diagnostic",
            "--prepared-root", str(PROJECT_ROOT / "Data" / "Prepared"),
            "--manifest-root", str(PROJECT_ROOT / "Data" / "ExperimentManifests" / "v1"),
            "--experiment-root", str(PRIMARY_ROOT),
            "--eval-embedding-root", str(PRIMARY_ROOT / "Embeddings" / "Qwen3Eval"),
            "--weight-grid-root", str(weight_root),
            "--adaptive-root", str(adaptive_root),
            "--output-root", str(diagnostic_root),
        )
        verify_hash_lock(diagnostic_root, "ADAPTIVE_SELECTION_LOCK.json")

    one_shot_root = adaptive_root / "OneShotTest"
    one_common = [
        "--prepared-root", str(PROJECT_ROOT / "Data" / "Prepared"),
        "--manifest-root", str(PROJECT_ROOT / "Data" / "ExperimentManifests" / "v1"),
        "--experiment-root", str(PRIMARY_ROOT),
        "--eval-embedding-root", str(PRIMARY_ROOT / "Embeddings" / "Qwen3Eval"),
        "--weight-grid-root", str(weight_root),
        "--adaptive-diagnostic-root", str(diagnostic_root),
        "--output-root", str(one_shot_root),
    ]
    if lock_valid(one_shot_root, "PRETEST_THRESHOLD_LOCK.json"):
        print("SKIP primary pretest threshold freeze: verified lock")
    else:
        if one_shot_root.exists() and any(one_shot_root.iterdir()):
            raise RuntimeError(
                f"one-shot output exists without PRETEST_THRESHOLD_LOCK; refusing overwrite: {one_shot_root}"
            )
        run_stage("run_binary_adaptive_one_shot_test", *one_common, "--prepare-only")
        verify_hash_lock(one_shot_root, "PRETEST_THRESHOLD_LOCK.json")

    if lock_valid(one_shot_root, "ADAPTIVE_ONE_SHOT_TEST_LOCK.json"):
        print("SKIP primary one-shot evaluation: verified lock")
    else:
        if (one_shot_root / "TEST_ACCESS_STARTED.json").is_file():
            raise RuntimeError(
                "TEST_ACCESS_STARTED.json exists without a final one-shot lock; refusing automatic rerun"
            )
        run_stage("run_binary_adaptive_one_shot_test", *one_common, "--evaluate")
        verify_hash_lock(one_shot_root, "ADAPTIVE_ONE_SHOT_TEST_LOCK.json")

    if has_augmentation_on(diagnostic_root):
        analysis_root = one_shot_root / "FinalThreeConditionFriedmanV1"
        if lock_valid(analysis_root, "FRIEDMAN_ANALYSIS_LOCK.json"):
            print("SKIP primary Friedman analysis: verified lock")
        else:
            require_safe_missing_stage(analysis_root)
            run_stage("analyze_binary_friedman", "--input-root", str(one_shot_root))
            verify_hash_lock(analysis_root, "FRIEDMAN_ANALYSIS_LOCK.json")
    primary_lock_snapshot(PRIMARY_ROOT)


def verified_or_none(root: Path, lock_name: str) -> Path | None:
    if not (root / lock_name).is_file():
        return None
    verify_hash_lock(root, lock_name)
    return root


def endpoint_args(option: str, values: list[list[str]] | None) -> list[str]:
    result: list[str] = []
    append_endpoints(result, option, values)
    return result


def run_robustness(args: argparse.Namespace) -> list[Path]:
    locks: list[Path] = []
    main_generation = PRIMARY_ROOT / "Generation"
    main_qwen = PRIMARY_ROOT / "Embeddings" / "Qwen3"
    main_qwen_eval = PRIMARY_ROOT / "Embeddings" / "Qwen3Eval"
    main_bge = PRIMARY_ROOT / "Embeddings" / "BGE_M3"

    # Baseline fixed-condition LR arm from immutable primary artifacts.
    gemma_eval = ROBUSTNESS_ROOT / "Generator" / "GemmaOriginal" / "FixedEvaluation"
    run_stage(
        "run_fixed_robustness",
        "--arm", "sentiment_gemma_original_qwen_lr",
        "--task", "sentiment",
        "--generation-root", str(main_generation),
        "--real-embedding-root", str(main_qwen),
        "--synthetic-embedding-root", str(main_qwen),
        "--test-embedding-root", str(main_qwen_eval),
        "--output-root", str(gemma_eval),
        *( ["--dry-run"] if args.dry_run else [] ),
    )
    if not args.dry_run:
        locks.append(gemma_eval / "FIXED_ROBUSTNESS_LOCK.json")

    # Alternative generator: Qwen3.8-27B, exactly three candidates/seed, thinking OFF.
    qwen_arm = ROBUSTNESS_ROOT / "Generator" / "Qwen38"
    generation_args = [
        "--task", "sentiment",
        "--output-root", str(qwen_arm),
        "--timeout-seconds", str(args.timeout_seconds),
        "--export-shard-size", str(args.export_shard_size),
    ]
    if args.api_key:
        generation_args.extend(["--api-key", args.api_key])
    append_endpoints(generation_args, "--endpoint", args.qwen38_endpoint)
    if args.dry_run:
        generation_args.append("--dry-run")
    run_stage("generate_robustness_candidates", *generation_args)
    qwen_synth = qwen_arm / "Embeddings" / "Qwen3Synthetic"
    embed_args = [
        "--task", "sentiment", "--mode", "synthetic", "--family", "qwen",
        "--generation-root", str(qwen_arm / "Generation"),
        "--output-root", str(qwen_synth),
        "--timeout-seconds", str(args.timeout_seconds),
        "--embedding-batch-size", str(args.embedding_batch_size),
    ]
    if args.api_key:
        embed_args.extend(["--api-key", args.api_key])
    append_endpoints(embed_args, "--endpoint", args.qwen_embedding_endpoint)
    if args.dry_run:
        embed_args.append("--dry-run")
    run_stage("embed_robustness", *embed_args)
    qwen_eval = qwen_arm / "FixedEvaluation"
    run_stage(
        "run_fixed_robustness",
        "--arm", "sentiment_qwen38_qwen_lr",
        "--task", "sentiment",
        "--generation-root", str(qwen_arm / "Generation"),
        "--real-embedding-root", str(main_qwen),
        "--synthetic-embedding-root", str(qwen_synth),
        "--test-embedding-root", str(main_qwen_eval),
        "--output-root", str(qwen_eval),
        *( ["--dry-run"] if args.dry_run else [] ),
    )
    if not args.dry_run:
        locks.extend([
            qwen_arm / "Generation" / "GENERATION_LOCK.json",
            qwen_synth / "ROBUSTNESS_EMBEDDING_LOCK.json",
            qwen_eval / "FIXED_ROBUSTNESS_LOCK.json",
        ])

    # Alternative representation: BGE-M3 as downstream classifier representation.
    bge_arm = ROBUSTNESS_ROOT / "Embedding" / "BGE_M3"
    bge_test = bge_arm / "TestEmbeddings"
    bge_args = [
        "--task", "sentiment", "--mode", "test", "--family", "bge",
        "--output-root", str(bge_test),
        "--timeout-seconds", str(args.timeout_seconds),
        "--embedding-batch-size", str(args.embedding_batch_size),
    ]
    if args.api_key:
        bge_args.extend(["--api-key", args.api_key])
    append_endpoints(bge_args, "--endpoint", args.bge_embedding_endpoint)
    if args.dry_run:
        bge_args.append("--dry-run")
    run_stage("embed_robustness", *bge_args)
    bge_eval = bge_arm / "FixedEvaluation"
    run_stage(
        "run_fixed_robustness",
        "--arm", "sentiment_gemma_bge_lr",
        "--task", "sentiment",
        "--generation-root", str(main_generation),
        "--real-embedding-root", str(main_bge),
        "--synthetic-embedding-root", str(main_bge),
        "--test-embedding-root", str(bge_test),
        "--output-root", str(bge_eval),
        *( ["--dry-run"] if args.dry_run else [] ),
    )
    if not args.dry_run:
        locks.extend([
            bge_test / "ROBUSTNESS_EMBEDDING_LOCK.json",
            bge_eval / "FIXED_ROBUSTNESS_LOCK.json",
        ])

    # Transformer classifier sensitivity, directly addressing Reviewer 3 #4.
    transformer_root = ROBUSTNESS_ROOT / "Classifier" / "XLMRBase"
    transformer_args = [
        "--generation-root", str(main_generation),
        "--output-root", str(transformer_root),
        "--model-id", args.transformer_model,
        "--train-batch-size", str(args.transformer_train_batch_size),
        "--eval-batch-size", str(args.transformer_eval_batch_size),
        "--max-length", str(args.transformer_max_length),
    ]
    if args.dry_run:
        transformer_args.append("--dry-run")
    run_stage("run_transformer_robustness", *transformer_args)
    if not args.dry_run:
        locks.append(transformer_root / "TRANSFORMER_ROBUSTNESS_LOCK.json")

    # Second task: offensive-language detection.  Existing OFF generation is never used.
    offensive_arm = ROBUSTNESS_ROOT / "Task" / "OffensiveLanguage"
    offensive_generation_args = [
        "--task", "offensive",
        "--output-root", str(offensive_arm),
        "--timeout-seconds", str(args.timeout_seconds),
        "--export-shard-size", str(args.export_shard_size),
    ]
    if args.api_key:
        offensive_generation_args.extend(["--api-key", args.api_key])
    append_endpoints(offensive_generation_args, "--endpoint", args.gemma_endpoint)
    if args.dry_run:
        offensive_generation_args.append("--dry-run")
    run_stage("generate_robustness_candidates", *offensive_generation_args)

    offensive_synth = offensive_arm / "Embeddings" / "Qwen3Synthetic"
    offensive_synth_args = [
        "--task", "offensive", "--mode", "synthetic", "--family", "qwen",
        "--generation-root", str(offensive_arm / "Generation"),
        "--output-root", str(offensive_synth),
        "--timeout-seconds", str(args.timeout_seconds),
        "--embedding-batch-size", str(args.embedding_batch_size),
    ]
    if args.api_key:
        offensive_synth_args.extend(["--api-key", args.api_key])
    append_endpoints(offensive_synth_args, "--endpoint", args.qwen_embedding_endpoint)
    if args.dry_run:
        offensive_synth_args.append("--dry-run")
    run_stage("embed_robustness", *offensive_synth_args)

    # Real/test embeddings do not depend on the invalid OFF generation. Reuse only if their
    # locks verify; otherwise build clean robustness copies without deleting anything.
    pilot_real = verified_or_none(OFFENSIVE_PILOT_ROOT / "Embeddings" / "Qwen3", "QWEN_EMBEDDING_LOCK.json")
    pilot_test = verified_or_none(OFFENSIVE_PILOT_ROOT / "Embeddings" / "Qwen3Eval", "QWEN_EVAL_EMBEDDING_LOCK.json")
    if pilot_real is None:
        offensive_real = offensive_arm / "Embeddings" / "Qwen3Real"
        real_args = [
            "--task", "offensive", "--mode", "real", "--family", "qwen",
            "--output-root", str(offensive_real),
            "--timeout-seconds", str(args.timeout_seconds),
            "--embedding-batch-size", str(args.embedding_batch_size),
        ]
        if args.api_key:
            real_args.extend(["--api-key", args.api_key])
        append_endpoints(real_args, "--endpoint", args.qwen_embedding_endpoint)
        if args.dry_run:
            real_args.append("--dry-run")
        run_stage("embed_robustness", *real_args)
    else:
        offensive_real = pilot_real
        print(f"REUSE verified offensive real embeddings: {offensive_real}")
    if pilot_test is None:
        offensive_test = offensive_arm / "Embeddings" / "Qwen3Test"
        test_args = [
            "--task", "offensive", "--mode", "test", "--family", "qwen",
            "--output-root", str(offensive_test),
            "--timeout-seconds", str(args.timeout_seconds),
            "--embedding-batch-size", str(args.embedding_batch_size),
        ]
        if args.api_key:
            test_args.extend(["--api-key", args.api_key])
        append_endpoints(test_args, "--endpoint", args.qwen_embedding_endpoint)
        if args.dry_run:
            test_args.append("--dry-run")
        run_stage("embed_robustness", *test_args)
    else:
        offensive_test = pilot_test
        print(f"REUSE verified offensive test embeddings: {offensive_test}")

    offensive_eval = offensive_arm / "FixedEvaluation"
    run_stage(
        "run_fixed_robustness",
        "--arm", "offensive_gemma_qwen_lr",
        "--task", "offensive",
        "--generation-root", str(offensive_arm / "Generation"),
        "--real-embedding-root", str(offensive_real),
        "--synthetic-embedding-root", str(offensive_synth),
        "--test-embedding-root", str(offensive_test),
        "--output-root", str(offensive_eval),
        *( ["--dry-run"] if args.dry_run else [] ),
    )
    if not args.dry_run:
        locks.extend([
            offensive_arm / "Generation" / "GENERATION_LOCK.json",
            offensive_synth / "ROBUSTNESS_EMBEDDING_LOCK.json",
            offensive_eval / "FIXED_ROBUSTNESS_LOCK.json",
        ])
    return locks


def consolidate(locks: Sequence[Path], *, refresh_existing: bool = False) -> None:
    sources = (
        ("sentiment_gemma_original_qwen_lr", ROBUSTNESS_ROOT / "Generator" / "GemmaOriginal" / "FixedEvaluation" / "paired_summary.csv"),
        ("sentiment_qwen38_qwen_lr", ROBUSTNESS_ROOT / "Generator" / "Qwen38" / "FixedEvaluation" / "paired_summary.csv"),
        ("sentiment_gemma_bge_lr", ROBUSTNESS_ROOT / "Embedding" / "BGE_M3" / "FixedEvaluation" / "paired_summary.csv"),
        ("sentiment_gemma_xlmr", ROBUSTNESS_ROOT / "Classifier" / "XLMRBase" / "paired_summary.csv"),
        ("offensive_gemma_qwen_lr", ROBUSTNESS_ROOT / "Task" / "OffensiveLanguage" / "FixedEvaluation" / "paired_summary.csv"),
    )
    summary_csv = ROBUSTNESS_ROOT / "robustness_summary.csv"
    summary_json = ROBUSTNESS_ROOT / "robustness_summary.json"
    suite_lock = ROBUSTNESS_ROOT / SUITE_LOCK
    existing_summary = summary_csv.exists() or summary_json.exists() or suite_lock.exists()
    if existing_summary:
        if not refresh_existing or not suite_lock.is_file():
            raise RuntimeError(
                "robustness summary artifacts already exist without a verified suite skip; "
                "refusing automatic replacement"
            )
        for lock_path in locks:
            verify_hash_lock(lock_path.parent, lock_path.name)
        print(
            "REFRESH robustness summary: verified component locks changed after the previous "
            "suite consolidation; replacing derived summary artifacts only"
        )
    rows: list[dict[str, str]] = []
    for arm, path in sources:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"arm": arm, **row})
    fields = ("arm", "cell_id", "comparison", "metric", "mean_difference", "ci_lower", "ci_upper")
    write_csv(summary_csv, rows, fields)
    write_json(
        summary_json,
        {"robustness_version": ROBUSTNESS_VERSION, "rows": rows},
    )
    lock_files = {
        path.relative_to(ROBUSTNESS_ROOT).as_posix(): sha256_file(path)
        for path in locks
    }
    lock_files["robustness_summary.csv"] = sha256_file(summary_csv)
    lock_files["robustness_summary.json"] = sha256_file(summary_json)
    write_json(
        suite_lock,
        {
            "lock_version": "r3-robustness-suite-v1",
            "status": "completed",
            "robustness_version": ROBUSTNESS_VERSION,
            "files": lock_files,
        },
    )
    verify_hash_lock(ROBUSTNESS_ROOT, SUITE_LOCK)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--qwen38-endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--qwen-embedding-endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--bge-embedding-endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--export-shard-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--transformer-model", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--transformer-train-batch-size", type=int, default=16)
    parser.add_argument("--transformer-eval-batch-size", type=int, default=64)
    parser.add_argument("--transformer-max-length", type=int, default=256)
    parser.add_argument(
        "--candidate-sensitivity-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument("--candidate-sensitivity-progress-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate paper figures from existing result artifacts and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.figures_only:
        run_figures()
        return 0
    if min(
        args.timeout_seconds,
        args.embedding_batch_size,
        args.export_shard_size,
        args.workers,
        args.progress_every,
        args.transformer_train_batch_size,
        args.transformer_eval_batch_size,
        args.transformer_max_length,
        args.candidate_sensitivity_workers,
        args.candidate_sensitivity_progress_every,
    ) <= 0:
        raise SystemExit("numeric execution options must be positive")
    if args.dry_run:
        before = primary_lock_snapshot(PRIMARY_ROOT)
        print("PRIMARY COMPLETE: all required locks verified; dry-run will not modify primary outputs")
    else:
        run_primary_missing(args)
        before = primary_lock_snapshot(PRIMARY_ROOT)
        print("PRIMARY COMPLETE: every required stage is present and verified")

    if not args.dry_run:
        run_stage("run_qualitative_audit")

    candidate_args = [
        "--workers", str(args.candidate_sensitivity_workers),
        "--progress-every", str(args.candidate_sensitivity_progress_every),
    ]
    if args.dry_run:
        candidate_args.append("--dry-run")
    run_stage("run_candidate_count_sensitivity", *candidate_args)

    suite_lock = ROBUSTNESS_ROOT / SUITE_LOCK
    refresh_suite_summary = False
    if suite_lock.is_file() and not args.dry_run:
        try:
            verify_hash_lock(ROBUSTNESS_ROOT, SUITE_LOCK)
        except RuntimeError as exc:
            print(
                "ROBUSTNESS SUITE NEEDS REBUILD: existing suite lock no longer verifies; "
                f"continuing with stage-level validation ({exc})"
            )
            refresh_suite_summary = True
        else:
            require_current_transformer_lock(args)
            assert_primary_unchanged(PRIMARY_ROOT, before)
            run_figures()
            print(f"ALL COMPLETE: verified robustness suite lock {suite_lock}")
            return 0

    locks = run_robustness(args)
    if not args.dry_run:
        consolidate(locks, refresh_existing=refresh_suite_summary)
    assert_primary_unchanged(PRIMARY_ROOT, before)
    if not args.dry_run:
        run_figures()
    print(
        "DRY RUN COMPLETE"
        if args.dry_run
        else "PRIMARY + R1 CANDIDATE SENSITIVITY + R3 ROBUSTNESS COMPLETE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

