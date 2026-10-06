"""Generate compact, thinking-enabled candidates for Reviewer-3 robustness arms.

Only three candidates are generated for each locked experiment-bank row.  The
original 140k-candidate sentiment experiment is never modified by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
_SOURCES_ROOT = _MODULE_ROOT.parent
for candidate in (_MODULE_ROOT, _SOURCES_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from Common.lmstudio import (  # noqa: E402
    DEFAULT_LMSTUDIO_API_URL,
    LmStudioGenerationSettings,
    loaded_lmstudio_models,
    parse_lmstudio_endpoints,
)
import generate_binary_candidates as engine  # noqa: E402
from robustness_common import (  # noqa: E402
    CANDIDATES_PER_SEED,
    ROBUSTNESS_VERSION,
    canonical_json,
    sha256_bytes,
    task_spec,
    verify_hash_lock,
)


QWEN_MODEL_ID = "qwen/qwen3.8-27b"
GEMMA_MODEL_ID = "google/gemma-4-31b"
REQUESTS_PER_CELL = 1_400 * CANDIDATES_PER_SEED
TOTAL_REQUESTS = REQUESTS_PER_CELL * 5


def load_offensive_source() -> Any:
    path = _SOURCES_ROOT / "OffensiveLanguage" / "generate_binary_candidates.py"
    module_dir = str(path.parent)
    inserted = module_dir not in sys.path
    if inserted:
        sys.path.insert(0, module_dir)
    try:
        spec = importlib.util.spec_from_file_location("r3_offensive_generation_source", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load offensive generation source: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            sys.path.remove(module_dir)


def source_module(task: str) -> Any:
    return engine if task == "sentiment" else load_offensive_source()


def expected_model(task: str) -> str:
    return QWEN_MODEL_ID if task == "sentiment" else GEMMA_MODEL_ID


def expected_concurrency(task: str) -> int:
    return 24 if task == "sentiment" else 12


def expected_thinking(task: str) -> bool:
    # Only the alternative Qwen generator uses thinking disabled.  The primary
    # sentiment Gemma run is untouched, and the OffensiveLanguage Gemma
    # robustness arm remains thinking-enabled.
    return task != "sentiment"


def configure_engine(source: Any, task: str, model_id: str) -> None:
    """Reuse the audited primary generation engine under a compact contract."""
    spec = task_spec(task)
    experiment_id = (
        "r3_sentiment_qwen38_generator"
        if task == "sentiment"
        else "r3_offensive_gemma_task"
    )
    engine.EXPERIMENT_NAME = f"R3 Targeted Robustness Generation ({task})"
    engine.EXPERIMENT_ID = experiment_id
    engine.PATH_ID = "BinaryMatchedSizeExperiment/Robustness/v1"
    engine.PROTOCOL_VERSION = ROBUSTNESS_VERSION
    engine.DATABASE_SCHEMA_VERSION = "r3-targeted-generation-v1"
    engine.REQUESTS_PER_CELL = REQUESTS_PER_CELL
    engine.TOTAL_REQUESTS = TOTAL_REQUESTS
    engine.CANDIDATES_PER_SEED = CANDIDATES_PER_SEED
    engine.TARGET_CELLS = tuple(spec.cells)
    engine.CELL_LANGUAGE = dict(spec.languages)
    engine.INHERITED_LABEL = dict(spec.label_names)
    engine.ALLOWED_LABELS = set(spec.label_names.values())
    engine.PROMPT_ID = str(source.PROMPT_ID)
    engine.PROMPT_TEMPLATES = dict(source.PROMPT_TEMPLATES)
    engine.BINARY_OUTPUT_SCHEMA = dict(source.BINARY_OUTPUT_SCHEMA)
    engine.RESPONSE_FORMAT = dict(source.RESPONSE_FORMAT)
    engine.STRICT_JSON_WRAPPER = str(source.STRICT_JSON_WRAPPER)
    engine.LABEL_MISMATCH_RETRY = str(source.LABEL_MISMATCH_RETRY)
    engine.REPETITION_RETRY = str(source.REPETITION_RETRY)
    engine.WORD_LIMIT_RETRY = str(source.WORD_LIMIT_RETRY)
    engine.SURFACE_CORRUPTION_RETRY = str(source.SURFACE_CORRUPTION_RETRY)
    engine.MODEL_ID = model_id
    thinking = expected_thinking(task)
    engine.GENERATION_SETTINGS = LmStudioGenerationSettings(
        temperature=0.8,
        top_p=0.9,
        max_tokens=256,
        repeat_penalty=1.0,
        enable_thinking=thinking,
        stream=False,
    )
    engine.GENERATION_CONFIG = {
        "model": model_id,
        "enable_thinking": thinking,
        "temperature": 0.8,
        "top_p": 0.9,
        "repeat_penalty": 1.0,
        "max_tokens": 256,
        "stream": False,
        "response_format": "strict_json_schema",
        "seed_mode": "request_specific_attempt_seed",
        "robustness_version": ROBUSTNESS_VERSION,
        "candidates_per_seed": CANDIDATES_PER_SEED,
    }
    if engine.GENERATION_SETTINGS.enable_thinking is not thinking:
        raise RuntimeError("robustness generation settings lost the task-specific thinking contract")
    if engine.GENERATION_CONFIG.get("enable_thinking") is not thinking:
        raise RuntimeError("robustness generation config lost the task-specific thinking contract")


def compact_plan(source: Any, task: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = task_spec(task)
    lock_info, _banks, full_plan = source.load_and_validate_inputs(
        spec.prepared_root.resolve(), spec.manifest_root.resolve()
    )
    plan = [dict(row) for row in full_plan if int(row["candidate_index"]) < CANDIDATES_PER_SEED]
    if len(plan) != TOTAL_REQUESTS:
        raise RuntimeError(f"compact plan size {len(plan)} != {TOTAL_REQUESTS}")
    counts = Counter(str(row["cell_id"]) for row in plan)
    if any(counts[cell_id] != REQUESTS_PER_CELL for cell_id in spec.cells):
        raise RuntimeError(f"compact per-cell counts invalid: {dict(counts)}")
    # Hash exactly the compact protocol rows rather than the original 140k plan.
    plan_hash = sha256_bytes(
        ("\n".join(canonical_json(row) for row in plan) + "\n").encode("utf-8")
    )
    lock_info = dict(lock_info)
    lock_info["generation_plan_sha256"] = plan_hash
    return lock_info, plan


def prompt_hashes(source: Any, task: str) -> dict[str, str]:
    spec = task_spec(task)
    return {
        cell_id: sha256_bytes(
            str(source.PROMPT_TEMPLATES[spec.languages[cell_id]]).encode("utf-8")
        )
        for cell_id in spec.cells
    }


def validate_completed_generation(generation_root: Path, model_id: str, task: str) -> None:
    verify_hash_lock(generation_root, "GENERATION_LOCK.json")
    report_path = generation_root / "generation_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"completed generation is missing report: {report_path}")
    report = engine.read_json(report_path)
    config = report.get("generation_request_configuration") or {}
    if report.get("generation_model_identifier") != model_id:
        raise RuntimeError("existing generation lock belongs to a different model")
    if config.get("enable_thinking") is not expected_thinking(task):
        raise RuntimeError("existing robustness generation has the wrong thinking configuration")
    if int(config.get("candidates_per_seed", -1)) != CANDIDATES_PER_SEED:
        raise RuntimeError("existing robustness generation uses the wrong candidate count")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("sentiment", "offensive"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--export-shard-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(repository_root=root)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0 or args.export_shard_size <= 0:
        raise SystemExit("timeout and export shard size must be positive")
    task = str(args.task)
    model_id = expected_model(task)
    source = source_module(task)
    lock_info, plan = compact_plan(source, task)
    hashes = prompt_hashes(source, task)
    configure_engine(source, task, model_id)
    contract = engine.metadata_contract(lock_info, hashes)
    output_root = args.output_root.resolve()
    generation_root = output_root / "Generation"

    summary = {
        "robustness_version": ROBUSTNESS_VERSION,
        "task": task,
        "model": model_id,
        "enable_thinking": expected_thinking(task),
        "max_tokens": 256,
        "candidates_per_seed": CANDIDATES_PER_SEED,
        "requests": len(plan),
        "per_cell": dict(Counter(row["cell_id"] for row in plan)),
        "generation_root": str(generation_root),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    lock_path = generation_root / "GENERATION_LOCK.json"
    database_path = generation_root / "generation_status.sqlite3"
    if lock_path.is_file():
        validate_completed_generation(generation_root, model_id, task)
        print(f"SKIP generation: verified completed lock at {lock_path}")
        return 0
    if generation_root.exists() and any(generation_root.iterdir()) and not database_path.is_file():
        raise RuntimeError(
            f"non-empty generation root without resumable DB; refusing to overwrite: {generation_root}"
        )

    endpoint_args = args.endpoint or [
        [DEFAULT_LMSTUDIO_API_URL, model_id, str(expected_concurrency(task))]
    ]
    endpoints = parse_lmstudio_endpoints(endpoint_args, model_id, DEFAULT_LMSTUDIO_API_URL)
    if any(model != model_id for _, model, _ in endpoints):
        raise RuntimeError(f"all robustness endpoints must use {model_id}")

    generation_root.mkdir(parents=True, exist_ok=True)
    connection = engine.open_database(database_path)
    resume = database_path.is_file() and connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0] > 0
    try:
        # Contract verification happens before writing or replacing any derived artifact.
        engine.initialize_database(connection, plan, contract, resume)
        engine.write_prompt_templates(generation_root)
        compact_path = generation_root / "compact_generation_plan.jsonl"
        if compact_path.exists() and sha256_bytes(compact_path.read_bytes()) != lock_info["generation_plan_sha256"]:
            raise RuntimeError("existing compact generation plan does not match the resumable contract")
        if not compact_path.exists():
            payload = ("\n".join(canonical_json(row) for row in plan) + "\n").encode("utf-8")
            compact_path.write_bytes(payload)
        engine.progress_log(
            f"robustness generation {'resume' if resume else 'new run'} | {engine.status_summary(connection)}"
        )
        with loaded_lmstudio_models(
            endpoints,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            log=engine.progress_log,
        ):
            engine.run_pending_generation(connection, endpoints, args.api_key, args.timeout_seconds)
        derived_paths = [
            generation_root / "generation_report.json",
            generation_root / "generation_report.csv",
            generation_root / "raw_outputs",
            generation_root / "valid_outputs",
            generation_root / "failures",
        ]
        if any(path.exists() for path in derived_paths):
            raise RuntimeError(
                "derived generation artifacts already exist without a completed lock; "
                "refusing automatic replacement"
            )
        report = engine.export_generation(
            connection,
            generation_root,
            lock_info,
            hashes,
            endpoints,
            args.export_shard_size,
        )
    finally:
        connection.close()
    if report["generation_request_configuration"].get("enable_thinking") is not expected_thinking(task):
        raise RuntimeError("exported robustness generation does not certify the expected thinking setting")
    validate_completed_generation(generation_root, model_id, task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

