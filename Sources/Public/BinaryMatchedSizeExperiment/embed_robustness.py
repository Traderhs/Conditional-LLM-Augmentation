"""Embed only the texts needed by the compact Reviewer-3 robustness protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
_SOURCES_ROOT = _MODULE_ROOT.parent
for candidate in (_MODULE_ROOT, _SOURCES_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from Common.lmstudio import (  # noqa: E402
    DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL,
    loaded_lmstudio_models,
    parse_lmstudio_endpoints,
)
import embed_binary_bge as bge  # noqa: E402
import embed_binary_qwen as qwen  # noqa: E402
from robustness_common import (  # noqa: E402
    ROBUSTNESS_VERSION,
    load_bank,
    load_split,
    load_valid_generation,
    lock_file_map,
    sha256_bytes,
    task_spec,
    verify_hash_lock,
    write_csv,
    write_json,
)


LOCK_NAME = "ROBUSTNESS_EMBEDDING_LOCK.json"
FIELDS = (
    "embedding_row_index",
    "cell_id",
    "record_type",
    "real_row_id",
    "candidate_index",
    "global_generation_index",
    "split_row_index",
    "label",
    "text_sha256",
    "embedding_dimension",
)


def family_config(family: str) -> dict[str, Any]:
    if family == "qwen":
        return {
            "endpoint_model": qwen.ENDPOINT_MODEL_ID,
            "tokenizer_id": qwen.TOKENIZER_ID,
            "dimension": qwen.EXPECTED_DIMENSION,
        }
    if family == "bge":
        return {
            "endpoint_model": bge.ENDPOINT_MODEL_ID,
            "tokenizer_id": bge.TOKENIZER_ID,
            "dimension": bge.EXPECTED_DIMENSION,
        }
    raise ValueError(f"unsupported embedding family: {family}")


def build_entries(
    task: str,
    mode: str,
    generation_root: Path | None,
) -> dict[str, list[dict[str, Any]]]:
    spec = task_spec(task)
    result: dict[str, list[dict[str, Any]]] = {}
    if mode == "synthetic":
        if generation_root is None:
            raise ValueError("synthetic embedding mode requires --generation-root")
        valid = load_valid_generation(generation_root, spec.cells, candidate_limit=None)
        for cell_id in spec.cells:
            result[cell_id] = [
                {
                    "cell_id": cell_id,
                    "record_type": "synthetic",
                    "real_row_id": str(row["real_row_id"]),
                    "candidate_index": int(row["candidate_index"]),
                    "global_generation_index": int(row["global_generation_index"]),
                    "split_row_index": None,
                    "label": int(row["label_int"]),
                    "text": str(row["generated_text"]),
                }
                for row in sorted(valid[cell_id].values(), key=lambda value: int(value["global_generation_index"]))
            ]
        return result
    if mode == "real":
        for cell_id in spec.cells:
            bank = load_bank(spec, cell_id)
            result[cell_id] = [
                {
                    "cell_id": cell_id,
                    "record_type": "real",
                    "real_row_id": row_id,
                    "candidate_index": None,
                    "global_generation_index": None,
                    "split_row_index": None,
                    "label": int(row["label"]),
                    "text": str(row["text"]),
                }
                for row_id, row in bank.items()
            ]
        return result
    if mode == "test":
        for cell_id in spec.cells:
            result[cell_id] = [
                {
                    "cell_id": cell_id,
                    "record_type": "test",
                    "real_row_id": None,
                    "candidate_index": None,
                    "global_generation_index": None,
                    "split_row_index": int(row["split_row_index"]),
                    "label": int(row["label"]),
                    "text": str(row["text"]),
                }
                for row in load_split(spec, cell_id, "test")
            ]
        return result
    raise ValueError(f"unsupported embedding mode: {mode}")


def embedding_texts(
    family: str,
    tokenizer: Any,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    prepared: list[dict[str, Any]] = []
    for entry in entries:
        text = str(entry["text"])
        if family == "qwen":
            value = qwen.embedding_input(tokenizer, text)
        else:
            value, _original_count, _token_count, _truncated = bge.encode_for_bge(tokenizer, text)
        texts.append(value)
        prepared.append(dict(entry))
    return texts, prepared


def embed_cell(
    *,
    family: str,
    tokenizer: Any,
    entries: Sequence[Mapping[str, Any]],
    endpoints: Sequence[tuple[str, str, int]],
    api_key: str | None,
    timeout_seconds: int,
    batch_size: int,
    expected_dimension: int,
) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np

    texts, prepared = embedding_texts(family, tokenizer, entries)
    batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
    results = qwen.request_batches(batches, endpoints, api_key, timeout_seconds)
    vectors: list[Any] = []
    manifest: list[dict[str, Any]] = []
    row_offset = 0
    for batch_index, batch in enumerate(batches):
        raw_vectors, error = results[batch_index]
        batch_entries = prepared[row_offset : row_offset + len(batch)]
        row_offset += len(batch)
        if error is not None or raw_vectors is None or len(raw_vectors) != len(batch_entries):
            raise RuntimeError(f"embedding batch {batch_index} failed: {error or 'size mismatch'}")
        for entry, raw in zip(batch_entries, raw_vectors, strict=True):
            vector = np.asarray(raw, dtype=np.float32)
            if vector.shape != (expected_dimension,) or not np.all(np.isfinite(vector)):
                raise RuntimeError(f"invalid embedding vector shape/value: {vector.shape}")
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError("embedding vector has an invalid norm")
            vector = (vector / norm).astype(np.float32)
            index = len(vectors)
            vectors.append(vector)
            manifest.append(
                {
                    "embedding_row_index": index,
                    "cell_id": entry["cell_id"],
                    "record_type": entry["record_type"],
                    "real_row_id": entry.get("real_row_id"),
                    "candidate_index": entry.get("candidate_index"),
                    "global_generation_index": entry.get("global_generation_index"),
                    "split_row_index": entry.get("split_row_index"),
                    "label": entry.get("label"),
                    "text_sha256": sha256_bytes(str(entry["text"]).encode("utf-8")),
                    "embedding_dimension": expected_dimension,
                }
            )
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (len(entries), expected_dimension):
        raise RuntimeError(f"embedding row count mismatch: {array.shape}")
    return array, manifest


def expected_input_lock(task: str, mode: str, generation_root: Path | None) -> dict[str, str]:
    if mode == "synthetic":
        if generation_root is None:
            raise RuntimeError("synthetic embedding mode requires generation input")
        verify_hash_lock(generation_root, "GENERATION_LOCK.json")
        return {
            "generation": hashlib.sha256(
                (generation_root / "GENERATION_LOCK.json").read_bytes()
            ).hexdigest()
        }
    spec = task_spec(task)
    return {
        "manifest": hashlib.sha256(
            (spec.manifest_root / "MANIFEST_LOCK.json").read_bytes()
        ).hexdigest()
    }


def validate_existing(
    root: Path,
    task: str,
    family: str,
    mode: str,
    input_lock: Mapping[str, str],
) -> None:
    lock = verify_hash_lock(root, LOCK_NAME)
    config = lock.get("configuration") or {}
    expected = {"task": task, "family": family, "mode": mode, "robustness_version": ROBUSTNESS_VERSION}
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"existing embedding lock has incompatible {key}: {config.get(key)!r}")
    if config.get("input_lock_sha256") != dict(input_lock):
        raise RuntimeError("existing robustness embedding no longer matches its input lock")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("sentiment", "offensive"), required=True)
    parser.add_argument("--mode", choices=("synthetic", "real", "test"), required=True)
    parser.add_argument("--family", choices=("qwen", "bge"), required=True)
    parser.add_argument("--generation-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", nargs=3, metavar=("API_URL", "MODEL", "CONCURRENCY"))
    parser.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0 or args.embedding_batch_size <= 0:
        raise SystemExit("timeout and embedding batch size must be positive")
    task = str(args.task)
    family = str(args.family)
    mode = str(args.mode)
    generation_root = args.generation_root.resolve() if args.generation_root else None
    output_root = args.output_root.resolve()
    if (
        args.dry_run
        and mode == "synthetic"
        and (generation_root is None or not (generation_root / "GENERATION_LOCK.json").is_file())
    ):
        spec = task_spec(task)
        summary = {
            "robustness_version": ROBUSTNESS_VERSION,
            "task": task,
            "family": family,
            "mode": mode,
            "rows": {cell_id: 4200 for cell_id in spec.cells},
            "output_root": str(output_root),
            "source_generation": "planned_3_candidates_per_seed",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    entries = build_entries(task, mode, generation_root)
    config = family_config(family)
    summary = {
        "robustness_version": ROBUSTNESS_VERSION,
        "task": task,
        "family": family,
        "mode": mode,
        "rows": {cell_id: len(rows) for cell_id, rows in entries.items()},
        "output_root": str(output_root),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    lock_path = output_root / LOCK_NAME
    input_lock = expected_input_lock(task, mode, generation_root)
    if lock_path.is_file():
        validate_existing(output_root, task, family, mode, input_lock)
        print(f"SKIP embedding: verified completed lock at {lock_path}")
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"non-empty embedding root without completed lock; refusing overwrite: {output_root}")

    endpoints = parse_lmstudio_endpoints(
        args.endpoint,
        str(config["endpoint_model"]),
        DEFAULT_LMSTUDIO_EMBEDDINGS_API_URL,
    )
    if any(model != config["endpoint_model"] for _, model, _ in endpoints):
        raise RuntimeError(f"all embedding endpoints must use {config['endpoint_model']}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config["tokenizer_id"]),
        local_files_only=True,
        padding_side="left" if family == "qwen" else "right",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    with loaded_lmstudio_models(
        endpoints,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        log=qwen.progress_log,
    ):
        for cell_id, cell_entries in entries.items():
            array, manifest = embed_cell(
                family=family,
                tokenizer=tokenizer,
                entries=cell_entries,
                endpoints=endpoints,
                api_key=args.api_key,
                timeout_seconds=args.timeout_seconds,
                batch_size=args.embedding_batch_size,
                expected_dimension=int(config["dimension"]),
            )
            import numpy as np

            np.save(output_root / f"{cell_id}.npy", array)
            write_csv(output_root / f"{cell_id}_manifest.csv", manifest, FIELDS)
    metadata = {
        "robustness_version": ROBUSTNESS_VERSION,
        "task": task,
        "family": family,
        "mode": mode,
        "endpoint_model": config["endpoint_model"],
        "tokenizer_id": config["tokenizer_id"],
        "dimension": config["dimension"],
        "l2_normalized": True,
        "rows": {cell_id: len(rows) for cell_id, rows in entries.items()},
        "input_lock_sha256": input_lock,
    }
    write_json(output_root / "metadata.json", metadata)
    write_json(
        lock_path,
        {
            "lock_version": "r3-robustness-embedding-v1",
            "status": "completed",
            "configuration": metadata,
            "files": lock_file_map(output_root, excluded={LOCK_NAME}),
        },
    )
    validate_existing(output_root, task, family, mode, input_lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

