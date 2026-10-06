"""Transformer classifier sensitivity analysis for the fixed sentiment protocol.

This stage replaces only the downstream classifier.  It uses the same locked
sentiment bank, the same three-candidate subset, r=1, w=1, and 50 paired
repetitions as the compact logistic-regression robustness arm.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from robustness_common import (  # noqa: E402
    ROBUSTNESS_VERSION,
    canonical_json,
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


LOCK_NAME = "TRANSFORMER_ROBUSTNESS_LOCK.json"
DB_NAME = "transformer_status.sqlite3"
FIXED_EPOCHS = 20


def contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "robustness_version": ROBUSTNESS_VERSION,
        "task": "sentiment",
        "model_id": str(args.model_id),
        "epochs": FIXED_EPOCHS,
        "training_schedule": "fixed_epoch_budget_no_early_stopping",
        "checkpoint_selection": "development_macro_f1_then_auroc",
        "train_batch_size": int(args.train_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "max_length": int(args.max_length),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "ratio": 1.0,
        "synthetic_weight": 1.0,
        "similarity_filtering": False,
        "repetitions": 50,
        "conditions": ["Base", "Hybrid", "Matched All-real"],
    }


def open_database(path: Path, expected_contract: Mapping[str, Any]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS fits(
            cell_id TEXT NOT NULL,
            repeat_seed INTEGER NOT NULL,
            condition TEXT NOT NULL,
            train_n INTEGER NOT NULL,
            trained_epochs INTEGER NOT NULL,
            selected_epoch INTEGER NOT NULL,
            development_macro_f1 REAL NOT NULL,
            development_auroc REAL NOT NULL,
            macro_f1 REAL NOT NULL,
            auroc REAL NOT NULL,
            PRIMARY KEY(cell_id, repeat_seed, condition)
        );
        """
    )
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    serialized = canonical_json(expected_contract)
    if existing:
        if existing.get("contract") != serialized:
            raise RuntimeError("existing transformer DB uses a different robustness contract")
    else:
        with connection:
            connection.execute("INSERT INTO metadata(key, value) VALUES('contract', ?)", (serialized,))
    return connection


def selected_texts(
    condition: str,
    fixed: Mapping[str, Any],
    bank: Mapping[str, Mapping[str, Any]],
    synthetic: Mapping[int, Mapping[str, Any]],
) -> tuple[list[str], list[int]]:
    if condition == "Base":
        ids = list(fixed["base_ids"])
        return [str(bank[row_id]["text"]) for row_id in ids], list(fixed["base_labels"])
    if condition == "Matched All-real":
        ids = list(fixed["matched_ids"])
        return [str(bank[row_id]["text"]) for row_id in ids], list(fixed["matched_labels"])
    if condition == "Hybrid":
        texts = [str(bank[row_id]["text"]) for row_id in fixed["base_ids"]]
        texts.extend(str(synthetic[index]["generated_text"]) for index in fixed["selected_synthetic_indices"])
        return texts, list(fixed["hybrid_labels"])
    raise ValueError(condition)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_dataset(tokenizer: Any, texts: Sequence[str], labels: Sequence[int], max_length: int) -> Any:
    import torch
    from torch.utils.data import TensorDataset

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(list(labels), dtype=torch.long),
    )


def fit_one(
    *,
    model_id: str,
    tokenizer: Any,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    development_dataset: Any,
    test_dataset: Any,
    seed: int,
    train_batch_size: int,
    eval_batch_size: int,
    max_length: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[float, float, int, int, float, float]:
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, roc_auc_score
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    model.to(device)
    model.train()
    train_dataset = tensor_dataset(tokenizer, train_texts, train_labels, max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best_development_macro_f1 = float("-inf")
    best_development_auroc = float("-inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    trained_epochs = 0

    def evaluate(dataset: Any) -> tuple[float, float]:
        model.eval()
        loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        probabilities: list[float] = []
        targets: list[int] = []
        with torch.inference_mode():
            for input_ids, attention_mask, labels in loader:
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if use_bf16 else torch.float32,
                    enabled=use_bf16,
                ):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                probabilities.extend(float(value) for value in probs)
                targets.extend(int(value) for value in labels.numpy())
        probabilities_array = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(targets, dtype=np.int64)
        pred = (probabilities_array >= 0.5).astype(np.int64)
        macro_f1 = float(
            f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)
        )
        auroc = float(roc_auc_score(y, probabilities_array))
        model.train()
        return macro_f1, auroc

    for epoch in range(1, FIXED_EPOCHS + 1):
        trained_epochs = epoch
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if use_bf16 else torch.float32,
                enabled=use_bf16,
            ):
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss
            loss.backward()
            optimizer.step()
        development_macro_f1, development_auroc = evaluate(development_dataset)
        better_macro_f1 = development_macro_f1 > best_development_macro_f1 + 1e-12
        tied_macro_f1 = abs(development_macro_f1 - best_development_macro_f1) <= 1e-12
        better_auroc_tiebreak = (
            tied_macro_f1
            and development_auroc > best_development_auroc + 1e-12
        )
        if better_macro_f1 or better_auroc_tiebreak:
            best_development_macro_f1 = development_macro_f1
            best_development_auroc = development_auroc
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if trained_epochs != FIXED_EPOCHS:
        raise RuntimeError(
            f"transformer training stopped early: {trained_epochs}/{FIXED_EPOCHS} epochs"
        )
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("transformer training did not produce a selectable checkpoint")
    model.load_state_dict(best_state)
    macro_f1, auroc = evaluate(test_dataset)
    del best_state, model, optimizer, train_loader, train_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        macro_f1,
        auroc,
        trained_epochs,
        best_epoch,
        best_development_macro_f1,
        best_development_auroc,
    )


def summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cells = sorted({str(row["cell_id"]) for row in rows})
    lookup = {
        (str(row["cell_id"]), int(row["repeat_seed"]), str(row["condition"])): row
        for row in rows
    }
    for cell_id in cells:
        for comparison, left, right in (
            ("Hybrid - Base", "Hybrid", "Base"),
            ("Matched All-real - Base", "Matched All-real", "Base"),
            ("Matched All-real - Hybrid", "Matched All-real", "Hybrid"),
        ):
            for metric in ("macro_f1", "auroc"):
                values = [
                    float(lookup[(cell_id, seed, left)][metric])
                    - float(lookup[(cell_id, seed, right)][metric])
                    for seed in range(1000, 1050)
                ]
                mean, lower, upper = paired_bootstrap_ci(values)
                result.append(
                    {
                        "cell_id": cell_id,
                        "comparison": comparison,
                        "metric": metric,
                        "mean_difference": mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=root / "Results" / "BinaryMatchedSizeExperiment" / "Generation",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if min(
        args.train_batch_size,
        args.eval_batch_size,
        args.max_length,
    ) <= 0:
        raise SystemExit("batch sizes and max length must be positive")
    spec = task_spec("sentiment")
    generation_root = args.generation_root.resolve()
    output_root = args.output_root.resolve()
    synthetic = load_valid_generation(generation_root, spec.cells)
    expected_contract = {
        **contract(args),
        "input_generation_lock_sha256": sha256_file(generation_root / "GENERATION_LOCK.json"),
        "manifest_lock_sha256": sha256_file(spec.manifest_root / "MANIFEST_LOCK.json"),
    }
    print(json.dumps({**expected_contract, "planned_fits": len(spec.cells) * 50 * 3}, indent=2))
    # Always prove that the three-candidate frozen selection is feasible before training.
    for cell_id in spec.cells:
        bank = load_bank(spec, cell_id)
        for repeat in load_repetitions(spec, cell_id):
            fixed_condition_rows(spec, cell_id, repeat, bank, synthetic[cell_id])
    if args.dry_run:
        return 0
    lock_path = output_root / LOCK_NAME
    if lock_path.is_file():
        lock = verify_hash_lock(output_root, LOCK_NAME)
        if lock.get("configuration") != expected_contract:
            raise RuntimeError("existing transformer robustness lock uses a different contract")
        print(f"SKIP transformer robustness: verified {lock_path}")
        return 0
    db_path = output_root / DB_NAME
    if output_root.exists() and any(output_root.iterdir()) and not db_path.is_file():
        raise RuntimeError(f"non-empty transformer root without resumable DB; refusing overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    connection = open_database(db_path, expected_contract)
    try:
        completed = {
            (str(cell), int(seed), str(condition))
            for cell, seed, condition in connection.execute(
                "SELECT cell_id, repeat_seed, condition FROM fits"
            )
        }
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.model_id), use_fast=True)
        for cell_id in spec.cells:
            bank = load_bank(spec, cell_id)
            test_rows = load_split(spec, cell_id, "test")
            development_rows = load_split(spec, cell_id, "dev")
            development_dataset = tensor_dataset(
                tokenizer,
                [str(row["text"]) for row in development_rows],
                [int(row["label"]) for row in development_rows],
                args.max_length,
            )
            test_dataset = tensor_dataset(
                tokenizer,
                [str(row["text"]) for row in test_rows],
                [int(row["label"]) for row in test_rows],
                args.max_length,
            )
            for repeat in load_repetitions(spec, cell_id):
                seed = int(repeat["repeat_seed"])
                fixed = fixed_condition_rows(spec, cell_id, repeat, bank, synthetic[cell_id])
                for condition in ("Base", "Hybrid", "Matched All-real"):
                    key = (cell_id, seed, condition)
                    if key in completed:
                        continue
                    texts, labels = selected_texts(condition, fixed, bank, synthetic[cell_id])
                    (
                        macro_f1,
                        auroc,
                        trained_epochs,
                        selected_epoch,
                        development_macro_f1,
                        development_auroc,
                    ) = fit_one(
                        model_id=str(args.model_id),
                        tokenizer=tokenizer,
                        train_texts=texts,
                        train_labels=labels,
                        development_dataset=development_dataset,
                        test_dataset=test_dataset,
                        seed=seed,
                        train_batch_size=args.train_batch_size,
                        eval_batch_size=args.eval_batch_size,
                        max_length=args.max_length,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                    )
                    with connection:
                        connection.execute(
                            """INSERT INTO fits(
                                cell_id, repeat_seed, condition, train_n, trained_epochs,
                                selected_epoch, development_macro_f1, development_auroc,
                                macro_f1, auroc
                            ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                            (
                                cell_id,
                                seed,
                                condition,
                                len(labels),
                                trained_epochs,
                                selected_epoch,
                                development_macro_f1,
                                development_auroc,
                                macro_f1,
                                auroc,
                            ),
                        )
                    completed.add(key)
                    print(
                        f"transformer robustness | completed={len(completed)}/{len(spec.cells) * 50 * 3} "
                        f"| {cell_id} seed={seed} condition={condition} "
                        f"trained_epochs={trained_epochs} selected_epoch={selected_epoch} "
                        f"dev_macro_f1={development_macro_f1:.6f} "
                        f"dev_auroc={development_auroc:.6f}",
                        flush=True,
                    )
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(
            "SELECT * FROM fits ORDER BY cell_id, repeat_seed, condition"
        )]
        connection.row_factory = None
        if len(rows) != len(spec.cells) * 50 * 3:
            raise RuntimeError(f"transformer robustness incomplete: {len(rows)} rows")
        summary = summary_rows(rows)
        write_csv(
            output_root / "repeat_results.csv",
            rows,
            (
                "cell_id",
                "repeat_seed",
                "condition",
                "train_n",
                "trained_epochs",
                "selected_epoch",
                "development_macro_f1",
                "development_auroc",
                "macro_f1",
                "auroc",
            ),
        )
        write_csv(
            output_root / "paired_summary.csv",
            summary,
            ("cell_id", "comparison", "metric", "mean_difference", "ci_lower", "ci_upper"),
        )
        write_json(output_root / "report.json", {"configuration": expected_contract, "summary": summary})
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    write_json(
        lock_path,
        {
            "lock_version": "r3-transformer-robustness-v2",
            "status": "completed",
            "configuration": expected_contract,
            "files": lock_file_map(output_root, excluded={LOCK_NAME, DB_NAME, DB_NAME + "-wal", DB_NAME + "-shm"}),
        },
    )
    verify_hash_lock(output_root, LOCK_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

