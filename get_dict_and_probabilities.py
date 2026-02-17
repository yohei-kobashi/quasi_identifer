import argparse
import csv
import inspect
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

def _extract_uuid(record: Dict[str, Any]) -> Optional[str]:
    for key in ("uuid", "id", "record_id", "uid"):
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def _extract_text(record: Dict[str, Any], field: str) -> Optional[str]:
    if field not in record:
        return None
    value = record[field]
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _build_xy_from_streaming_source(
    target_dataset: Dict[str, Dict[str, float]],
    source_dataset_name: str,
    source_split: str,
) -> Tuple[List[str], List[str], List[str], List[float]]:
    target_uuids = set(str(uuid) for uuid in target_dataset.keys())
    remaining_pairs = {
        (str(uuid), field) for uuid, fields in target_dataset.items() for field in fields.keys()
    }
    uuid_list: List[str] = []
    field_list: List[str] = []
    text_list: List[str] = []
    y_list: List[float] = []

    stream = load_dataset(source_dataset_name, split=source_split, streaming=True)
    scan_bar = tqdm(stream, desc="Scanning source stream")
    matched_uuids: set[str] = set()
    for record in scan_bar:
        uuid = _extract_uuid(record)
        if uuid is None or uuid not in target_uuids:
            continue
        fields = target_dataset.get(uuid, {})
        for field, y in fields.items():
            pair = (uuid, field)
            if pair not in remaining_pairs:
                continue
            text = _extract_text(record, field)
            if text is None or not text.strip():
                continue
            uuid_list.append(uuid)
            field_list.append(field)
            text_list.append(text)
            y_list.append(float(y))
            remaining_pairs.remove(pair)
            matched_uuids.add(uuid)
            if len(text_list) % 100 == 0:
                scan_bar.set_postfix(matched=len(text_list), uuids=len(matched_uuids), left=len(remaining_pairs))
        if not remaining_pairs:
            break
    return uuid_list, field_list, text_list, y_list


def _compute_metrics(eval_pred: Tuple[Any, Any]) -> Dict[str, float]:
    predictions, labels = eval_pred
    preds = predictions.reshape(-1)
    labels = labels.reshape(-1)
    if preds.size > 1 and float(np.std(preds)) > 0.0 and float(np.std(labels)) > 0.0:
        pearson = float(np.corrcoef(preds, labels)[0, 1])
    else:
        pearson = 0.0
    mse = float(((preds - labels) ** 2).mean())
    mae = float(abs(preds - labels).mean())
    return {"mse": mse, "mae": mae, "pearson": pearson}


class StepMetricsPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step)
        train_loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        eval_pearson = logs.get("eval_pearson")
        if train_loss is not None:
            print(f"[step {step}] train_loss={train_loss:.6f}")
        if eval_loss is not None or eval_pearson is not None:
            msg = f"[step {step}]"
            if eval_loss is not None:
                msg += f" eval_loss={eval_loss:.6f}"
            if eval_pearson is not None:
                msg += f" eval_pearson={eval_pearson:.6f}"
            print(msg)


def main() -> None:
    load_dotenv(dotenv_path=".env", override=False)

    parser = argparse.ArgumentParser(description="Fine-tune BertForSequenceClassification for y regression.")
    parser.add_argument("--source-dataset", default="nvidia/Nemotron-Personas-Japan")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--model-name", default="sbintuitions/modernbert-ja-310m")
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", default="./outputs/bert_regression")
    parser.add_argument("--hub-repo", default=os.environ.get("HF_OUTPUT_REPO"))
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--hub-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    set_seed(args.seed)
    print("[1/6] Building y targets from local files...")

    # terms_json,set_size,text_count,probability
    freq = {}
    mor2n = {}
    max_p = 0
    for i,row in enumerate(tqdm(csv.reader(open("nemotron_term_set_probabilities.csv", "r")), desc="Reading probabilities CSV")):
        if i:
            n = float(row[2])
            if not n in freq:
                freq[n] = {"morphemes":[], "prob":float(row[3]), "samples":[]}
            mor = json.loads(row[0])
            freq[n]["morphemes"].append(mor)
            mor2n[tuple(sorted(mor))] = n
            max_p = max(max_p, freq[n]["prob"])

    no_PIIs = []
    for row in tqdm(open("nemotron_text_dictionary_terms.jsonl", "r"), desc="Reading dictionary terms JSONL"):
        data = json.loads(row)
        n = mor2n[tuple(sorted(data["matched_terms"]))]
        if len(data["matched_terms"]):
            freq[n]["samples"].append((data["uuid"], data["field"]))
        else:
            no_PIIs.append((data["uuid"], data["field"]))

    total_sample_n = 0
    dataset = {}
    for n,v in tqdm(freq.items(), total=len(freq), desc="Sampling PII candidates"):
        # random.sample expects an integer k and k must not exceed len(samples)
        sample_base = max(1.0, n * len(v["morphemes"]))
        sample_n = int(math.log(sample_base) * 15)
        sample_n = min(sample_n, len(freq[n]["samples"]))
        freq[n]["sample_n"] = sample_n
        freq[n]["samples"] = random.sample(freq[n]["samples"], sample_n)
        for uuid, field in freq[n]["samples"]:
            if not uuid in dataset:
                dataset[uuid] = {}
            dataset[uuid][field] = (max_p - freq[n]["prob"]) / max_p
        total_sample_n += freq[n]["sample_n"]

    total_sample_n = min(total_sample_n, len(no_PIIs))
    for uuid, field in tqdm(random.sample(no_PIIs, total_sample_n), desc="Sampling non-PII candidates"):
        if not uuid in dataset:
            dataset[uuid] = {}
        dataset[uuid][field] = 0
    print(f"Built target dataset: {len(dataset)} uuids")

    print("[2/6] Streaming source dataset and collecting matched texts...")
    uuid_list, field_list, text_list, y_list = _build_xy_from_streaming_source(
        target_dataset=dataset,
        source_dataset_name=args.source_dataset,
        source_split=args.source_split,
    )
    if not text_list:
        raise RuntimeError("No training records were matched from the streaming source dataset.")
    print(f"Collected pairs: {len(text_list)}")

    print("[3/6] Building train/eval dataset...")
    raw_dataset = Dataset.from_dict(
        {
            "uuid": uuid_list,
            "field": field_list,
            "text": text_list,
            "labels": [float(v) for v in y_list],
        }
    )
    split = raw_dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed, shuffle=True)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"train={len(train_ds)} eval={len(eval_ds)}")

    print("[4/6] Loading tokenizer/model and tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(_tokenize, batched=True, desc="Tokenizing train split")
    eval_ds = eval_ds.map(_tokenize, batched=True, desc="Tokenizing eval split")
    train_ds = train_ds.remove_columns(["uuid", "field", "text"])
    eval_ds = eval_ds.remove_columns(["uuid", "field", "text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    steps_per_epoch = max(1, len(train_ds) // max(1, args.train_batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "bf16": torch.cuda.is_available(),
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "mse",
        "greater_is_better": False,
        "push_to_hub": bool(args.hub_repo),
        "hub_model_id": args.hub_repo,
        "hub_private_repo": args.hub_private,
        "hub_token": args.hub_token,
        "seed": args.seed,
    }
    if "evaluation_strategy" in ta_sig:
        ta_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_sig:
        ta_kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in ta_sig:
        ta_kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in ta_sig:
        ta_kwargs["logging_strategy"] = "epoch"
    training_args = TrainingArguments(**{k: v for k, v in ta_kwargs.items() if k in ta_sig})

    print("[5/6] Fine-tuning BertForSequenceClassification...")
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": data_collator,
        "compute_metrics": _compute_metrics,
        "callbacks": [StepMetricsPrinterCallback()],
    }
    if "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    print(f"eval metrics: {metrics}")

    print("[6/6] Saving and uploading model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.hub_repo:
        trainer.push_to_hub()
        print(f"Uploaded fine-tuned model to {args.hub_repo}")
    else:
        print(f"Model saved locally at {args.output_dir} (set HF_OUTPUT_REPO or --hub-repo to upload)")


if __name__ == "__main__":
    main()
