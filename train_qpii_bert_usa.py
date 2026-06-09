"""Fine-tune (English) ModernBERT to classify clause → label on Nemotron-USA.

`train_qpii_bert.py`(日本語・回帰版)を参考にした英語・分類版。

相違点
------
* タスク: 回帰(確率スコア) → **3クラス分類**(PII / PROFILE / NONE)。
* データ: 共起CSVから targets を組み立てる方式 → `annotations_usa2.jsonl` の
  各行 (`clause`, `label`) を直接学習データに使う。
* モデル: `sbintuitions/modernbert-ja-310m`(日本語) → 英語 ModernBERT。
  既定 `answerdotai/ModernBERT-base`(約150M)。`answerdotai/ModernBERT-large`
  (約395M, JA 310m に近い規模)も選択可。
* ラベル不均衡(PROFILE が約73%)に対応するため、層化K分割CVと
  クラス重み付き損失(balanced)を既定で使用。

要件: transformers >= 4.48.0 (ModernBERT ネイティブ対応), scikit-learn, torch。

使い方
------
  python train_qpii_bert_usa.py                         # 既定: base, 5-fold CV + 最終モデル
  python train_qpii_bert_usa.py --model-name answerdotai/ModernBERT-large
  python train_qpii_bert_usa.py --skip-cv               # CV を飛ばして全データ学習のみ
  python train_qpii_bert_usa.py --class-weights none    # クラス重みを使わない
"""

import argparse
import inspect
import json
import os
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv は任意
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold


# ── データ読み込み ────────────────────────────────────────────────────────────

def load_clause_label(path: str) -> Tuple[List[str], List[str]]:
    """JSONL から (clause, label) を読み、空 clause / 欠損 label を除外する。"""
    clauses: List[str] = []
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(path)}"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            clause = (rec.get("clause") or "").strip()
            label = rec.get("label")
            if not clause or label is None:
                continue
            clauses.append(clause)
            labels.append(str(label))
    return clauses, labels


# ── メトリクス ────────────────────────────────────────────────────────────────

def make_compute_metrics(label_names: List[str]) -> Callable[[Tuple[Any, Any]], Dict[str, float]]:
    def _compute_metrics(eval_pred: Tuple[Any, Any]) -> Dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.asarray(logits).argmax(axis=-1)
        labels = np.asarray(labels)
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }
        # クラス別 F1(ラベル名付き)
        _, _, f1s, _ = precision_recall_fscore_support(
            labels, preds, labels=list(range(len(label_names))),
            average=None, zero_division=0,
        )
        for name, f1 in zip(label_names, f1s):
            metrics[f"f1_{name}"] = float(f1)
        return metrics

    return _compute_metrics


class StepMetricsPrinterCallback(TrainerCallback):
    def __init__(self, fold: int):
        self.fold = fold

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step)
        if logs.get("loss") is not None:
            print(f"[fold {self.fold} step {step}] train_loss={logs['loss']:.6f}")
        if "eval_loss" in logs or "eval_macro_f1" in logs:
            msg = f"[fold {self.fold} step {step}]"
            if "eval_loss" in logs:
                msg += f" eval_loss={logs['eval_loss']:.6f}"
            if "eval_accuracy" in logs:
                msg += f" acc={logs['eval_accuracy']:.4f}"
            if "eval_macro_f1" in logs:
                msg += f" macro_f1={logs['eval_macro_f1']:.4f}"
            print(msg)


# ── クラス重み付き Trainer ────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """class_weights を渡すと重み付き CrossEntropy で学習する Trainer。"""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), weight=weight
        )
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(label_ids: List[int], num_labels: int) -> torch.Tensor:
    """balanced 重み: w_c = N / (num_labels * count_c)。"""
    counts = Counter(label_ids)
    total = len(label_ids)
    weights = [
        total / (num_labels * max(1, counts.get(c, 0))) for c in range(num_labels)
    ]
    return torch.tensor(weights, dtype=torch.float32)


# ── モデル / TrainingArguments 生成ヘルパー ───────────────────────────────────

def build_model(model_name: str, num_labels: int, id2label: Dict[int, str],
                label2id: Dict[str, int]):
    # 重みは fp32 でロードし、混合精度は TrainingArguments の bf16=True(autocast)で行う。
    # こうすると fp32 マスター重みが保たれ、DeBERTa-v3 等でも安定して収束する
    # (torch_dtype=bf16 でロードすると純bf16学習になり不安定になりやすい)。
    extra: Dict[str, Any] = {}
    # ModernBERT は既定で reference_compile(torch.compile)を使い、Triton/Inductor が
    # gcc で Python.h(開発ヘッダ)を要求する。ヘッダが無い環境ではJITに失敗するため無効化。
    # (eager 実行になり多少遅くなるだけで精度には影響しない)
    if "modernbert" in model_name.lower():
        extra["reference_compile"] = False
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        trust_remote_code=True,
        **extra,
    )


def build_training_args(ta_sig, base_kwargs: Dict[str, Any], do_eval: bool) -> TrainingArguments:
    kwargs = dict(base_kwargs)
    eval_value = "epoch" if do_eval else "no"
    if "evaluation_strategy" in ta_sig:
        kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in ta_sig:
        kwargs["eval_strategy"] = eval_value
    if "save_strategy" in ta_sig:
        kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in ta_sig:
        kwargs["logging_strategy"] = "epoch"
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in ta_sig})


def attach_tokenizer(trainer_sig, kwargs: Dict[str, Any], tokenizer) -> None:
    if "processing_class" in trainer_sig:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        kwargs["tokenizer"] = tokenizer


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(dotenv_path=".env", override=False)

    parser = argparse.ArgumentParser(
        description="Fine-tune English ModernBERT for clause→label (PII/PROFILE/NONE) classification."
    )
    parser.add_argument("--data-path", default="annotations_usa2.jsonl")
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base",
                        help="英語 ModernBERT。大きめは answerdotai/ModernBERT-large")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256,
                        help="clause は短いので256で十分(ModernBERT は最大8192)")
    parser.add_argument("--class-weights", choices=["balanced", "none"], default="balanced",
                        help="ラベル不均衡対策。balanced=逆頻度重み, none=重みなし")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="0で無効。>0 で macro_f1 が改善しないエポック数で早期終了")
    parser.add_argument("--output-dir", default="./outputs/qpii_modernbert_usa")
    parser.add_argument("--hub-repo", default=os.environ.get("HF_OUTPUT_REPO"))
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--hub-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    set_seed(args.seed)

    # ── [1/5] データ読み込み + ラベル符号化 ────────────────────────────────
    print("[1/5] Loading data ...")
    clauses, label_strs = load_clause_label(args.data_path)
    if not clauses:
        raise RuntimeError(f"No usable (clause,label) rows in {args.data_path}")

    label_names = sorted(set(label_strs))            # 決定的な順序
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    num_labels = len(label_names)
    y = [label2id[s] for s in label_strs]

    dist = Counter(label_strs)
    print(f"  samples={len(clauses)}  labels={label_names}")
    print(f"  distribution={dict(dist)}")

    # ── [2/5] トークナイズ ────────────────────────────────────────────────
    print("[2/5] Loading tokenizer and tokenizing ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    raw_ds = Dataset.from_dict({"text": clauses, "labels": y})

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized_ds = raw_ds.map(_tokenize, batched=True, desc="Tokenizing")
    tokenized_ds = tokenized_ds.remove_columns(["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    compute_metrics = make_compute_metrics(label_names)
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    trainer_sig = inspect.signature(Trainer.__init__).parameters

    def make_weights(label_ids: List[int]) -> Optional[torch.Tensor]:
        if args.class_weights == "none":
            return None
        return compute_class_weights(label_ids, num_labels)

    def fold_callbacks(fold: int) -> List[TrainerCallback]:
        cbs: List[TrainerCallback] = [StepMetricsPrinterCallback(fold=fold)]
        if args.early_stopping_patience > 0:
            cbs.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        return cbs

    common_ta: Dict[str, Any] = {
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "weight_decay": args.weight_decay,
        "bf16": torch.cuda.is_available(),
        "report_to": "none",
    }

    # ── [3/5] 層化K分割CV ──────────────────────────────────────────────────
    all_metrics: List[Dict[str, float]] = []
    if args.skip_cv:
        print("[3/5] Skipping stratified K-fold CV (--skip-cv)")
    else:
        print(f"[3/5] {args.num_folds}-fold stratified CV ...")
        skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        for fold_idx, (train_idx, eval_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            fold_seed = args.seed + fold_idx
            set_seed(fold_seed)
            train_ds = tokenized_ds.select(train_idx.tolist())
            eval_ds = tokenized_ds.select(eval_idx.tolist())
            run_dir = os.path.join(args.output_dir, f"fold_{fold_idx + 1}")
            steps_per_epoch = max(1, len(train_ds) // max(1, args.train_batch_size * args.grad_accum))
            warmup_steps = int(steps_per_epoch * args.epochs * args.warmup_ratio)

            model = build_model(args.model_name, num_labels, id2label, label2id)
            ta_kwargs = {
                **common_ta,
                "output_dir": run_dir,
                "warmup_steps": warmup_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "macro_f1",
                "greater_is_better": True,
                "push_to_hub": False,
                "seed": fold_seed,
            }
            training_args = build_training_args(ta_sig, ta_kwargs, do_eval=True)

            trainer_kwargs: Dict[str, Any] = {
                "model": model,
                "args": training_args,
                "train_dataset": train_ds,
                "eval_dataset": eval_ds,
                "data_collator": data_collator,
                "compute_metrics": compute_metrics,
                "callbacks": fold_callbacks(fold_idx + 1),
                "class_weights": make_weights([y[i] for i in train_idx]),
            }
            attach_tokenizer(trainer_sig, trainer_kwargs, tokenizer)
            trainer = WeightedTrainer(**trainer_kwargs)
            trainer.train()
            metrics = trainer.evaluate()
            print(f"[fold {fold_idx + 1}] eval: {metrics}")
            all_metrics.append(metrics)

        if all_metrics:
            keys = ["eval_accuracy", "eval_macro_f1", "eval_weighted_f1"]
            print("\n=== CV averages ===")
            for k in keys:
                vals = [m[k] for m in all_metrics if k in m]
                if vals:
                    print(f"  {k}: {sum(vals) / len(vals):.4f}")

    # ── [4/5] 全データで最終モデルを学習 ──────────────────────────────────
    print("[4/5] Training final model on full data ...")
    final_dir = os.path.join(args.output_dir, "final_full_data")
    set_seed(args.seed)
    final_model = build_model(args.model_name, num_labels, id2label, label2id)
    steps_per_epoch = max(1, len(tokenized_ds) // max(1, args.train_batch_size * args.grad_accum))
    warmup_steps = int(steps_per_epoch * args.epochs * args.warmup_ratio)
    final_ta_kwargs = {
        **common_ta,
        "output_dir": final_dir,
        "warmup_steps": warmup_steps,
        "load_best_model_at_end": False,
        "push_to_hub": bool(args.hub_repo),
        "hub_model_id": args.hub_repo,
        "hub_private_repo": args.hub_private,
        "hub_token": args.hub_token,
        "seed": args.seed,
    }
    final_training_args = build_training_args(ta_sig, final_ta_kwargs, do_eval=False)
    final_trainer_kwargs: Dict[str, Any] = {
        "model": final_model,
        "args": final_training_args,
        "train_dataset": tokenized_ds,
        "data_collator": data_collator,
        "callbacks": [StepMetricsPrinterCallback(fold=0)],
        "class_weights": make_weights(y),
    }
    attach_tokenizer(trainer_sig, final_trainer_kwargs, tokenizer)
    final_trainer = WeightedTrainer(**final_trainer_kwargs)
    final_trainer.train()

    # ── [5/5] 保存 / アップロード ──────────────────────────────────────────
    print("[5/5] Saving ...")
    os.makedirs(final_dir, exist_ok=True)
    final_trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    if args.hub_repo:
        final_trainer.push_to_hub()
        print(f"Uploaded final model to {args.hub_repo}")
    else:
        print(f"Final model saved under {final_dir} "
              f"(set HF_OUTPUT_REPO or --hub-repo to upload)")


if __name__ == "__main__":
    main()
