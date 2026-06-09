"""DeBERTa-v3-large と ModernBERT-large(他)を層化K分割CVで公平比較する。

`train_qpii_bert_usa.py` のヘルパー(データ読込・メトリクス・WeightedTrainer・
クラス重み・モデル/引数生成)を再利用する。比較の公平性のため:

  * fold 分割は1度だけ計算し、全モデルで**同一の分割**を使う(ペア比較)。
  * データ・ハイパラ・クラス重み方針は全モデル共通。
  * トークナイズのみモデル個別(各自のトークナイザを使用)。

各モデルについて accuracy / macro-F1 / weighted-F1 / クラス別F1 の
fold平均±標準偏差と、CV全体の所要時間(効率指標)を出力する。

出力:
  ./outputs/compare_usa/<model>/fold_*/   各foldの学習成果物
  --results-json (既定 compare_results.json)  全fold生メトリクス
  --results-csv  (既定 compare_summary.csv)   モデル×指標の集計表
  + 標準出力に比較表

使い方:
  python compare_models_usa.py                           # 既定の2モデルを比較
  python compare_models_usa.py --max-samples 4000        # 層化サブサンプルで時短ドライラン
  python compare_models_usa.py --models microsoft/deberta-v3-large answerdotai/ModernBERT-large answerdotai/ModernBERT-base
"""

import argparse
import csv
import gc
import inspect
import json
import os
import time
from collections import Counter
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

import train_qpii_bert_usa as T  # ヘルパー再利用

DEFAULT_MODELS = [
    "microsoft/deberta-v3-large",
    "answerdotai/ModernBERT-large",
]

# 集計対象の指標キー(eval_ プレフィックス付き)
SUMMARY_KEYS = ["eval_accuracy", "eval_macro_f1", "eval_weighted_f1"]


def stratified_subsample(y: List[int], max_samples: int, seed: int) -> np.ndarray:
    """層化を保ったまま max_samples 件のインデックスを返す。"""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)
    n = len(y_arr)
    if max_samples >= n:
        return np.arange(n)
    keep: List[int] = []
    for cls in np.unique(y_arr):
        idx = np.where(y_arr == cls)[0]
        k = max(1, round(len(idx) * max_samples / n))
        k = min(k, len(idx))
        keep.extend(rng.choice(idx, size=k, replace=False).tolist())
    return np.array(sorted(keep))


def run_cv_for_model(
    model_name: str,
    clauses: List[str],
    y: List[int],
    label_names: List[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    ta_sig,
    trainer_sig,
) -> Tuple[List[Dict[str, float]], float]:
    """1モデルについて共有 splits で CV を回し、(fold毎メトリクス, 所要秒) を返す。"""
    num_labels = len(label_names)
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    raw = Dataset.from_dict({"text": clauses, "labels": y})
    tokenized = raw.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_length),
        batched=True, desc=f"Tokenizing ({model_name})",
    ).remove_columns(["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    compute_metrics = T.make_compute_metrics(label_names)

    safe_name = model_name.replace("/", "_")
    common_ta: Dict[str, Any] = {
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "weight_decay": args.weight_decay,
        "bf16": torch.cuda.is_available(),
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "push_to_hub": False,
    }

    fold_metrics: List[Dict[str, float]] = []
    started = time.time()
    for fold_idx, (train_idx, eval_idx) in enumerate(splits):
        fold_seed = args.seed + fold_idx
        set_seed(fold_seed)
        train_ds = tokenized.select(train_idx.tolist())
        eval_ds = tokenized.select(eval_idx.tolist())
        run_dir = os.path.join(args.output_dir, safe_name, f"fold_{fold_idx + 1}")
        steps_per_epoch = max(1, len(train_ds) // max(1, args.train_batch_size * args.grad_accum))
        warmup_steps = int(steps_per_epoch * args.epochs * args.warmup_ratio)

        model = T.build_model(model_name, num_labels, id2label, label2id)
        ta_kwargs = {**common_ta, "output_dir": run_dir,
                     "warmup_steps": warmup_steps, "seed": fold_seed}
        training_args = T.build_training_args(ta_sig, ta_kwargs, do_eval=True)

        class_weights = None
        if args.class_weights == "balanced":
            class_weights = T.compute_class_weights([y[i] for i in train_idx], num_labels)

        tk: Dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "data_collator": collator,
            "compute_metrics": compute_metrics,
            "callbacks": [T.StepMetricsPrinterCallback(fold=fold_idx + 1)],
            "class_weights": class_weights,
        }
        T.attach_tokenizer(trainer_sig, tk, tokenizer)
        trainer = T.WeightedTrainer(**tk)
        trainer.train()
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)
        print(f"  [{model_name} fold {fold_idx + 1}] "
              f"acc={metrics.get('eval_accuracy', 0):.4f} "
              f"macro_f1={metrics.get('eval_macro_f1', 0):.4f}")

        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return fold_metrics, time.time() - started


def aggregate(fold_metrics: List[Dict[str, float]], label_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """指標キーごとに (mean, std) を返す。"""
    keys = list(SUMMARY_KEYS) + [f"eval_f1_{n}" for n in label_names]
    out: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        if vals:
            out[k] = (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DeBERTa-v3-large vs ModernBERT-large (and others) via stratified CV.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--data-path", default="annotations_usa2.jsonl")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0,
                        help=">0 で層化サブサンプル(時短ドライラン用)")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="large系の既定。両モデル共通で公平比較")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--class-weights", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--output-dir", default="./outputs/compare_usa")
    parser.add_argument("--results-json", default="compare_results.json")
    parser.add_argument("--results-csv", default="compare_summary.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # ── データ ──────────────────────────────────────────────────────────────
    clauses, label_strs = T.load_clause_label(args.data_path)
    if not clauses:
        raise RuntimeError(f"No usable rows in {args.data_path}")
    label_names = sorted(set(label_strs))
    label2id = {n: i for i, n in enumerate(label_names)}
    y = [label2id[s] for s in label_strs]

    if args.max_samples and args.max_samples > 0:
        keep = stratified_subsample(y, args.max_samples, args.seed)
        clauses = [clauses[i] for i in keep]
        y = [y[i] for i in keep]
        print(f"[subsample] {len(clauses)} 件に層化サブサンプル")

    print(f"samples={len(clauses)} labels={label_names} dist={dict(Counter(label_strs))}")

    # ── 共有 fold 分割(全モデル同一) ─────────────────────────────────────────
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    splits = [(tr, ev) for tr, ev in skf.split(np.zeros(len(y)), y)]
    print(f"{args.num_folds}-fold stratified splits を全モデルで共有")

    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    trainer_sig = inspect.signature(Trainer.__init__).parameters

    # ── 各モデルを CV ─────────────────────────────────────────────────────────
    all_results: Dict[str, Any] = {}
    for model_name in args.models:
        print(f"\n{'='*70}\n  CV: {model_name}\n{'='*70}")
        try:
            fold_metrics, elapsed = run_cv_for_model(
                model_name, clauses, y, label_names, splits, args, ta_sig, trainer_sig)
        except Exception as exc:  # 1モデルが失敗しても他は続行
            print(f"  [SKIP] {model_name} failed: {type(exc).__name__}: {exc}")
            all_results[model_name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        agg = aggregate(fold_metrics, label_names)
        all_results[model_name] = {
            "fold_metrics": fold_metrics,
            "aggregate": {k: {"mean": v[0], "std": v[1]} for k, v in agg.items()},
            "cv_seconds": elapsed,
        }

    # ── 結果保存 ──────────────────────────────────────────────────────────────
    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    summary_keys = list(SUMMARY_KEYS) + [f"eval_f1_{n}" for n in label_names]
    with open(args.results_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + [k.replace("eval_", "") for k in summary_keys] + ["cv_seconds"])
        for model_name in args.models:
            res = all_results.get(model_name, {})
            if "error" in res:
                writer.writerow([model_name] + ["ERROR"] * len(summary_keys) + [res["error"]])
                continue
            agg = res["aggregate"]
            row = [model_name]
            for k in summary_keys:
                if k in agg:
                    row.append(f"{agg[k]['mean']:.4f}±{agg[k]['std']:.4f}")
                else:
                    row.append("-")
            row.append(f"{res['cv_seconds']:.1f}")
            writer.writerow(row)

    # ── 比較表(標準出力) ────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  比較サマリ ({args.num_folds}-fold CV, mean±std)\n{'='*70}")
    header = f"{'model':40s} {'accuracy':>16s} {'macro_f1':>16s} {'weighted_f1':>16s} {'sec':>8s}"
    print(header)
    print("-" * len(header))
    best_model, best_f1 = None, -1.0
    for model_name in args.models:
        res = all_results.get(model_name, {})
        if "error" in res:
            print(f"{model_name:40s}   ERROR: {res['error'][:40]}")
            continue
        agg = res["aggregate"]
        def cell(k):
            return f"{agg[k]['mean']:.4f}±{agg[k]['std']:.4f}" if k in agg else "-"
        print(f"{model_name:40s} {cell('eval_accuracy'):>16s} "
              f"{cell('eval_macro_f1'):>16s} {cell('eval_weighted_f1'):>16s} "
              f"{res['cv_seconds']:>8.1f}")
        mf1 = agg.get("eval_macro_f1", {}).get("mean", -1.0)
        if mf1 > best_f1:
            best_f1, best_model = mf1, model_name

    if best_model:
        print(f"\n  → macro-F1 最良: {best_model} ({best_f1:.4f})")
    print(f"\n保存: {args.results_json} / {args.results_csv}")


if __name__ == "__main__":
    main()
