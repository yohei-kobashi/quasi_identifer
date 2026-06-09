import argparse
import csv
from typing import List

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def batched(iterable: List[str], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with train_qpii_bert.py model and append outputs to CSV."
    )
    parser.add_argument("--input-csv", default="data.csv")
    parser.add_argument("--output-csv", default="data_with_pred.csv")
    parser.add_argument("--model-dir", default="./outputs/bert_regression/final_full_data")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    rows = []
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")
        if args.text_col not in reader.fieldnames:
            raise ValueError(f"Column '{args.text_col}' not found in {args.input_csv}.")
        for row in reader:
            rows.append(row)
        fieldnames = list(reader.fieldnames)

    texts = [str(r.get(args.text_col, "") or "") for r in rows]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    raw_scores: List[float] = []
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for text_batch in tqdm(batched(texts, args.batch_size), total=total_batches, desc="Inferring"):
            encoded = tokenizer(
                text_batch,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = logits.squeeze(1)
            else:
                batch_scores = logits.reshape(-1)
            raw_scores.extend(batch_scores.detach().float().cpu().tolist())

    out_fields = fieldnames + ["qpii_score_raw", "qpii_score_clipped", "qpii_pred"]
    for row, raw in zip(rows, raw_scores):
        clipped = max(0.0, min(1.0, float(raw)))
        pred = 1 if clipped >= args.threshold else 0
        row["qpii_score_raw"] = f"{raw:.6f}"
        row["qpii_score_clipped"] = f"{clipped:.6f}"
        row["qpii_pred"] = str(pred)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
