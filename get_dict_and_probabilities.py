import argparse
import csv
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

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


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _embed_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int,
    max_length: int,
) -> List[List[float]]:
    vectors: List[List[float]] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            vectors.extend(pooled.cpu().tolist())
    return vectors


def _build_xy_from_streaming_source(
    target_dataset: Dict[str, Dict[str, float]],
    source_dataset_name: str,
    source_split: str,
) -> Tuple[List[str], List[str], List[str], List[float]]:
    target_uuids = set(str(uuid) for uuid in target_dataset.keys())
    uuid_list: List[str] = []
    field_list: List[str] = []
    text_list: List[str] = []
    y_list: List[float] = []

    stream = load_dataset(source_dataset_name, split=source_split, streaming=True)
    for record in stream:
        uuid = _extract_uuid(record)
        if uuid is None or uuid not in target_uuids:
            continue
        fields = target_dataset.get(uuid, {})
        for field, y in fields.items():
            text = _extract_text(record, field)
            if text is None or not text.strip():
                continue
            uuid_list.append(uuid)
            field_list.append(field)
            text_list.append(text)
            y_list.append(float(y))
    return uuid_list, field_list, text_list, y_list


def main() -> None:
    load_dotenv(dotenv_path=".env", override=False)

    parser = argparse.ArgumentParser(description="Build X/y and upload to Hugging Face Hub.")
    parser.add_argument("--source-dataset", default="nvidia/Nemotron-Personas-Japan")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--embedding-model", default="sbintuitions/modernbert-ja-310m")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hub-repo", default=os.environ.get("HF_OUTPUT_REPO"))
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--hub-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    random.seed(42)

    # terms_json,set_size,text_count,probability
    freq = {}
    mor2n = {}
    max_p = 0
    for i,row in enumerate(csv.reader(open("nemotron_term_set_probabilities.csv", "r"))):
        if i:
            n = float(row[2])
            if not n in freq:
                freq[n] = {"morphemes":[], "prob":float(row[3]), "samples":[]}
            mor = json.loads(row[0])
            freq[n]["morphemes"].append(mor)
            mor2n[tuple(mor)] = n
            max_p = max(max_p, freq[n]["prob"])

    no_PIIs = []
    for row in open("nemotron_text_dictionary_terms.jsonl", "r"):
        data = json.loads(row)
        n = mor2n[tuple(data["matched_terms"])]
        if len(data["matched_terms"]):
            freq[n]["samples"].append((data["uuid"], data["field"]))
        else:
            no_PIIs.append((data["uuid"], data["field"]))

    total_sample_n = 0
    dataset = {}
    for n,v in freq.items():
        # random.sample expects an integer k and k must not exceed len(samples)
        sample_base = max(1.0, n * len(v["morphemes"]))
        sample_n = int(max(1, math.log(sample_base)))
        sample_n = min(sample_n, len(freq[n]["samples"]))
        freq[n]["sample_n"] = sample_n
        freq[n]["samples"] = random.sample(freq[n]["samples"], sample_n)
        for uuid, field in freq[n]["samples"]:
            if not uuid in dataset:
                dataset[uuid] = {}
            dataset[uuid][field] = (max_p - freq[n]["prob"]) / max_p
        total_sample_n += freq[n]["sample_n"]

    total_sample_n = min(total_sample_n, len(no_PIIs))
    for uuid, field in random.sample(no_PIIs, total_sample_n):
        if not uuid in dataset:
            dataset[uuid] = {}
        dataset[uuid][field] = 0

    uuid_list, field_list, text_list, y_list = _build_xy_from_streaming_source(
        target_dataset=dataset,
        source_dataset_name=args.source_dataset,
        source_split=args.source_split,
    )
    if not text_list:
        raise RuntimeError("No training records were matched from the streaming source dataset.")

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(args.embedding_model).to(args.device)
    x_vectors = _embed_texts(
        texts=text_list,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    rows = []
    for uuid, field, text, x, y in zip(uuid_list, field_list, text_list, x_vectors, y_list):
        rows.append({"uuid": uuid, "field": field, "text": text, "x": x, "y": y})
    feature_size = len(x_vectors[0])
    hf_dataset = Dataset.from_list(
        rows,
        features=Features(
            {
                "uuid": Value("string"),
                "field": Value("string"),
                "text": Value("string"),
                "x": Sequence(Value("float32"), length=feature_size),
                "y": Value("float32"),
            }
        ),
    )

    if not args.hub_repo:
        raise ValueError("Please set --hub-repo or HF_OUTPUT_REPO to upload the dataset.")
    hf_dataset.push_to_hub(args.hub_repo, private=args.hub_private, token=args.hub_token)
    print(f"Uploaded {len(hf_dataset)} rows to {args.hub_repo}")


if __name__ == "__main__":
    main()
