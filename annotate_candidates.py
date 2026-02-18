#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nemotron ペルソナ文を句単位に分割し、PII/PROFILE/NONE を付与して JSONL に追記するスクリプト。

処理概要:
1. Hugging Face `nvidia/Nemotron-Personas-Japan` を読み込み、対象 split を選択する。
2. 指定フィールドのテキストを正規化し、句点ベースで clause に分割する。
3. 各 clause を OpenAI API に送り、`PII` / `PROFILE` / `NONE` を1ラベル分類する。
4. 既存出力ファイルを読み込んで `(uuid, clause)` の重複をスキップする。
5. `row_index`, `uuid`, `field`, `clause`, `label` を JSONL 形式で追記保存する。

主な用途:
- ペルソナ文の準識別子アノテーション作成
- 後続の語彙統計・有意差分析用データの作成
"""

import argparse
import json
import os
import re
import time
from itertools import islice
from typing import Iterable, List, Tuple

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


PUNCT_SPLIT_RE = re.compile(r"[。、]+")


FIELDS = [
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "cultural_background",
]


def normalize_field(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            if v is None:
                continue
            if isinstance(v, str):
                out.append(v)
            else:
                out.append(str(v))
        return out
    return [str(value)]


def split_clauses(text: str) -> List[str]:
    clauses = [c.strip() for c in PUNCT_SPLIT_RE.split(text) if c.strip()]
    return clauses


def iter_clauses(dataset, split: str, limit: int, max_rows: int) -> Iterable[Tuple[int, str, str]]:
    ds = dataset[split]
    if max_rows and max_rows > 0:
        ds = islice(ds, max_rows)
    count = 0
    for idx, row in enumerate(ds):
        uuid = row.get("uuid")
        for field in FIELDS:
            values = normalize_field(row.get(field))
            for v in values:
                for clause in split_clauses(v):
                    yield idx, uuid, field, clause
                    count += 1
                    if limit and count >= limit:
                        return


def classify_clause(client: OpenAI, model: str, clause: str, sleep_s: float, temperature) -> str:
    prompt = (
        "Output must be exactly one of: PII, PROFILE, NONE (no reasons, no extra text).\n"
        "Label definitions:\n"
        "- PII: Information that can identify a specific person (names, unique contacts, identifiers).\n"
        "- PROFILE: Quasi-identifiers like hobbies, occupation, or region that narrow a person but do not uniquely identify.\n"
        "- NONE: Other information unrelated to identification (including values or internal states).\n"
        "If uncertain, prefer the label more likely to aid identification.\n\n"
        "Examples\n"
        "Phrase: 横田 嘉宏は\n"
        "PII\n"
        "Phrase: 埼玉県の関東地方で昭和期に育ち\n"
        "PROFILE\n"
        "Phrase: 共同体の絆と自然への感謝を学び\n"
        "NONE\n"
        "Phrase: 鈴木 世愛は\n"
        "PII\n"
        "Phrase: 四国・愛媛県出身で\n"
        "PROFILE\n"
        "Phrase: 伝統的な共同作業の価値観を根底に\n"
        "NONE\n\n"
        "Input\n"
        f"Phrase: {clause}\n"
    )
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict classifier for short Japanese clauses."},
            {"role": "user", "content": prompt},
        ],
    }
    if temperature is not None:
        params["temperature"] = temperature
    resp = client.chat.completions.create(**params)
    label = resp.choices[0].message.content.strip()
    if sleep_s:
        time.sleep(sleep_s)
    return label


def main():
    parser = argparse.ArgumentParser(description="Annotate Nemotron personas clauses with PII/PROFILE/NONE.")
    parser.add_argument("--dataset", default="nvidia/Nemotron-Personas-Japan")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit (clauses)")
    parser.add_argument("--max_rows", type=int, default=2000, help="0 means no limit (rows)")
    parser.add_argument("--streaming", action="store_true", help="stream dataset to avoid full download")
    parser.add_argument("--output", default="annotations.jsonl")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="If omitted, do not send temperature (model default).",
    )
    parser.add_argument("--sleep", type=float, default=0.1, help="sleep seconds between requests")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, streaming=args.streaming)
    if args.split not in dataset:
        args.split = list(dataset.keys())[0]
    client = OpenAI()

    annotated_clauses = set([])
    if os.path.exists(args.output):
        for row in open(args.output, encoding="utf-8"):
            data = json.loads(row)
            annotated_clauses.add((data["uuid"], data["clause"]))

    with open(args.output, "a", encoding="utf-8") as f:
        for idx, uuid, field, clause in tqdm(iter_clauses(dataset, args.split, args.limit, args.max_rows)):
            if (uuid, clause) in annotated_clauses:
                continue
            label = classify_clause(client, args.model, clause, args.sleep, args.temperature)
            record = {
                "row_index": idx,
                "uuid": uuid,
                "field": field,
                "clause": clause,
                "label": label,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
