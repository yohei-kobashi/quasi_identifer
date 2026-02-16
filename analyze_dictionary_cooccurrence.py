#!/usr/bin/env python3
"""辞書形態素のテキスト内共起を Nemotron-Personas-Japan で集計する。"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional


_WORKER_TAGGER = None
_WORKER_DICTIONARY: set[str] = set()


def normalize_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).lower().strip()


def is_independent_word(pos1: str, pos2: str, lemma: str) -> bool:
    if not lemma or lemma == "*":
        return False

    excluded_pos1 = {
        "助詞",
        "助動詞",
        "記号",
        "補助記号",
        "接続詞",
        "接頭詞",
        "感動詞",
        "連体詞",
        "空白",
        "フィラー",
    }
    if pos1 in excluded_pos1:
        return False

    excluded_pos2 = {"非自立", "代名詞"}
    if pos2 in excluded_pos2:
        return False

    return pos1 in {"名詞", "動詞", "形容詞", "副詞", "接尾辞"}


def normalize_to_text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                v = item.strip()
            else:
                v = str(item).strip()
            if v:
                out.append(v)
        return out
    v = str(value).strip()
    return [v] if v else []


def load_dictionary_terms(
    dictionary_csv: Path,
    min_count_profile: int,
    min_odds_ratio: float,
) -> set[str]:
    terms: set[str] = set()
    with dictionary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = normalize_token(str(row.get("token", "")))
            if not token:
                continue
            try:
                c = int(float(row.get("count_PROFILE", "0")))
                odds = float(row.get("odds_ratio", "0"))
            except ValueError:
                continue
            if c >= min_count_profile and odds >= min_odds_ratio:
                terms.add(token)
    return terms


def _init_worker(dictionary_terms: list[str]) -> None:
    global _WORKER_TAGGER, _WORKER_DICTIONARY
    import fugashi  # type: ignore

    _WORKER_TAGGER = fugashi.Tagger()
    _WORKER_DICTIONARY = set(dictionary_terms)


def extract_terms_from_text(text: str) -> list[str]:
    seen: set[str] = set()
    matched: list[str] = []
    for w in _WORKER_TAGGER(text):  # type: ignore[misc]
        feat = w.feature
        pos1 = getattr(feat, "pos1", "") or ""
        pos2 = getattr(feat, "pos2", "") or ""
        lemma = str(getattr(feat, "lemma", "") or "")
        lemma = w.surface if lemma in {"", "*"} else lemma
        if not is_independent_word(pos1, pos2, lemma):
            continue
        token = normalize_token(lemma)
        if token in _WORKER_DICTIONARY and token not in seen:
            seen.add(token)
            matched.append(token)
    return matched


def _process_row_task(task: dict) -> list[dict]:
    out: list[dict] = []
    row_index = task["row_index"]
    uuid = task["uuid"]
    for item in task["texts"]:
        matched_terms = extract_terms_from_text(item["text"])
        out.append(
            {
                "row_index": row_index,
                "uuid": uuid,
                "field": item["field"],
                "matched_terms": matched_terms,
            }
        )
    return out


def _resolve_field_texts(row: dict, requested_fields: list[str]) -> list[dict]:
    out: list[dict] = []
    for field in requested_fields:
        candidates = [field]
        if field in {"skills_and_expertise", "hobbies_and_interests"}:
            candidates.append(f"{field}_list")
        elif field.endswith("_list"):
            candidates.append(field[: -len("_list")])

        value = None
        for name in candidates:
            if name in row:
                value = row.get(name)
                break

        for text in normalize_to_text_list(value):
            out.append({"field": field, "text": text})
    return out


def iter_sampled_row_tasks(
    dataset_name: str,
    split: str,
    streaming: bool,
    sample_ratio: float,
    seed: Optional[int],
    max_rows: int,
    requested_fields: list[str],
) -> Iterable[dict]:
    from datasets import load_dataset  # type: ignore

    rng = random.Random(seed)
    ds = load_dataset(dataset_name, split=split, streaming=streaming)

    for i, row in enumerate(ds):
        if max_rows > 0 and i >= max_rows:
            break
        if sample_ratio < 1.0 and rng.random() > sample_ratio:
            continue

        texts = _resolve_field_texts(row, requested_fields)
        if not texts:
            continue

        yield {
            "row_index": i,
            "uuid": row.get("uuid"),
            "texts": texts,
        }


def run(args: argparse.Namespace) -> None:
    dictionary_terms = load_dictionary_terms(
        dictionary_csv=args.dictionary_csv,
        min_count_profile=args.min_count_profile,
        min_odds_ratio=args.min_odds_ratio,
    )
    if not dictionary_terms:
        raise RuntimeError("辞書条件に一致する形態素が0件です。閾値を見直してください。")

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, workers)

    requested_fields = [
        "professional_persona",
        "sports_persona",
        "arts_persona",
        "travel_persona",
        "culinary_persona",
        "persona",
        "cultural_background",
        "skills_and_expertise",
        "hobbies_and_interests",
    ]

    term_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    total_texts = 0
    sampled_rows = 0
    started = time.time()

    row_tasks = iter_sampled_row_tasks(
        dataset_name=args.dataset,
        split=args.split,
        streaming=not args.no_streaming,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        max_rows=args.max_rows,
        requested_fields=requested_fields,
    )

    with args.text_output.open("w", encoding="utf-8") as fw:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(sorted(dictionary_terms),),
        ) as pool:
            for row_result in pool.imap_unordered(_process_row_task, row_tasks, chunksize=1):
                sampled_rows += 1
                for rec in row_result:
                    terms = rec["matched_terms"]
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_texts += 1
                    if not terms:
                        continue
                    unique_sorted = sorted(set(terms))
                    for t in unique_sorted:
                        term_counts[t] += 1
                    for a, b in itertools.combinations(unique_sorted, 2):
                        pair_counts[(a, b)] += 1

                if args.progress_every > 0 and sampled_rows % args.progress_every == 0:
                    elapsed = max(1e-9, time.time() - started)
                    rps = sampled_rows / elapsed
                    print(
                        f"[progress] sampled_rows={sampled_rows} total_texts={total_texts} "
                        f"pair_types={len(pair_counts)} speed={rps:.2f} rows/s"
                    )

    with args.term_output.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["token", "text_count", "probability"])
        for token, cnt in sorted(term_counts.items(), key=lambda x: (-x[1], x[0])):
            p = (cnt / total_texts) if total_texts > 0 else 0.0
            writer.writerow([token, cnt, p])

    with args.pair_output.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["token_a", "token_b", "cooccurrence_count", "probability"])
        for (a, b), cnt in sorted(pair_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            p = (cnt / total_texts) if total_texts > 0 else 0.0
            writer.writerow([a, b, cnt, p])

    elapsed = time.time() - started
    print(f"dictionary_terms={len(dictionary_terms)}")
    print(f"workers={workers}")
    print(f"sampled_rows={sampled_rows}")
    print(f"total_texts={total_texts}")
    print(f"term_types={len(term_counts)}")
    print(f"pair_types={len(pair_counts)}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"text_output={args.text_output}")
    print(f"term_output={args.term_output}")
    print(f"pair_output={args.pair_output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "辞書形態素を閾値で絞り込み、Nemotron各テキスト内の共起語一覧と"
            "同時共起確率(テキスト単位)を集計する"
        )
    )
    p.add_argument("--dictionary-csv", type=Path, default=Path("profile_significant_morphemes.csv"))
    p.add_argument("--min-count-profile", type=int, default=1)
    p.add_argument("--min-odds-ratio", type=float, default=1.0)
    p.add_argument("--dataset", default="nvidia/Nemotron-Personas-Japan")
    p.add_argument("--split", default="train")
    p.add_argument("--no-streaming", action="store_true")
    p.add_argument("--sample-ratio", type=float, default=0.01, help="0.0-1.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0, help="0で上限なし")
    p.add_argument("--workers", type=int, default=0, help="0以下でCPUコア数")
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument(
        "--text-output",
        type=Path,
        default=Path("nemotron_text_dictionary_terms.jsonl"),
    )
    p.add_argument(
        "--term-output",
        type=Path,
        default=Path("nemotron_term_probabilities.csv"),
    )
    p.add_argument(
        "--pair-output",
        type=Path,
        default=Path("nemotron_pair_cooccurrence_probabilities.csv"),
    )
    args = p.parse_args()

    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("--sample-ratio は 0.0 より大きく 1.0 以下で指定してください。")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
