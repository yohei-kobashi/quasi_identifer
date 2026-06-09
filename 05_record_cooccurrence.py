#!/usr/bin/env python3
"""quasi identifier のレコード内共起セットを Nemotron-Personas-USA で集計する。

`analyze_dictionary_cooccurrence.py` を下敷きにしつつ、2点を変更している。

1. 集計単位を「テキスト単位」から「レコード(persona 1行)単位」へ変更。
   元コードは各テキストフィールドごとに共起セットを作っていたが、本コードは
   1レコードに含まれる全テキストフィールドを統合して 1つの共起セットを作る。

2. 対象表現の特定方法を「辞書形態素マッチ」から
   「03_method_a_extract.py → 04_method_b_phrase.py を順に適用する方法」へ変更。
   - Method A: data/03_method_a_candidates.csv の候補トークン集合を辞書として読み込む。
   - Method B: 各レコードのテキストから名詞句(NP)を抽出し、Method A の候補トークンを
     含む最長 NP に展開する(04 のロジックをレコード本文へオンライン適用)。
   レコードごとに得られた表現の集合を共起セットとして集計する。

出力
----
  --text-output (jsonl) : レコードごとの matched_terms 明細
  --term-output (csv)   : 共起セットの出現確率(レコード単位)
  --sqlite-path         : セット集計に使う一時 SQLite DB

使い方
------
  # 先に Method A / Method B を流して候補を生成しておく
  python 03_method_a_extract.py
  python 04_method_b_phrase.py    # (任意。本コードは 03 の候補トークンを使う)

  # 共起セット集計(レコード単位)
  python 05_record_cooccurrence.py --sample-ratio 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    sys.path.insert(0, str(Path.cwd()))
from config import DATA_DIR, TEXT_FIELDS

DEFAULT_CANDIDATES_CSV: Path = DATA_DIR / "03_method_a_candidates.csv"

# 04_method_b_phrase.py と同一の NP 文法(longest enclosing NP 抽出用)
NP_GRAMMAR: str = r"""
  NP: {<DT>?<JJ.*>*<NN.*>+(<IN><DT>?<JJ.*>*<NN.*>+)*}
"""

# ── ワーカーごとのグローバル状態 ───────────────────────────────────────────────
_WORKER_PARSER = None
_WORKER_TOKENS: set[str] = set()
_WORKER_DROP_SINGLE_WORD: bool = False


# ── Method A: 候補トークン辞書の読み込み ───────────────────────────────────────

def load_candidate_tokens(
    candidates_csv: Path,
    min_freq_profile: int,
    min_effect_size: float,
) -> set[str]:
    """03_method_a_candidates.csv から候補トークン集合を読み込む。

    元コードの load_dictionary_terms に対応。effect_size(odds ratio)と
    freq_profile で追加の絞り込みができる。
    """
    if not candidates_csv.exists():
        raise FileNotFoundError(
            f"Method A の候補が見つかりません: {candidates_csv}\n"
            f"先に `python 03_method_a_extract.py` を実行してください。"
        )

    tokens: set[str] = set()
    with candidates_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = str(row.get("token", "")).strip().lower()
            if not token:
                continue
            try:
                freq = int(float(row.get("freq_profile", "0")))
                effect = float(row.get("effect_size", "0"))
            except (ValueError, TypeError):
                continue
            if freq >= min_freq_profile and effect >= min_effect_size:
                tokens.add(token)
    return tokens


# ── Method B: NP 抽出 + 最長 NP 展開(04 のロジック) ─────────────────────────

def extract_nps_from_text(text: str) -> list[tuple[str, str]]:
    """テキストから (np_string, head_noun) のリストを返す(04 と同一)。"""
    import nltk

    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    tree = _WORKER_PARSER.parse(tagged)  # type: ignore[union-attr]

    nps: list[tuple[str, str]] = []
    for subtree in tree.subtrees(lambda t: t.label() == "NP"):
        leaves = subtree.leaves()
        np_str = " ".join(word for word, _ in leaves).lower()
        nn_words = [word.lower() for word, pos in leaves if pos.startswith("NN")]
        if nn_words:
            nps.append((np_str, nn_words[-1]))
    return nps


def find_longest_np_containing(
    token: str,
    nps: list[tuple[str, str]],
) -> Optional[tuple[str, str]]:
    """token を語境界で含む最長 NP を返す(04 と同一)。"""
    candidates = [
        (np_str, head) for np_str, head in nps if token in np_str.split()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda x: len(x[0]))


# ── ワーカー本体: 1レコード分の対象表現を抽出 ─────────────────────────────────

def _init_worker(
    candidate_tokens: list[str],
    drop_single_word: bool,
) -> None:
    global _WORKER_PARSER, _WORKER_TOKENS, _WORKER_DROP_SINGLE_WORD
    import nltk

    _WORKER_PARSER = nltk.RegexpParser(NP_GRAMMAR)
    _WORKER_TOKENS = set(candidate_tokens)
    _WORKER_DROP_SINGLE_WORD = drop_single_word


def extract_record_terms(texts: list[str]) -> list[str]:
    """レコードの全テキストへ Method A → Method B を順に適用し、表現集合を返す。

    Method A : 候補トークン(_WORKER_TOKENS)が本文に出現するか判定。
    Method B : 出現した各トークンを、それを含む最長 NP に展開(04 main と同一の
               フォールバック: 該当 NP が無ければトークン自身を採用)。
    """
    phrases: set[str] = set()
    for text in texts:
        if not text:
            continue
        nps = extract_nps_from_text(text)
        text_lower = text.lower()
        matched_tokens = [t for t in _WORKER_TOKENS if t in text_lower]
        for token in matched_tokens:
            result = find_longest_np_containing(token, nps)
            phrase = token if result is None else result[0]
            if _WORKER_DROP_SINGLE_WORD and len(phrase.split()) < 2:
                continue
            phrases.add(phrase)
    return sorted(phrases)


def _process_row_task(task: dict) -> dict:
    """1レコードを処理し、レコード単位の共起セットを返す。"""
    texts = [item["text"] for item in task["texts"]]
    matched_terms = extract_record_terms(texts)
    return {
        "row_index": task["row_index"],
        "uuid": task["uuid"],
        "matched_terms": matched_terms,
    }


def _process_row_batch_task(tasks: list[dict]) -> tuple[list[dict], int]:
    out = [_process_row_task(task) for task in tasks]
    return out, len(tasks)


# ── データセット読み込み(元コードを踏襲) ─────────────────────────────────────

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
            v = item.strip() if isinstance(item, str) else str(item).strip()
            if v:
                out.append(v)
        return out
    v = str(value).strip()
    return [v] if v else []


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


def iter_row_task_batches(row_tasks: Iterable[dict], batch_size: int) -> Iterable[list[dict]]:
    batch: list[dict] = []
    for task in row_tasks:
        batch.append(task)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ── SQLite による共起セット集計(元コードを踏襲) ─────────────────────────────

def setup_sqlite(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("CREATE TABLE raw_sets (term_set_key TEXT NOT NULL);")
    conn.commit()
    return conn


def flush_set_buffer(conn: sqlite3.Connection, buf: list[tuple[str]]) -> None:
    if not buf:
        return
    conn.executemany("INSERT INTO raw_sets (term_set_key) VALUES (?)", buf)
    conn.commit()
    buf.clear()


def export_set_probabilities(
    conn: sqlite3.Connection,
    total_records: int,
    output_path: Path,
) -> int:
    conn.execute("DROP TABLE IF EXISTS set_counts;")
    conn.execute(
        """
        CREATE TABLE set_counts AS
        SELECT term_set_key, COUNT(*) AS record_count
        FROM raw_sets
        GROUP BY term_set_key
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_set_counts_count ON set_counts(record_count DESC);"
    )
    conn.commit()

    cur = conn.execute("SELECT COUNT(*) FROM set_counts")
    set_types = int(cur.fetchone()[0])

    with output_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["terms_json", "set_size", "record_count", "probability"])
        for term_set_key, cnt in conn.execute(
            "SELECT term_set_key, record_count FROM set_counts "
            "ORDER BY record_count DESC, term_set_key ASC"
        ):
            try:
                set_size = len(json.loads(term_set_key))
            except json.JSONDecodeError:
                set_size = 0
            p = (cnt / total_records) if total_records > 0 else 0.0
            writer.writerow([term_set_key, set_size, cnt, p])

    return set_types


# ── 実行本体 ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    candidate_tokens = load_candidate_tokens(
        candidates_csv=args.candidates_csv,
        min_freq_profile=args.min_freq_profile,
        min_effect_size=args.min_effect_size,
    )
    if not candidate_tokens:
        raise RuntimeError(
            "条件に一致する候補トークンが0件です。閾値を見直してください。"
        )

    # NLTK データを親プロセスで一度だけ取得(ワーカーはキャッシュを共有)
    import nltk
    for resource in ("punkt_tab", "averaged_perceptron_tagger_eng"):
        nltk.download(resource, quiet=True)

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, workers)

    requested_fields = list(TEXT_FIELDS)

    conn = setup_sqlite(args.sqlite_path)
    set_insert_buffer: list[tuple[str]] = []
    inserted_set_rows = 0
    total_records = 0
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
    row_task_batches = iter_row_task_batches(
        row_tasks=row_tasks,
        batch_size=max(1, args.row_batch_size),
    )

    with args.text_output.open("w", encoding="utf-8") as fw:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(sorted(candidate_tokens), args.drop_single_word),
        ) as pool:
            for row_result, rows_done in pool.imap_unordered(
                _process_row_batch_task,
                row_task_batches,
                chunksize=max(1, args.pool_chunksize),
            ):
                sampled_rows += rows_done
                for rec in row_result:
                    terms = rec["matched_terms"]
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_records += 1
                    unique_sorted = sorted(set(terms))
                    term_set_key = json.dumps(unique_sorted, ensure_ascii=False)
                    set_insert_buffer.append((term_set_key,))
                    if len(set_insert_buffer) >= args.sqlite_insert_buffer:
                        flush_set_buffer(conn, set_insert_buffer)
                        inserted_set_rows += args.sqlite_insert_buffer

                if args.progress_every > 0 and sampled_rows % args.progress_every == 0:
                    elapsed = max(1e-9, time.time() - started)
                    rps = sampled_rows / elapsed
                    print(
                        f"[progress] sampled_rows={sampled_rows} "
                        f"total_records={total_records} "
                        f"inserted_sets={inserted_set_rows + len(set_insert_buffer)} "
                        f"speed={rps:.2f} rows/s"
                    )

    inserted_set_rows += len(set_insert_buffer)
    flush_set_buffer(conn, set_insert_buffer)
    set_types = export_set_probabilities(
        conn, total_records=total_records, output_path=args.term_output
    )
    conn.close()

    elapsed = time.time() - started
    print(f"candidate_tokens={len(candidate_tokens)}")
    print(f"workers={workers}")
    print(f"sampled_rows={sampled_rows}")
    print(f"total_records={total_records}")
    print(f"set_rows_inserted={inserted_set_rows}")
    print(f"set_types={set_types}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"text_output={args.text_output}")
    print(f"term_output={args.term_output}")
    print(f"sqlite_path={args.sqlite_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Method A → Method B を順に適用して対象表現を特定し、"
            "Nemotron-Personas-USA のレコード単位で共起語セットの出現確率を集計する"
        )
    )
    p.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV,
                   help="Method A の候補トークン CSV (03_method_a_candidates.csv)")
    p.add_argument("--min-freq-profile", type=int, default=1,
                   help="候補トークンの最小 freq_profile")
    p.add_argument("--min-effect-size", type=float, default=1.0,
                   help="候補トークンの最小 effect_size (odds ratio)")
    p.add_argument("--drop-single-word", action="store_true",
                   help="1語の表現を除外する(04 の最終出力と同じ挙動)")
    p.add_argument("--dataset", default="nvidia/Nemotron-Personas-USA")
    p.add_argument("--split", default="train")
    p.add_argument("--no-streaming", action="store_true")
    p.add_argument("--sample-ratio", type=float, default=0.01, help="0.0-1.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0, help="0で上限なし")
    p.add_argument("--workers", type=int, default=0, help="0以下でCPUコア数")
    p.add_argument("--row-batch-size", type=int, default=128,
                   help="ワーカーへ渡す行バッチサイズ")
    p.add_argument("--pool-chunksize", type=int, default=64,
                   help="imap_unordered の chunksize")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--text-output", type=Path,
                   default=DATA_DIR / "05_record_matched_terms.jsonl")
    p.add_argument("--term-output", type=Path,
                   default=DATA_DIR / "05_record_set_probabilities.csv")
    p.add_argument("--sqlite-path", type=Path,
                   default=DATA_DIR / "05_record_sets.sqlite3",
                   help="セット集計に使うSQLite DBパス")
    p.add_argument("--sqlite-insert-buffer", type=int, default=20000,
                   help="SQLiteへ一括INSERTするバッファ行数")
    args = p.parse_args()

    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("--sample-ratio は 0.0 より大きく 1.0 以下で指定してください。")
    if args.sqlite_insert_buffer <= 0:
        raise ValueError("--sqlite-insert-buffer は1以上で指定してください。")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
