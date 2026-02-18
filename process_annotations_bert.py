#!/usr/bin/env python3
"""
annotations_BERT.jsonl を形態素解析し、PROFILE 側で有意に多い語を抽出するスクリプト。

処理概要:
1. JSONL の各行から `label` と `clause` を読み取り、対象ラベル (PROFILE/NONE) を集計する。
2. fugashi + UniDic でトークン化し、機能語や非自立語を除外して自立語のみを数える。
3. 必要に応じて Nemotron データセットの skills/hobbies リストを PROFILE 側語彙に加算する。
4. 各語について 2x2 分割表を作り、G-test (LLR)・p 値・オッズ比を計算する。
5. PROFILE 比率が NONE を上回り、有意水準 5% を満たす語だけを CSV に出力する。

主な出力:
- `profile_significant_morphemes.csv` (語、各ラベルの頻度/比率、odds ratio、LLR、p 値)
- 実行ログ (処理件数、スキップ件数、トークン総数、抽出語数 など)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import os
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class Morpheme:
    surface: str
    lemma: str
    pos1: str
    pos2: str


class FugashiTokenizer:
    name = "fugashi"

    def __init__(self) -> None:
        import fugashi  # type: ignore

        self._tagger = fugashi.Tagger()

    def _extract_pos(self, feature) -> tuple[str, str]:
        return (getattr(feature, "pos1", "") or "", getattr(feature, "pos2", "") or "")

    def _extract_lemma(self, word) -> str:
        feat = word.feature
        lemma = str(getattr(feat, "lemma", "") or "")
        return word.surface if lemma == "*" or not lemma else lemma

    def tokenize(self, text: str) -> Iterable[Morpheme]:
        for word in self._tagger(text):
            pos1, pos2 = self._extract_pos(word.feature)
            lemma = self._extract_lemma(word)
            yield Morpheme(surface=word.surface, lemma=lemma, pos1=pos1, pos2=pos2)


def load_tokenizer() -> FugashiTokenizer:
    try:
        return FugashiTokenizer()
    except Exception as e:  # pragma: no cover
        msg = "fugashi + UniDic の初期化に失敗しました。"
        msg += " `pip install fugashi unidic-lite` などで依存を導入してください。"
        msg += f"\nFugashiTokenizer: {e}"
        raise RuntimeError(msg) from e


def is_independent_word(m: Morpheme) -> bool:
    if not m.lemma or m.lemma == "*":
        return False

    # 機能語・記号を除外
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
    if m.pos1 in excluded_pos1:
        return False

    # 非自立や形式語的なものを除外
    excluded_pos2 = {"非自立", "代名詞"}
    if m.pos2 in excluded_pos2:
        return False

    # UniDic では「歳」などが pos1=接尾辞 になるため許可する
    return m.pos1 in {"名詞", "動詞", "形容詞", "副詞", "接尾辞"}


def normalize_token(token: str) -> str:
    # 全角/半角の揺れを吸収し、英字の大文字小文字を統一する
    normalized = unicodedata.normalize("NFKC", token)
    return normalized.lower().strip()


def g_test_llr(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """2x2 分割表の尤度比統計量 G と p値(自由度1)を返す。"""
    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0

    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d

    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 0.0, 1.0

    expected = (
        row1 * col1 / n,
        row1 * col2 / n,
        row2 * col1 / n,
        row2 * col2 / n,
    )
    observed = (a, b, c, d)

    g = 0.0
    for o, e in zip(observed, expected):
        if o > 0 and e > 0:
            g += 2.0 * o * math.log(o / e)

    # df=1 のカイ二乗分布の上側確率: sf(x)=erfc(sqrt(x/2))
    p_value = math.erfc(math.sqrt(max(g, 0.0) / 2.0))
    return g, p_value


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    """Haldane-Anscombe補正付きオッズ比。"""
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


@dataclass
class ChunkStats:
    start: int
    end: int
    profile_counts: Counter[str]
    none_counts: Counter[str]
    profile_total: int
    none_total: int
    lines: int
    malformed: int
    skipped: int


@dataclass
class NemotronChunkStats:
    profile_counts: Counter[str]
    profile_total: int
    rows: int
    texts: int


_NEMOTRON_WORKER_TOKENIZER: Optional[FugashiTokenizer] = None


def _get_nemotron_worker_tokenizer() -> FugashiTokenizer:
    global _NEMOTRON_WORKER_TOKENIZER
    if _NEMOTRON_WORKER_TOKENIZER is None:
        _NEMOTRON_WORKER_TOKENIZER = load_tokenizer()
    return _NEMOTRON_WORKER_TOKENIZER


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


def iter_nemotron_text_chunks(
    rows_iter: Iterable[dict],
    chunk_rows: int,
    target_fields: tuple[str, str],
) -> Iterable[tuple[list[str], int]]:
    buf_texts: list[str] = []
    buf_rows = 0

    for row in rows_iter:
        buf_rows += 1
        for field in target_fields:
            buf_texts.extend(normalize_to_text_list(row.get(field)))

        if buf_rows >= chunk_rows:
            yield buf_texts, buf_rows
            buf_texts = []
            buf_rows = 0

    if buf_rows > 0:
        yield buf_texts, buf_rows


def _process_nemotron_text_chunk(args: tuple[list[str], int]) -> NemotronChunkStats:
    texts, rows = args
    tokenizer = _get_nemotron_worker_tokenizer()
    profile_counts: Counter[str] = Counter()
    profile_total = 0

    for text in texts:
        for m in tokenizer.tokenize(text):
            if not is_independent_word(m):
                continue
            token = normalize_token(m.lemma)
            if not token:
                continue
            profile_counts[token] += 1
            profile_total += 1

    return NemotronChunkStats(
        profile_counts=profile_counts,
        profile_total=profile_total,
        rows=rows,
        texts=len(texts),
    )


def augment_profile_from_nemotron_lists(
    profile_counts: Counter[str],
    profile_total: int,
    dataset_name: str,
    dataset_split: str,
    dataset_streaming: bool,
    dataset_max_rows: int,
    workers: int,
    show_progress: bool,
    chunk_rows: int,
) -> tuple[int, int, int]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split=dataset_split, streaming=dataset_streaming)
    rows = ds if dataset_max_rows <= 0 else itertools.islice(ds, dataset_max_rows)
    target_fields = ("skills_and_expertise_list", "hobbies_and_interests_list")
    total_rows_hint: Optional[int] = None
    if dataset_max_rows > 0:
        total_rows_hint = dataset_max_rows
    elif not dataset_streaming:
        try:
            total_rows_hint = len(ds)
        except TypeError:
            total_rows_hint = None

    rows_count = 0
    added_tokens = 0
    started_at = time.time()
    done_chunks = 0
    done_texts = 0
    chunk_rows = max(1, chunk_rows)
    workers = max(1, workers)

    chunk_iter = iter_nemotron_text_chunks(rows, chunk_rows=chunk_rows, target_fields=target_fields)

    if workers == 1:
        nemotron_stats_iter = (_process_nemotron_text_chunk(chunk) for chunk in chunk_iter)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            nemotron_stats_iter = pool.imap_unordered(
                _process_nemotron_text_chunk,
                chunk_iter,
                chunksize=1,
            )
            for st in nemotron_stats_iter:
                profile_counts.update(st.profile_counts)
                profile_total += st.profile_total
                added_tokens += st.profile_total
                rows_count += st.rows
                done_chunks += 1
                done_texts += st.texts
                if show_progress:
                    elapsed = max(1e-9, time.time() - started_at)
                    rows_per_sec = rows_count / elapsed
                    msg = (
                        f"[progress:nemotron] chunks={done_chunks} rows={rows_count}"
                        f" texts={done_texts} added_tokens={added_tokens}"
                        f" speed={rows_per_sec:.2f} rows/s"
                    )
                    if total_rows_hint:
                        pct = min(100.0, rows_count / total_rows_hint * 100.0)
                        msg += f" ({pct:.1f}%)"
                    print(msg)
            return profile_total, rows_count, added_tokens

    for st in nemotron_stats_iter:
        profile_counts.update(st.profile_counts)
        profile_total += st.profile_total
        added_tokens += st.profile_total
        rows_count += st.rows
        done_chunks += 1
        done_texts += st.texts
        if show_progress:
            elapsed = max(1e-9, time.time() - started_at)
            rows_per_sec = rows_count / elapsed
            msg = (
                f"[progress:nemotron] chunks={done_chunks} rows={rows_count}"
                f" texts={done_texts} added_tokens={added_tokens}"
                f" speed={rows_per_sec:.2f} rows/s"
            )
            if total_rows_hint:
                pct = min(100.0, rows_count / total_rows_hint * 100.0)
                msg += f" ({pct:.1f}%)"
            print(msg)

    return profile_total, rows_count, added_tokens


def split_byte_ranges(
    input_path: Path, workers: int, chunks_per_worker: int = 8
) -> list[tuple[int, int]]:
    file_size = input_path.stat().st_size
    if file_size == 0:
        return [(0, 0)]

    workers = max(1, workers)
    num_chunks = max(1, workers * max(1, chunks_per_worker))
    num_chunks = min(num_chunks, file_size)
    chunk_size = max(1, file_size // num_chunks)
    ranges: list[tuple[int, int]] = []
    start = 0

    for i in range(num_chunks):
        if i == num_chunks - 1:
            end = file_size
        else:
            end = start + chunk_size
        ranges.append((start, end))
        start = end

    return ranges


def process_chunk(
    input_path: str,
    start: int,
    end: int,
    profile_label: str,
    none_label: str,
) -> ChunkStats:
    tokenizer = load_tokenizer()
    profile_counts: Counter[str] = Counter()
    none_counts: Counter[str] = Counter()
    profile_total = 0
    none_total = 0
    lines = 0
    malformed = 0
    skipped = 0

    with open(input_path, "rb") as f:
        f.seek(start)
        if start != 0:
            f.readline()

        while True:
            pos = f.tell()
            if pos >= end and end != 0:
                break

            raw = f.readline()
            if not raw:
                break

            lines += 1
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                malformed += 1
                continue

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            label = obj.get("label")
            clause = obj.get("clause")
            if not isinstance(clause, str) or label not in {profile_label, none_label}:
                skipped += 1
                continue

            target_counter: Optional[Counter[str]] = None
            if label == profile_label:
                target_counter = profile_counts
            elif label == none_label:
                target_counter = none_counts

            for m in tokenizer.tokenize(clause):
                if not is_independent_word(m):
                    continue
                token = normalize_token(m.lemma)
                if not token:
                    continue
                target_counter[token] += 1
                if label == profile_label:
                    profile_total += 1
                else:
                    none_total += 1

    return ChunkStats(
        start=start,
        end=end,
        profile_counts=profile_counts,
        none_counts=none_counts,
        profile_total=profile_total,
        none_total=none_total,
        lines=lines,
        malformed=malformed,
        skipped=skipped,
    )


def process(
    input_path: Path,
    output_path: Path,
    profile_label: str = "PROFILE",
    none_label: str = "NONE",
    min_count_profile: int = 1,
    workers: Optional[int] = None,
    show_progress: bool = True,
    chunks_per_worker: int = 8,
    augment_nemotron_lists: bool = True,
    nemotron_dataset: str = "nvidia/Nemotron-Personas-Japan",
    nemotron_split: str = "train",
    nemotron_streaming: bool = True,
    nemotron_max_rows: int = 0,
    nemotron_workers: Optional[int] = None,
    nemotron_chunk_rows: int = 500,
) -> None:
    profile_counts: Counter[str] = Counter()
    none_counts: Counter[str] = Counter()
    profile_total = 0
    none_total = 0

    lines = 0
    malformed = 0
    skipped = 0

    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    workers = max(1, workers)

    ranges = split_byte_ranges(input_path, workers, chunks_per_worker=chunks_per_worker)
    use_workers = min(workers, len(ranges))
    total_chunks = len(ranges)
    file_size = input_path.stat().st_size

    started_at = time.time()
    completed_chunks = 0
    completed_bytes = 0

    if use_workers == 1 and total_chunks == 1:
        st = process_chunk(
            str(input_path), ranges[0][0], ranges[0][1], profile_label, none_label
        )
        chunk_stats_iter = [st]
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=use_workers)
        chunk_stats_iter = pool.imap_unordered(
            _process_chunk_star,
            [
                (str(input_path), start, end, profile_label, none_label)
                for start, end in ranges
            ],
        )

    for st in chunk_stats_iter:
        profile_counts.update(st.profile_counts)
        none_counts.update(st.none_counts)
        profile_total += st.profile_total
        none_total += st.none_total
        lines += st.lines
        malformed += st.malformed
        skipped += st.skipped

        completed_chunks += 1
        completed_bytes += max(0, st.end - st.start)
        if show_progress:
            pct = (completed_bytes / file_size * 100.0) if file_size > 0 else 100.0
            elapsed = max(1e-9, time.time() - started_at)
            mbps = (completed_bytes / (1024 * 1024)) / elapsed
            print(
                f"[progress] chunks={completed_chunks}/{total_chunks} "
                f"bytes={completed_bytes}/{file_size} ({pct:.1f}%) "
                f"speed={mbps:.2f} MB/s"
            )

    if not (use_workers == 1 and total_chunks == 1):
        pool.close()
        pool.join()

    nemotron_rows = 0
    nemotron_added_tokens = 0
    use_nemotron_workers = 0
    if augment_nemotron_lists:
        if nemotron_workers is None or nemotron_workers <= 0:
            nemotron_workers = workers
        use_nemotron_workers = max(1, nemotron_workers)
        profile_total, nemotron_rows, nemotron_added_tokens = (
            augment_profile_from_nemotron_lists(
                profile_counts=profile_counts,
                profile_total=profile_total,
                dataset_name=nemotron_dataset,
                dataset_split=nemotron_split,
                dataset_streaming=nemotron_streaming,
                dataset_max_rows=nemotron_max_rows,
                workers=use_nemotron_workers,
                show_progress=show_progress,
                chunk_rows=nemotron_chunk_rows,
            )
        )

    # 片側条件: PROFILE 側が大きい語のみ採用
    alpha = 0.05
    results = []
    all_tokens = set(profile_counts.keys()) | set(none_counts.keys())

    for token in all_tokens:
        a = profile_counts[token]
        c = none_counts[token]
        if a < min_count_profile:
            continue

        b = profile_total - a
        d = none_total - c
        if b < 0 or d < 0:
            continue

        g, p = g_test_llr(a, b, c, d)
        oratio = odds_ratio(a, b, c, d)

        # 比率とオッズ比が PROFILE > NONE で、5%有意
        prof_rate = a / profile_total if profile_total > 0 else 0.0
        none_rate = c / none_total if none_total > 0 else 0.0

        if prof_rate > none_rate and oratio > 1.0 and p < alpha:
            results.append((token, a, c, prof_rate, none_rate, oratio, g, p))

    results.sort(key=lambda x: (-x[5], -x[1], x[0]))

    with output_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(
            [
                "token",
                f"count_{profile_label}",
                f"count_{none_label}",
                f"rate_{profile_label}",
                f"rate_{none_label}",
                "odds_ratio",
                "llr_g",
                "p_value",
            ]
        )
        writer.writerows(results)

    print("tokenizer=fugashi")
    print(f"workers={use_workers}")
    print(f"lines={lines}")
    print(f"malformed_json={malformed}")
    print(f"skipped_non_target={skipped}")
    print(f"total_tokens_{profile_label}={profile_total}")
    print(f"total_tokens_{none_label}={none_total}")
    if augment_nemotron_lists:
        print(f"nemotron_dataset={nemotron_dataset}")
        print(f"nemotron_workers={use_nemotron_workers}")
        print(f"nemotron_rows={nemotron_rows}")
        print(f"nemotron_added_tokens_to_{profile_label}={nemotron_added_tokens}")
    print(f"significant_tokens={len(results)}")
    print(f"output={output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="JSONLを形態素解析し、PROFILE vs NONEで有意に多い語を抽出する"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("annotations_BERT.jsonl"),
        help="入力JSONL",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("profile_significant_morphemes.csv"),
        help="出力CSV",
    )
    p.add_argument("--profile-label", default="PROFILE", help="PROFILE側ラベル名")
    p.add_argument("--none-label", default="NONE", help="NONE側ラベル名")
    p.add_argument(
        "--min-count-profile",
        type=int,
        default=1,
        help="PROFILE側の最低出現回数",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="並列プロセス数 (0以下でCPUコア数を使用)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="進行状況の表示を無効化",
    )
    p.add_argument(
        "--chunks-per-worker",
        type=int,
        default=8,
        help="ワーカーあたりのチャンク数（大きいほど進捗表示が細かい）",
    )
    p.add_argument(
        "--no-augment-nemotron-lists",
        action="store_true",
        help="Nemotronのskills/hobbiesリストをPROFILE集計に追加しない",
    )
    p.add_argument(
        "--nemotron-dataset",
        default="nvidia/Nemotron-Personas-Japan",
        help="追加集計に使うNemotronデータセット名",
    )
    p.add_argument(
        "--nemotron-split",
        default="train",
        help="追加集計に使うNemotron split 名",
    )
    p.add_argument(
        "--nemotron-no-streaming",
        action="store_true",
        help="Nemotron読み込みでstreamingを使わない",
    )
    p.add_argument(
        "--nemotron-max-rows",
        type=int,
        default=0,
        help="Nemotron追加集計の最大行数 (0以下で全件)",
    )
    p.add_argument(
        "--nemotron-workers",
        type=int,
        default=0,
        help="Nemotron追加集計の並列プロセス数 (0以下で--workersと同値)",
    )
    p.add_argument(
        "--nemotron-chunk-rows",
        type=int,
        default=500,
        help="Nemotron追加集計の1チャンクあたり行数",
    )
    return p.parse_args()


def _process_chunk_star(args: tuple[str, int, int, str, str]) -> ChunkStats:
    return process_chunk(*args)


def main() -> None:
    args = parse_args()
    process(
        input_path=args.input,
        output_path=args.output,
        profile_label=args.profile_label,
        none_label=args.none_label,
        min_count_profile=args.min_count_profile,
        workers=args.workers,
        show_progress=not args.no_progress,
        chunks_per_worker=args.chunks_per_worker,
        augment_nemotron_lists=not args.no_augment_nemotron_lists,
        nemotron_dataset=args.nemotron_dataset,
        nemotron_split=args.nemotron_split,
        nemotron_streaming=not args.nemotron_no_streaming,
        nemotron_max_rows=args.nemotron_max_rows,
        nemotron_workers=args.nemotron_workers,
        nemotron_chunk_rows=args.nemotron_chunk_rows,
    )


if __name__ == "__main__":
    main()
