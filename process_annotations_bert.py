#!/usr/bin/env python3
"""annotations_BERT.jsonl を処理して PROFILE 優位語を抽出する。"""

from __future__ import annotations

import argparse
import csv
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
    )


if __name__ == "__main__":
    main()
