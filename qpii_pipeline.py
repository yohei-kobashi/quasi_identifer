#!/usr/bin/env python3
"""QPII 抽出パイプライン(Method A → Method B → レコード内共起集計)を1本に統合。

3つのステージを1スクリプトにまとめたもの。

  method-a       形態素ベースの統計抽出(Fisher の正確確率検定)で候補トークンを得る。
                 旧 03_method_a_extract.py 相当。
  method-b       候補トークンを最長名詞句(NP)へ展開して候補フレーズを得る。
                 旧 04_method_b_phrase.py 相当。
  cooccurrence   Method A → Method B を順に適用して対象表現を特定し、
                 Nemotron-Personas-USA の「レコード単位」で共起セットを集計する。
                 旧 05_record_cooccurrence.py 相当。
  all            method-a → method-b → cooccurrence を一気通貫で実行する。
                 候補トークンはメモリ上で受け渡すため CSV 経由のやり取りは不要。

入力
----
  data/02_labeled_clauses.csv   (label ∈ {PROFILE, NONE}, clause)

出力
----
  data/03_method_a_stats.csv          Method A 全統計(α≤0.10)
  data/03_method_a_candidates.csv     Method A 候補トークン(+ alpha_tier 列)
  data/04_method_b_candidates.csv     Method B 候補フレーズ(+ alpha_tier/source_tokens 列)
  data/05_record_matched_terms.jsonl  レコード別の対象表現明細
  data/05_record_set_probabilities.csv 共起セットの出現確率(+ tiers_json/n_tier_* 列)

alpha_tier は Bonferroni 補正後に通過する最も厳しい有意水準("0.01"/"0.05"/"0.10")。
フレーズ/共起の tier は「そのフレーズに含まれる候補トークンの最も厳しい tier」。
全3階層を得るには method-a を α=0.10 で実行する(例: `method-a --alpha 0.10`,
`all --alpha 0.10`)。

使い方
------
  python qpii_pipeline.py method-a                 # 既定 ALPHA で候補トークン
  python qpii_pipeline.py method-a --filter 0.01   # 保存済み統計から再フィルタ
  python qpii_pipeline.py method-a --compare       # α=0.01/0.05/0.10 比較
  python qpii_pipeline.py method-b                 # 候補フレーズ
  python qpii_pipeline.py cooccurrence --sample-ratio 0.01
  python qpii_pipeline.py all --sample-ratio 0.01  # 全ステージ
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import nltk
import pandas as pd
from scipy.stats import fisher_exact
from tqdm import tqdm

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    sys.path.insert(0, str(Path.cwd()))
from config import ALPHA, DATA_DIR, KEEP_POS_PREFIXES, TEXT_FIELDS

# ── パス ────────────────────────────────────────────────────────────────────
LABELED_CLAUSES_PATH: Path = DATA_DIR / "02_labeled_clauses.csv"

A_STATS_PATH: Path        = DATA_DIR / "03_method_a_stats.csv"
A_CANDIDATES_PATH: Path   = DATA_DIR / "03_method_a_candidates.csv"
B_CANDIDATES_PATH: Path   = DATA_DIR / "04_method_b_candidates.csv"
COOC_TEXT_OUTPUT: Path    = DATA_DIR / "05_record_matched_terms.jsonl"
COOC_TERM_OUTPUT: Path    = DATA_DIR / "05_record_set_probabilities.csv"
COOC_SQLITE_PATH: Path    = DATA_DIR / "05_record_sets.sqlite3"

# Method A: 統計キャッシュを作る最も緩い閾値
MAX_ALPHA: float = 0.10
COMPARE_ALPHAS: list[float] = [0.01, 0.05, 0.10]

# Method B / cooccurrence: NP 文法(longest enclosing NP 抽出用)
NP_GRAMMAR: str = r"""
  NP: {<DT>?<JJ.*>*<NN.*>+(<IN><DT>?<JJ.*>*<NN.*>+)*}
"""


# ════════════════════════════════════════════════════════════════════════════
# 共有: NLTK ヘルパー
# ════════════════════════════════════════════════════════════════════════════

def download_nltk_data(with_stopwords: bool = False) -> None:
    resources = ["punkt_tab", "averaged_perceptron_tagger_eng"]
    if with_stopwords:
        resources.append("stopwords")
    for resource in resources:
        nltk.download(resource, quiet=True)


def make_np_parser() -> "nltk.RegexpParser":
    return nltk.RegexpParser(NP_GRAMMAR)


def extract_nps(text: str, parser: "nltk.RegexpParser") -> list[tuple[str, str]]:
    """テキストから (np_string(lower), head_noun) のリストを返す。

    head_noun = NP 内の末尾 NN*(英語の主辞右則)。
    """
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    tree = parser.parse(tagged)

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
    """token を語境界で含む最長 NP を返す。無ければ None。"""
    candidates = [(np_str, head) for np_str, head in nps if token in np_str.split()]
    if not candidates:
        return None
    return max(candidates, key=lambda x: len(x[0]))


# ════════════════════════════════════════════════════════════════════════════
# 共有: alpha tier(有意水準の階層)
# ════════════════════════════════════════════════════════════════════════════

TIER_ALPHAS: tuple[float, ...] = (0.01, 0.05, 0.10)  # 昇順 = 厳しい順
TIER_ORDER: dict[str, int] = {"0.01": 0, "0.05": 1, "0.10": 2}


def compute_alpha_tier(p_value: float, n_tokens: int) -> Optional[str]:
    """p値が Bonferroni 補正後に通過する最も厳しい α を返す("0.01"/"0.05"/"0.10")。"""
    for alpha in TIER_ALPHAS:
        if p_value < alpha / max(n_tokens, 1):
            return f"{alpha:.2f}"
    return None


def strictest_tier(tiers: Iterable[Optional[str]]) -> Optional[str]:
    """複数 tier のうち最も厳しい(小さい α)ものを返す。"""
    valid = [t for t in tiers if t in TIER_ORDER]
    if not valid:
        return None
    return min(valid, key=lambda t: TIER_ORDER[t])


def phrase_alpha_tier(phrase: str, token_tier: dict[str, str]) -> Optional[str]:
    """フレーズに含まれる候補トークンのうち最も厳しい tier を返す。"""
    return strictest_tier(token_tier.get(w) for w in phrase.split())


def add_alpha_tier_column(df: pd.DataFrame) -> pd.DataFrame:
    """候補 DataFrame に alpha_tier 列を付与して返す。"""
    df = df.copy()
    df["alpha_tier"] = [
        compute_alpha_tier(p, int(n)) for p, n in zip(df["p_value"], df["n_tokens"])
    ]
    return df


def build_token_tier_map(df_candidates: pd.DataFrame) -> dict[str, str]:
    """候補 DataFrame から token→alpha_tier マップを作る。"""
    out: dict[str, str] = {}
    if "alpha_tier" not in df_candidates.columns:
        return out
    for token, tier in zip(df_candidates["token"], df_candidates["alpha_tier"]):
        if isinstance(tier, str) and tier:
            out[str(token).lower()] = tier
    return out


# ════════════════════════════════════════════════════════════════════════════
# 共有: 並列実行ヘルパー
# ════════════════════════════════════════════════════════════════════════════

def resolve_workers(workers: int) -> int:
    w = workers if workers and workers > 0 else (os.cpu_count() or 1)
    return max(1, w)


def chunked(seq: list, size: int) -> list[list]:
    size = max(1, size)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def parallel_map(
    work_fn,
    batches: list,
    workers: int,
    desc: str,
    init_fn=None,
    init_args: tuple = (),
) -> list:
    """batches を work_fn でマップする。workers<=1 ならプール無しで逐次実行。

    spawn プールを使うため work_fn / init_fn はモジュールトップレベル関数であること。
    """
    w = resolve_workers(workers)
    if w == 1:
        if init_fn is not None:
            init_fn(*init_args)
        return [work_fn(b) for b in tqdm(batches, desc=desc)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(w, initializer=init_fn, initargs=init_args) as pool:
        return list(tqdm(pool.imap(work_fn, batches), total=len(batches), desc=desc))


# ════════════════════════════════════════════════════════════════════════════
# Method A: 形態素ベースの統計抽出
# ════════════════════════════════════════════════════════════════════════════

def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    """add-0.5 平滑化付きオッズ比。"""
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


# ── 並列: トークン化ワーカー ──────────────────────────────────────────────────
_MA_STOPWORDS: set[str] = set()


def _ma_count_init() -> None:
    global _MA_STOPWORDS
    download_nltk_data(with_stopwords=True)
    _MA_STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def _ma_count_batch(clauses: list[str]) -> tuple[Counter, dict[str, str]]:
    freq: Counter[str] = Counter()
    pos_map: dict[str, str] = {}
    for clause in clauses:
        for token, pos in nltk.pos_tag(nltk.word_tokenize(clause.lower())):
            if not token.isalpha() or len(token) < 3:
                continue
            if not any(pos.startswith(p) for p in KEEP_POS_PREFIXES):
                continue
            if token in _MA_STOPWORDS:
                continue
            freq[token] += 1
            pos_map.setdefault(token, pos)
    return freq, pos_map


def count_tokens_in_clauses(
    clauses: list[str],
    workers: int = 1,
    batch_size: int = 256,
) -> tuple[Counter[str], dict[str, str]]:
    """(トークン頻度 Counter, トークン→POS マップ) を並列で集計する。"""
    results = parallel_map(
        _ma_count_batch, chunked(clauses, batch_size), workers,
        desc="  Tokenizing", init_fn=_ma_count_init,
    )
    freq: Counter[str] = Counter()
    pos_map: dict[str, str] = {}
    for f, pm in results:
        freq.update(f)
        for k, v in pm.items():
            pos_map.setdefault(k, v)
    return freq, pos_map


# ── 並列: Fisher 検定ワーカー ─────────────────────────────────────────────────
_MA_FREQ_PROFILE: dict[str, int] = {}
_MA_FREQ_NONE: dict[str, int] = {}
_MA_N_PROFILE: int = 0
_MA_N_NONE: int = 0
_MA_POS_MAP: dict[str, str] = {}
_MA_CORRECTED_MAX: float = 0.0
_MA_N_TOKENS: int = 0


def _ma_fisher_init(fp, fn, n_profile, n_none, pos_map, corrected_max, n_tokens) -> None:
    global _MA_FREQ_PROFILE, _MA_FREQ_NONE, _MA_N_PROFILE, _MA_N_NONE
    global _MA_POS_MAP, _MA_CORRECTED_MAX, _MA_N_TOKENS
    _MA_FREQ_PROFILE = fp
    _MA_FREQ_NONE = fn
    _MA_N_PROFILE = n_profile
    _MA_N_NONE = n_none
    _MA_POS_MAP = pos_map
    _MA_CORRECTED_MAX = corrected_max
    _MA_N_TOKENS = n_tokens


def _ma_fisher_batch(tokens: list[str]) -> list[dict]:
    rows: list[dict] = []
    for token in tokens:
        a = _MA_FREQ_PROFILE.get(token, 0)
        if a == 0:
            continue
        b = _MA_N_PROFILE - a
        c = _MA_FREQ_NONE.get(token, 0)
        d = _MA_N_NONE - c
        _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        if p_value < _MA_CORRECTED_MAX:
            rows.append({
                "token":        token,
                "pos":          _MA_POS_MAP.get(token, ""),
                "freq_profile": a,
                "freq_none":    c,
                "p_value":      p_value,
                "effect_size":  odds_ratio(a, b, c, d),
                "n_tokens":     _MA_N_TOKENS,
            })
    return rows


def build_stats_cache(workers: int = 1, batch_size: int = 256) -> pd.DataFrame:
    """トークン化 + Fisher 検定を並列実行し、全統計を A_STATS_PATH へ保存。

    Bonferroni 補正 α=MAX_ALPHA を通過したトークンのみ保持する。
    """
    if not LABELED_CLAUSES_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {LABELED_CLAUSES_PATH}")

    df = pd.read_csv(LABELED_CLAUSES_PATH)
    profile_texts: list[str] = df[df["label"] == "PROFILE"]["clause"].tolist()
    none_texts: list[str]    = df[df["label"] == "NONE"]["clause"].tolist()
    print(f"PROFILE clauses : {len(profile_texts)}")
    print(f"NONE clauses    : {len(none_texts)}")

    print("Counting PROFILE tokens ...")
    freq_profile, pos_map = count_tokens_in_clauses(profile_texts, workers, batch_size)
    print("Counting NONE tokens ...")
    freq_none, pos_map_none = count_tokens_in_clauses(none_texts, workers, batch_size)
    pos_map.update(pos_map_none)

    n_profile = len(profile_texts)
    n_none    = len(none_texts)
    all_tokens: list[str] = sorted(set(freq_profile.keys()) | set(freq_none.keys()))
    n_tokens = len(all_tokens)
    corrected_max = MAX_ALPHA / max(n_tokens, 1)

    print(f"Running Fisher's exact test on {n_tokens} tokens ...")
    results = parallel_map(
        _ma_fisher_batch, chunked(all_tokens, batch_size), workers,
        desc="Fisher's exact test", init_fn=_ma_fisher_init,
        init_args=(dict(freq_profile), dict(freq_none), n_profile, n_none,
                   pos_map, corrected_max, n_tokens),
    )
    rows: list[dict] = [r for batch in results for r in batch]

    df_stats = (
        pd.DataFrame(rows)
        .sort_values("effect_size", ascending=False)
        .reset_index(drop=True)
    )
    df_stats.to_csv(A_STATS_PATH, index=False)
    print(f"Stats cache saved ({len(df_stats)} tokens, α≤{MAX_ALPHA}) → {A_STATS_PATH}")
    return df_stats


def filter_at_alpha(df_stats: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """指定 alpha で Bonferroni 補正を適用し候補を返す。"""
    n_tokens = int(df_stats["n_tokens"].iloc[0])
    corrected = alpha / max(n_tokens, 1)
    df_pass = df_stats[df_stats["p_value"] < corrected]
    if df_pass.empty:
        print(f"  No tokens pass Bonferroni-corrected α={corrected:.2e}. "
              f"Falling back to uncorrected α={alpha}.")
        df_pass = df_stats[df_stats["p_value"] < alpha]
    return df_pass.sort_values("effect_size", ascending=False).reset_index(drop=True)


def load_or_build_stats(
    force_build: bool = False,
    workers: int = 1,
    batch_size: int = 256,
) -> pd.DataFrame:
    if not force_build and A_STATS_PATH.exists():
        df_stats = pd.read_csv(A_STATS_PATH)
        print(f"Loaded stats cache: {len(df_stats)} tokens → {A_STATS_PATH}")
        return df_stats
    return build_stats_cache(workers=workers, batch_size=batch_size)


def compute_method_a(
    alpha: float = ALPHA,
    force_build: bool = False,
    workers: int = 1,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Method A を実行し、候補トークン DataFrame を返す(候補 CSV も保存)。"""
    download_nltk_data(with_stopwords=True)
    df_stats = load_or_build_stats(force_build=force_build, workers=workers, batch_size=batch_size)
    df_out = add_alpha_tier_column(filter_at_alpha(df_stats, alpha))
    df_out.to_csv(A_CANDIDATES_PATH, index=False)
    print(f"Method A candidates (α={alpha}): {len(df_out)} → {A_CANDIDATES_PATH}")
    return df_out


# ── サブコマンド: method-a ────────────────────────────────────────────────────

def cmd_method_a(args: argparse.Namespace) -> None:
    download_nltk_data(with_stopwords=True)

    if args.filter or args.compare:
        df_stats = load_or_build_stats(
            force_build=False, workers=args.workers, batch_size=args.batch_size)
    else:
        print("Building stats cache (α≤0.10) ...")
        df_stats = build_stats_cache(workers=args.workers, batch_size=args.batch_size)

    if args.filter:
        alpha = args.filter
        df_out = add_alpha_tier_column(filter_at_alpha(df_stats, alpha))
        label = str(alpha).replace(".", "")
        out_path = DATA_DIR / f"03_method_a_alpha{label}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\nα={alpha} → {len(df_out)} candidates")
        print(df_out.head(20).to_string(index=False))
        print(f"\nSaved → {out_path}")
        return

    if args.compare:
        print(f"\n{'═'*60}")
        print("  Alpha comparison  (no recomputation – using saved stats)")
        print(f"{'═'*60}")
        base_n = len(filter_at_alpha(df_stats, 0.05))
        rows_cmp: list[dict] = []
        sets: dict[float, set[str]] = {}
        for alpha in COMPARE_ALPHAS:
            df_pass = filter_at_alpha(df_stats, alpha)
            sets[alpha] = set(df_pass["token"])
            diff = len(df_pass) - base_n
            rows_cmp.append({
                "alpha":      f"α={alpha}",
                "candidates": len(df_pass),
                "vs α=0.05":  f"{diff:+d}",
            })
        print(pd.DataFrame(rows_cmp).to_string(index=False))
        print(f"\n  Tokens in α=0.01 only : {len(sets[0.01])}")
        print(f"  Added at α=0.05       : {len(sets[0.05] - sets[0.01])}  "
              f"→ {sorted(sets[0.05] - sets[0.01])}")
        print(f"  Added at α=0.10       : {len(sets[0.10] - sets[0.05])}  "
              f"→ {sorted(sets[0.10] - sets[0.05])}")
        return

    alpha = args.alpha
    df_out = add_alpha_tier_column(filter_at_alpha(df_stats, alpha))
    df_out.to_csv(A_CANDIDATES_PATH, index=False)
    print(f"\nCandidates found (α={alpha}) : {len(df_out)}")
    print(df_out.head(20).to_string(index=False))
    print(f"\nSaved → {A_CANDIDATES_PATH}")


# ════════════════════════════════════════════════════════════════════════════
# Method B: 候補トークン → 最長名詞句
# ════════════════════════════════════════════════════════════════════════════

# ── 並列: NP 抽出ワーカー ─────────────────────────────────────────────────────
_MB_PARSER = None
_MB_TOKENS: set[str] = set()


def _mb_init(candidate_tokens: list[str]) -> None:
    global _MB_PARSER, _MB_TOKENS
    download_nltk_data()
    _MB_PARSER = make_np_parser()
    _MB_TOKENS = set(candidate_tokens)


def _mb_batch(
    clauses: list[str],
) -> tuple[Counter, dict[str, str], dict[str, list[str]]]:
    """節バッチを処理し (phrase 頻度, phrase→head, phrase→例文<=3) を返す。"""
    freq: Counter[str] = Counter()
    head: dict[str, str] = {}
    examples: dict[str, list[str]] = defaultdict(list)
    for clause in clauses:
        nps = extract_nps(clause, _MB_PARSER)
        clause_lower = clause.lower()
        matched_tokens = [t for t in _MB_TOKENS if t in clause_lower]
        if not matched_tokens:
            continue
        for token in matched_tokens:
            result = find_longest_np_containing(token, nps)
            if result is None:
                longest, h = token, token
            else:
                longest, h = result
            freq[longest] += 1
            head[longest] = h
            if len(examples[longest]) < 3:
                examples[longest].append(clause)
    return freq, head, dict(examples)


def run_method_b(
    candidate_tokens: set[str],
    token_tier: Optional[dict[str, str]] = None,
    workers: int = 1,
    batch_size: int = 256,
) -> pd.DataFrame:
    """候補トークンを PROFILE 節中の最長 NP に展開し、フレーズ CSV を保存。"""
    token_tier = token_tier or {}
    download_nltk_data()
    if not LABELED_CLAUSES_PATH.exists():
        raise FileNotFoundError(f"Run labeling first: {LABELED_CLAUSES_PATH}")

    print(f"Candidate tokens (Method A): {len(candidate_tokens)}")
    df_clauses = pd.read_csv(LABELED_CLAUSES_PATH)
    profile_clauses: list[str] = (
        df_clauses[df_clauses["label"] == "PROFILE"]["clause"].tolist()
    )
    print(f"PROFILE clauses to scan: {len(profile_clauses)}")

    results = parallel_map(
        _mb_batch, chunked(profile_clauses, batch_size), workers,
        desc="Extracting NPs", init_fn=_mb_init,
        init_args=(sorted(candidate_tokens),),
    )

    phrase_freq: Counter[str] = Counter()
    phrase_examples: defaultdict[str, list[str]] = defaultdict(list)
    phrase_head: dict[str, str] = {}
    for freq, head, examples in results:
        phrase_freq.update(freq)
        phrase_head.update(head)
        for ph, ex in examples.items():
            room = 3 - len(phrase_examples[ph])
            if room > 0:
                phrase_examples[ph].extend(ex[:room])

    rows: list[dict] = []
    for phrase, freq in phrase_freq.most_common():
        if len(phrase.split()) < 2:  # 1語は Method A でカバー済み
            continue
        examples = phrase_examples[phrase]
        tier = phrase_alpha_tier(phrase, token_tier)
        source_tokens = [w for w in phrase.split() if w in token_tier]
        rows.append({
            "phrase":          phrase,
            "head_token":      phrase_head.get(phrase, ""),
            "alpha_tier":      tier or "",
            "source_tokens":   " ".join(source_tokens),
            "freq":            freq,
            "example_clause_1": examples[0] if len(examples) > 0 else "",
            "example_clause_2": examples[1] if len(examples) > 1 else "",
            "example_clause_3": examples[2] if len(examples) > 2 else "",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(B_CANDIDATES_PATH, index=False)
    print(f"\nUnique phrases : {len(df_out)}")
    if not df_out.empty:
        print(df_out.head(20)[["phrase", "head_token", "alpha_tier", "freq"]].to_string(index=False))
    print(f"\nSaved → {B_CANDIDATES_PATH}")
    return df_out


def load_candidates_with_tier(
    candidates_csv: Path,
    min_freq_profile: int = 1,
    min_effect_size: float = 1.0,
) -> tuple[set[str], dict[str, str]]:
    """03_method_a_candidates.csv から (候補トークン集合, token→alpha_tier マップ) を読む。

    CSV に alpha_tier 列が無い場合は p_value / n_tokens から復元する。
    """
    if not candidates_csv.exists():
        raise FileNotFoundError(
            f"Method A の候補が見つかりません: {candidates_csv}\n"
            f"先に `python qpii_pipeline.py method-a` を実行してください。"
        )
    tokens: set[str] = set()
    token_tier: dict[str, str] = {}
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
            if freq < min_freq_profile or effect < min_effect_size:
                continue
            tokens.add(token)

            tier = str(row.get("alpha_tier", "") or "").strip()
            if not tier:  # alpha_tier 列が無い古い CSV からの復元
                try:
                    tier = compute_alpha_tier(
                        float(row["p_value"]), int(float(row["n_tokens"]))
                    ) or ""
                except (KeyError, ValueError, TypeError):
                    tier = ""
            if tier:
                token_tier[token] = tier
    return tokens, token_tier


# ── サブコマンド: method-b ────────────────────────────────────────────────────

def cmd_method_b(args: argparse.Namespace) -> None:
    candidate_tokens, token_tier = load_candidates_with_tier(A_CANDIDATES_PATH)
    run_method_b(candidate_tokens, token_tier,
                 workers=args.workers, batch_size=args.batch_size)


# ════════════════════════════════════════════════════════════════════════════
# レコード内共起集計(Method A → Method B をレコード本文へオンライン適用)
# ════════════════════════════════════════════════════════════════════════════

_WORKER_PARSER = None
_WORKER_TOKENS: set[str] = set()
_WORKER_DROP_SINGLE_WORD: bool = False


def _init_worker(candidate_tokens: list[str], drop_single_word: bool) -> None:
    global _WORKER_PARSER, _WORKER_TOKENS, _WORKER_DROP_SINGLE_WORD
    import nltk as _nltk  # spawn 後の各ワーカーで読み込み

    for resource in ("punkt_tab", "averaged_perceptron_tagger_eng"):
        _nltk.download(resource, quiet=True)
    _WORKER_PARSER = _nltk.RegexpParser(NP_GRAMMAR)
    _WORKER_TOKENS = set(candidate_tokens)
    _WORKER_DROP_SINGLE_WORD = drop_single_word


def extract_record_terms(texts: list[str]) -> list[str]:
    """レコードの全テキストへ Method A → Method B を順に適用し、表現集合を返す。

    Method A : 候補トークン(_WORKER_TOKENS)が本文に出現するか判定。
    Method B : 出現した各トークンを、それを含む最長 NP に展開
               (該当 NP が無ければトークン自身を採用 = 04 main と同じ挙動)。
    """
    phrases: set[str] = set()
    for text in texts:
        if not text:
            continue
        nps = extract_nps(text, _WORKER_PARSER)
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
    texts = [item["text"] for item in task["texts"]]
    return {
        "row_index": task["row_index"],
        "uuid": task["uuid"],
        "matched_terms": extract_record_terms(texts),
    }


def _process_row_batch_task(tasks: list[dict]) -> tuple[list[dict], int]:
    out = [_process_row_task(task) for task in tasks]
    return out, len(tasks)


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
        yield {"row_index": i, "uuid": row.get("uuid"), "texts": texts}


def iter_row_task_batches(row_tasks: Iterable[dict], batch_size: int) -> Iterable[list[dict]]:
    batch: list[dict] = []
    for task in row_tasks:
        batch.append(task)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
    token_tier: Optional[dict[str, str]] = None,
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

    token_tier = token_tier or {}
    with output_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow([
            "terms_json", "tiers_json", "set_size", "record_count", "probability",
            "n_tier_001", "n_tier_005", "n_tier_010",
        ])
        for term_set_key, cnt in conn.execute(
            "SELECT term_set_key, record_count FROM set_counts "
            "ORDER BY record_count DESC, term_set_key ASC"
        ):
            try:
                terms = json.loads(term_set_key)
            except json.JSONDecodeError:
                terms = []
            tiers = [phrase_alpha_tier(t, token_tier) for t in terms]
            tiers_json = json.dumps([t or "" for t in tiers], ensure_ascii=False)
            n001 = sum(1 for t in tiers if t == "0.01")
            n005 = sum(1 for t in tiers if t == "0.05")
            n010 = sum(1 for t in tiers if t == "0.10")
            p = (cnt / total_records) if total_records > 0 else 0.0
            writer.writerow([
                term_set_key, tiers_json, len(terms), cnt, p, n001, n005, n010,
            ])
    return set_types


def run_cooccurrence(
    args: argparse.Namespace,
    candidate_tokens: set[str],
    token_tier: Optional[dict[str, str]] = None,
) -> None:
    if not candidate_tokens:
        raise RuntimeError("候補トークンが0件です。閾値や Method A の結果を見直してください。")

    download_nltk_data()

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
    row_task_batches = iter_row_task_batches(row_tasks, batch_size=max(1, args.row_batch_size))

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
        conn, total_records=total_records, output_path=args.term_output,
        token_tier=token_tier,
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


# ── サブコマンド: cooccurrence ────────────────────────────────────────────────

def cmd_cooccurrence(args: argparse.Namespace) -> None:
    candidate_tokens, token_tier = load_candidates_with_tier(
        args.candidates_csv,
        min_freq_profile=args.min_freq_profile,
        min_effect_size=args.min_effect_size,
    )
    run_cooccurrence(args, candidate_tokens, token_tier)


# ── サブコマンド: all(全ステージを一気通貫) ─────────────────────────────────

def cmd_all(args: argparse.Namespace) -> None:
    print("════ Stage 1/3: Method A ════")
    df_a = compute_method_a(alpha=args.alpha, force_build=args.force_build,
                            workers=args.workers, batch_size=args.batch_size)

    if args.min_freq_profile > 1 or args.min_effect_size > 1.0:
        df_a = df_a[
            (df_a["freq_profile"] >= args.min_freq_profile)
            & (df_a["effect_size"] >= args.min_effect_size)
        ]
        print(f"  filtered candidate tokens: {len(df_a)}")

    candidate_tokens: set[str] = set(df_a["token"].str.lower().tolist())
    token_tier = build_token_tier_map(df_a)

    print("\n════ Stage 2/3: Method B ════")
    run_method_b(candidate_tokens, token_tier,
                 workers=args.workers, batch_size=args.batch_size)

    print("\n════ Stage 3/3: Record co-occurrence ════")
    run_cooccurrence(args, candidate_tokens, token_tier)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _add_ab_parallel_args(p: argparse.ArgumentParser) -> None:
    """Method A / Method B 用の並列化オプション(節・トークンのバッチ並列)。"""
    p.add_argument("--workers", type=int, default=0, help="0以下でCPUコア数")
    p.add_argument("--batch-size", type=int, default=256,
                   help="ワーカーへ渡す節/トークンのバッチサイズ")


def _add_cooccurrence_args(p: argparse.ArgumentParser, with_candidates_csv: bool) -> None:
    if with_candidates_csv:
        p.add_argument("--candidates-csv", type=Path, default=A_CANDIDATES_PATH,
                       help="Method A の候補トークン CSV")
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
    p.add_argument("--row-batch-size", type=int, default=128, help="ワーカーへ渡す行バッチサイズ")
    p.add_argument("--pool-chunksize", type=int, default=64, help="imap_unordered の chunksize")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--text-output", type=Path, default=COOC_TEXT_OUTPUT)
    p.add_argument("--term-output", type=Path, default=COOC_TERM_OUTPUT)
    p.add_argument("--sqlite-path", type=Path, default=COOC_SQLITE_PATH,
                   help="セット集計に使うSQLite DBパス")
    p.add_argument("--sqlite-insert-buffer", type=int, default=20000,
                   help="SQLiteへ一括INSERTするバッファ行数")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QPII 抽出パイプライン(Method A → Method B → レコード内共起集計)"
    )
    sub = parser.add_subparsers(dest="command")

    # method-a
    pa = sub.add_parser("method-a", help="形態素ベースの統計抽出で候補トークンを得る")
    pa.add_argument("--alpha", type=float, default=ALPHA,
                    help="候補トークンの有意水準(既定 config.ALPHA)。"
                         "3階層すべてを得たい場合は 0.10 を指定。")
    pa.add_argument("--filter", type=float, metavar="ALPHA",
                    help="保存済み統計から指定 alpha で再フィルタ(再計算なし)")
    pa.add_argument("--compare", action="store_true",
                    help="α=0.01/0.05/0.10 の比較表を出力(保存済み統計を使用)")
    _add_ab_parallel_args(pa)
    pa.set_defaults(func=cmd_method_a)

    # method-b
    pb = sub.add_parser("method-b", help="候補トークンを最長 NP へ展開して候補フレーズを得る")
    _add_ab_parallel_args(pb)
    pb.set_defaults(func=cmd_method_b)

    # cooccurrence
    pc = sub.add_parser("cooccurrence", help="レコード単位で共起セットを集計する")
    _add_cooccurrence_args(pc, with_candidates_csv=True)
    pc.set_defaults(func=cmd_cooccurrence)

    # all
    pall = sub.add_parser("all", help="method-a → method-b → cooccurrence を一気通貫で実行")
    pall.add_argument("--alpha", type=float, default=ALPHA, help="Method A の有意水準")
    pall.add_argument("--force-build", action="store_true",
                      help="統計キャッシュを無視して Method A を再計算する")
    pall.add_argument("--batch-size", type=int, default=256,
                      help="Method A/B のワーカーへ渡す節/トークンのバッチサイズ")
    _add_cooccurrence_args(pall, with_candidates_csv=False)
    pall.set_defaults(func=cmd_all)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # 検証(cooccurrence / all のみ)
    if args.command in {"cooccurrence", "all"}:
        if not (0.0 < args.sample_ratio <= 1.0):
            raise ValueError("--sample-ratio は 0.0 より大きく 1.0 以下で指定してください。")
        if args.sqlite_insert_buffer <= 0:
            raise ValueError("--sqlite-insert-buffer は1以上で指定してください。")
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
