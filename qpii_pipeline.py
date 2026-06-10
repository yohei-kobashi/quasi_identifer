#!/usr/bin/env python3
"""QPII 抽出パイプライン(Method A → Method B → レコード内共起集計)を1本に統合。

3つのステージを1スクリプトにまとめたもの。

  method-a       形態素ベースの統計抽出(Fisher の正確確率検定)で候補トークンを得る。
                 旧 03_method_a_extract.py 相当。
  method-b       候補トークンを最長名詞句(NP)へ展開して候補フレーズを得る。
                 旧 04_method_b_phrase.py 相当。
  cooccurrence   Method A → Method B を順に適用して対象表現を特定し、annotations_pred_usa.jsonl
                 (予測アノテーション)の「テキスト(フィールド)単位」で共起セットを集計する。
                 同一(record, field)の clause をまとめて1テキストとして扱う。旧 05 相当。
  benchmark      先頭 N 件(既定1万)で試走し、目標件数(既定100万)の所要時間・
                 延べ/ユニーク phrase 数・共起セット数を外挿推定する(出力は書かない)。
  all            method-a → method-b → cooccurrence を一気通貫で実行する。
                 候補トークンはメモリ上で受け渡すため CSV 経由のやり取りは不要。

入力
----
  annotations_pred_usa.jsonl    (annotate_with_deberta_usa.py の出力; clause/label を含む。
                                 label ∈ {PROFILE, NONE, PII}。Method A/B は PROFILE と NONE を使う)

出力
----
  data/03_method_a_stats.csv          Method A 全統計(α≤0.10)
  data/03_method_a_candidates.csv     Method A 候補トークン(+ alpha_tier 列)
  data/04_method_b_candidates.csv     Method B 候補フレーズ(+ alpha_tier/source_tokens 列)
  data/05_clause_matched_terms.jsonl    clause別の対象表現明細(row_index/uuid/field 付き)
  data/05_uuid_set_probabilities.csv    共起セット出現確率(uuid単位; uuid_count + tiers/n_tier_*)
  data/05_field_set_probabilities.csv   共起セット出現確率(field単位; field_count + ...)
  data/05_clause_set_probabilities.csv  共起セット出現確率(clause単位; clause_count + ...)
  data/05_<unit>_pairs.csv              phrase ペア共起(co_count/count_a/count_b/probability/pmi/tier)
  (--unit で出力単位を選択可。既定は3つすべて。--no-pairs でペア出力を無効)

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
  python qpii_pipeline.py cooccurrence              # annotations_pred_usa.jsonl を全件集計
  python qpii_pipeline.py cooccurrence --input my_preds.jsonl --sample-ratio 0.1
  python qpii_pipeline.py benchmark --test-rows 10000 --target-rows 1000000
  python qpii_pipeline.py all                        # 全ステージ
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional

import nltk
import pandas as pd
from scipy.stats import fisher_exact
from tqdm import tqdm

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    sys.path.insert(0, str(Path.cwd()))
from config import ALPHA, DATA_DIR, KEEP_POS_PREFIXES, MIN_PHRASE_FREQ, TEXT_FIELDS

# ── パス ────────────────────────────────────────────────────────────────────
# 入力は annotate_with_deberta_usa.py の予測アノテーション(JSONL, clause/label を含む)。
# .csv を渡せば従来の data/02_labeled_clauses.csv 形式も読める(read_labeled_clauses 参照)。
LABELED_CLAUSES_PATH: Path = DATA_DIR.parent / "annotations_pred_usa.jsonl"

A_STATS_PATH: Path        = DATA_DIR / "03_method_a_stats.csv"
A_CANDIDATES_PATH: Path   = DATA_DIR / "03_method_a_candidates.csv"
B_CANDIDATES_PATH: Path   = DATA_DIR / "04_method_b_candidates.csv"
COOC_OUT_PREFIX: Path     = DATA_DIR / "05"   # <prefix>_<unit>_set_probabilities.csv 等
COOC_UNITS: tuple[str, ...] = ("uuid", "field", "clause")  # 共起セットの集計単位

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
    # wordnet/omw は phrase 正規化(lemma化)に使う
    resources = ["punkt_tab", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"]
    if with_stopwords:
        resources.append("stopwords")
    for resource in resources:
        nltk.download(resource, quiet=True)


# ── phrase 正規化(案2) ───────────────────────────────────────────────────────
_DETERMINERS: frozenset[str] = frozenset({"a", "an", "the"})
_LEMMA_CACHE: dict[str, str] = {}


def normalize_phrase(phrase: str, lemmatizer) -> str:
    """冠詞/限定詞を除去し、各語を名詞 lemma 化して小文字で返す(表記ゆれ統合)。

    例: 'a registered nurses' → 'registered nurse', 'weekend getaways' → 'weekend getaway'。
    lemmatizer=None なら lemma 化せず冠詞除去+小文字のみ。
    """
    out: list[str] = []
    for w in phrase.lower().split():
        if w in _DETERMINERS:
            continue
        if lemmatizer is not None:
            lw = _LEMMA_CACHE.get(w)
            if lw is None:
                lw = lemmatizer.lemmatize(w)
                _LEMMA_CACHE[w] = lw
            w = lw
        out.append(w)
    return " ".join(out)


def augment_token_tier_with_lemmas(token_tier: dict[str, str]) -> dict[str, str]:
    """token→tier に lemma キーも足す(正規化 phrase の tier 照合用)。

    'gatherings'→tier に加えて 'gathering'→tier も引けるようにする。
    同一 lemma に複数 tier が来たら最も厳しいものを採用。
    """
    if not token_tier:
        return dict(token_tier)
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    out: dict[str, str] = dict(token_tier)
    for tok, tier in token_tier.items():
        lt = lem.lemmatize(tok)
        if lt == tok:
            continue
        prev = out.get(lt)
        out[lt] = tier if prev is None else (strictest_tier([prev, tier]) or tier)
    return out


def load_allowed_phrases(method_b_csv: Path, min_phrase_freq: int) -> Optional[set[str]]:
    """Method B 辞書(04 CSV)を freq>=K で絞り、許可フレーズ集合を返す(案1)。

    min_phrase_freq<=0 なら None(フィルタ無効=自由NP)を返す。
    """
    if min_phrase_freq <= 0:
        return None
    if not method_b_csv.exists():
        raise FileNotFoundError(
            f"頻度フィルタ(--min-phrase-freq {min_phrase_freq})に必要な Method B 辞書が"
            f"見つかりません: {method_b_csv}\n先に `qpii_pipeline.py method-b` を実行してください。"
        )
    df = pd.read_csv(method_b_csv)
    if "phrase" not in df.columns or "freq" not in df.columns:
        raise ValueError(f"{method_b_csv} に phrase/freq 列が必要です")
    return set(df[df["freq"] >= min_phrase_freq]["phrase"].astype(str))


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


def read_labeled_clauses(path: Path) -> pd.DataFrame:
    """clause/label を持つ入力を DataFrame で返す。

    .jsonl/.ndjson は逐次パースで clause/label のみ抽出(巨大ファイルでも省メモリ)。
    .csv は従来どおり pandas で読む(旧 data/02_labeled_clauses.csv 互換)。
    """
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        clauses: list[str] = []
        labels: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {path.name}", leave=False):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                clause = rec.get("clause")
                label = rec.get("label")
                if clause is None or label is None:
                    continue
                clauses.append(str(clause))
                labels.append(str(label))
        df = pd.DataFrame({"clause": clauses, "label": labels})
    else:
        df = pd.read_csv(path)
    if "clause" not in df.columns or "label" not in df.columns:
        raise ValueError(f"入力に 'clause' と 'label' 列が必要です: {path}")
    return df


def build_stats_cache(workers: int = 1, batch_size: int = 256) -> pd.DataFrame:
    """トークン化 + Fisher 検定を並列実行し、全統計を A_STATS_PATH へ保存。

    Bonferroni 補正 α=MAX_ALPHA を通過したトークンのみ保持する。
    """
    if not LABELED_CLAUSES_PATH.exists():
        raise FileNotFoundError(
            f"入力が見つかりません: {LABELED_CLAUSES_PATH}\n"
            f"先に annotate_with_deberta_usa.py で予測アノテーションを作成してください。"
        )

    df = read_labeled_clauses(LABELED_CLAUSES_PATH)
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
_MB_LEMMATIZER = None


def _mb_init(candidate_tokens: list[str]) -> None:
    global _MB_PARSER, _MB_TOKENS, _MB_LEMMATIZER
    import nltk as _nltk
    download_nltk_data()
    _MB_PARSER = make_np_parser()
    _MB_TOKENS = set(candidate_tokens)
    _MB_LEMMATIZER = _nltk.stem.WordNetLemmatizer()


def _mb_batch(
    clauses: list[str],
) -> tuple[Counter, dict[str, str], dict[str, list[str]]]:
    """節バッチを処理し (phrase 頻度, phrase→head, phrase→例文<=3) を返す。

    候補トークンは単語境界一致(案4)、phrase は正規化(案2)してから集計する。
    """
    freq: Counter[str] = Counter()
    head: dict[str, str] = {}
    examples: dict[str, list[str]] = defaultdict(list)
    for clause in clauses:
        nps = extract_nps(clause, _MB_PARSER)
        words = set(re.findall(r"[a-z]+", clause.lower()))   # 案4: 単語境界
        matched_tokens = _MB_TOKENS & words
        if not matched_tokens:
            continue
        for token in matched_tokens:
            result = find_longest_np_containing(token, nps)
            raw, raw_head = (token, token) if result is None else result
            phrase = normalize_phrase(raw, _MB_LEMMATIZER)   # 案2: 正規化
            if not phrase:
                continue
            freq[phrase] += 1
            head[phrase] = normalize_phrase(raw_head, _MB_LEMMATIZER)
            if len(examples[phrase]) < 3:
                examples[phrase].append(clause)
    return freq, head, dict(examples)


def run_method_b(
    candidate_tokens: set[str],
    token_tier: Optional[dict[str, str]] = None,
    workers: int = 1,
    batch_size: int = 256,
) -> pd.DataFrame:
    """候補トークンを PROFILE 節中の最長 NP に展開し、フレーズ CSV を保存。"""
    download_nltk_data()
    token_tier = augment_token_tier_with_lemmas(token_tier or {})  # 正規化phraseの tier 照合用
    if not LABELED_CLAUSES_PATH.exists():
        raise FileNotFoundError(
            f"入力が見つかりません: {LABELED_CLAUSES_PATH}\n"
            f"先に annotate_with_deberta_usa.py で予測アノテーションを作成してください。"
        )

    print(f"Candidate tokens (Method A): {len(candidate_tokens)}")
    df_clauses = read_labeled_clauses(LABELED_CLAUSES_PATH)
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
# テキスト内共起集計(Method A → Method B を各テキストフィールドへオンライン適用)
# ════════════════════════════════════════════════════════════════════════════

_WORKER_PARSER = None
_WORKER_TOKENS: set[str] = set()
_WORKER_DROP_SINGLE_WORD: bool = False
_WORKER_ALLOWED: Optional[set[str]] = None   # 案1: 許可フレーズ(None=フィルタ無効)
_WORKER_LEMMATIZER = None


def _init_worker(
    candidate_tokens: list[str],
    drop_single_word: bool,
    allowed_phrases: Optional[list[str]] = None,
) -> None:
    global _WORKER_PARSER, _WORKER_TOKENS, _WORKER_DROP_SINGLE_WORD
    global _WORKER_ALLOWED, _WORKER_LEMMATIZER
    import nltk as _nltk  # spawn 後の各ワーカーで読み込み

    for resource in ("punkt_tab", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"):
        _nltk.download(resource, quiet=True)
    _WORKER_PARSER = _nltk.RegexpParser(NP_GRAMMAR)
    _WORKER_TOKENS = set(candidate_tokens)
    _WORKER_DROP_SINGLE_WORD = drop_single_word
    _WORKER_ALLOWED = set(allowed_phrases) if allowed_phrases is not None else None
    _WORKER_LEMMATIZER = _nltk.stem.WordNetLemmatizer()


def extract_text_terms(text: str) -> list[str]:
    """1テキスト(=1フィールド値/clause)へ Method A → Method B を適用し、表現集合を返す。

    案4: 候補トークン一致は単語境界。案2: 最長 NP を正規化。
    案1: 許可フレーズ(_WORKER_ALLOWED)が設定されていればそれに含まれる phrase のみ採用。
    """
    if not text:
        return []
    phrases: set[str] = set()
    nps = extract_nps(text, _WORKER_PARSER)
    words = set(re.findall(r"[a-z]+", text.lower()))   # 案4: 単語境界
    matched_tokens = _WORKER_TOKENS & words
    for token in matched_tokens:
        result = find_longest_np_containing(token, nps)
        raw = token if result is None else result[0]
        phrase = normalize_phrase(raw, _WORKER_LEMMATIZER)   # 案2: 正規化
        if not phrase:
            continue
        if _WORKER_DROP_SINGLE_WORD and len(phrase.split()) < 2:
            continue
        if _WORKER_ALLOWED is not None and phrase not in _WORKER_ALLOWED:   # 案1: 頻度フィルタ
            continue
        phrases.add(phrase)
    return sorted(phrases)


def _process_record_task(task: dict) -> dict:
    """1レコードを処理し、各 clause の (field, 表現集合) を返す。

    抽出は clause 単位で行い、field/uuid 単位はその和集合として後段で集計する
    (3粒度で一貫させるため)。
    """
    clauses_terms = [
        (field, extract_text_terms(clause)) for field, clause in task["items"]
    ]
    return {"row_index": task["row_index"], "uuid": task["uuid"], "clauses": clauses_terms}


def _process_record_batch(tasks: list[dict]) -> tuple[list[dict], int]:
    out = [_process_record_task(t) for t in tasks]
    return out, len(tasks)  # 第2要素は処理した「レコード数」


def record_unit_sets(record_result: dict, units: list[str]) -> dict[str, list[list[str]]]:
    """worker 出力から、指定単位ごとの term set(sorted unique)のリストを返す。

    clause: 各 clause が1セット / field: 同一 field の clause を和集合して1セット /
    uuid: レコード全体(全 field・全 clause)を和集合して1セット。
    """
    clauses = record_result["clauses"]  # [(field, terms), ...]
    out: dict[str, list[list[str]]] = {}
    if "clause" in units:
        out["clause"] = [sorted(set(terms)) for _f, terms in clauses]
    if "field" in units:
        acc: dict[str, set] = {}
        for field, terms in clauses:
            acc.setdefault(field, set()).update(terms)
        out["field"] = [sorted(s) for s in acc.values()]
    if "uuid" in units:
        u: set = set()
        for _f, terms in clauses:
            u.update(terms)
        out["uuid"] = [sorted(u)]
    return out


_NO_UUID = object()  # iter_jsonl_record_tasks の番兵


def iter_jsonl_record_tasks(
    path: Path,
    sample_ratio: float,
    seed: Optional[int],
    max_rows: int,
    requested_fields: list[str],
) -> Iterable[dict]:
    """annotations_pred_usa.jsonl(clause 単位)を読み、レコード(uuid)単位でまとめて

    {row_index, uuid, texts:[{field, text}]} を生成する。
    text は同一(record, field)の clause を ' . ' で連結したもの(=テキスト/フィールド単位)。
    入力は同一レコードが連続して並んでいる前提(annotate_with_deberta_usa.py の出力順)。
    sample_ratio / max_rows はレコード(uuid)単位で適用する。
    """
    rng = random.Random(seed)
    field_order = list(requested_fields)
    field_set = set(requested_fields)

    cur_uuid: Any = _NO_UUID
    cur_row_index = None
    cur_keep = False
    cur_fields: dict[str, list[str]] = {}
    yielded = 0

    def build_task() -> dict:
        # clause 単位で抽出するため、joinせず (field, clause) のまま渡す。
        items = [
            (f, c) for f in field_order for c in cur_fields.get(f, [])
        ]
        return {"row_index": cur_row_index, "uuid": cur_uuid, "items": items}

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uuid = rec.get("uuid")
            if uuid != cur_uuid:
                # 直前レコードを確定
                if cur_uuid is not _NO_UUID and cur_keep:
                    task = build_task()
                    if task["items"]:
                        yield task
                        yielded += 1
                        if max_rows > 0 and yielded >= max_rows:
                            return
                # 新レコード開始(サンプリング判定は uuid 単位で1回)
                cur_uuid = uuid
                cur_row_index = rec.get("row_index")
                cur_fields = {}
                cur_keep = sample_ratio >= 1.0 or rng.random() <= sample_ratio
            if cur_keep:
                field = rec.get("field")
                clause = rec.get("clause")
                if field in field_set and clause:
                    cur_fields.setdefault(field, []).append(str(clause))
        # 最後のレコードを確定
        if cur_uuid is not _NO_UUID and cur_keep:
            task = build_task()
            if task["items"]:
                yield task


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
    total_units: int,
    output_path: Path,
    token_tier: Optional[dict[str, str]] = None,
    count_col: str = "text_count",
) -> int:
    conn.execute("DROP TABLE IF EXISTS set_counts;")
    conn.execute(
        """
        CREATE TABLE set_counts AS
        SELECT term_set_key, COUNT(*) AS cnt
        FROM raw_sets
        GROUP BY term_set_key
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_set_counts_count ON set_counts(cnt DESC);")
    conn.commit()

    cur = conn.execute("SELECT COUNT(*) FROM set_counts")
    set_types = int(cur.fetchone()[0])

    token_tier = token_tier or {}
    with output_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow([
            "terms_json", "tiers_json", "set_size", count_col, "probability",
            "n_tier_001", "n_tier_005", "n_tier_010",
        ])
        for term_set_key, cnt in conn.execute(
            "SELECT term_set_key, cnt FROM set_counts "
            "ORDER BY cnt DESC, term_set_key ASC"
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
            p = (cnt / total_units) if total_units > 0 else 0.0
            writer.writerow([
                term_set_key, tiers_json, len(terms), cnt, p, n001, n005, n010,
            ])
    return set_types


def export_pairs(
    conn: sqlite3.Connection,
    total_units: int,
    output_path: Path,
    min_pair_count: int,
    token_tier: Optional[dict[str, str]] = None,
) -> int:
    """phrase ペアの共起回数 + PMI を出力する(案B)。

    phrase_occ(p) と pair_occ(a,b) を GROUP BY して集計。co_count>=min_pair_count のみ出力。
    PMI = log2( co_count * N / (count_a * count_b) )、N=total_units。
    """
    token_tier = token_tier or {}
    # 単独 phrase 出現数(=その phrase を含む単位数)
    conn.execute("DROP TABLE IF EXISTS phrase_counts;")
    conn.execute("CREATE TABLE phrase_counts AS SELECT p, COUNT(*) AS c FROM phrase_occ GROUP BY p")
    conn.commit()
    phrase_count: dict[str, int] = {p: c for p, c in conn.execute("SELECT p, c FROM phrase_counts")}

    conn.execute("DROP TABLE IF EXISTS pair_counts;")
    conn.execute("CREATE TABLE pair_counts AS SELECT a, b, COUNT(*) AS c FROM pair_occ GROUP BY a, b")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pair_c ON pair_counts(c DESC);")
    conn.commit()

    n_pairs = 0
    with output_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow([
            "phrase_a", "phrase_b", "co_count", "count_a", "count_b",
            "probability", "pmi", "tier_a", "tier_b",
        ])
        for a, b, c in conn.execute(
            "SELECT a, b, c FROM pair_counts WHERE c >= ? ORDER BY c DESC, a ASC, b ASC",
            (min_pair_count,),
        ):
            ca = phrase_count.get(a, 0)
            cb = phrase_count.get(b, 0)
            prob = c / total_units if total_units > 0 else 0.0
            if total_units > 0 and ca > 0 and cb > 0:
                pmi = math.log2((c * total_units) / (ca * cb))
            else:
                pmi = 0.0
            writer.writerow([
                a, b, c, ca, cb, prob, round(pmi, 4),
                phrase_alpha_tier(a, token_tier) or "",
                phrase_alpha_tier(b, token_tier) or "",
            ])
            n_pairs += 1
    return n_pairs


def run_cooccurrence(
    args: argparse.Namespace,
    candidate_tokens: set[str],
    token_tier: Optional[dict[str, str]] = None,
    allowed_phrases: Optional[set[str]] = None,
) -> None:
    if not candidate_tokens:
        raise RuntimeError("候補トークンが0件です。閾値や Method A の結果を見直してください。")
    if not args.input.exists():
        raise FileNotFoundError(
            f"共起集計の入力が見つかりません: {args.input}\n"
            f"先に annotate_with_deberta_usa.py で予測アノテーションを作成してください。"
        )

    download_nltk_data()
    token_tier = augment_token_tier_with_lemmas(token_tier or {})  # 正規化phraseの tier 照合用
    allowed_list = sorted(allowed_phrases) if allowed_phrases is not None else None
    if allowed_phrases is not None:
        print(f"許可フレーズ(freq≥{args.min_phrase_freq}): {len(allowed_phrases)}")

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, workers)
    requested_fields = list(TEXT_FIELDS)
    units = list(args.unit)
    prefix = str(args.out_prefix)

    emit_pairs = args.pairs

    # 単位ごとに SQLite / バッファ / カウントを用意
    conns: dict[str, sqlite3.Connection] = {}
    buffers: dict[str, list[tuple[str]]] = {}
    phrase_buffers: dict[str, list[tuple[str]]] = {}
    pair_buffers: dict[str, list[tuple[str, str]]] = {}
    totals: dict[str, int] = {}
    for u in units:
        conns[u] = setup_sqlite(Path(f"{prefix}_{u}_sets.sqlite3"))
        if emit_pairs:  # 案B: ペア共起用テーブル
            conns[u].execute("CREATE TABLE phrase_occ (p TEXT NOT NULL);")
            conns[u].execute("CREATE TABLE pair_occ (a TEXT NOT NULL, b TEXT NOT NULL);")
            conns[u].commit()
        buffers[u] = []
        phrase_buffers[u] = []
        pair_buffers[u] = []
        totals[u] = 0

    def flush_unit(u: str) -> None:
        flush_set_buffer(conns[u], buffers[u])
        if emit_pairs:
            if phrase_buffers[u]:
                conns[u].executemany("INSERT INTO phrase_occ (p) VALUES (?)", phrase_buffers[u])
                phrase_buffers[u].clear()
            if pair_buffers[u]:
                conns[u].executemany("INSERT INTO pair_occ (a, b) VALUES (?, ?)", pair_buffers[u])
                pair_buffers[u].clear()
            conns[u].commit()

    detail_path = Path(f"{prefix}_clause_matched_terms.jsonl")
    sampled_rows = 0
    started = time.time()

    row_tasks = iter_jsonl_record_tasks(
        path=args.input,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        max_rows=args.max_rows,
        requested_fields=requested_fields,
    )
    row_task_batches = iter_row_task_batches(row_tasks, batch_size=max(1, args.row_batch_size))

    with detail_path.open("w", encoding="utf-8") as fw:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(sorted(candidate_tokens), args.drop_single_word, allowed_list),
        ) as pool:
            for row_result, rows_done in pool.imap_unordered(
                _process_record_batch,
                row_task_batches,
                chunksize=max(1, args.pool_chunksize),
            ):
                sampled_rows += rows_done
                for rec in row_result:
                    # clause 単位の明細を書き出し
                    for field, terms in rec["clauses"]:
                        fw.write(json.dumps({
                            "row_index": rec["row_index"], "uuid": rec["uuid"],
                            "field": field, "matched_terms": terms,
                        }, ensure_ascii=False) + "\n")
                    # 各単位のセットを SQLite へ(＋ペア共起)
                    sets = record_unit_sets(rec, units)
                    for u in units:
                        for s in sets[u]:
                            buffers[u].append((json.dumps(s, ensure_ascii=False),))
                            totals[u] += 1
                            if emit_pairs:
                                for p in s:
                                    phrase_buffers[u].append((p,))
                                for a, b in combinations(s, 2):  # s はソート済み
                                    pair_buffers[u].append((a, b))
                            if len(buffers[u]) >= args.sqlite_insert_buffer:
                                flush_unit(u)

                if args.progress_every > 0 and sampled_rows % args.progress_every == 0:
                    elapsed = max(1e-9, time.time() - started)
                    counts = " ".join(f"{u}={totals[u]}" for u in units)
                    print(f"[progress] records={sampled_rows} {counts} "
                          f"speed={sampled_rows / elapsed:.2f} rec/s")

    # 単位ごとに確率表 + ペア共起を出力
    results: dict[str, tuple] = {}
    for u in units:
        flush_unit(u)
        out_csv = Path(f"{prefix}_{u}_set_probabilities.csv")
        set_types = export_set_probabilities(
            conns[u], total_units=totals[u], output_path=out_csv,
            token_tier=token_tier, count_col=f"{u}_count",
        )
        pairs_path = None
        n_pairs = 0
        if emit_pairs:
            pairs_path = Path(f"{prefix}_{u}_pairs.csv")
            n_pairs = export_pairs(
                conns[u], total_units=totals[u], output_path=pairs_path,
                min_pair_count=args.min_pair_count, token_tier=token_tier,
            )
        conns[u].close()
        results[u] = (totals[u], set_types, out_csv, n_pairs, pairs_path)

    elapsed = time.time() - started
    print(f"\ncandidate_tokens={len(candidate_tokens)}  workers={workers}  "
          f"records={sampled_rows}  elapsed_sec={elapsed:.2f}")
    for u in units:
        total, set_types, out_csv, n_pairs, pairs_path = results[u]
        print(f"  [{u:6s}] sets={total}  unique_sets={set_types}  → {out_csv}")
        if emit_pairs:
            print(f"           pairs(co_count≥{args.min_pair_count})={n_pairs}  → {pairs_path}")
    print(f"  detail(clause) → {detail_path}")


# ── サブコマンド: cooccurrence ────────────────────────────────────────────────

def cmd_cooccurrence(args: argparse.Namespace) -> None:
    candidate_tokens, token_tier = load_candidates_with_tier(
        args.candidates_csv,
        min_freq_profile=args.min_freq_profile,
        min_effect_size=args.min_effect_size,
    )
    allowed = load_allowed_phrases(args.method_b_csv, args.min_phrase_freq)
    run_cooccurrence(args, candidate_tokens, token_tier, allowed)


# ════════════════════════════════════════════════════════════════════════════
# ベンチマーク / 推定モード
# ════════════════════════════════════════════════════════════════════════════

def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f} 秒"
    if seconds < 5400:
        return f"{seconds / 60:.1f} 分"
    return f"{seconds / 3600:.2f} 時間"


def _heaps_estimate(
    points: list[tuple[int, int]], target: int
) -> tuple[float, Optional[float]]:
    """points=[(n, value), ...] を Heaps 則 value=K*n^β で最小二乗フィットし、

    target における推定値と β を返す。有効点が2未満なら線形外挿(β=None)。
    """
    import numpy as np

    pts = [(n, v) for n, v in points if n > 0 and v > 0]
    if len(pts) < 2:
        n, v = points[-1]
        return (v / n * target if n else 0.0), None
    ns = np.array([p[0] for p in pts], dtype=float)
    vs = np.array([p[1] for p in pts], dtype=float)
    beta, log_k = np.polyfit(np.log(ns), np.log(vs), 1)
    return float(np.exp(log_k)) * (target ** beta), float(beta)


def run_benchmark(
    args: argparse.Namespace,
    candidate_tokens: set[str],
    allowed_phrases: Optional[set[str]] = None,
) -> None:
    if not candidate_tokens:
        raise RuntimeError("候補トークンが0件です。先に method-a を実行してください。")
    if not args.input.exists():
        raise FileNotFoundError(
            f"共起集計の入力が見つかりません: {args.input}\n"
            f"先に annotate_with_deberta_usa.py で予測アノテーションを作成してください。"
        )
    download_nltk_data()
    allowed_list = sorted(allowed_phrases) if allowed_phrases is not None else None
    workers = resolve_workers(args.workers)
    test_rows = max(1, args.test_rows)
    target_rows = max(1, args.target_rows)
    requested_fields = list(TEXT_FIELDS)
    units = list(args.unit)

    # Heaps 則フィット用のチェックポイント(レコード数)
    fracs = (0.1, 0.25, 0.5, 0.75, 1.0)
    pending = sorted({max(1, int(test_rows * f)) for f in fracs})

    row_tasks = iter_jsonl_record_tasks(
        path=args.input,
        sample_ratio=1.0,          # 先頭から連続 test_rows 件を処理
        seed=args.seed, max_rows=test_rows,
        requested_fields=requested_fields,
    )
    batches = iter_row_task_batches(row_tasks, batch_size=max(1, args.row_batch_size))

    emit_pairs = bool(args.pairs)
    min_pair_count = max(1, args.min_pair_count)

    seen_phrases: set[str] = set()
    seen_sets: dict[str, set] = {u: set() for u in units}     # 単位ごとのユニークセット
    unit_counts: dict[str, int] = {u: 0 for u in units}        # 単位ごとの延べセット数
    pair_counts: dict[str, Counter] = {u: Counter() for u in units}  # (a,b)->共起回数
    pair_occ_total: dict[str, int] = {u: 0 for u in units}     # 延べペア出現(=ΣC(|s|,2))
    total_occ = 0
    rows = 0
    phrase_snaps: list[tuple[int, int]] = []
    set_snaps: dict[str, list[tuple[int, int]]] = {u: [] for u in units}
    # ペアの Heaps 用スナップショット: ユニークペア数 / co_count>=K のペア数
    pair_uniq_snaps: dict[str, list[tuple[int, int]]] = {u: [] for u in units}
    pair_keep_snaps: dict[str, list[tuple[int, int]]] = {u: [] for u in units}
    t_first: Optional[float] = None

    print(f"[benchmark] test_rows={test_rows} workers={workers} units={units} input={args.input}")
    started = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers, initializer=_init_worker,
        initargs=(sorted(candidate_tokens), args.drop_single_word, allowed_list),
    ) as pool:
        for row_result, rows_done in pool.imap_unordered(
            _process_record_batch, batches, chunksize=max(1, args.pool_chunksize)
        ):
            if t_first is None:
                t_first = time.time() - started
            for rec in row_result:
                for _field, terms in rec["clauses"]:
                    total_occ += len(terms)
                    seen_phrases.update(terms)
                sets = record_unit_sets(rec, units)
                for u in units:
                    pc = pair_counts[u]
                    for s in sets[u]:
                        unit_counts[u] += 1
                        seen_sets[u].add(json.dumps(s, ensure_ascii=False))
                        if emit_pairs and len(s) >= 2:
                            for a, b in combinations(s, 2):  # s はソート済み
                                pc[(a, b)] += 1
                                pair_occ_total[u] += 1
            rows += rows_done
            if pending and rows >= pending[0]:
                while pending and rows >= pending[0]:
                    pending.pop(0)
                phrase_snaps.append((rows, len(seen_phrases)))
                for u in units:
                    set_snaps[u].append((rows, len(seen_sets[u])))
                    if emit_pairs:
                        pc = pair_counts[u]
                        keep = sum(1 for c in pc.values() if c >= min_pair_count)
                        pair_uniq_snaps[u].append((rows, len(pc)))
                        pair_keep_snaps[u].append((rows, keep))
    elapsed = time.time() - started

    if rows == 0:
        print("[benchmark] 処理レコードが0件でした。入力/フィールド設定を確認してください。")
        return

    if not phrase_snaps or phrase_snaps[-1][0] != rows:
        phrase_snaps.append((rows, len(seen_phrases)))
        for u in units:
            set_snaps[u].append((rows, len(seen_sets[u])))
            if emit_pairs:
                pc = pair_counts[u]
                keep = sum(1 for c in pc.values() if c >= min_pair_count)
                pair_uniq_snaps[u].append((rows, len(pc)))
                pair_keep_snaps[u].append((rows, keep))

    rate = rows / elapsed if elapsed > 0 else float("inf")
    est_time = target_rows / rate if rate > 0 else float("inf")
    est_occ = (total_occ / rows) * target_rows
    est_uphr, beta_p = _heaps_estimate(phrase_snaps, target_rows)
    bp = f"Heaps β={beta_p:.3f}" if beta_p is not None else "線形"
    warm = t_first if t_first is not None else 0.0

    print("\n" + "=" * 64)
    print(f"  ベンチマーク実測 ({rows:,} レコード)")
    print("=" * 64)
    print(f"  処理時間              : {_fmt_duration(elapsed)} ({elapsed:.1f}s)")
    print(f"   うちウォームアップ   : {warm:.1f}s (初回DL+spawn)")
    print(f"  全体スループット      : {rate:.1f} レコード/秒")
    print(f"  ユニーク phrase       : {len(seen_phrases):,}")
    print(f"  延べ phrase 出現      : {total_occ:,}")
    print(f"  単位別 セット数(延べ/ユニーク):")
    for u in units:
        print(f"    {u:6s}: {unit_counts[u]:,} / {len(seen_sets[u]):,}  "
              f"({unit_counts[u] / rows:.2f} セット/レコード)")
    if emit_pairs:
        print(f"  単位別 ペア数(延べ / ユニーク / co_count≥{min_pair_count}):")
        for u in units:
            pc = pair_counts[u]
            keep = sum(1 for c in pc.values() if c >= min_pair_count)
            print(f"    {u:6s}: {pair_occ_total[u]:,} / {len(pc):,} / {keep:,}")
    print("\n" + "-" * 64)
    print(f"  {target_rows:,} レコードへの外挿")
    print("-" * 64)
    print(f"  推定処理時間          : {_fmt_duration(est_time)}  (@{rate:.0f} rec/s, 線形)")
    print(f"  延べ phrase 出現      : 約 {est_occ:,.0f}  (線形)")
    print(f"  ユニーク phrase       : 約 {est_uphr:,.0f}  ({bp})")
    for u in units:
        est_sets, beta_u = _heaps_estimate(set_snaps[u], target_rows)
        bu = f"Heaps β={beta_u:.3f}" if beta_u is not None else "線形"
        est_total = (unit_counts[u] / rows) * target_rows
        print(f"  [{u:6s}] 延べセット 約 {est_total:,.0f} / "
              f"ユニーク 約 {est_sets:,.0f} ({bu})")
    if emit_pairs:
        print("-" * 64)
        print(f"  ペア共起の外挿 (出力CSV行数 = co_count≥{min_pair_count} のユニークペア)")
        print("-" * 64)
        for u in units:
            est_pocc = (pair_occ_total[u] / rows) * target_rows
            est_puniq, beta_pu = _heaps_estimate(pair_uniq_snaps[u], target_rows)
            est_pkeep, beta_pk = _heaps_estimate(pair_keep_snaps[u], target_rows)
            bpu = f"Heaps β={beta_pu:.3f}" if beta_pu is not None else "線形"
            bpk = f"Heaps β={beta_pk:.3f}" if beta_pk is not None else "線形"
            print(f"  [{u:6s}] 延べペア 約 {est_pocc:,.0f} (線形) / "
                  f"ユニーク 約 {est_puniq:,.0f} ({bpu})")
            print(f"           └ CSV行数(co_count≥{min_pair_count}) 約 {est_pkeep:,.0f} ({bpk})")
    print("=" * 64)
    print("注: ユニーク数は語彙成長(Heaps則)で逓減するため Heaps 推定が主。")
    print("    時間推定はウォームアップ(初回DL+spawn)を含む全体スループットの線形外挿。")
    if emit_pairs:
        print("    ペアCSV行数は co_count≥K のユニークペア数。閾値超えは n とともに増えるため")
        print("    Heaps外挿(逓増/逓減)。--no-pairs でペア集計を省略可。")


def cmd_benchmark(args: argparse.Namespace) -> None:
    allowed = load_allowed_phrases(args.method_b_csv, args.min_phrase_freq)
    candidate_tokens, _ = load_candidates_with_tier(
        args.candidates_csv,
        min_freq_profile=args.min_freq_profile,
        min_effect_size=args.min_effect_size,
    )
    run_benchmark(args, candidate_tokens, allowed)


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
    df_b = run_method_b(candidate_tokens, token_tier,
                        workers=args.workers, batch_size=args.batch_size)

    # 案1: Method B 辞書(正規化phrase + freq)を freq≥K で絞り共起の許可語彙にする
    allowed: Optional[set[str]] = None
    if args.min_phrase_freq > 0 and not df_b.empty:
        allowed = set(df_b[df_b["freq"] >= args.min_phrase_freq]["phrase"].astype(str))

    print("\n════ Stage 3/3: Record co-occurrence ════")
    run_cooccurrence(args, candidate_tokens, token_tier, allowed)


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
    p.add_argument("--min-phrase-freq", type=int, default=MIN_PHRASE_FREQ,
                   help="案1 頻度フィルタ: Method B 辞書で freq≥K の phrase のみ共起に採用。"
                        "0でフィルタ無効(自由NP)")
    p.add_argument("--method-b-csv", type=Path, default=B_CANDIDATES_PATH,
                   help="許可フレーズ語彙に使う Method B 辞書 CSV")
    p.add_argument("--input", type=Path, default=LABELED_CLAUSES_PATH,
                   help="共起集計の入力 JSONL(annotate_with_deberta_usa.py の出力)")
    p.add_argument("--sample-ratio", type=float, default=1.0,
                   help="レコード(uuid)単位のサンプリング率 0.0-1.0(既定 1.0=全件)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0, help="0で上限なし(レコード数)")
    p.add_argument("--workers", type=int, default=0, help="0以下でCPUコア数")
    p.add_argument("--row-batch-size", type=int, default=128, help="ワーカーへ渡す行バッチサイズ")
    p.add_argument("--pool-chunksize", type=int, default=64, help="imap_unordered の chunksize")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--unit", nargs="+", choices=["uuid", "field", "clause"],
                   default=list(COOC_UNITS),
                   help="共起セットの集計単位(複数可)。既定は uuid/field/clause すべて")
    p.add_argument("--pairs", action=argparse.BooleanOptionalAction, default=True,
                   help="案B: phrase ペアの共起(co_count+PMI)も出力する。--no-pairs で無効")
    p.add_argument("--min-pair-count", type=int, default=2,
                   help="ペア出力の最小共起回数(これ未満のペアは出力しない=裾の間引き)")
    p.add_argument("--out-prefix", type=Path, default=COOC_OUT_PREFIX,
                   help="出力接頭辞。<prefix>_<unit>_set_probabilities.csv 等を生成")
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
    pc = sub.add_parser("cooccurrence", help="テキスト(フィールド)単位で共起セットを集計する")
    _add_cooccurrence_args(pc, with_candidates_csv=True)
    pc.set_defaults(func=cmd_cooccurrence)

    # benchmark
    pbm = sub.add_parser(
        "benchmark", help="N件で試走し、目標件数の所要時間/phrase数を推定する")
    pbm.add_argument("--test-rows", type=int, default=10000,
                     help="試走するレコード数(先頭から連続)")
    pbm.add_argument("--target-rows", type=int, default=1_000_000,
                     help="外挿先のレコード数")
    _add_cooccurrence_args(pbm, with_candidates_csv=True)
    pbm.set_defaults(func=cmd_benchmark)

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
