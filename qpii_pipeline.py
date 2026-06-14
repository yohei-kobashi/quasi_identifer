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
  data/06_uuid_combos.csv               レコード単位 QPII 組み合わせの部分含有率(頻出アイテムセット)
  data/06_phrase_set_records.csv        phrase(タグ付き範囲=clause)集合の出現レコード数(risk-sets)
  data/06_field_set_records.csv         field 集合の出現レコード数(risk-sets)
  data/06_term_records.csv              単独語の出現レコード数(risk-sets; 特徴量 const_support 用)
  data/06_risk_sets_meta.json           risk-sets のメタ(総レコード N など)
  data/06_span_records.csv              span(text/label/QI集合/freq)素材(span-join; 案2)
  data/07_reid_samples.csv              再識別リスク学習データ(y_risk + LDS 重み + train/test)
                                        sample-spans 版は text 列(X=埋め込みの元)を含む
  (--unit で出力単位を選択可。既定は3つすべて。--no-pairs でペア出力を無効)

単語の有意性は FDR(Benjamini-Hochberg)の q 値で判定する(主解析 q=0.05)。
alpha_tier は q 値が通過する最も厳しい水準("0.01"/"0.05"/"0.10")で、0.01/0.10 は
主結果に対する感度分析(頑健性確認)に使う。フレーズ/共起の tier は「そのフレーズに
含まれる候補トークンの最も厳しい tier」。

頻度フィルタの閾値は3層とも学術的基準で決められる(select-thresholds / "auto"):
  層1 単語   --min-freq-profile : オッズ比95%CI下限>1(Woolf)が95%を満たす最小頻度
  層2 フレーズ --min-phrase-freq : 2-fold 保留集合再現率 ρ(r)>=0.5(Good-Turing)の最小頻度
  層3 ペア   --pair-q           : G²尤度比検定(Dunning)+BH-FDR で有意な共起のみ採用

使い方
------
  python qpii_pipeline.py method-a                 # FDR q=0.05 で候補トークン
  python qpii_pipeline.py method-a --filter 0.01   # 保存済み統計から q=0.01 で再フィルタ
  python qpii_pipeline.py method-a --compare       # 感度分析 q=0.01/0.05/0.10
  python qpii_pipeline.py method-b                 # 候補フレーズ
  python qpii_pipeline.py select-thresholds        # 3層の頻度閾値を推奨
  python qpii_pipeline.py cooccurrence              # annotations_pred_usa.jsonl を全件集計
  python qpii_pipeline.py cooccurrence --min-freq-profile auto --min-phrase-freq auto
  python qpii_pipeline.py benchmark --test-rows 10000 --target-rows 1000000
  python qpii_pipeline.py combos --min-support 0.001 # uuid内QPII組み合わせ(k≥3)の部分含有率
  python qpii_pipeline.py risk-sets                  # phrase/field 集合の出現レコード数(05明細から)
  python qpii_pipeline.py span-join                  # 案2: clause に text/label/QI集合を紐づけ
  python qpii_pipeline.py sample-spans --max-support1 2000000 --max-none-empty 2000000
  python qpii_pipeline.py sample-balanced            # (集合中心)再識別リスクの LDS 学習データ
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
from scipy.stats import chi2, fisher_exact
from tqdm import tqdm

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    sys.path.insert(0, str(Path.cwd()))
from config import DATA_DIR, KEEP_POS_PREFIXES, MIN_PHRASE_FREQ, TEXT_FIELDS

# ── パス ────────────────────────────────────────────────────────────────────
# 入力は annotate_with_deberta_usa.py の予測アノテーション(JSONL, clause/label を含む)。
# .csv を渡せば従来の data/02_labeled_clauses.csv 形式も読める(read_labeled_clauses 参照)。
LABELED_CLAUSES_PATH: Path = DATA_DIR.parent / "annotations_pred_usa.jsonl"

A_STATS_PATH: Path        = DATA_DIR / "03_method_a_stats.csv"
A_CANDIDATES_PATH: Path   = DATA_DIR / "03_method_a_candidates.csv"
B_CANDIDATES_PATH: Path   = DATA_DIR / "04_method_b_candidates.csv"
COOC_OUT_PREFIX: Path     = DATA_DIR / "05"   # <prefix>_<unit>_set_probabilities.csv 等
COMBO_OUT_PATH: Path      = DATA_DIR / "06_uuid_combos.csv"  # 部分含有率(頻出アイテムセット)
# risk-sets: clause(=タグ付き範囲=phrase) / field 単位の表現集合と「出現レコード数」
PHRASE_SET_PATH: Path     = DATA_DIR / "06_phrase_set_records.csv"
FIELD_SET_PATH: Path      = DATA_DIR / "06_field_set_records.csv"
TERM_REC_PATH: Path       = DATA_DIR / "06_term_records.csv"
RISK_SETS_META_PATH: Path = DATA_DIR / "06_risk_sets_meta.json"
# span-join: タグ付き span(clause)に text/label/QI集合を紐づけた学習素材(案2)
SPAN_REC_PATH: Path       = DATA_DIR / "06_span_records.csv"
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
# 共有: 有意水準の階層(tier)と FDR(Benjamini-Hochberg)
# ════════════════════════════════════════════════════════════════════════════
#
# 主基準は FDR(BH)の q 値。tier はこの q 値が通過する最も厳しい水準
# ("0.01"/"0.05"/"0.10")を表す。0.01/0.10 は主結果(既定 q=0.05)に対する
# 感度分析(頑健性確認)として使う。旧 Bonferroni 由来の tier も後方互換で残す。

TIER_ALPHAS: tuple[float, ...] = (0.01, 0.05, 0.10)  # 昇順 = 厳しい順
TIER_ORDER: dict[str, int] = {"0.01": 0, "0.05": 1, "0.10": 2}
DEFAULT_Q: float = 0.05  # FDR の既定水準(主解析)


def bh_qvalues(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg の q 値(調整 p 値)を入力順で返す。

    q_(i) = min_{k>=i} ( p_(k) * m / k )  (昇順ランク i, 検定数 m)。
    単調性を保つよう後ろから累積 min を取り、[0,1] にクリップする。
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])  # p 昇順のインデックス
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):                          # m..1
        idx = order[rank - 1]
        val = pvalues[idx] * m / rank
        prev = min(prev, val)
        q[idx] = min(1.0, prev)
    return q


def compute_q_tier(q_value: float) -> Optional[str]:
    """FDR の q 値が通過する最も厳しい水準を返す("0.01"/"0.05"/"0.10")。"""
    for level in TIER_ALPHAS:
        if q_value <= level:
            return f"{level:.2f}"
    return None


def compute_alpha_tier(p_value: float, n_tokens: int) -> Optional[str]:
    """[後方互換] p値が Bonferroni 補正後に通過する最も厳しい α を返す。

    q_value 列を持たない旧 CSV からの tier 復元用。新規は compute_q_tier を使う。
    """
    for alpha in TIER_ALPHAS:
        if p_value < alpha / max(n_tokens, 1):
            return f"{alpha:.2f}"
    return None


def strictest_tier(tiers: Iterable[Optional[str]]) -> Optional[str]:
    """複数 tier のうち最も厳しい(小さい水準)ものを返す。"""
    valid = [t for t in tiers if t in TIER_ORDER]
    if not valid:
        return None
    return min(valid, key=lambda t: TIER_ORDER[t])


def phrase_alpha_tier(phrase: str, token_tier: dict[str, str]) -> Optional[str]:
    """フレーズに含まれる候補トークンのうち最も厳しい tier を返す。"""
    return strictest_tier(token_tier.get(w) for w in phrase.split())


def add_alpha_tier_column(df: pd.DataFrame) -> pd.DataFrame:
    """候補 DataFrame に alpha_tier 列(FDR q 由来の tier)を付与して返す。

    q_value 列があれば FDR ベース、無ければ旧 Bonferroni(p_value/n_tokens)で復元。
    列名は後方互換のため alpha_tier のまま(意味は「FDR q が通過する最厳水準」)。
    """
    df = df.copy()
    if "q_value" in df.columns:
        df["alpha_tier"] = [compute_q_tier(float(q)) for q in df["q_value"]]
    else:
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
    # FDR(BH)では全検定の p 値が必要なため、Bonferroni による事前足切りはしない。
    # 検定対象は a>0(PROFILE に1回以上出た)トークンのみ(片側 greater で a=0 は p≈1)。
    rows: list[dict] = []
    for token in tokens:
        a = _MA_FREQ_PROFILE.get(token, 0)
        if a == 0:
            continue
        b = _MA_N_PROFILE - a
        c = _MA_FREQ_NONE.get(token, 0)
        d = _MA_N_NONE - c
        _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({
            "token":        token,
            "pos":          _MA_POS_MAP.get(token, ""),
            "freq_profile": a,
            "freq_none":    c,
            "n_profile":    _MA_N_PROFILE,
            "n_none":       _MA_N_NONE,
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

    検定した全トークン(a>0)を保持し、Benjamini-Hochberg の q 値(FDR)を付与する。
    トークンの採否は後段で q<=Q により決める(主基準 FDR)。
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
    corrected_max = MAX_ALPHA / max(n_tokens, 1)  # 互換のため init へ渡すのみ(未使用)

    print(f"Running Fisher's exact test on {n_tokens} tokens ...")
    results = parallel_map(
        _ma_fisher_batch, chunked(all_tokens, batch_size), workers,
        desc="Fisher's exact test", init_fn=_ma_fisher_init,
        init_args=(dict(freq_profile), dict(freq_none), n_profile, n_none,
                   pos_map, corrected_max, n_tokens),
    )
    rows: list[dict] = [r for batch in results for r in batch]

    # BH(FDR)の q 値を「実際に検定したトークン(a>0)」の集合 m 上で計算する。
    m_tested = len(rows)
    pvals = [r["p_value"] for r in rows]
    qvals = bh_qvalues(pvals)
    for r, q in zip(rows, qvals):
        r["q_value"] = q
        r["n_tested"] = m_tested

    df_stats = (
        pd.DataFrame(rows)
        .sort_values(["q_value", "p_value"], ascending=True)
        .reset_index(drop=True)
    )
    df_stats.to_csv(A_STATS_PATH, index=False)
    n_sig = int((df_stats["q_value"] <= DEFAULT_Q).sum())
    print(f"Stats cache saved ({len(df_stats)} tested tokens; "
          f"q<= {DEFAULT_Q}: {n_sig}) → {A_STATS_PATH}")
    return df_stats


def select_significant(df_stats: pd.DataFrame, q: float) -> pd.DataFrame:
    """FDR(BH)の q 値が q 以下のトークンを返す(主基準)。

    q_value 列が無い旧キャッシュは Bonferroni(filter_at_alpha)へフォールバック。
    """
    if "q_value" not in df_stats.columns:
        return filter_at_alpha(df_stats, q)
    df_pass = df_stats[df_stats["q_value"] <= q]
    if df_pass.empty:
        print(f"  FDR q<={q} を満たすトークンがありません。最小 q={df_stats['q_value'].min():.3g}")
    return df_pass.sort_values("effect_size", ascending=False).reset_index(drop=True)


def filter_at_alpha(df_stats: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """[後方互換] 指定 alpha で Bonferroni 補正を適用し候補を返す。"""
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
        if "q_value" not in df_stats.columns:  # 旧 Bonferroni キャッシュは作り直す
            print("Old stats cache lacks q_value (FDR). Rebuilding ...")
            return build_stats_cache(workers=workers, batch_size=batch_size)
        print(f"Loaded stats cache: {len(df_stats)} tokens → {A_STATS_PATH}")
        return df_stats
    return build_stats_cache(workers=workers, batch_size=batch_size)


def compute_method_a(
    q: float = DEFAULT_Q,
    force_build: bool = False,
    workers: int = 1,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Method A を実行し、FDR q<=q の候補トークン DataFrame を返す(候補 CSV も保存)。"""
    download_nltk_data(with_stopwords=True)
    df_stats = load_or_build_stats(force_build=force_build, workers=workers, batch_size=batch_size)
    df_out = add_alpha_tier_column(select_significant(df_stats, q))
    df_out.to_csv(A_CANDIDATES_PATH, index=False)
    print(f"Method A candidates (FDR q<={q}): {len(df_out)} → {A_CANDIDATES_PATH}")
    return df_out


# ── サブコマンド: method-a ────────────────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def cmd_method_a(args: argparse.Namespace) -> None:
    download_nltk_data(with_stopwords=True)

    if args.filter or args.compare:
        df_stats = load_or_build_stats(
            force_build=False, workers=args.workers, batch_size=args.batch_size)
    else:
        print("Building stats cache (FDR) ...")
        df_stats = build_stats_cache(workers=args.workers, batch_size=args.batch_size)

    if args.filter is not None:
        q = args.filter
        df_out = add_alpha_tier_column(select_significant(df_stats, q))
        label = str(q).replace(".", "")
        out_path = DATA_DIR / f"03_method_a_q{label}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\nFDR q<={q} → {len(df_out)} candidates")
        print(df_out.head(20).to_string(index=False))
        print(f"\nSaved → {out_path}")
        return

    if args.compare:
        # 感度分析: 主基準 q=0.05 に対し q=0.01/0.10 で候補集合がどれだけ動くか
        print(f"\n{'═'*64}")
        print("  Sensitivity analysis — FDR q ∈ {0.01, 0.05, 0.10}")
        print("  (主解析 q=0.05。0.01/0.10 で結論が頑健かを確認する)")
        print(f"{'═'*64}")
        sets: dict[float, set[str]] = {}
        for q in COMPARE_ALPHAS:
            sets[q] = set(select_significant(df_stats, q)["token"])
        base = sets[DEFAULT_Q]
        rows_cmp: list[dict] = []
        for q in COMPARE_ALPHAS:
            rows_cmp.append({
                "q":            f"q={q}",
                "candidates":   len(sets[q]),
                "vs q=0.05":    f"{len(sets[q]) - len(base):+d}",
                "Jaccard(0.05)": f"{_jaccard(sets[q], base):.3f}",
            })
        print(pd.DataFrame(rows_cmp).to_string(index=False))
        print(f"\n  q=0.01 のみ採用       : {len(sets[0.01])}")
        print(f"  q=0.05 で追加         : {len(sets[0.05] - sets[0.01])}  "
              f"→ {sorted(sets[0.05] - sets[0.01])[:30]}")
        print(f"  q=0.10 で追加         : {len(sets[0.10] - sets[0.05])}  "
              f"→ {sorted(sets[0.10] - sets[0.05])[:30]}")
        print("\n  解釈: Jaccard が高い(≈1)ほど閾値に頑健。低ければ q の選択が結果を左右する。")
        return

    q = args.q
    df_out = add_alpha_tier_column(select_significant(df_stats, q))
    df_out.to_csv(A_CANDIDATES_PATH, index=False)
    print(f"\nCandidates found (FDR q<={q}) : {len(df_out)}")
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
            if not tier:  # alpha_tier 列が無い CSV からの復元(q_value 優先)
                try:
                    if row.get("q_value") not in (None, ""):
                        tier = compute_q_tier(float(row["q_value"])) or ""
                    else:
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


def _llr_2x2(n11: float, n12: float, n21: float, n22: float) -> float:
    """2x2 分割表の対数尤度比 G²(Dunning 1993)。漸近的に χ²(df=1)。"""
    n = n11 + n12 + n21 + n22
    if n <= 0:
        return 0.0
    r1, r2 = n11 + n12, n21 + n22
    c1, c2 = n11 + n21, n12 + n22
    g = 0.0
    for obs, ri, cj in ((n11, r1, c1), (n12, r1, c2), (n21, r2, c1), (n22, r2, c2)):
        e = ri * cj / n
        if obs > 0 and e > 0:
            g += obs * math.log(obs / e)
    return 2.0 * g


def export_pairs(
    conn: sqlite3.Connection,
    total_units: int,
    output_path: Path,
    min_pair_count: int,
    token_tier: Optional[dict[str, str]] = None,
    pair_q: float = DEFAULT_Q,
) -> int:
    """phrase ペアの共起 + PMI + G²/FDR を出力する(案B + 層3)。

    phrase_occ(p)/pair_occ(a,b) を GROUP BY 集計。min_pair_count は計算量の事前足切り。
    各ペアに 2x2 分割表の G²(Dunning)→ χ²(df=1) 片側 p 値 → BH-FDR の q 値を付与。
    pair_q>0 のとき「q<=pair_q かつ 正の関連(観測>期待)」のみ出力(層3の本選別)。
    PMI = log2( co_count * N / (count_a * count_b) )、N=total_units。
    """
    token_tier = token_tier or {}
    N = total_units
    conn.execute("DROP TABLE IF EXISTS phrase_counts;")
    conn.execute("CREATE TABLE phrase_counts AS SELECT p, COUNT(*) AS c FROM phrase_occ GROUP BY p")
    conn.commit()
    phrase_count: dict[str, int] = {p: c for p, c in conn.execute("SELECT p, c FROM phrase_counts")}

    conn.execute("DROP TABLE IF EXISTS pair_counts;")
    conn.execute("CREATE TABLE pair_counts AS SELECT a, b, COUNT(*) AS c FROM pair_occ GROUP BY a, b")
    conn.commit()

    # 1パス目: 事前足切りを通ったペアの統計(G², 片側p値, 正の関連か)を集める
    recs: list[tuple] = []     # (a, b, co, ca, cb, pmi, g2, positive)
    pvals: list[float] = []
    for a, b, co in conn.execute(
        "SELECT a, b, c FROM pair_counts WHERE c >= ?", (min_pair_count,)
    ):
        ca = phrase_count.get(a, 0)
        cb = phrase_count.get(b, 0)
        n11 = co
        n12 = ca - co
        n21 = cb - co
        n22 = N - ca - cb + co
        e11 = (ca * cb / N) if N > 0 else 0.0
        positive = n11 > e11
        g2 = _llr_2x2(n11, n12, n21, n22)
        # χ²(df=1) 両側 p を方向で片側化(正なら p/2、負なら 1-p/2)
        p_two = float(chi2.sf(g2, 1))
        p_one = (p_two / 2.0) if positive else (1.0 - p_two / 2.0)
        pmi = math.log2((co * N) / (ca * cb)) if (N > 0 and ca > 0 and cb > 0) else 0.0
        recs.append((a, b, co, ca, cb, pmi, g2, positive))
        pvals.append(p_one)

    qvals = bh_qvalues(pvals)  # 足切り後のペア集合上で BH-FDR

    # 2パス目: q でフィルタしつつ書き出し(co_count 降順)
    order = sorted(range(len(recs)), key=lambda i: (-recs[i][2], recs[i][0], recs[i][1]))
    n_pairs = 0
    with output_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow([
            "phrase_a", "phrase_b", "co_count", "count_a", "count_b",
            "probability", "pmi", "g2", "q_value", "tier_a", "tier_b",
        ])
        for i in order:
            a, b, co, ca, cb, pmi, g2, positive = recs[i]
            q = qvals[i]
            if pair_q > 0 and not (positive and q <= pair_q):
                continue
            prob = co / N if N > 0 else 0.0
            writer.writerow([
                a, b, co, ca, cb, prob, round(pmi, 4), round(g2, 4), round(q, 6),
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
                pair_q=args.pair_q,
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
            crit = (f"q≤{args.pair_q} & co_count≥{args.min_pair_count}"
                    if args.pair_q > 0 else f"co_count≥{args.min_pair_count}")
            print(f"           pairs({crit})={n_pairs}  → {pairs_path}")
    print(f"  detail(clause) → {detail_path}")


def resume_cooccurrence_export(
    args: argparse.Namespace,
    token_tier: Optional[dict[str, str]] = None,
) -> None:
    """ingestion 済みの既存 *_sets.sqlite3 を使い、export 段階だけ再開する。

    sqlite を作り直さず開き、total_units を raw_sets から再計算して set確率 /
    ペア共起 を再生成する(export 系は集計テーブルを DROP→CREATE するので冪等)。
    --skip-set-prob で set確率 export をスキップ(完成済みの単位のペアのみ補完する用)。
    """
    download_nltk_data()
    token_tier = augment_token_tier_with_lemmas(token_tier or {})
    units = list(args.unit)
    prefix = str(args.out_prefix)
    emit_pairs = args.pairs
    print(f"[export-only] 既存 sqlite から export 再開: units={units}  "
          f"pairs={'on' if emit_pairs else 'off'}  skip_set_prob={args.skip_set_prob}")
    for u in units:
        db_path = Path(f"{prefix}_{u}_sets.sqlite3")
        if not db_path.exists():
            print(f"  [skip] {db_path.name} が無いので {u} をスキップ")
            continue
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-200000;")
        total_units = int(conn.execute("SELECT COUNT(*) FROM raw_sets").fetchone()[0])
        print(f"  [{u:6s}] total_units={total_units}  ({db_path.name})")
        if not args.skip_set_prob:
            out_csv = Path(f"{prefix}_{u}_set_probabilities.csv")
            t0 = time.time()
            set_types = export_set_probabilities(
                conn, total_units=total_units, output_path=out_csv,
                token_tier=token_tier, count_col=f"{u}_count",
            )
            print(f"          set確率: unique_sets={set_types}  "
                  f"({time.time() - t0:.1f}s) → {out_csv}")
        if emit_pairs:
            pairs_path = Path(f"{prefix}_{u}_pairs.csv")
            t0 = time.time()
            n_pairs = export_pairs(
                conn, total_units=total_units, output_path=pairs_path,
                min_pair_count=args.min_pair_count, token_tier=token_tier,
                pair_q=args.pair_q,
            )
            crit = (f"q≤{args.pair_q} & co_count≥{args.min_pair_count}"
                    if args.pair_q > 0 else f"co_count≥{args.min_pair_count}")
            print(f"          ペア({crit})={n_pairs}  ({time.time() - t0:.1f}s) → {pairs_path}")
        conn.close()
    print("[export-only] 完了")


# ── サブコマンド: cooccurrence ────────────────────────────────────────────────

def cmd_cooccurrence(args: argparse.Namespace) -> None:
    if getattr(args, "export_only", False):
        # ingestion 済みの sqlite から export だけ再開する。method-b/許可フレーズ(層2)は
        # 不要で、tier 注釈用の token_tier だけ候補CSVから用意する。
        min_freq_profile = resolve_min_freq_profile(
            args.min_freq_profile, candidates_csv=args.candidates_csv)
        _cand, token_tier = load_candidates_with_tier(
            args.candidates_csv,
            min_freq_profile=min_freq_profile,
            min_effect_size=args.min_effect_size,
        )
        resume_cooccurrence_export(args, token_tier)
        return
    # 層1: 単語頻度 min(auto=オッズ比CI、候補CSVから算出)
    min_freq_profile = resolve_min_freq_profile(
        args.min_freq_profile, candidates_csv=args.candidates_csv)
    candidate_tokens, token_tier = load_candidates_with_tier(
        args.candidates_csv,
        min_freq_profile=min_freq_profile,
        min_effect_size=args.min_effect_size,
    )
    # 層2: フレーズ頻度 min(auto=2-fold 再現率)
    min_phrase_freq = resolve_min_phrase_freq(
        args.min_phrase_freq, candidate_tokens=candidate_tokens,
        workers=resolve_workers(args.workers), batch_size=256)
    allowed = load_allowed_phrases(args.method_b_csv, min_phrase_freq)
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
    min_freq_profile = resolve_min_freq_profile(
        args.min_freq_profile, candidates_csv=args.candidates_csv)
    candidate_tokens, _ = load_candidates_with_tier(
        args.candidates_csv,
        min_freq_profile=min_freq_profile,
        min_effect_size=args.min_effect_size,
    )
    min_phrase_freq = resolve_min_phrase_freq(
        args.min_phrase_freq, candidate_tokens=candidate_tokens,
        workers=resolve_workers(args.workers), batch_size=256)
    allowed = load_allowed_phrases(args.method_b_csv, min_phrase_freq)
    run_benchmark(args, candidate_tokens, allowed)


# ── サブコマンド: all(全ステージを一気通貫) ─────────────────────────────────

def cmd_all(args: argparse.Namespace) -> None:
    print("════ Stage 1/3: Method A ════")
    df_a = compute_method_a(q=args.q, force_build=args.force_build,
                            workers=args.workers, batch_size=args.batch_size)

    # 層1: 単語頻度 min を解決(auto = オッズ比CI下限>1 が target 割合を満たす最小頻度)
    min_freq_profile = resolve_min_freq_profile(args.min_freq_profile, df_a)
    if min_freq_profile > 1 or args.min_effect_size > 1.0:
        df_a = df_a[
            (df_a["freq_profile"] >= min_freq_profile)
            & (df_a["effect_size"] >= args.min_effect_size)
        ]
        print(f"  filtered candidate tokens (min_freq_profile={min_freq_profile}): {len(df_a)}")

    candidate_tokens: set[str] = set(df_a["token"].str.lower().tolist())
    token_tier = build_token_tier_map(df_a)

    print("\n════ Stage 2/3: Method B ════")
    df_b = run_method_b(candidate_tokens, token_tier,
                        workers=args.workers, batch_size=args.batch_size)

    # 層2: フレーズ頻度 min を解決(auto = 2-fold 再現率 ρ(r)>=target の最小 r)
    min_phrase_freq = resolve_min_phrase_freq(
        args.min_phrase_freq,
        candidate_tokens=candidate_tokens, workers=args.workers, batch_size=args.batch_size,
    )
    # 案1: Method B 辞書(正規化phrase + freq)を freq≥K で絞り共起の許可語彙にする
    allowed: Optional[set[str]] = None
    if min_phrase_freq > 0 and not df_b.empty:
        allowed = set(df_b[df_b["freq"] >= min_phrase_freq]["phrase"].astype(str))

    print("\n════ Stage 3/3: Record co-occurrence ════")
    run_cooccurrence(args, candidate_tokens, token_tier, allowed)


# ════════════════════════════════════════════════════════════════════════════
# 頻度フィルタ閾値の選択アルゴリズム(層1: 単語 / 層2: フレーズ / 層3: ペア)
# ════════════════════════════════════════════════════════════════════════════
#
# 層1 単語頻度  : オッズ比 95%CI 下限 > 1(Woolf)が target 割合を満たす最小頻度
# 層2 フレーズ  : 2-fold 保留集合再現率 ρ(r) >= target を満たす最小頻度(Good-Turing)
# 層3 ペア      : G²(尤度比検定, Dunning) + BH-FDR(export_pairs 内で適用)
#
# 層1/層2 は推奨整数を返し、--min-freq-profile/--min-phrase-freq の "auto" で解決する。
# 層3 は整数ではなく有意性で切るため export_pairs の --pair-q で制御する。

def odds_ratio_ci_lower(a: int, b: int, c: int, d: int, z: float = 1.96) -> float:
    """オッズ比の (1-α) 信頼区間下限。Woolf 法 + Haldane-Anscombe 0.5 補正。

    SE(lnOR) = sqrt(1/(a+.5)+1/(b+.5)+1/(c+.5)+1/(d+.5))、下限 = exp(lnOR - z·SE)。
    """
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    ln_or = math.log((a_ * d_) / (b_ * c_))
    se = math.sqrt(1.0 / a_ + 1.0 / b_ + 1.0 / c_ + 1.0 / d_)
    return math.exp(ln_or - z * se)


def recommend_min_freq_profile(
    df: pd.DataFrame, z: float = 1.96, target: float = 0.95, max_k: int = 30,
) -> tuple[int, list[tuple[int, int, float]]]:
    """層1: オッズ比CI下限>1 を満たすトークン割合が target 以上になる最小 freq を返す。

    返り値: (推奨 K, [(K, 該当トークン数, CI下限>1 の割合), ...])。
    df は freq_profile/freq_none/n_profile/n_none 列を要する(FDR版 method-a の出力)。
    """
    need = {"freq_profile", "freq_none", "n_profile", "n_none"}
    if not need.issubset(df.columns):
        raise ValueError(
            "min-freq-profile auto には n_profile/n_none 列が必要です。"
            "FDR版 method-a を再実行して 03_method_a_candidates.csv を作り直してください。"
        )
    a = df["freq_profile"].astype(int).tolist()
    c = df["freq_none"].astype(int).tolist()
    npf = df["n_profile"].astype(int).tolist()
    nnn = df["n_none"].astype(int).tolist()
    lowers = [
        odds_ratio_ci_lower(ai, npf_i - ai, ci, nnn_i - ci, z)
        for ai, ci, npf_i, nnn_i in zip(a, c, npf, nnn)
    ]
    table: list[tuple[int, int, float]] = []
    rec: Optional[int] = None
    for k in range(1, max_k + 1):
        idx = [i for i, ai in enumerate(a) if ai >= k]
        if not idx:
            break
        passrate = sum(1 for i in idx if lowers[i] > 1.0) / len(idx)
        table.append((k, len(idx), passrate))
        if rec is None and passrate >= target:
            rec = k
    if rec is None:
        rec = table[-1][0] if table else 1
    return rec, table


def resolve_min_freq_profile(
    value, df: Optional[pd.DataFrame] = None, candidates_csv: Optional[Path] = None,
) -> int:
    """--min-freq-profile の値(int または "auto")を整数へ解決する。"""
    s = str(value).strip().lower()
    if s != "auto":
        return int(float(s))
    if df is None:
        df = pd.read_csv(candidates_csv)
    rec, table = recommend_min_freq_profile(df)
    print(f"[auto:層1] min-freq-profile = {rec}  "
          f"(OR 95%CI下限>1 の割合 ≥ 0.95 となる最小頻度)")
    for k, n, pr in table[:rec + 2]:
        print(f"          freq≥{k:2d}: tokens={n:5d}  CI下限>1の割合={pr:.3f}")
    return rec


def _phrase_freq_for_clauses(
    clauses: list[str], candidate_tokens: set[str], workers: int, batch_size: int, desc: str,
) -> Counter:
    """節リストへ Method B の NP 抽出を適用し、正規化phrase の頻度 Counter を返す。"""
    results = parallel_map(
        _mb_batch, chunked(clauses, batch_size), workers,
        desc=desc, init_fn=_mb_init, init_args=(sorted(candidate_tokens),),
    )
    freq: Counter = Counter()
    for f, _h, _ex in results:
        freq.update(f)
    return freq


def recommend_min_phrase_freq(
    candidate_tokens: set[str], workers: int = 1, batch_size: int = 256,
    target: float = 0.5, seed: int = 42, max_k: int = 30,
) -> tuple[int, list[tuple[int, int, float]]]:
    """層2: 2-fold 保留集合再現率 ρ(K) >= target を満たす最小頻度 K を返す。

    PROFILE 節を2分割し、片側で「頻度 >= K」のフレーズが他方にも出現する累積割合
    ρ(K) を測る(両方向を合算)。ρ(K) は K について単調増加で、許可語彙が freq>=K で
    切られる挙動と一致する。返り値: (推奨 K*, [(K, 該当フレーズ数, ρ(K)), ...])。
    """
    download_nltk_data()
    df_clauses = read_labeled_clauses(LABELED_CLAUSES_PATH)
    profile = df_clauses[df_clauses["label"] == "PROFILE"]["clause"].astype(str).tolist()
    rng = random.Random(seed)
    rng.shuffle(profile)
    mid = len(profile) // 2
    fold_a, fold_b = profile[:mid], profile[mid:]
    print(f"[auto:層2] 2-fold 再現率: PROFILE節 {len(profile)} を {len(fold_a)}/{len(fold_b)} に分割")
    freq_a = _phrase_freq_for_clauses(fold_a, candidate_tokens, workers, batch_size, "  fold A")
    freq_b = _phrase_freq_for_clauses(fold_b, candidate_tokens, workers, batch_size, "  fold B")

    table: list[tuple[int, int, float]] = []
    rec: Optional[int] = None
    for k in range(1, max_k + 1):
        # 「頻度>=K」のフレーズが他方に出る累積割合(両方向を合算)
        a_ge = [p for p, ct in freq_a.items() if ct >= k]
        b_ge = [p for p, ct in freq_b.items() if ct >= k]
        n = len(a_ge) + len(b_ge)
        if n == 0:
            table.append((k, 0, float("nan")))
            continue
        hit = sum(1 for p in a_ge if p in freq_b) + sum(1 for p in b_ge if p in freq_a)
        rho = hit / n
        table.append((k, n, rho))
        if rec is None and rho >= target:
            rec = k
    if rec is None:
        rec = max_k
    return rec, table


def resolve_min_phrase_freq(
    value, candidate_tokens: Optional[set[str]] = None,
    workers: int = 1, batch_size: int = 256,
) -> int:
    """--min-phrase-freq の値(int または "auto")を整数へ解決する。"""
    s = str(value).strip().lower()
    if s != "auto":
        return int(float(s))
    if candidate_tokens is None:
        candidate_tokens, _ = load_candidates_with_tier(A_CANDIDATES_PATH)
    rec, table = recommend_min_phrase_freq(candidate_tokens, workers, batch_size)
    print(f"[auto:層2] min-phrase-freq = {rec}  (2-fold 累積再現率 ρ(K) ≥ 0.5 となる最小頻度)")
    for k, n, rho in table[:rec + 2]:
        rho_s = "  n/a" if rho != rho else f"{rho:.3f}"  # nan チェック
        print(f"          freq≥{k:2d}: phrases={n:6d}  再現率ρ={rho_s}")
    return rec


def cmd_select_thresholds(args: argparse.Namespace) -> None:
    """3層の頻度フィルタ閾値を学術的基準で推奨する(層1/層2を実測、層3は方針提示)。"""
    workers = resolve_workers(args.workers)
    print("=" * 64)
    print("  頻度フィルタ閾値の選択(学術的基準)")
    print("=" * 64)

    # 層1: 単語頻度(オッズ比CI)
    df_a = pd.read_csv(args.candidates_csv)
    rec1, t1 = recommend_min_freq_profile(df_a, target=args.ci_target)
    print(f"\n[層1] 単語頻度 min-freq-profile = {rec1}")
    print(f"      基準: オッズ比95%CI下限>1 のトークン割合 ≥ {args.ci_target}(Woolf)")
    for k, n, pr in t1[:rec1 + 3]:
        mark = " ←推奨" if k == rec1 else ""
        print(f"        freq≥{k:2d}: tokens={n:5d}  CI下限>1割合={pr:.3f}{mark}")

    # 層2: フレーズ頻度(2-fold 再現率)
    if not args.no_layer2:
        cand_tokens, _ = load_candidates_with_tier(
            args.candidates_csv, min_freq_profile=rec1, min_effect_size=args.min_effect_size)
        rec2, t2 = recommend_min_phrase_freq(
            cand_tokens, workers=workers, batch_size=args.batch_size, target=args.rho_target)
        print(f"\n[層2] フレーズ頻度 min-phrase-freq = {rec2}")
        print(f"      基準: 2-fold 累積保留集合再現率 ρ(K) ≥ {args.rho_target}(Good-Turing)")
        for k, n, rho in t2[:rec2 + 3]:
            rho_s = "  n/a" if rho != rho else f"{rho:.3f}"
            mark = " ←推奨" if k == rec2 else ""
            print(f"        freq≥{k:2d}: phrases={n:6d}  再現率ρ={rho_s}{mark}")
    else:
        rec2 = None
        print("\n[層2] スキップ(--no-layer2)")

    # 層3: ペア(G²+FDR の方針)
    print(f"\n[層3] ペア共起 --min-pair-count は計算量の事前足切りのみ。")
    print(f"      本選別は cooccurrence --pair-q {args.pair_q}(G² 尤度比検定 + BH-FDR)で行う。")
    print(f"      → 偶然共起を有意性で除くため、整数閾値の最適化は不要。")

    print("\n" + "=" * 64)
    print("  推奨コマンド例:")
    rec2_disp = rec2 if rec2 is not None else "auto"
    print(f"    python qpii_pipeline.py cooccurrence \\")
    print(f"        --min-freq-profile {rec1} --min-phrase-freq {rec2_disp} "
          f"--pair-q {args.pair_q}")
    print("=" * 64)


# ════════════════════════════════════════════════════════════════════════════
# 部分含有率: レコード(uuid)単位の QPII 組み合わせ(頻出アイテムセット)
# ════════════════════════════════════════════════════════════════════════════

def load_uuid_sets(
    sets_csv: Path,
) -> tuple[list[tuple[frozenset, int]], int, dict[str, str]]:
    """05_uuid_set_probabilities.csv を (集合, レコード数) のリストへ読み込む。

    返り値: ([(frozenset(terms), count), ...], 総レコード数, phrase→tier)。
    uuid 単位は 1 レコード=1 集合なので、総レコード数 = uuid_count 列の総和。
    各行の count は「ちょうどその集合を持つレコード数」なので、ある組み合わせ C を
    部分集合に含む全行の count を合算すれば「C を含むレコード数」になる。
    """
    df = pd.read_csv(sets_csv)
    count_col = next((c for c in ("uuid_count", "text_count") if c in df.columns), None)
    if count_col is None:
        raise ValueError(
            f"{sets_csv} に uuid_count 列がありません。"
            "cooccurrence の uuid 単位出力(05_uuid_set_probabilities.csv)を指定してください。")
    tiers_series = df["tiers_json"] if "tiers_json" in df.columns else [None] * len(df)
    sets_counts: list[tuple[frozenset, int]] = []
    tier_map: dict[str, str] = {}
    for terms_json, tiers_json, cnt in zip(df["terms_json"], tiers_series, df[count_col]):
        try:
            terms = json.loads(terms_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if not terms:
            continue
        sets_counts.append((frozenset(terms), int(cnt)))
        if isinstance(tiers_json, str):
            try:
                for t, ti in zip(terms, json.loads(tiers_json)):
                    if ti and t not in tier_map:
                        tier_map[t] = ti
            except json.JSONDecodeError:
                pass
    total_records = sum(c for _s, c in sets_counts)
    return sets_counts, total_records, tier_map


def mine_frequent_itemsets(
    sets_counts: list[tuple[frozenset, int]],
    min_support_count: int,
    max_size: int = 0,
) -> dict[frozenset, int]:
    """Apriori で「部分集合として含む」レコード数(支持度)≥ min_support_count の
    アイテムセットを列挙する。max_size=0 で頻出が尽きるまで探索する。

    返り値: {frozenset(組み合わせ): その組み合わせを部分集合に含むレコード数}。
    各レコード集合 S は自身の全部分集合に count を寄与する。
    """
    item_support: dict[str, int] = defaultdict(int)           # L1: 単一フレーズの支持度
    for s, c in sets_counts:
        for it in s:
            item_support[it] += c
    freq: dict[frozenset, int] = {
        frozenset((it,)): sup for it, sup in item_support.items() if sup >= min_support_count
    }
    all_freq: dict[frozenset, int] = dict(freq)
    freq1 = set(freq)                                          # 頻出 1-項目(レコード前処理用)

    k = 1
    while freq and (max_size == 0 or k < max_size):
        k += 1
        # Apriori-gen: 先頭 k-2 要素を共有する頻出 (k-1)-集合どうしを結合
        prev_sorted = sorted(tuple(sorted(s)) for s in freq)
        freq_prev = set(freq)
        candidates: set[frozenset] = set()
        for i in range(len(prev_sorted)):
            a = prev_sorted[i]
            for j in range(i + 1, len(prev_sorted)):
                b = prev_sorted[j]
                if a[: k - 2] != b[: k - 2]:
                    break                                      # ソート済 → 接頭辞が変われば以降も不一致
                cand = frozenset(a) | frozenset(b)
                if len(cand) != k:
                    continue
                if all(frozenset(cand - {x}) in freq_prev for x in cand):  # 全 (k-1)-部分集合が頻出か
                    candidates.add(cand)
        if not candidates:
            break
        # 候補の支持度を数える(各レコードの頻出項目だけから k-組合せを列挙)
        cand_support: dict[frozenset, int] = {c: 0 for c in candidates}
        for s, cnt in sets_counts:
            if len(s) < k:
                continue
            items = [it for it in s if frozenset((it,)) in freq1]
            if len(items) < k:
                continue
            for combo in combinations(sorted(items), k):
                fc = frozenset(combo)
                if fc in cand_support:
                    cand_support[fc] += cnt
        freq = {c: sup for c, sup in cand_support.items() if sup >= min_support_count}
        all_freq.update(freq)
    return all_freq


def cmd_combos(args: argparse.Namespace) -> None:
    """レコード(uuid)単位で QPII 組み合わせの部分含有率を頻出アイテムセットとして算出する。"""
    sets_counts, total_records, tier_map = load_uuid_sets(args.sets_csv)
    if total_records == 0:
        print("レコードがありません。")
        return
    ms = args.min_support                                     # <1 は割合、>=1 は絶対件数
    if ms < 1.0:
        min_count = max(1, math.ceil(ms * total_records))
        sup_desc = f"{ms:.4g} ({min_count}/{total_records}件)"
    else:
        min_count = int(ms)
        sup_desc = f"{min_count}件"

    print("=" * 64)
    print("  レコード(uuid)単位 QPII 組み合わせの部分含有率(頻出アイテムセット)")
    print("=" * 64)
    print(f"  総レコード数={total_records}  ユニーク集合={len(sets_counts)}  "
          f"min-support={sup_desc}  size≥{args.min_size}"
          + (f"  max-size={args.max_size}" if args.max_size else ""))

    itemsets = mine_frequent_itemsets(sets_counts, min_count, max_size=args.max_size)

    by_size: dict[int, int] = defaultdict(int)                # サイズ別頻出数(1,2 も文脈用に表示)
    for fs in itemsets:
        by_size[len(fs)] += 1
    for sz in sorted(by_size):
        tag = "  ←出力対象" if sz >= args.min_size else ""
        print(f"    size={sz}: {by_size[sz]} 個{tag}")

    rows = [
        {"combo": sorted(fs), "size": len(fs), "support_count": sup,
         "support_pct": sup / total_records}
        for fs, sup in itemsets.items() if len(fs) >= args.min_size
    ]
    rows.sort(key=lambda r: (-r["support_count"], r["size"]))

    with args.output.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["combo_json", "tiers_json", "size", "support_count", "support_pct"])
        for r in rows:
            tiers = [tier_map.get(t, "") for t in r["combo"]]
            writer.writerow([
                json.dumps(r["combo"], ensure_ascii=False),
                json.dumps(tiers, ensure_ascii=False),
                r["size"], r["support_count"], f"{r['support_pct']:.6f}",
            ])
    print(f"\n  size≥{args.min_size} の組み合わせ {len(rows)} 件 → {args.output}")
    if rows:
        print("\n  上位(部分含有レコード割合):")
        for r in rows[: min(args.top, len(rows))]:
            print(f"    {r['support_pct'] * 100:7.3f}%  (n={r['support_count']:>8})  "
                  f"size={r['size']}  {r['combo']}")


# ════════════════════════════════════════════════════════════════════════════
# risk-sets: phrase(=タグ付き範囲=clause) / field 単位の表現集合と出現レコード数
# ════════════════════════════════════════════════════════════════════════════
#
# 05_clause_matched_terms.jsonl(clause ごとに uuid/field/matched_terms を持ち、
# レコード順に連続)を1パスで集計する。レコード単位で「そのレコード内に現れた集合」を
# 重複排除してから +1 することで、出現回数(COUNT(*))ではなく出現レコード数
# (COUNT(DISTINCT uuid))を厳密に得る。これにより support が 1〜N に分布し、
# uuid 全体和集合のようにレコード数で頭打ちにならない(再識別リスク分布の学習用)。

def _write_set_records(
    path: Path, ctr: "Counter[str]", n_records: int, token_tier: dict[str, str],
) -> None:
    """{集合キー(JSON): 出現レコード数} を CSV 出力(record_count 降順)。"""
    rows = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))
    with path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["terms_json", "tiers_json", "set_size", "record_count", "probability"])
        for key, cnt in rows:
            try:
                terms = json.loads(key)
            except json.JSONDecodeError:
                continue
            tiers = [phrase_alpha_tier(t, token_tier) or "" for t in terms]
            p = (cnt / n_records) if n_records > 0 else 0.0
            writer.writerow([
                key, json.dumps(tiers, ensure_ascii=False), len(terms), cnt, f"{p:.8f}",
            ])


def cmd_risk_sets(args: argparse.Namespace) -> None:
    """05_clause_matched_terms.jsonl から phrase/field 集合の出現レコード数を集計する。"""
    detail = Path(args.detail)
    if not detail.exists():
        raise FileNotFoundError(
            f"明細 JSONL が見つかりません: {detail}\n"
            f"先に `python qpii_pipeline.py cooccurrence`(または all)を実行してください。")

    download_nltk_data()
    _, token_tier = load_candidates_with_tier(args.candidates_csv)
    token_tier = augment_token_tier_with_lemmas(token_tier)  # 05 の tier 計算と整合

    phrase_ctr: "Counter[str]" = Counter()   # clause(タグ付き範囲)単位の表現集合
    field_ctr: "Counter[str]" = Counter()    # field 単位(同一レコード内 field の和集合)
    term_ctr: "Counter[str]" = Counter()     # 単独語の出現レコード数(特徴量 const_support 用)
    n_records = 0

    cur_uuid: Any = _NO_UUID
    rec_phrase: set[str] = set()              # レコード内に現れた phrase 集合(JSON キー)
    rec_field_acc: dict[str, set] = {}        # field -> 表現の和集合
    rec_terms: set[str] = set()               # レコード内の単独語

    def flush_record() -> None:
        nonlocal n_records
        if cur_uuid is _NO_UUID:
            return
        n_records += 1                        # QPII が0件でもレコードは母集団 N に数える
        for k in rec_phrase:
            phrase_ctr[k] += 1
        for ts in rec_field_acc.values():
            if ts:
                field_ctr[json.dumps(sorted(ts), ensure_ascii=False)] += 1
        for t in rec_terms:
            term_ctr[t] += 1

    print("=" * 64)
    print("  risk-sets: phrase/field 集合の出現レコード数を集計")
    print("=" * 64)
    print(f"  入力: {detail}")
    started = time.time()
    n_lines = 0
    with detail.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uuid = rec.get("uuid")
            if uuid != cur_uuid:
                flush_record()
                cur_uuid = uuid
                rec_phrase = set()
                rec_field_acc = {}
                rec_terms = set()
            terms = rec.get("matched_terms") or []
            if not terms:                     # 空マッチ clause は集合に寄与しない
                continue
            st = sorted(set(terms))
            rec_phrase.add(json.dumps(st, ensure_ascii=False))
            rec_field_acc.setdefault(rec.get("field"), set()).update(terms)
            rec_terms.update(terms)
            if args.progress_every > 0 and n_lines % args.progress_every == 0:
                el = max(1e-9, time.time() - started)
                print(f"[progress] lines={n_lines} records={n_records} "
                      f"phrase_sets={len(phrase_ctr)} speed={n_lines / el:.0f} line/s")
        flush_record()

    N = n_records
    if N == 0:
        print("レコードがありません。")
        return

    _write_set_records(args.phrase_out, phrase_ctr, N, token_tier)
    _write_set_records(args.field_out, field_ctr, N, token_tier)
    with args.term_out.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["term", "tier", "record_count", "probability"])
        for t, c in sorted(term_ctr.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow([t, phrase_alpha_tier(t, token_tier) or "", c, f"{c / N:.8f}"])

    meta = {
        "N": N, "n_lines": n_lines,
        "n_phrase_sets": len(phrase_ctr), "n_field_sets": len(field_ctr),
        "n_terms": len(term_ctr),
    }
    args.meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  N(総レコード)={N}  clause行={n_lines}")
    print(f"  phrase集合(distinct)={len(phrase_ctr)} → {args.phrase_out}")
    print(f"  field 集合(distinct)={len(field_ctr)} → {args.field_out}")
    print(f"  単独語(distinct)    ={len(term_ctr)} → {args.term_out}")
    print(f"  meta → {args.meta_out}")


# ════════════════════════════════════════════════════════════════════════════
# 再識別リスク学習データ: LDS 重み付きバランスサンプリング(目的B)
# ════════════════════════════════════════════════════════════════════════════
#
# 目的変数 Y = log10(N / support)  (N=総レコード数, support=匿名集合サイズ)。
#   rare な QI ほど support 小 → Y 大(高リスク)。common/NONE は Y≈0(低リスク)。
#   情報量(self-information)/IDF と整合し、全単位がレコード母集団基準で通約可能。
# バランスは既定で LDS(Label Distribution Smoothing, Yang et al. ICML2021)に基づく
#   密度逆数の標本重み lds_weight を付与する(部分抽出ではなく重み付けで均す)。

REID_OUT_PATH: Path = DATA_DIR / "07_reid_samples.csv"


def lds_weights(
    y_values: list[float], n_bins: int = 50, sigma_bins: float = 2.0, clip: float = 10.0,
) -> list[float]:
    """LDS(Yang et al. 2021)による密度逆数の標本重み。平均1に正規化し clip 倍で頭打ち。

    目的値を n_bins ビンに離散化→経験密度をガウス核で平滑化(Label Distribution
    Smoothing)→各標本の重み = 1/平滑化密度。重い裾(高リスク稀少域)を相対的に重く扱う。
    """
    import numpy as np

    y = np.asarray(y_values, dtype=float)
    if y.size == 0:
        return []
    ymin, ymax = float(y.min()), float(y.max())
    if ymax <= ymin:
        return [1.0] * y.size
    edges = np.linspace(ymin, ymax, n_bins + 1)
    idx = np.clip(np.digitize(y, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    half = max(1, int(round(3 * sigma_bins)))                 # ガウス核(±3σ)
    ks = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-(ks ** 2) / (2.0 * sigma_bins ** 2))
    kernel /= kernel.sum()
    smoothed = np.maximum(np.convolve(counts, kernel, mode="same"), 1e-12)
    w = 1.0 / smoothed[idx]
    w *= y.size / w.sum()                                     # 平均1
    if clip and clip > 0:
        w = np.clip(w, 1.0 / clip, clip)
        w *= y.size / w.sum()
    return w.tolist()


def _reid_features(terms: list[str], item_support: dict[str, int]) -> dict:
    """QI(フレーズ集合)から support 以外の予測特徴を作る(丸暗記でなく汎化用)。"""
    lens = [len(t) for t in terms]
    nwords = [len(t.split()) for t in terms]
    cs = [item_support.get(t, 0) for t in terms]
    return {
        "char_len_max": max(lens),
        "char_len_mean": round(sum(lens) / len(lens), 3),
        "n_words_mean": round(sum(nwords) / len(nwords), 3),
        "has_digit": int(any(ch.isdigit() for t in terms for ch in t)),
        "const_support_min": min(cs) if cs else 0,
        "const_support_mean": round(sum(cs) / len(cs), 3) if cs else 0.0,
    }


def _make_reid_sample(
    terms: list[str], support: int, unit_type: str, N: int,
    tier_map: dict[str, str], item_support: dict[str, int],
    provenance: str = "PROFILE", risk_zero: bool = False,
) -> dict:
    support = max(1, int(support))
    y = 0.0 if risk_zero else math.log10(N / support)
    strict = strictest_tier(tier_map.get(t, "") for t in terms) or ""
    s = {
        "provenance": provenance, "unit_type": unit_type, "size": len(terms),
        "qi_json": json.dumps(terms, ensure_ascii=False),
        "support": support, "N": N, "y_risk": round(y, 6),
        "tier_strictest": strict,
    }
    s.update(_reid_features(terms, item_support))
    return s


def _load_none_anchors(
    stats_csv: Path, n_target: int, N: int, item_support: dict[str, int], seed: int,
) -> list[dict]:
    """NONE(非PII)アンカーを method-a 統計から採る。非有意(q大)かつ NONE 優勢な語。

    y_risk=0(NONE は support に依らずリスク0)。運用で非PII入力も来る前提の零点アンカー。
    """
    if n_target <= 0 or not stats_csv.exists():
        if n_target > 0:
            print(f"  [none] {stats_csv} が無いため NONE アンカーをスキップ")
        return []
    df = pd.read_csv(stats_csv)
    need = {"token", "q_value", "freq_profile", "freq_none"}
    if not need.issubset(df.columns):
        print(f"  [none] {stats_csv} に必要列が無いため NONE アンカーをスキップ")
        return []
    cand = df[(df["q_value"] > 0.5) & (df["freq_none"] >= df["freq_profile"])]
    if cand.empty:
        print("  [none] 条件(非有意 & NONE優勢)に合うトークンが無く NONE アンカーをスキップ")
        return []
    cand = cand.sample(n=min(n_target, len(cand)), random_state=seed)
    out: list[dict] = []
    for tok, fn in zip(cand["token"].astype(str), cand["freq_none"].astype(int)):
        out.append(_make_reid_sample(
            [tok], max(1, int(fn)), "phrase", N, {}, item_support,
            provenance="NONE", risk_zero=True))
    print(f"  [none] NONE アンカー {len(out)} 件(q>0.5 & freq_none≥freq_profile から抽出)")
    return out


def _assign_reid_split(
    samples: list[dict], item_support: dict[str, int], test_frac: float, seed: int,
) -> None:
    """リーク回避: 各標本を「最もレアな構成フレーズ(最小support)」でグループ化し、

    そのキーのハッシュで train/test を分ける。識別力を担うレア語が両 split に跨らない。
    """
    import hashlib

    def group_key(terms: list[str]) -> str:
        return min(terms, key=lambda t: (item_support.get(t, 0), t)) if terms else ""

    thr = int(test_frac * 1000)
    for s in samples:
        terms = json.loads(s["qi_json"])
        gk = group_key(terms)
        h = int(hashlib.md5(f"{seed}:{gk}".encode()).hexdigest(), 16) % 1000
        s["split"] = "test" if h < thr else "train"


def _load_set_records(
    path: Path,
) -> tuple[list[tuple[list[str], int]], dict[str, str]]:
    """06_*_set_records.csv を (集合, 出現レコード数) のリストと phrase→tier に読む。"""
    if not path.exists():
        raise FileNotFoundError(
            f"集合-レコード数 CSV が見つかりません: {path}\n"
            f"先に `python qpii_pipeline.py risk-sets` を実行してください。")
    df = pd.read_csv(path)
    if "record_count" not in df.columns:
        raise ValueError(f"{path} に record_count 列がありません(risk-sets の出力を指定してください)。")
    tiers_series = df["tiers_json"] if "tiers_json" in df.columns else [None] * len(df)
    out: list[tuple[list[str], int]] = []
    tier_map: dict[str, str] = {}
    for tj, tij, cnt in zip(df["terms_json"], tiers_series, df["record_count"]):
        try:
            terms = json.loads(tj)
        except (TypeError, json.JSONDecodeError):
            continue
        if not terms:
            continue
        out.append((sorted(terms), int(cnt)))
        if isinstance(tij, str):
            try:
                for t, ti in zip(terms, json.loads(tij)):
                    if ti and t not in tier_map:
                        tier_map[t] = ti
            except json.JSONDecodeError:
                pass
    return out, tier_map


def _build_samples_phrase_field(
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, int], int, str]:
    """risk-sets 出力(phrase/field 集合 + 単独語レコード数)から PROFILE サンプルを作る。

    各 distinct 集合が1サンプル。support=出現レコード数 → y_risk=log10(N/support)。
    uuid 全体和集合と違い phrase/field 集合は複数レコードに再出現するので support が散らばる。
    """
    # item_support(単独語の出現レコード数)+ tier を term_records から
    tdf = pd.read_csv(args.term_records_csv)
    item_support: dict[str, int] = {
        str(t): int(c) for t, c in zip(tdf["term"], tdf["record_count"])
    }
    tier_map: dict[str, str] = {}
    if "tier" in tdf.columns:
        for t, ti in zip(tdf["term"].astype(str), tdf["tier"]):
            if isinstance(ti, str) and ti:
                tier_map.setdefault(t, ti)

    # N(総レコード): meta 優先、無ければ probability から復元
    N = 0
    meta_path = Path(args.meta_json)
    if meta_path.exists():
        try:
            N = int(json.loads(meta_path.read_text(encoding="utf-8")).get("N", 0))
        except (ValueError, json.JSONDecodeError):
            N = 0
    if N <= 0 and "probability" in tdf.columns:
        for c, p in zip(tdf["record_count"], tdf["probability"]):
            if p and float(p) > 0:
                N = int(round(int(c) / float(p)))
                break

    phrase_recs, tm1 = _load_set_records(args.phrase_sets_csv)
    field_recs, tm2 = _load_set_records(args.field_sets_csv)
    for src in (tm1, tm2):
        for k, v in src.items():
            tier_map.setdefault(k, v)

    min_rc = max(1, args.min_record_count)
    samples: list[dict] = []
    for terms, cnt in phrase_recs:
        if cnt >= min_rc:
            samples.append(_make_reid_sample(terms, cnt, "phrase", N, tier_map, item_support))
    n_phrase = len(samples)
    for terms, cnt in field_recs:
        if cnt >= min_rc:
            samples.append(_make_reid_sample(terms, cnt, "field", N, tier_map, item_support))
    info = f"phrase={n_phrase} field={len(samples) - n_phrase} (min_record_count={min_rc})"
    return samples, item_support, N, info


def _build_samples_uuid(
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, int], int, str]:
    """(旧)uuid 全体和集合ソース。phrase=単独語 / combination=mining / record_set=完全集合。

    全レコードが固有集合化し support がレコード数で頭打ちになるため比較用。
    """
    sets_counts, N, tier_map = load_uuid_sets(args.sets_csv)
    item_support: dict[str, int] = defaultdict(int)
    for s, c in sets_counts:
        for it in s:
            item_support[it] += c
    samples: list[dict] = []
    for ph, sup in item_support.items():
        samples.append(_make_reid_sample([ph], sup, "phrase", N, tier_map, item_support))
    n_phrase = len(samples)
    n_combo = 0
    if args.max_size >= 2:
        itemsets = mine_frequent_itemsets(sets_counts, args.combo_min_support, max_size=args.max_size)
        for fs, sup in itemsets.items():
            if len(fs) >= 2:
                samples.append(_make_reid_sample(
                    sorted(fs), sup, "combination", N, tier_map, item_support))
                n_combo += 1
    n_rec = 0
    if not args.no_record_sets:
        for s, c in sets_counts:
            samples.append(_make_reid_sample(
                sorted(s), c, "record_set", N, tier_map, item_support))
            n_rec += 1
    info = f"phrase={n_phrase} combination={n_combo} record_set={n_rec}"
    return samples, dict(item_support), N, info


def cmd_sample_balanced(args: argparse.Namespace) -> None:
    """目的B: 再識別リスク(y_risk=log10(N/support))の LDS 重み付きバランス学習データを作る。"""
    import numpy as np

    print("=" * 64)
    print("  再識別リスク学習データ生成(目的B / LDS 重み付け)")
    print("=" * 64)
    if args.source == "uuid":
        samples, item_support, N, info = _build_samples_uuid(args)
    else:  # phrase-field(既定): risk-sets 出力を使用
        samples, item_support, N, info = _build_samples_phrase_field(args)
    if N <= 0 or not samples:
        print("サンプルがありません(N=0 か集合0件)。入力 CSV を確認してください。")
        return
    print(f"  source={args.source}  N(総レコード)={N}  "
          f"y_risk=log10(N/support) 最大={math.log10(N):.3f}")

    # NONE アンカー(y_risk=0): 目的Bでは勾配を潰さぬよう少量のみ
    n_profile = len(samples)
    if args.none_fraction > 0:
        f = min(0.9, args.none_fraction)
        n_none_target = int(round(f / (1.0 - f) * n_profile))
        samples += _load_none_anchors(args.stats_csv, n_none_target, N, item_support, args.seed)

    print(f"  サンプル: {info}  NONE={len(samples) - n_profile}  計={len(samples)}")

    # ── バランス化 ──────────────────────────────────────────────
    ys = [s["y_risk"] for s in samples]
    if args.balance_mode == "lds":
        w = lds_weights(ys, args.lds_bins, args.lds_sigma, args.weight_clip)
        for s, wi in zip(samples, w):
            s["lds_weight"] = round(wi, 6)
        # max-samples 指定時は重み比例の非復元抽出で物理的にも均す
        if args.max_samples and 0 < args.max_samples < len(samples):
            rng = np.random.default_rng(args.seed)
            p = np.asarray(w, dtype=float); p = p / p.sum()
            sel = rng.choice(len(samples), size=args.max_samples, replace=False, p=p)
            samples = [samples[i] for i in sorted(sel.tolist())]
            ys = [s["y_risk"] for s in samples]
            w2 = lds_weights(ys, args.lds_bins, args.lds_sigma, args.weight_clip)
            for s, wi in zip(samples, w2):
                s["lds_weight"] = round(wi, 6)
            print(f"  max-samples={args.max_samples}: 重み比例抽出で {len(samples)} 件に縮小")
    else:  # bin-equal: 分位ビン等数抽出(従来法の比較用)。重みは一律1
        order = sorted(range(len(samples)), key=lambda i: ys[i])
        nb = max(1, args.lds_bins)
        per = (args.max_samples // nb) if args.max_samples else None
        bins = np.array_split(np.array(order), nb)
        rng = np.random.default_rng(args.seed)
        keep: list[int] = []
        target = per if per else min(len(b) for b in bins if len(b) > 0)
        for b in bins:
            b = b.tolist()
            if not b:
                continue
            k = min(target, len(b))
            keep += rng.choice(b, size=k, replace=False).tolist()
        samples = [samples[i] for i in sorted(keep)]
        for s in samples:
            s["lds_weight"] = 1.0
        print(f"  bin-equal: {nb}ビン×{target}件 → {len(samples)} 件")

    # ── train/test 分割(レア構成フレーズでグループ化しリーク回避)──
    _assign_reid_split(samples, item_support, args.test_frac, args.seed)
    n_test = sum(1 for s in samples if s["split"] == "test")

    # ── 出力 ────────────────────────────────────────────────────
    cols = [
        "provenance", "unit_type", "size", "qi_json", "support", "N", "y_risk",
        "lds_weight", "split", "tier_strictest",
        "char_len_max", "char_len_mean", "n_words_mean", "has_digit",
        "const_support_min", "const_support_mean",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=cols)
        writer.writeheader()
        for s in samples:
            writer.writerow({c: s.get(c, "") for c in cols})

    wvals = [s["lds_weight"] for s in samples]
    print(f"\n  y_risk: min={min(ys):.3f} max={max(ys):.3f}")
    print(f"  lds_weight: min={min(wvals):.3f} max={max(wvals):.3f} (平均≈1)")
    print(f"  split: train={len(samples) - n_test}  test={n_test} (test_frac={args.test_frac})")
    print(f"  → {len(samples)} 件 / {len(cols)} 列  → {args.output}")


# ════════════════════════════════════════════════════════════════════════════
# span-join + sample-spans: タグ付き span(text)中心の学習データ(案2)
# ════════════════════════════════════════════════════════════════════════════
#
# サンプル単位 = 元データでタグ付けされた clause(=span)。X はその text(後段で
# BERT 等の埋め込みを別コードで取得)。y_risk は:
#   PROFILE かつ QI 抽出あり → log10(N/support)  (support=その QI 集合の出現レコード数)
#   NONE                      → 0                (非PII=リスク0。マッチ有はハードネガ)
#   PROFILE かつ QI 抽出なし   → 除外(rarity 由来の y を付けられないため)
# ラベル/テキストは元 annotations_pred_usa.jsonl、QI 集合(matched_terms)は
# 05_clause_matched_terms.jsonl にあり、両者を (uuid→field順) の位置整合で join する
# (NLTK 再抽出は不要)。span-join が distinct text 単位に集約して 06_span_records.csv を
# 出力し、sample-spans が support 参照・バランス化して 07_reid_samples.csv を作る。


def _iter_original_clause_groups(
    path: Path, field_order: list[str],
) -> Iterable[tuple[Any, list[tuple[str, str]]]]:
    """元 jsonl を uuid 単位でまとめ、(uuid, [(field, text, label), ...]) を yield する。

    iter_jsonl_record_tasks と同じ整列(field in TEXT_FIELDS / 非空 clause / field_order 順)
    にして 05_clause_matched_terms.jsonl の clause 並びと一致させる。
    """
    field_set = set(field_order)
    cur_uuid: Any = _NO_UUID
    cur_fields: dict[str, list[tuple[str, str]]] = {}

    def build() -> list[tuple[str, str, str]]:
        return [(f, t, lab) for f in field_order for (t, lab) in cur_fields.get(f, [])]

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
                if cur_uuid is not _NO_UUID:
                    seq = build()
                    if seq:
                        yield cur_uuid, seq
                cur_uuid = uuid
                cur_fields = {}
            field = rec.get("field")
            clause = rec.get("clause")
            if field in field_set and clause:
                cur_fields.setdefault(field, []).append((str(clause), str(rec.get("label"))))
        if cur_uuid is not _NO_UUID:
            seq = build()
            if seq:
                yield cur_uuid, seq


def _iter_detail_clause_groups(
    path: Path,
) -> Iterable[tuple[Any, list[tuple[Any, list[str]]]]]:
    """05_clause_matched_terms.jsonl を uuid 単位でまとめ (uuid, [(field, matched_terms),...])。"""
    cur_uuid: Any = _NO_UUID
    seq: list[tuple[Any, list[str]]] = []
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
                if cur_uuid is not _NO_UUID and seq:
                    yield cur_uuid, seq
                cur_uuid = uuid
                seq = []
            seq.append((rec.get("field"), rec.get("matched_terms") or []))
        if cur_uuid is not _NO_UUID and seq:
            yield cur_uuid, seq


def _join_clause_spans(
    original_path: Path, detail_path: Path, field_order: list[str],
) -> Iterable[tuple[str, str, list[str]]]:
    """元 jsonl と detail jsonl を位置整合 join し (text, label, matched_terms) を yield。

    両ファイルとも uuid 連続・同順・同じ clause 集合(field in TEXT_FIELDS / 非空)である前提。
    長さ不一致の uuid は安全のためスキップ(件数は呼び出し側で集計)。
    """
    orig = _iter_original_clause_groups(original_path, field_order)
    o_uuid, o_seq = next(orig, (None, None))
    skipped = 0
    for d_uuid, d_seq in _iter_detail_clause_groups(detail_path):
        while o_uuid is not None and o_uuid != d_uuid:
            o_uuid, o_seq = next(orig, (None, None))
        if o_uuid is None:
            break
        if o_seq is None or len(o_seq) != len(d_seq):
            skipped += 1
            o_uuid, o_seq = next(orig, (None, None))
            continue
        for (_of, text, label), (_df, terms) in zip(o_seq, d_seq):
            yield text, label, terms
        o_uuid, o_seq = next(orig, (None, None))
    if skipped:
        print(f"  [warn] clause 数不一致でスキップした uuid: {skipped}")


def cmd_span_join(args: argparse.Namespace) -> None:
    """元 jsonl × detail jsonl を join し、distinct text 単位の span 学習素材を作る。"""
    original = Path(args.input)
    detail = Path(args.detail)
    for p in (original, detail):
        if not p.exists():
            raise FileNotFoundError(f"入力が見つかりません: {p}")

    db_path = Path(args.work_db)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute("CREATE TABLE spans (text TEXT NOT NULL, label TEXT NOT NULL, qi_json TEXT NOT NULL);")
    conn.commit()

    print("=" * 64)
    print("  span-join: clause(text/label/QI集合)を join して span 素材を作成")
    print("=" * 64)
    print(f"  元: {original}\n  detail: {detail}")

    field_order = list(TEXT_FIELDS)
    buf: list[tuple[str, str, str]] = []
    n_clause = n_profile = n_none = 0
    started = time.time()
    for text, label, terms in _join_clause_spans(original, detail, field_order):
        qi = json.dumps(sorted(set(terms)), ensure_ascii=False)  # 空集合は "[]"
        buf.append((text, label, qi))
        n_clause += 1
        if label == "PROFILE":
            n_profile += 1
        elif label == "NONE":
            n_none += 1
        if len(buf) >= args.insert_buffer:
            conn.executemany("INSERT INTO spans (text, label, qi_json) VALUES (?, ?, ?)", buf)
            conn.commit()
            buf.clear()
            if args.progress_every > 0 and n_clause % args.progress_every == 0:
                el = max(1e-9, time.time() - started)
                print(f"[progress] clause={n_clause:,} speed={n_clause / el:.0f}/s")
    if buf:
        conn.executemany("INSERT INTO spans (text, label, qi_json) VALUES (?, ?, ?)", buf)
        conn.commit()

    # distinct text に集約。曖昧ラベルは MAX(label)= PROFILE 優先(リスク過小評価を避ける)
    conn.execute("DROP TABLE IF EXISTS span_uniq;")
    conn.execute(
        "CREATE TABLE span_uniq AS "
        "SELECT text, MAX(label) AS label, MAX(qi_json) AS qi_json, COUNT(*) AS freq "
        "FROM spans GROUP BY text"
    )
    conn.commit()
    n_uniq = int(conn.execute("SELECT COUNT(*) FROM span_uniq").fetchone()[0])

    with args.output.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["text", "label", "qi_json", "size", "freq"])
        for text, label, qi_json, freq in conn.execute(
            "SELECT text, label, qi_json, freq FROM span_uniq ORDER BY freq DESC"
        ):
            try:
                size = len(json.loads(qi_json))
            except json.JSONDecodeError:
                size = 0
            writer.writerow([text, label, qi_json, size, freq])
    conn.close()

    print(f"\n  clause 総数={n_clause:,} (PROFILE={n_profile:,} NONE={n_none:,})")
    print(f"  distinct text={n_uniq:,} → {args.output}")


def _load_support_index(
    phrase_sets_csv: Path,
) -> tuple[dict[str, int], dict[str, str]]:
    """06_phrase_set_records.csv を {qi_json: record_count} と {qi_json: strictest_tier} に読む。"""
    support: dict[str, int] = {}
    strict: dict[str, str] = {}
    df = pd.read_csv(phrase_sets_csv)
    tiers_series = df["tiers_json"] if "tiers_json" in df.columns else [None] * len(df)
    for tj, tij, cnt in zip(df["terms_json"], tiers_series, df["record_count"]):
        support[str(tj)] = int(cnt)
        if isinstance(tij, str):
            try:
                strict[str(tj)] = strictest_tier(json.loads(tij)) or ""
            except json.JSONDecodeError:
                strict[str(tj)] = ""
    return support, strict


def cmd_sample_spans(args: argparse.Namespace) -> None:
    """案2: span(text)中心の再識別リスク学習データを LDS バランスで作る。"""
    import numpy as np

    # N(総レコード)
    N = 0
    meta_path = Path(args.meta_json)
    if meta_path.exists():
        try:
            N = int(json.loads(meta_path.read_text(encoding="utf-8")).get("N", 0))
        except (ValueError, json.JSONDecodeError):
            N = 0
    if N <= 0:
        raise ValueError(f"{meta_path} から N を取得できません。risk-sets を先に実行してください。")

    support, strict = _load_support_index(args.phrase_sets_csv)
    item_support: dict[str, int] = {}
    if Path(args.term_records_csv).exists():
        tdf = pd.read_csv(args.term_records_csv)
        item_support = {str(t): int(c) for t, c in zip(tdf["term"], tdf["record_count"])}

    print("=" * 64)
    print("  span 中心 再識別リスク学習データ生成(案2 / LDS)")
    print("=" * 64)
    spans = pd.read_csv(args.span_records_csv,
                        usecols=["text", "label", "qi_json", "size", "freq"],
                        dtype={"label": "category", "size": "int32", "freq": "int64"},
                        keep_default_na=False)
    print(f"  N={N}  span(distinct text)={len(spans):,}  最大 y=log10(N)={math.log10(N):.3f}")

    # support / y_risk を算出
    sup = spans["qi_json"].map(support)               # 非空PROFILEは必ずヒット
    is_profile = spans["label"].astype(str) == "PROFILE"
    is_none = spans["label"].astype(str) == "NONE"
    has_qi = spans["size"] > 0

    drop = is_profile & ~has_qi                        # PROFILE だが QI 抽出なし → 除外
    n_drop = int(drop.sum())
    spans = spans[~drop].copy()
    sup = sup[~drop]
    is_profile = is_profile[~drop]
    is_none = is_none[~drop]
    has_qi = has_qi[~drop]

    spans["support"] = sup.fillna(0).astype("int64").values
    y = np.where(is_none.values, 0.0,
                 np.log10(N / np.maximum(1, spans["support"].values)))
    spans["y_risk"] = np.round(y, 6)
    spans["provenance"] = np.where(is_none.values, "NONE", "PROFILE")

    # カテゴリ分け
    cat_profile = spans[spans["provenance"] == "PROFILE"]
    none_all = spans[spans["provenance"] == "NONE"]
    none_match = none_all[none_all["size"] > 0]
    none_empty = none_all[none_all["size"] == 0]
    n_sup1 = int((cat_profile["support"] == 1).sum())
    print(f"  除外(PROFILE/QI無)={n_drop:,}")
    print(f"  PROFILE={len(cat_profile):,} (うち support=1: {n_sup1:,})  "
          f"NONE_match={len(none_match):,}  NONE_empty={len(none_empty):,}")

    rng = np.random.default_rng(args.seed)

    def subsample(d: pd.DataFrame, cap: int) -> pd.DataFrame:
        if cap and 0 < cap < len(d):
            idx = rng.choice(len(d), size=cap, replace=False)
            return d.iloc[np.sort(idx)]
        return d

    # PROFILE: support=1 のスパイクのみ間引き、support>=2 は全保持
    prof_sup1 = subsample(cat_profile[cat_profile["support"] == 1], args.max_support1)
    prof_rest = cat_profile[cat_profile["support"] >= 2]
    # NONE: マッチ有(ハードネガ)は全保持、空マッチは上限
    none_empty = subsample(none_empty, args.max_none_empty)

    out = pd.concat([prof_rest, prof_sup1, none_match, none_empty], ignore_index=True)
    print(f"  → バランス後: PROFILE={len(prof_rest) + len(prof_sup1):,} "
          f"(support=1 を {len(prof_sup1):,} に間引き)  "
          f"NONE={len(none_match) + len(none_empty):,}  計={len(out):,}")

    # LDS 重み
    ys = out["y_risk"].tolist()
    out["lds_weight"] = [round(w, 6) for w in
                         lds_weights(ys, args.lds_bins, args.lds_sigma, args.weight_clip)]

    # tier_strictest
    out["tier_strictest"] = out["qi_json"].map(lambda q: strict.get(q, ""))

    # train/test 分割(最レア構成語でグループ化, 空QI は text ハッシュ)
    import hashlib

    def split_key(qi_json: str, text: str) -> str:
        try:
            terms = json.loads(qi_json)
        except json.JSONDecodeError:
            terms = []
        if terms:
            return min(terms, key=lambda t: (item_support.get(t, 0), t))
        return "EMPTY:" + hashlib.md5(text.encode()).hexdigest()[:12]

    thr = int(args.test_frac * 1000)
    splits = []
    for qi_json, text in zip(out["qi_json"], out["text"]):
        gk = split_key(qi_json, text)
        h = int(hashlib.md5(f"{args.seed}:{gk}".encode()).hexdigest(), 16) % 1000
        splits.append("test" if h < thr else "train")
    out["split"] = splits
    out["N"] = N

    # max-samples(任意・LDS重み比例の物理間引き)
    if args.max_samples and 0 < args.max_samples < len(out):
        w = np.asarray(out["lds_weight"].values, dtype=float)
        w = w / w.sum()
        sel = rng.choice(len(out), size=args.max_samples, replace=False, p=w)
        out = out.iloc[np.sort(sel)].copy()
        out["lds_weight"] = [round(v, 6) for v in
                             lds_weights(out["y_risk"].tolist(), args.lds_bins,
                                         args.lds_sigma, args.weight_clip)]
        print(f"  max-samples={args.max_samples}: 重み比例抽出で {len(out):,} 件に縮小")

    cols = ["provenance", "text", "qi_json", "size", "support", "N", "y_risk",
            "lds_weight", "split", "tier_strictest", "freq"]
    out[cols].to_csv(args.output, index=False)

    n_test = int((out["split"] == "test").sum())
    print(f"\n  y_risk: min={out['y_risk'].min():.3f} max={out['y_risk'].max():.3f}")
    print(f"  lds_weight: min={out['lds_weight'].min():.3f} max={out['lds_weight'].max():.3f}")
    print(f"  split: train={len(out) - n_test:,}  test={n_test:,} (test_frac={args.test_frac})")
    print(f"  → {len(out):,} 件 / {len(cols)} 列(text 含む) → {args.output}")


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
    p.add_argument("--min-freq-profile", type=str, default="1",
                   help="層1 単語頻度フィルタ: 候補トークンの最小 freq_profile。"
                        "整数 or 'auto'(オッズ比95%%CI下限>1 が95%%を満たす最小頻度)")
    p.add_argument("--min-effect-size", type=float, default=1.0,
                   help="候補トークンの最小 effect_size (odds ratio)")
    p.add_argument("--drop-single-word", action="store_true",
                   help="1語の表現を除外する(04 の最終出力と同じ挙動)")
    p.add_argument("--min-phrase-freq", type=str, default=str(MIN_PHRASE_FREQ),
                   help="層2 フレーズ頻度フィルタ(案1): Method B 辞書で freq≥K の phrase のみ採用。"
                        "整数 or 'auto'(2-fold 再現率 ρ(r)>=0.5 の最小頻度)。0でフィルタ無効")
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
                   help="層3 ペアの事前足切り: 共起回数がこれ未満のペアは出力しない(計算量制御)")
    p.add_argument("--pair-q", type=float, default=DEFAULT_Q,
                   help="層3 ペアの本選別: G²尤度比検定+BH-FDR の q 値がこれ以下かつ正の関連の"
                        "ペアのみ出力。0以下でFDR無効(min-pair-countのみ)")
    p.add_argument("--out-prefix", type=Path, default=COOC_OUT_PREFIX,
                   help="出力接頭辞。<prefix>_<unit>_set_probabilities.csv 等を生成")
    p.add_argument("--sqlite-insert-buffer", type=int, default=20000,
                   help="SQLiteへ一括INSERTするバッファ行数")
    p.add_argument("--export-only", action="store_true",
                   help="ingestionをスキップし、既存 <prefix>_<unit>_sets.sqlite3 から "
                        "set確率/ペアの export だけ再開する(再開モード)")
    p.add_argument("--skip-set-prob", action="store_true",
                   help="--export-only 時に set確率CSVをスキップしペアのみ再生成する"
                        "(set確率が完成済みの単位用)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QPII 抽出パイプライン(Method A → Method B → レコード内共起集計)"
    )
    sub = parser.add_subparsers(dest="command")

    # method-a
    pa = sub.add_parser("method-a", help="形態素ベースの統計抽出で候補トークンを得る(FDR)")
    pa.add_argument("--q", type=float, default=DEFAULT_Q,
                    help="FDR(Benjamini-Hochberg)の q 値しきい値(既定 0.05=主解析)")
    pa.add_argument("--filter", type=float, metavar="Q",
                    help="保存済み統計から指定 q で再フィルタ(再計算なし)")
    pa.add_argument("--compare", action="store_true",
                    help="感度分析: q=0.01/0.05/0.10 で候補集合の変動と Jaccard を出力")
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
    pall.add_argument("--q", type=float, default=DEFAULT_Q, help="Method A の FDR q 値しきい値")
    pall.add_argument("--force-build", action="store_true",
                      help="統計キャッシュを無視して Method A を再計算する")
    pall.add_argument("--batch-size", type=int, default=256,
                      help="Method A/B のワーカーへ渡す節/トークンのバッチサイズ")
    _add_cooccurrence_args(pall, with_candidates_csv=False)
    pall.set_defaults(func=cmd_all)

    # select-thresholds(3層の頻度フィルタ閾値を学術的基準で推奨)
    pst = sub.add_parser(
        "select-thresholds",
        help="層1単語/層2フレーズ/層3ペアの頻度フィルタ閾値を学術的基準で推奨する")
    pst.add_argument("--candidates-csv", type=Path, default=A_CANDIDATES_PATH,
                     help="Method A の候補トークン CSV(FDR版)")
    pst.add_argument("--ci-target", type=float, default=0.95,
                     help="層1: オッズ比CI下限>1 を満たすべきトークン割合(既定0.95)")
    pst.add_argument("--rho-target", type=float, default=0.5,
                     help="層2: 2-fold 再現率 ρ(r) のしきい値(既定0.5)")
    pst.add_argument("--pair-q", type=float, default=DEFAULT_Q,
                     help="層3: 推奨コマンドに載せる FDR q 値(既定0.05)")
    pst.add_argument("--min-effect-size", type=float, default=1.0)
    pst.add_argument("--no-layer2", action="store_true",
                     help="層2(2-fold 再現率, 重い)をスキップする")
    pst.add_argument("--workers", type=int, default=0, help="0以下でCPUコア数")
    pst.add_argument("--batch-size", type=int, default=256)
    pst.set_defaults(func=cmd_select_thresholds)

    # combos(レコード単位 QPII 組み合わせの部分含有率)
    pco = sub.add_parser(
        "combos",
        help="uuid 単位で QPII 組み合わせ(k≥3)を部分集合に含むレコード割合を算出する")
    pco.add_argument("--sets-csv", type=Path,
                     default=DATA_DIR / "05_uuid_set_probabilities.csv",
                     help="cooccurrence の uuid 集合出力 CSV(uuid_count 列が必要)")
    pco.add_argument("--min-size", type=int, default=3,
                     help="出力する組み合わせの最小サイズ(既定3)")
    pco.add_argument("--max-size", type=int, default=0,
                     help="探索する最大サイズ(0で頻出が尽きるまで)")
    pco.add_argument("--min-support", type=float, default=0.001,
                     help="最小支持度。<1で割合(既定0.001=0.1%%)、>=1で絶対レコード数")
    pco.add_argument("--top", type=int, default=30, help="標準出力に表示する上位件数")
    pco.add_argument("--output", type=Path, default=COMBO_OUT_PATH)
    pco.set_defaults(func=cmd_combos)

    # risk-sets(phrase/field 単位の表現集合と「出現レコード数」を算出)
    prs = sub.add_parser(
        "risk-sets",
        help="05_clause_matched_terms.jsonl から phrase/field 集合の出現レコード数を集計する")
    prs.add_argument("--detail", type=Path,
                     default=DATA_DIR / "05_clause_matched_terms.jsonl",
                     help="cooccurrence の clause 明細 JSONL(uuid/field/matched_terms)")
    prs.add_argument("--candidates-csv", type=Path, default=A_CANDIDATES_PATH,
                     help="tier 付与に使う Method A 候補トークン CSV")
    prs.add_argument("--phrase-out", type=Path, default=PHRASE_SET_PATH)
    prs.add_argument("--field-out", type=Path, default=FIELD_SET_PATH)
    prs.add_argument("--term-out", type=Path, default=TERM_REC_PATH)
    prs.add_argument("--meta-out", type=Path, default=RISK_SETS_META_PATH)
    prs.add_argument("--progress-every", type=int, default=1_000_000,
                     help="進捗表示する明細行間隔(0で無効)")
    prs.set_defaults(func=cmd_risk_sets)

    # span-join(案2: clause に text/label/QI集合を紐づけ distinct text へ集約)
    psj = sub.add_parser(
        "span-join",
        help="元jsonl×detail jsonl を join し span(text/label/QI集合)素材を作る(案2)")
    psj.add_argument("--input", type=Path, default=LABELED_CLAUSES_PATH,
                     help="元アノテーション JSONL(clause/label/uuid/field)")
    psj.add_argument("--detail", type=Path,
                     default=DATA_DIR / "05_clause_matched_terms.jsonl",
                     help="cooccurrence の clause 明細 JSONL(uuid/field/matched_terms)")
    psj.add_argument("--work-db", type=Path, default=DATA_DIR / "06_span_join.sqlite3",
                     help="集約用の一時 SQLite(実行ごとに作り直し)")
    psj.add_argument("--insert-buffer", type=int, default=50000,
                     help="SQLite へ一括INSERTするバッファ行数")
    psj.add_argument("--progress-every", type=int, default=2_000_000,
                     help="進捗表示する clause 間隔(0で無効)")
    psj.add_argument("--output", type=Path, default=SPAN_REC_PATH)
    psj.set_defaults(func=cmd_span_join)

    # sample-spans(案2: span(text)中心の LDS バランス学習データ)
    psp = sub.add_parser(
        "sample-spans",
        help="span(text)中心の再識別リスク学習データを作る(X=text, 埋め込みは別コード)")
    psp.add_argument("--span-records-csv", type=Path, default=SPAN_REC_PATH,
                     help="span-join の出力(text/label/qi_json/size/freq)")
    psp.add_argument("--phrase-sets-csv", type=Path, default=PHRASE_SET_PATH,
                     help="support 参照用 risk-sets の phrase 集合 CSV")
    psp.add_argument("--term-records-csv", type=Path, default=TERM_REC_PATH,
                     help="split グループ化(最レア構成語)用の単独語レコード数 CSV")
    psp.add_argument("--meta-json", type=Path, default=RISK_SETS_META_PATH,
                     help="総レコード N を含む risk-sets meta")
    psp.add_argument("--max-support1", type=int, default=0,
                     help="PROFILE の support=1(ユニーク=最大リスク)の保持上限。0で全保持。"
                          "スパイク間引き量は検証メトリクスでスイープして決める")
    psp.add_argument("--max-none-empty", type=int, default=0,
                     help="NONE 空マッチ span の保持上限。0で全保持(マッチ有NONEは常に全保持)")
    psp.add_argument("--lds-bins", type=int, default=50, help="LDS の分割数")
    psp.add_argument("--lds-sigma", type=float, default=2.0, help="LDS ガウス平滑核の σ")
    psp.add_argument("--weight-clip", type=float, default=10.0,
                     help="lds_weight の頭打ち倍率(0で無効)")
    psp.add_argument("--max-samples", type=int, default=0,
                     help="出力上限(0=全件)。指定時は LDS 重み比例の非復元抽出で物理縮小")
    psp.add_argument("--test-frac", type=float, default=0.2, help="test 分割比率")
    psp.add_argument("--seed", type=int, default=42)
    psp.add_argument("--output", type=Path, default=REID_OUT_PATH)
    psp.set_defaults(func=cmd_sample_spans)

    # sample-balanced(目的B: 再識別リスクの LDS 重み付きバランス学習データ)
    psb = sub.add_parser(
        "sample-balanced",
        help="再識別リスク y_risk=log10(N/support) の LDS 重み付きバランス学習データを作る")
    psb.add_argument("--source", choices=["phrase-field", "uuid"], default="phrase-field",
                     help="既定 phrase-field=risk-sets の phrase/field 集合(support が分布する)"
                          " / uuid=旧 全体和集合(レコード数で頭打ち, 比較用)")
    psb.add_argument("--phrase-sets-csv", type=Path, default=PHRASE_SET_PATH,
                     help="[phrase-field] risk-sets の phrase 集合 CSV")
    psb.add_argument("--field-sets-csv", type=Path, default=FIELD_SET_PATH,
                     help="[phrase-field] risk-sets の field 集合 CSV")
    psb.add_argument("--term-records-csv", type=Path, default=TERM_REC_PATH,
                     help="[phrase-field] risk-sets の単独語レコード数 CSV(特徴量 const_support 用)")
    psb.add_argument("--meta-json", type=Path, default=RISK_SETS_META_PATH,
                     help="[phrase-field] risk-sets の meta(総レコード N)")
    psb.add_argument("--min-record-count", type=int, default=1,
                     help="[phrase-field] この出現レコード数未満の集合を除外(既定1=全件)")
    psb.add_argument("--sets-csv", type=Path,
                     default=DATA_DIR / "05_uuid_set_probabilities.csv",
                     help="[uuid] 全体和集合 CSV(uuid_count 列が必要)")
    psb.add_argument("--stats-csv", type=Path, default=A_STATS_PATH,
                     help="NONE アンカー抽出元の method-a 統計 CSV")
    psb.add_argument("--max-size", type=int, default=4,
                     help="[uuid] 組み合わせ(size≥2)を探索する最大サイズ(既定4)")
    psb.add_argument("--combo-min-support", type=int, default=2,
                     help="[uuid] 組み合わせ mining の事前足切り(計算量制御, 既定2)")
    psb.add_argument("--no-record-sets", action="store_true",
                     help="[uuid] 観測レコード完全集合(k-匿名性等価クラス)を含めない")
    psb.add_argument("--none-fraction", type=float, default=0.15,
                     help="NONE(リスク0)アンカーの目標割合(既定0.15)。0で無効")
    psb.add_argument("--balance-mode", choices=["lds", "bin-equal"], default="lds",
                     help="既定 lds=密度逆数重み付け / bin-equal=分位ビン等数抽出")
    psb.add_argument("--lds-bins", type=int, default=50, help="LDS/ビンの分割数")
    psb.add_argument("--lds-sigma", type=float, default=2.0,
                     help="LDS ガウス平滑核の σ(ビン単位)")
    psb.add_argument("--weight-clip", type=float, default=10.0,
                     help="lds_weight の頭打ち倍率(0で無効)")
    psb.add_argument("--max-samples", type=int, default=0,
                     help="出力上限(既定0=全件)。LDSは重みで分布を均すので通常は間引き不要。"
                          "巨大すぎる場合のみ指定→lds は重み比例の非復元抽出で物理バランス化")
    psb.add_argument("--test-frac", type=float, default=0.2, help="test 分割比率")
    psb.add_argument("--seed", type=int, default=42)
    psb.add_argument("--output", type=Path, default=REID_OUT_PATH)
    psb.set_defaults(func=cmd_sample_balanced)

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
