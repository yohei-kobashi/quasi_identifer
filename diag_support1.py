#!/usr/bin/env python
"""support=1(ユニーク=最大リスク)の内訳を診断し、method-b 見直し(①②)の効果を事前見積もる。

再生成を一切せず、既存の 07_reid_samples.csv(support/size/qi_json 付き) と
06_term_records.csv(phrase→record_count) だけで以下を測る:

  1. support 別の分布と、support=1 が PROFILE に占める割合
  2. support=1 の **size 分布**:
       size=1 が多い → ②(phrase 粗化)が強く効く (人工的一意性が主因)
       size>=2 が多い → 組合せ一意性(本物の再識別シグナル) → ②は限定的・温存すべき
  3. support=1 の構成 phrase が **②で粗化される対象か**:
       語数分布 / 前置詞(of,at,in,…)を含む割合(=NP文法の PP連結が作った具体phrase)
  4. support=1 の構成 phrase 自体の **希少度**(06_term_records の record_count):
       ①(MIN_PHRASE_FREQ=K)で「構成語が落ちる/集合が空になる」support=1 の割合を K 別に見積り

使い方:
    python diag_support1.py [07_CSV] [06_term_records_CSV]
        既定: data/07_reid_samples.csv  data/06_term_records.csv
"""
import sys
import json
from collections import Counter

import numpy as np
import pandas as pd

PREPS = {"of", "at", "in", "from", "for", "with", "on", "to", "by",
         "as", "near", "about", "into", "over", "under"}
K_GRID = [3, 5, 10, 20, 50]
CHUNK = 2_000_000


def bar(frac: float, width: int = 40) -> str:
    return "#" * int(round(frac * width))


def main() -> None:
    reid = sys.argv[1] if len(sys.argv) > 1 else "data/07_reid_samples.csv"
    term_csv = sys.argv[2] if len(sys.argv) > 2 else "data/06_term_records.csv"

    # phrase -> 出現レコード数(自身の support)
    term_rc: dict[str, int] = {}
    try:
        tdf = pd.read_csv(term_csv)
        term_rc = {str(t): int(c) for t, c in zip(tdf["term"], tdf["record_count"])}
        print(f"term_records: {len(term_rc):,} phrase 読込 ({term_csv})")
    except (FileNotFoundError, KeyError) as e:
        print(f"[warn] term_records を読めません({e}) → 希少度/①見積りはスキップ")

    # 集計器
    n_profile = 0
    support_hist: Counter[int] = Counter()        # support 値 -> PROFILE span 数
    size_all: Counter[int] = Counter()            # 全 PROFILE の size
    s1_size: Counter[int] = Counter()             # support=1 の size
    s1_nwords: Counter[int] = Counter()           # support=1 構成 phrase の語数(phrase 単位)
    s1_with_prep = 0                              # support=1 で前置詞 phrase を含む span 数
    s1_with_multiword = 0                         # support=1 で複数語 phrase を含む span 数
    s1_total = 0
    s1_minrc: list[int] = []                      # support=1 の「最レア構成語の record_count」
    s2_size: Counter[int] = Counter()             # support>=2 の size(対比用)

    reader = pd.read_csv(
        reid, usecols=["provenance", "support", "size", "qi_json"],
        dtype={"provenance": "category", "support": "int64", "size": "int32"},
        keep_default_na=False, chunksize=CHUNK,
    )
    for ci, chunk in enumerate(reader):
        prof = chunk[chunk["provenance"] == "PROFILE"]
        n_profile += len(prof)
        support_hist.update(prof["support"].values.tolist())
        for s, c in prof["size"].value_counts().items():
            size_all[int(s)] += int(c)

        s1 = prof[prof["support"] == 1]
        s1_total += len(s1)
        for s, c in s1["size"].value_counts().items():
            s1_size[int(s)] += int(c)

        s2 = prof[prof["support"] >= 2]
        for s, c in s2["size"].value_counts().items():
            s2_size[int(s)] += int(c)

        # support=1 の構成 phrase を解析(②の対象判定 + 希少度)
        for qi in s1["qi_json"].values:
            try:
                terms = json.loads(qi)
            except json.JSONDecodeError:
                continue
            if not terms:
                continue
            has_prep = has_multi = False
            rcs = []
            for ph in terms:
                toks = str(ph).split()
                s1_nwords[len(toks)] += 1
                if len(toks) >= 2:
                    has_multi = True
                if any(t in PREPS for t in toks):
                    has_prep = True
                if term_rc:
                    rcs.append(term_rc.get(str(ph), 0))
            if has_prep:
                s1_with_prep += 1
            if has_multi:
                s1_with_multiword += 1
            if rcs:
                s1_minrc.append(min(rcs))
        print(f"  ...chunk {ci + 1} 処理(PROFILE累計 {n_profile:,})", flush=True)

    # ---------- 出力 ----------
    print("\n" + "=" * 64)
    print(f"  PROFILE span 総数={n_profile:,}")
    n_s1 = support_hist.get(1, 0)
    print(f"  support=1: {n_s1:,} ({100 * n_s1 / max(1, n_profile):.2f}% of PROFILE)")
    print("  support 値 上位:")
    for s, c in support_hist.most_common(8):
        print(f"    support={s:>8}  n={c:,} ({100 * c / max(1, n_profile):.2f}%)")

    def size_table(title: str, ctr: Counter, tot: int) -> None:
        print(f"\n  [{title}] n={tot:,}")
        cum = 0
        for s in sorted(ctr):
            c = ctr[s]
            cum += c
            print(f"    size={s:>3} {c:>12,} ({100 * c / max(1, tot):>6.2f}%) {bar(c / max(1, tot))}")

    size_table("全PROFILE size分布", size_all, n_profile)
    size_table("support=1 size分布", s1_size, s1_total)
    size_table("support>=2 size分布(対比:組合せ一意性=本物シグナル)", s2_size,
               sum(s2_size.values()))

    # ② 効果見積り
    print("\n" + "=" * 64)
    print("  ②(NP文法のPP連結除去/基底NP化)の見積り — support=1 のうち:")
    print(f"    複数語 phrase を含む span : {s1_with_multiword:,} "
          f"({100 * s1_with_multiword / max(1, s1_total):.2f}%) … ②で粗化対象")
    print(f"    前置詞 phrase を含む span : {s1_with_prep:,} "
          f"({100 * s1_with_prep / max(1, s1_total):.2f}%) … ②(PP除去)が直接分解")
    print("  support=1 構成 phrase の語数分布:")
    tot_ph = sum(s1_nwords.values())
    for w in sorted(s1_nwords):
        c = s1_nwords[w]
        print(f"    {w}語 {c:>12,} ({100 * c / max(1, tot_ph):>6.2f}%) {bar(c / max(1, tot_ph))}")
    print("  → 複数語/前置詞比率が高いほど ② で support=1 が大きく減る見込み。")

    # ① 効果見積り
    if s1_minrc:
        arr = np.array(s1_minrc)
        print("\n" + "=" * 64)
        print("  ①(MIN_PHRASE_FREQ=K)の見積り — support=1 のうち最レア構成語が K 未満:")
        print(f"    (= その構成語が allowed から除外され、集合が変化/空化する span 割合)")
        n = len(arr)
        for K in K_GRID:
            removed = int((arr < K).sum())
            print(f"    K={K:>3}: {removed:,} ({100 * removed / n:.2f}%) の support=1 が影響を受ける "
                  f"{bar(removed / n)}")
        print("  最レア構成語 record_count 分布(支持=1):")
        for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 9), (10, 49), (50, 10**9)]:
            c = int(((arr >= lo) & (arr <= hi)).sum())
            lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f">={lo}")
            print(f"    rc={lab:>7}: {c:>12,} ({100 * c / n:>6.2f}%)")

    print("\n読み方:")
    print("  - support=1 が size=1 主体 かつ 複数語/前置詞比率が高い → ②が最大効果(人工的一意性)。")
    print("  - support=1 が size>=2 主体 → 組合せ一意性(本物シグナル)。②は控えめに、温存を優先。")
    print("  - ①は『最レア構成語<K』の割合で効き、K は希少度分布の谷を見て決める。")


if __name__ == "__main__":
    main()
