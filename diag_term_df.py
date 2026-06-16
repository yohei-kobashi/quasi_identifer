#!/usr/bin/env python
"""term/phrase の出現頻度(DF)分布を見て、QI バンドパス上限 MAX_PHRASE_FREQ を決める。

MAX_PHRASE_FREQ は Method B 辞書(04_method_b_candidates.csv)の freq 列(=phrase を
含む PROFILE clause 数)を閾値に、ありふれ過ぎて非識別な phrase(enjoys/love 等)を
許可語彙から除外する。本診断は **その freq 軸**で分布・上裾・K_high 別の除外数/代表語を
出す(=フィルタが実際に切る量と一致)。06_term_records があれば record 出現率も併記する。

使い方:
    python diag_term_df.py [04_CSV] [06_term_records_CSV] [N]
        既定: data/04_method_b_candidates.csv  data/06_term_records.csv  N=metaから/1000000
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# K_high 候補グリッド(04 freq 軸)。上裾を細かく見るため対数的に配置。
K_GRID = [500_000, 300_000, 200_000, 100_000, 50_000, 30_000, 20_000, 10_000, 5_000]
# freq ヒストグラムの対数ビン境界
HIST_EDGES = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000,
              100_000, 300_000, 1_000_000, 3_000_000, 10**9]


def bar(frac: float, width: int = 40) -> str:
    return "#" * int(round(frac * width))


def main() -> None:
    mb = Path(sys.argv[1] if len(sys.argv) > 1 else "data/04_method_b_candidates.csv")
    tr = Path(sys.argv[2] if len(sys.argv) > 2 else "data/06_term_records.csv")
    if len(sys.argv) > 3:
        N = int(sys.argv[3])
    else:
        meta = Path("data/06_risk_sets_meta.json")
        N = int(json.loads(meta.read_text()).get("N", 1_000_000)) if meta.exists() else 1_000_000

    df = pd.read_csv(mb, usecols=["phrase", "freq"])
    df = df.dropna(subset=["freq"])
    df["freq"] = df["freq"].astype("int64")
    n = len(df)
    print(f"Method B 辞書: {n:,} phrase ({mb})   N(レコード総数)={N:,}")

    # record 出現率(任意): 06_term_records の record_count を join
    rec_rate: dict[str, int] = {}
    if tr.exists():
        try:
            t = pd.read_csv(tr, usecols=["term", "record_count"])
            rec_rate = {str(k): int(v) for k, v in zip(t["term"], t["record_count"])}
            print(f"06_term_records: {len(rec_rate):,} term の record_count を併記")
        except (KeyError, ValueError) as e:
            print(f"[warn] term_records 読込失敗({e}) → record率はスキップ")

    def rr(phrase: str) -> str:
        if not rec_rate:
            return ""
        c = rec_rate.get(str(phrase))
        return f"  rec={c:,}({100 * c / N:.1f}%)" if c is not None else "  rec=?"

    # freq 統計
    fq = df["freq"].to_numpy()
    qs = np.percentile(fq, [50, 90, 95, 99, 99.9])
    print(f"\nfreq: min={fq.min():,} median={int(qs[0]):,} "
          f"p90={int(qs[1]):,} p95={int(qs[2]):,} p99={int(qs[3]):,} "
          f"p99.9={int(qs[4]):,} max={fq.max():,}")

    # 対数ヒストグラム
    print("\n[freq 分布(対数ビン)]  ← 上裾(ありふれ語)とバルク(QI候補)の谷を見る")
    edges = np.array(HIST_EDGES)
    idx = np.clip(np.digitize(fq, edges[1:-1]), 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        lab = f"{lo:,}–{hi:,}" if hi < 10**9 else f">={lo:,}"
        print(f"  {lab:>20} {c:>10,} ({100 * c / n:>6.2f}%) {bar(c / n)}")

    # 上裾の代表語(=非識別マーカー候補)
    print("\n[最頻 phrase 上位30(=ありふれ過ぎ=非識別の候補)]")
    top = df.sort_values("freq", ascending=False).head(30)
    for ph, fr in zip(top["phrase"], top["freq"]):
        print(f"  freq={fr:>10,}  {ph}{rr(ph)}")

    # K_high 別の除外数 + 代表語(しきい値直上=切り始める語)
    print("\n[MAX_PHRASE_FREQ=K_high 別の除外見積り(freq>K を許可語彙から除外)]")
    fqs = df.sort_values("freq", ascending=False).reset_index(drop=True)
    for K in K_GRID:
        dropped = fqs[fqs["freq"] > K]
        nd = len(dropped)
        # しきい値直上(=最も freq の小さい除外語)= K を上げ下げする判断材料
        borderline = dropped.tail(6)
        bl = ", ".join(f"{p}({f:,})" for p, f in
                       zip(borderline["phrase"], borderline["freq"]))
        print(f"  K={K:>8,}: 除外 {nd:>7,} 語 ({100 * nd / n:>5.2f}%)  "
              f"境界付近: {bl}")

    # record 出現率ヒスト(任意・意味軸)
    if rec_rate:
        rates = np.array([100 * v / N for v in rec_rate.values()])
        print(f"\n[record 出現率 分布]  term={len(rates):,}  "
              f"(>50%={int((rates > 50).sum()):,}  >30%={int((rates > 30).sum()):,}  "
              f">10%={int((rates > 10).sum()):,})")
        for lo, hi in [(50, 101), (30, 50), (20, 30), (10, 20), (5, 10), (1, 5), (0, 1)]:
            c = int(((rates >= lo) & (rates < hi)).sum())
            print(f"  {lo:>3}–{hi if hi <= 100 else 100:>3}% : {c:>10,} "
                  f"({100 * c / len(rates):>6.2f}%) {bar(c / len(rates))}")

    print("\n読み方:")
    print("  - freq 分布の『上裾』と『バルク』の谷あたりに K_high を置く(谷で切ると恣意性が小)。")
    print("  - K_high 別の『境界付近』語が真の QI(残したい属性)なら K を上げ、")
    print("    非識別マーカー(enjoys/love/general等)ばかりなら K を下げてよい。")
    print("  - record 出現率が併記されていれば『>30〜50%は非識別』の目安と照合する。")


if __name__ == "__main__":
    main()
