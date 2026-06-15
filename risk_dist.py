#!/usr/bin/env python
"""07_reid_samples.csv(span版) の再識別リスク y_risk=log10(N/support) の分布を見る。

使い方:
    python risk_dist.py [CSV] [PNG]
        CSV: 既定 data/07_reid_samples.csv
        PNG: 第2引数を渡すと train(LDS有/無)の密度ヒストグラムを保存

span版スキーマ(列): provenance,text,qi_json,size,support,N,y_risk,lds_weight,split,
tier_strictest,freq。NONE は provenance=="NONE"(y=0)。
train は --max-support1/--max-none-empty でリバランス、test は自然分布で固定。
n%(生の件数割合) と w%(LDS 重み後) を並べ、support=1(=ユニーク=最大リスク)割合に注目。
"""
import sys
import math

import numpy as np
import pandas as pd


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/07_reid_samples.csv"
    png = sys.argv[2] if len(sys.argv) > 2 else ""

    cols = pd.read_csv(path, nrows=0).columns
    want = [c for c in ["provenance", "support", "y_risk", "lds_weight", "split", "size"]
            if c in cols]
    df = pd.read_csv(path, usecols=want)
    n = len(df)
    ymax = float(df["y_risk"].max())
    n_none = int((df["provenance"] == "NONE").sum()) if "provenance" in df else 0
    print(f"rows={n:,}  y_risk: min={df['y_risk'].min():.3f} max={ymax:.3f}  "
          f"NONE(risk0)={n_none:,}")

    bins = np.arange(0, math.ceil(ymax * 2) / 2 + 1e-9, 0.5)

    def hist(d: pd.DataFrame, title: str) -> None:
        if not len(d):
            return
        yb = pd.cut(d["y_risk"], bins=bins, include_lowest=True)
        agg = {"n": ("y_risk", "size")}
        if "lds_weight" in d:
            agg["w"] = ("lds_weight", "sum")
        g = d.groupby(yb, observed=False).agg(**agg)
        npc = 100 * g["n"] / max(1, g["n"].sum())
        wpc = 100 * g["w"] / max(1e-9, g["w"].sum()) if "w" in g else npc * 0
        print(f"\n[{title}] n={len(d):,}")
        print(f"{'y_risk bin':>13} {'count':>12} {'n%':>6} {'w%(LDS)':>8}  bar:n%(.) w%(#)")
        for iv in g.index:
            print(f"{str(iv):>13} {int(g['n'][iv]):>12,} {npc[iv]:>6.2f} {wpc[iv]:>8.2f}  "
                  + "." * int(npc[iv] / 2) + "#" * int(wpc[iv] / 2))

    hist(df, "ALL")
    if "split" in df:
        for sp in ["train", "test"]:
            hist(df[df["split"] == sp], f"split={sp}")
    if "provenance" in df:
        for pv in df["provenance"].dropna().unique():
            hist(df[df["provenance"] == pv], f"provenance={pv}")

    print("\n[support 別] 上位:")
    for s, c in df["support"].value_counts().head(10).items():
        print(f"  support={int(s):>8}  y={ymax - math.log10(max(1, s)):.3f}  "
              f"n={c:,} ({100 * c / n:.2f}%)")
    print(f"\n  support=1(ユニーク, y={ymax:.2f}): {int((df['support'] == 1).sum()):,} "
          f"({100 * (df['support'] == 1).mean():.2f}%)")
    if "split" in df:
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "test"]
        print(f"    train: support=1 {100 * (tr['support'] == 1).mean():.2f}%  "
              f"(n={len(tr):,})")
        print(f"    test : support=1 {100 * (te['support'] == 1).mean():.2f}%  "
              f"(n={len(te):,})  ← 自然分布(評価用)")

    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tr = df[df["split"] == "train"] if "split" in df else df
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(tr["y_risk"], bins=60, alpha=.5, label="train unweighted", density=True)
        if "lds_weight" in tr:
            ax.hist(tr["y_risk"], bins=60, weights=tr["lds_weight"], alpha=.5,
                    label="train LDS-weighted", density=True)
        if "split" in df:
            ax.hist(df[df["split"] == "test"]["y_risk"], bins=60, histtype="step",
                    label="test (natural)", density=True, color="k")
        ax.set_xlabel("y_risk = log10(N/support)")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title("Re-identification risk distribution (span)")
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        print(f"\n  PNG → {png}")


if __name__ == "__main__":
    main()
