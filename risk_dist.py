#!/usr/bin/env python
"""07_reid_samples.csv の再識別リスク y_risk=log10(N/support) の分布を確認する。

使い方:
    python risk_dist.py [CSV] [PNG]
        CSV: 既定 data/07_reid_samples.csv
        PNG: 第2引数を渡すと unweighted/LDS-weighted の密度ヒストグラムを保存

n%(生の件数割合) と w%(LDS 重み後) を並べて表示し、LDS がどれだけ分布を
平坦化したかを見る。support=1(=データ中で一意 → 最大リスク)の割合にも注目。
"""
import sys
import math

import numpy as np
import pandas as pd


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/07_reid_samples.csv"
    png = sys.argv[2] if len(sys.argv) > 2 else ""

    df = pd.read_csv(
        path,
        usecols=["unit_type", "provenance", "support", "y_risk", "lds_weight"],
        dtype={"unit_type": "category", "provenance": "category",
               "support": "int64", "y_risk": "float32", "lds_weight": "float32"},
    )
    n = len(df)
    ymax = float(df["y_risk"].max())
    print(f"rows={n:,}  y_risk: min={df['y_risk'].min():.3f} max={ymax:.3f}  "
          f"NONE(risk0)={int((df['provenance'] == 'NONE').sum()):,}")

    bins = np.arange(0, math.ceil(ymax * 2) / 2 + 1e-9, 0.5)

    def hist(d: pd.DataFrame, title: str) -> None:
        yb = pd.cut(d["y_risk"], bins=bins, include_lowest=True)
        g = d.groupby(yb, observed=False).agg(n=("y_risk", "size"), w=("lds_weight", "sum"))
        npc = 100 * g["n"] / max(1, g["n"].sum())
        wpc = 100 * g["w"] / max(1e-9, g["w"].sum())
        print(f"\n[{title}] n={len(d):,}")
        print(f"{'y_risk bin':>13} {'count':>12} {'n%':>6} {'w%(LDS)':>8}  bar:n%(.) w%(#)")
        for iv in g.index:
            print(f"{str(iv):>13} {int(g['n'][iv]):>12,} {npc[iv]:>6.2f} {wpc[iv]:>8.2f}  "
                  + "." * int(npc[iv] / 2) + "#" * int(wpc[iv] / 2))

    hist(df, "ALL")
    for ut in df["unit_type"].cat.categories:
        hist(df[df["unit_type"] == ut], f"unit_type={ut}")

    print("\n[support 別] 上位:")
    for s, c in df["support"].value_counts().head(10).items():
        print(f"  support={int(s):>8}  y={ymax - math.log10(s):.3f}  n={c:,} ({100 * c / n:.2f}%)")
    print(f"\n  support=1(ユニーク, y={ymax:.2f}): {int((df['support'] == 1).sum()):,} "
          f"({100 * (df['support'] == 1).mean():.2f}%)")

    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["y_risk"], bins=60, alpha=.5, label="unweighted", density=True)
        ax.hist(df["y_risk"], bins=60, weights=df["lds_weight"], alpha=.5,
                label="LDS-weighted", density=True)
        ax.set_xlabel("y_risk = log10(N/support)")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title("Re-identification risk distribution")
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        print(f"\n  PNG → {png}")


if __name__ == "__main__":
    main()
