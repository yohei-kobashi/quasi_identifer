#!/usr/bin/env python
"""07 の学習ターゲット分布(y_risk / y_combined / y_bits)を診断する。

特に support=1 の塊が y_combined で 6 超へどう段階化されたか(6付近に圧縮 vs 十分広がり)、
provenance(PROFILE/PII/NONE)・support 領域(>=2 / =1)別の分布を見る。これを基に、
回帰ターゲットに対数/順位変換が要るか、PII の ceiling 位置が妥当かを判断する。

使い方:
    python diag_targets.py [CSV] [PNG]
        既定 CSV = data/07_reid_samples.csv
        PNG を渡すと y_combined(全体, 対数軸)と support=1 部分のヒストグラムを保存
"""
import sys
import math

import numpy as np
import pandas as pd


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/07_reid_samples.csv"
    png = sys.argv[2] if len(sys.argv) > 2 else ""

    cols = pd.read_csv(path, nrows=0).columns
    want = [c for c in ["provenance", "support", "y_risk", "y_bits", "y_combined",
                        "split", "size"] if c in cols]
    df = pd.read_csv(path, usecols=want, dtype={
        "provenance": "category", "support": "int64", "y_risk": "float32",
        "y_bits": "float32", "y_combined": "float32", "split": "category",
        "size": "int32"})
    n = len(df)
    print(f"rows={n:,}  cols={want}")
    if "y_combined" not in df:
        print("[!] y_combined 列がありません。sample-spans を再実行してください。")
        return

    def stats(name: str) -> None:
        s = df[name].to_numpy(dtype=float)
        qs = np.percentile(s, [50, 90, 99, 99.9])
        print(f"  {name:11}: min={s.min():.3f} median={qs[0]:.3f} p90={qs[1]:.3f} "
              f"p99={qs[2]:.3f} p99.9={qs[3]:.3f} max={s.max():.3f}")

    print("\n[ターゲット統計]")
    for c in ["y_risk", "y_combined", "y_bits"]:
        if c in df:
            stats(c)

    print("\n[provenance 内訳]")
    for pv, c in df["provenance"].value_counts().items():
        print(f"  {pv:8}: {c:>12,} ({100 * c / n:5.2f}%)")

    ymax = float(df["y_combined"].max())
    edges = np.concatenate([np.arange(0, 6, 0.5),
                            np.arange(6, math.ceil(ymax / 2) * 2 + 2, 2)])

    def hist(name: str, mask=None, title: str = "") -> None:
        d = df[name] if mask is None else df.loc[mask, name]
        m = len(d)
        if m == 0:
            return
        cnt, _ = np.histogram(d.to_numpy(dtype=float), bins=edges)
        print(f"\n[{name} 分布 {title}] n={m:,}")
        for i, c in enumerate(cnt):
            if c == 0:
                continue
            lo, hi = edges[i], edges[i + 1]
            print(f"  [{lo:5.1f},{hi:5.1f}) {c:>12,} ({100 * c / m:5.2f}%) "
                  + "#" * int(50 * c / m))

    prof = (df["provenance"] == "PROFILE").to_numpy()
    sup1 = (df["support"] == 1).to_numpy()
    hist("y_combined", None, "ALL")
    hist("y_combined", prof & ~sup1, "PROFILE support>=2(経験的 y_risk)")
    hist("y_combined", prof & sup1, "PROFILE support=1(y_bits で段階化)")
    hist("y_combined", (df["provenance"] == "PII").to_numpy(), "PII(最大リスク=ceiling)")

    # support=1 の広がり(6付近圧縮か)
    s1 = df.loc[prof & sup1, "y_combined"].to_numpy(dtype=float)
    if len(s1):
        qs = np.percentile(s1, [10, 25, 50, 75, 90, 99])
        print(f"\n[PROFILE support=1 の y_combined 広がり] n={len(s1):,}")
        print(f"  p10={qs[0]:.2f} p25={qs[1]:.2f} median={qs[2]:.2f} "
              f"p75={qs[3]:.2f} p90={qs[4]:.2f} p99={qs[5]:.2f} max={s1.max():.2f}")
        print(f"  <7 の割合: {100 * (s1 < 7).mean():.1f}%  "
              f"(高いほど 6 付近に圧縮 → 対数/順位変換を検討)")

    if "split" in df:
        print("\n[split 別 y_combined]")
        for sp in ["train", "test"]:
            d = df.loc[df["split"] == sp, "y_combined"]
            if len(d):
                print(f"  {sp}: n={len(d):,} median={d.median():.3f} "
                      f"(>=6 の割合={100 * (d >= 6).mean():.1f}%)")

    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(df["y_combined"], bins=80)
        ax[0].set_yscale("log")
        ax[0].set_title("y_combined ALL (log y)")
        ax[0].set_xlabel("y_combined")
        if len(s1):
            ax[1].hist(s1, bins=80, color="C1")
            ax[1].set_title("y_combined (PROFILE support=1)")
            ax[1].set_xlabel("y_combined")
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        print(f"\nPNG → {png}")


if __name__ == "__main__":
    main()
