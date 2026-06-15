#!/usr/bin/env python
"""再識別リスク回帰モデルの評価(不均衡回帰向けの層別メトリクス)。

目的: support=1(y=6) が多数を占める自然分布で、単純な MAE/RMSE は「常に最大値を
予測する自明モデル」を過大評価してしまう。そこで y_true の層ごとに誤差を出し、
その macro 平均(各層を等価値とみなす)を主指標にする。識別力は Spearman と
unicity AUC で別評価する。

使い方:
    python eval_risk.py PRED.csv
        PRED.csv に必要な列: y_true, y_pred
        任意列:
          split   -> あれば split=="test" の行のみで評価(自然分布の固定テスト)
          support -> ベースライン(support から y を逆算)と整合チェックに使用

主な出力:
  - 全体 MAE/RMSE(自然分布)
  - 層別(y_true のビン)MAE/RMSE/件数  と  macro 平均(層を等重み)
  - Spearman 順位相関(識別力 / 校正に依らない)
  - unicity AUC: 「y_true==ymax(=ユニーク)」を y_pred で当てる二値 AUC
  - 自明ベースライン(平均予測 / 常に ymax / support→y)の同指標
"""
import sys

import numpy as np
import pandas as pd


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def auc_binary(score: np.ndarray, pos: np.ndarray) -> float:
    """正例 pos(bool) を score で順位付けしたときの ROC-AUC(Mann-Whitney U / rank法)。"""
    pos = pos.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # 同点は平均順位に補正
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def strata(y_true: np.ndarray, ymax: float) -> "list[tuple[str, np.ndarray]]":
    """y_true を層に分割。NONE(=0)、(0,ymax) を 1.0 幅、ymax(=ユニーク)を独立層に。"""
    out: list[tuple[str, np.ndarray]] = []
    out.append(("NONE(y=0)", np.isclose(y_true, 0.0)))
    edges = np.arange(0.0, ymax, 1.0)  # 0,1,2,...,<ymax
    for lo in edges:
        hi = lo + 1.0
        if lo == 0.0:
            m = (y_true > 0.0) & (y_true < hi)
            lab = f"(0,{hi:.0f})"
        elif hi >= ymax:
            m = (y_true >= lo) & (y_true < ymax)
            lab = f"[{lo:.0f},{ymax:.2f})"
        else:
            m = (y_true >= lo) & (y_true < hi)
            lab = f"[{lo:.0f},{hi:.0f})"
        if m.any():
            out.append((lab, m))
    out.append((f"UNIQUE(y={ymax:.2f})", np.isclose(y_true, ymax)))
    return out


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray, ymax: float) -> None:
    print(f"\n===== {name} =====")
    print(f"  全体: MAE={mae(y_true, y_pred):.4f}  RMSE={rmse(y_true, y_pred):.4f}  "
          f"n={len(y_true):,}")
    print(f"  {'stratum':>16} {'n':>10} {'MAE':>8} {'RMSE':>8} "
          f"{'mean_true':>9} {'mean_pred':>9}")
    smae, srmse = [], []
    for lab, m in strata(y_true, ymax):
        n = int(m.sum())
        if n == 0:
            continue
        a, b = y_true[m], y_pred[m]
        em, er = mae(a, b), rmse(a, b)
        smae.append(em)
        srmse.append(er)
        print(f"  {lab:>16} {n:>10,} {em:>8.4f} {er:>8.4f} "
              f"{a.mean():>9.4f} {b.mean():>9.4f}")
    print(f"  --> macro(層等重み): MAE={np.mean(smae):.4f}  RMSE={np.mean(srmse):.4f}  "
          f"({len(smae)} 層)")
    # 識別力(定数予測のときは Spearman 未定義 → nan。警告は抑制)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")
    auc = auc_binary(y_pred, np.isclose(y_true, ymax))
    print(f"  Spearman={sp:.4f}   unicity-AUC(y==ymax)={auc:.4f}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    for c in ("y_true", "y_pred"):
        if c not in df.columns:
            raise SystemExit(f"必要な列がありません: {c}")
    if "split" in df.columns:
        before = len(df)
        df = df[df["split"] == "test"].copy()
        print(f"split 列あり → test {len(df):,}/{before:,} 行で評価(自然分布)")
    yt = df["y_true"].to_numpy(dtype=float)
    yp = df["y_pred"].to_numpy(dtype=float)
    ymax = float(np.nanmax(yt))
    print(f"y_true: min={yt.min():.3f} max={ymax:.3f}  n={len(yt):,}")

    report("model", yt, yp, ymax)

    # ---- 自明ベースライン ----
    report("baseline: 常に平均", yt, np.full_like(yt, yt.mean()), ymax)
    report("baseline: 常に ymax", yt, np.full_like(yt, ymax), ymax)
    if "support" in df.columns:
        N = float(np.power(10.0, ymax))  # ymax=log10(N)
        sup = np.maximum(1.0, df["support"].to_numpy(dtype=float))
        y_from_sup = np.where(np.isclose(yt, 0.0), 0.0, np.log10(N / sup))
        report("baseline: support→y(リーク上限)", yt, y_from_sup, ymax)

    print("\n読み方: モデルの macro-MAE が『常に平均』『常に ymax』より十分小さく、"
          "Spearman/AUC が高ければ、多数派(ユニーク)に引きずられず識別できている。")


if __name__ == "__main__":
    main()
