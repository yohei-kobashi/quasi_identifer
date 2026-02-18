import argparse
import csv
import math
from typing import List, Tuple


def rankdata(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank
        i = j
    return ranks


def pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 rows")

    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def load_columns(csv_path: str, x_col: str, y_col: str) -> Tuple[List[float], List[float], int]:
    x_vals: List[float] = []
    y_vals: List[float] = []
    skipped = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV header is missing")
        if x_col not in reader.fieldnames:
            raise ValueError(f"Column '{x_col}' not found")
        if y_col not in reader.fieldnames:
            raise ValueError(f"Column '{y_col}' not found")

        for row in reader:
            sx = (row.get(x_col) or "").strip()
            sy = (row.get(y_col) or "").strip()
            if not sx or not sy:
                skipped += 1
                continue
            try:
                xv = float(sx)
                yv = float(sy)
            except ValueError:
                skipped += 1
                continue
            if math.isnan(xv) or math.isnan(yv):
                skipped += 1
                continue
            x_vals.append(xv)
            y_vals.append(yv)

    return x_vals, y_vals, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Spearman correlation from CSV columns")
    parser.add_argument("--csv", default="data_with_pred.csv", help="Input CSV path")
    parser.add_argument("--x-col", default="is_qpii", help="First column")
    parser.add_argument("--y-col", default="qpii_score_raw", help="Second column")
    args = parser.parse_args()

    x_vals, y_vals, skipped = load_columns(args.csv, args.x_col, args.y_col)
    if len(x_vals) < 2:
        raise ValueError("Not enough valid rows to compute correlation")

    rx = rankdata(x_vals)
    ry = rankdata(y_vals)
    rho = pearson(rx, ry)

    print(f"csv={args.csv}")
    print(f"x_col={args.x_col}, y_col={args.y_col}")
    print(f"used_rows={len(x_vals)}, skipped_rows={skipped}")
    print(f"spearman_rho={rho:.6f}")


if __name__ == "__main__":
    main()
