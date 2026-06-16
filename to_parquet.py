#!/usr/bin/env python
"""07_reid_samples.csv を Parquet へ変換する(チャンク処理・大規模対応)。

38M 行・text 列を含み数〜十数 GB になるため、CSV をチャンク読みして
pyarrow.ParquetWriter で逐次書き出し、メモリを一定に保つ。列ごとに型を固定
(provenance/text/split 等は string→自動 dictionary 圧縮, y_* は float32)し、
列指向＋圧縮で配布・読込を高速化する。

使い方:
    python to_parquet.py [IN_CSV] [OUT_PARQUET]
                         [--compression zstd|snappy|gzip|none] [--chunksize N]
        既定: data/07_reid_samples.csv → data/07_reid_samples.parquet
              compression=zstd, chunksize=1,000,000
"""
import argparse
import time
from pathlib import Path

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    raise SystemExit("pyarrow が必要です: pip install pyarrow")

# 列 → (pandas 読込 dtype, pyarrow 型)。CSV に在る列だけ採用する。
COLS: "dict[str, tuple[str, object]]" = {
    "provenance":         ("str",     pa.string()),
    "text":               ("str",     pa.string()),
    "qi_json":            ("str",     pa.string()),
    "size":               ("int32",   pa.int32()),
    "support":            ("int64",   pa.int64()),
    "N":                  ("int64",   pa.int64()),
    "y_risk":             ("float32", pa.float32()),
    "y_bits":             ("float32", pa.float32()),
    "y_bits_capped":      ("float32", pa.float32()),
    "min_constituent_df": ("int64",   pa.int64()),
    "y_combined":         ("float32", pa.float32()),
    "lds_weight":         ("float32", pa.float32()),
    "split":              ("str",     pa.string()),
    "tier_strictest":     ("str",     pa.string()),
    "freq":               ("int64",   pa.int64()),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv", nargs="?", default="data/07_reid_samples.csv", type=Path)
    ap.add_argument("out_parquet", nargs="?", default=None, type=Path)
    ap.add_argument("--compression", default="zstd",
                    choices=["zstd", "snappy", "gzip", "none"])
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    in_csv: Path = args.in_csv
    if not in_csv.exists():
        raise SystemExit(f"入力が見つかりません: {in_csv}")
    out: Path = args.out_parquet or in_csv.with_suffix(".parquet")
    comp = None if args.compression == "none" else args.compression

    header = list(pd.read_csv(in_csv, nrows=0).columns)
    use = [c for c in header if c in COLS]
    missing = [c for c in header if c not in COLS]
    if missing:
        print(f"[warn] 未知の列は string として取り込みます: {missing}")
    pd_dtypes = {c: COLS[c][0] for c in use}
    fields = [(c, COLS[c][1]) for c in use]
    fields += [(c, pa.string()) for c in missing]
    schema = pa.schema(fields)
    read_cols = use + missing

    print(f"  入力 : {in_csv}")
    print(f"  出力 : {out}  (compression={args.compression}, chunksize={args.chunksize:,})")
    print(f"  列   : {len(schema.names)}  {schema.names}")

    writer = pq.ParquetWriter(str(out), schema, compression=comp)
    n_rows = 0
    started = time.time()
    reader = pd.read_csv(
        in_csv, usecols=read_cols, dtype=pd_dtypes,
        keep_default_na=False, na_filter=False, chunksize=args.chunksize,
    )
    for ci, chunk in enumerate(reader):
        chunk = chunk[read_cols]                      # 列順を schema に合わせる
        table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
        writer.write_table(table)
        n_rows += len(chunk)
        el = max(1e-9, time.time() - started)
        print(f"    chunk {ci + 1}: 累計 {n_rows:,} 行  ({n_rows / el:,.0f} 行/s)", flush=True)
    writer.close()

    in_sz = in_csv.stat().st_size
    out_sz = out.stat().st_size
    print(f"\n  完了: {n_rows:,} 行 → {out}")
    print(f"  サイズ: CSV {in_sz / 1e9:.2f} GB → Parquet {out_sz / 1e9:.2f} GB "
          f"(圧縮率 {out_sz / max(1, in_sz):.1%})")


if __name__ == "__main__":
    main()
