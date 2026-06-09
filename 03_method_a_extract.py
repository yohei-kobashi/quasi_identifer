"""Step 3 – Method A: Morpheme-based statistical extraction.

Tokens that appear significantly more often in PROFILE clauses than in NONE
clauses (Fisher's exact test, Bonferroni-corrected p < ALPHA) are QPII candidates.

Design
------
The expensive part (tokenisation + Fisher's test) is run ONCE at the most lenient
threshold (α=0.10).  All passing tokens are saved WITH their statistics so that
tighter thresholds (α=0.05, α=0.01) can be explored by simple CSV filtering,
with no recomputation required.

Output files
------------
  data/03_method_a_stats.csv          full stats for every token that passes
                                       Bonferroni-corrected α=0.10  (run once)
  data/03_method_a_candidates.csv     tokens filtered at config ALPHA (default)

Usage
-----
  # First run – compute everything and save stats + default candidates
  python 03_method_a_extract.py

  # Re-filter from saved stats at a different alpha (fast, no recomputation)
  python 03_method_a_extract.py --filter 0.01
  python 03_method_a_extract.py --filter 0.05
  python 03_method_a_extract.py --filter 0.10

  # Print comparison table across all three alphas (reads saved stats)
  python 03_method_a_extract.py --compare
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from scipy.stats import fisher_exact
from tqdm import tqdm

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    sys.path.insert(0, str(Path.cwd()))
from config import ALPHA, DATA_DIR, KEEP_POS_PREFIXES

INPUT_PATH: Path  = DATA_DIR / "02_labeled_clauses.csv"
STATS_PATH: Path  = DATA_DIR / "03_method_a_stats.csv"   # full stats (α≤0.10)
OUTPUT_PATH: Path = DATA_DIR / "03_method_a_candidates.csv"

# Most lenient threshold used to build the stats cache
MAX_ALPHA: float = 0.10
COMPARE_ALPHAS: list[float] = [0.01, 0.05, 0.10]


# ── NLTK helpers ──────────────────────────────────────────────────────────────

def download_nltk_data() -> None:
    for resource in ("punkt_tab", "averaged_perceptron_tagger_eng", "stopwords"):
        nltk.download(resource, quiet=True)


def tokenize_and_tag(text: str) -> list[tuple[str, str]]:
    return nltk.pos_tag(nltk.word_tokenize(text.lower()))


def is_content_token(token: str, pos: str) -> bool:
    if not token.isalpha() or len(token) < 3:
        return False
    if not any(pos.startswith(p) for p in KEEP_POS_PREFIXES):
        return False
    return token not in nltk.corpus.stopwords.words("english")


def count_tokens_in_clauses(
    clauses: list[str],
) -> tuple[Counter[str], dict[str, str]]:
    """Return (token freq Counter, token→POS mapping)."""
    freq: Counter[str] = Counter()
    pos_map: dict[str, str] = {}
    for clause in tqdm(clauses, desc="  Tokenizing", leave=False):
        for token, pos in tokenize_and_tag(clause):
            if is_content_token(token, pos):
                freq[token] += 1
                pos_map.setdefault(token, pos)
    return freq, pos_map


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    """Odds ratio with add-0.5 smoothing."""
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


# ── Core computation ──────────────────────────────────────────────────────────

def build_stats_cache() -> pd.DataFrame:
    """Run tokenisation + Fisher's test once and save full stats to STATS_PATH.

    Only tokens that pass Bonferroni-corrected α=MAX_ALPHA are kept,
    so the cache is compact while still covering all useful thresholds.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    profile_texts: list[str] = df[df["label"] == "PROFILE"]["clause"].tolist()
    none_texts: list[str]    = df[df["label"] == "NONE"]["clause"].tolist()
    print(f"PROFILE clauses : {len(profile_texts)}")
    print(f"NONE clauses    : {len(none_texts)}")

    print("Counting PROFILE tokens ...")
    freq_profile, pos_map = count_tokens_in_clauses(profile_texts)
    print("Counting NONE tokens ...")
    freq_none, pos_map_none = count_tokens_in_clauses(none_texts)
    pos_map.update(pos_map_none)

    n_profile = len(profile_texts)
    n_none    = len(none_texts)
    all_tokens: set[str] = set(freq_profile.keys()) | set(freq_none.keys())
    n_tokens = len(all_tokens)
    corrected_max = MAX_ALPHA / max(n_tokens, 1)

    print(f"Running Fisher's exact test on {n_tokens} tokens ...")
    rows: list[dict] = []
    for token in tqdm(all_tokens, desc="Fisher's exact test"):
        a = freq_profile[token]
        if a == 0:
            continue
        b = n_profile - a
        c = freq_none[token]
        d = n_none - c
        _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        if p_value < corrected_max:
            rows.append({
                "token":        token,
                "pos":          pos_map.get(token, ""),
                "freq_profile": a,
                "freq_none":    c,
                "p_value":      p_value,
                "effect_size":  odds_ratio(a, b, c, d),
                "n_tokens":     n_tokens,   # store for correct Bonferroni later
            })

    df_stats = (
        pd.DataFrame(rows)
        .sort_values("effect_size", ascending=False)
        .reset_index(drop=True)
    )
    df_stats.to_csv(STATS_PATH, index=False)
    print(f"Stats cache saved ({len(df_stats)} tokens, α≤{MAX_ALPHA}) → {STATS_PATH}")
    return df_stats


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_at_alpha(df_stats: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Apply Bonferroni correction for the given alpha and return candidates."""
    n_tokens = int(df_stats["n_tokens"].iloc[0])
    corrected = alpha / max(n_tokens, 1)
    df_pass = df_stats[df_stats["p_value"] < corrected]
    if df_pass.empty:
        print(f"  No tokens pass Bonferroni-corrected α={corrected:.2e}. "
              f"Falling back to uncorrected α={alpha}.")
        df_pass = df_stats[df_stats["p_value"] < alpha]
    return df_pass.sort_values("effect_size", ascending=False).reset_index(drop=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", type=float, metavar="ALPHA",
        help="Re-filter from saved stats at this alpha (e.g. 0.01). "
             "Skips expensive recomputation."
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Print comparison table across α=0.01/0.05/0.10 using saved stats."
    )
    args, _ = parser.parse_known_args()

    download_nltk_data()

    # ── Load or build stats cache ─────────────────────────────────────────
    if args.filter or args.compare:
        if not STATS_PATH.exists():
            print(f"Stats cache not found. Running full extraction first ...")
            df_stats = build_stats_cache()
        else:
            df_stats = pd.read_csv(STATS_PATH)
            print(f"Loaded stats cache: {len(df_stats)} tokens → {STATS_PATH}")
    else:
        print("Building stats cache (α≤0.10) ...")
        df_stats = build_stats_cache()

    # ── Mode: --filter ALPHA ──────────────────────────────────────────────
    if args.filter:
        alpha = args.filter
        df_out = filter_at_alpha(df_stats, alpha)
        label = str(alpha).replace(".", "")
        out_path = DATA_DIR / f"03_method_a_alpha{label}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\nα={alpha} → {len(df_out)} candidates")
        print(df_out.head(20).to_string(index=False))
        print(f"\nSaved → {out_path}")
        return

    # ── Mode: --compare ───────────────────────────────────────────────────
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

    # ── Default: run with config ALPHA ────────────────────────────────────
    df_out = filter_at_alpha(df_stats, ALPHA)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCandidates found (α={ALPHA}) : {len(df_out)}")
    print(df_out.head(20).to_string(index=False))
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
