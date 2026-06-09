"""Step 4 – Method B: Expand candidate tokens to longest enclosing noun phrase.

For each candidate token from Method A, scan PROFILE clauses and extract the
longest NP that contains the token using an NLTK RegexpParser grammar.

Reads:  data/03_method_a_candidates.csv
        data/02_labeled_clauses.csv
Output: data/04_method_b_candidates.csv
Columns: phrase, head_token, freq, example_clause_1/2/3
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import pandas as pd
from tqdm import tqdm

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    sys.path.insert(0, str(Path.cwd()))
from config import DATA_DIR

INPUT_TOKENS: Path  = DATA_DIR / "03_method_a_candidates.csv"
INPUT_CLAUSES: Path = DATA_DIR / "02_labeled_clauses.csv"
OUTPUT_PATH: Path   = DATA_DIR / "04_method_b_candidates.csv"

# Allows "graduate of MIT", "nurse at a hospital", "senior walking club" etc.
NP_GRAMMAR: str = r"""
  NP: {<DT>?<JJ.*>*<NN.*>+(<IN><DT>?<JJ.*>*<NN.*>+)*}
"""


def download_nltk_data() -> None:
    for resource in ("punkt_tab", "averaged_perceptron_tagger_eng"):
        nltk.download(resource, quiet=True)


def extract_nps_from_clause(clause: str) -> list[tuple[str, str]]:
    """Return list of (np_string, head_noun) from a clause.

    head_noun = last NN* token in the NP (standard English head-right rule).
    """
    tokens = nltk.word_tokenize(clause)
    tagged: list[tuple[str, str]] = nltk.pos_tag(tokens)
    parser = nltk.RegexpParser(NP_GRAMMAR)
    tree = parser.parse(tagged)

    nps: list[tuple[str, str]] = []
    for subtree in tree.subtrees(lambda t: t.label() == "NP"):
        leaves: list[tuple[str, str]] = subtree.leaves()
        np_str = " ".join(word for word, _ in leaves).lower()
        nn_words = [word.lower() for word, pos in leaves if pos.startswith("NN")]
        if nn_words:
            nps.append((np_str, nn_words[-1]))
    return nps


def find_longest_np_containing(
    token: str,
    nps: list[tuple[str, str]],
) -> tuple[str, str] | None:
    """Return (longest NP string, head_noun) where token appears anywhere in NP.

    Matches token at word boundary (split-based) so "art" does not match "party".
    Returns None if no NP contains the token.
    """
    candidates = [
        (np_str, head) for np_str, head in nps if token in np_str.split()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda x: len(x[0]))


def main() -> None:
    download_nltk_data()

    if not INPUT_TOKENS.exists():
        raise FileNotFoundError(f"Run 03_method_a_extract.py first: {INPUT_TOKENS}")
    if not INPUT_CLAUSES.exists():
        raise FileNotFoundError(f"Run 02_labeling.py first: {INPUT_CLAUSES}")

    df_tokens = pd.read_csv(INPUT_TOKENS)
    candidate_tokens: set[str] = set(df_tokens["token"].tolist())
    print(f"Candidate tokens (Method A): {len(candidate_tokens)}")

    df_clauses = pd.read_csv(INPUT_CLAUSES)
    profile_clauses: list[str] = (
        df_clauses[df_clauses["label"] == "PROFILE"]["clause"].tolist()
    )
    print(f"PROFILE clauses to scan: {len(profile_clauses)}")

    phrase_freq: Counter[str] = Counter()
    phrase_examples: defaultdict[str, list[str]] = defaultdict(list)
    phrase_head: dict[str, str] = {}

    for clause in tqdm(profile_clauses, desc="Extracting NPs"):
        nps = extract_nps_from_clause(clause)
        clause_lower = clause.lower()
        matched_tokens = [t for t in candidate_tokens if t in clause_lower]
        if not matched_tokens:
            continue
        for token in matched_tokens:
            result = find_longest_np_containing(token, nps)
            if result is None:
                longest, head = token, token
            else:
                longest, head = result
            phrase_freq[longest] += 1
            phrase_head[longest] = head
            if len(phrase_examples[longest]) < 3:
                phrase_examples[longest].append(clause)

    rows: list[dict] = []
    for phrase, freq in phrase_freq.most_common():
        if len(phrase.split()) < 2:  # single-word fallbacks are covered by Method A
            continue
        examples = phrase_examples[phrase]
        rows.append({
            "phrase":          phrase,
            "head_token":      phrase_head.get(phrase, ""),
            "freq":            freq,
            "example_clause_1": examples[0] if len(examples) > 0 else "",
            "example_clause_2": examples[1] if len(examples) > 1 else "",
            "example_clause_3": examples[2] if len(examples) > 2 else "",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nUnique phrases : {len(df_out)}")
    print(df_out.head(20)[["phrase", "head_token", "freq"]].to_string(index=False))
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
