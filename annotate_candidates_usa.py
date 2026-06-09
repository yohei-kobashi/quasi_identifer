#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate Nemotron USA persona text clauses with PII/PROFILE/NONE labels.

Processing summary:
1. Load `nvidia/Nemotron-Personas-USA` from Hugging Face and choose a split.
2. Resolve persona-like text fields from each row and normalize them.
3. Split each English text into smaller clauses using dependency parsing when spaCy
   is available, otherwise fall back to a rule-based splitter.
4. Send each clause to the OpenAI API for 3-way classification: `PII` / `PROFILE` / `NONE`.
5. Skip clauses already present in the output JSONL based on `(uuid, clause)`.
6. Append `row_index`, `uuid`, `field`, `clause`, `label` records to the output file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from itertools import islice
from typing import Iterable, List, Optional, Sequence

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


SENTENCE_SPLIT_RE = re.compile(r"(?:\r?\n)+|[.!?;]+")
SUBCLAUSE_SPLIT_RE = re.compile(
    r"""
    \s*(?:,\s*)?(?:
        and|but|while|although|though|whereas|however|because|since|when|after|before
    )\s+
    """,
    re.IGNORECASE | re.VERBOSE,
)
LEADING_PUNCT_RE = re.compile(r"^[\s,;:.\-]+")
TRAILING_PUNCT_RE = re.compile(r"[\s,;:.\-]+$")
MULTISPACE_RE = re.compile(r"\s+")

DEFAULT_FIELDS = [
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "cultural_background",
]

FIELD_ALIASES = {
    "skills_and_expertise": ["skills_and_expertise", "skills_and_expertise_list"],
    "hobbies_and_interests": ["hobbies_and_interests", "hobbies_and_interests_list"],
}

AUTO_FIELD_EXCLUDE = {
    "uuid",
    "id",
    "index",
    "row_index",
    "prompt",
    "messages",
    "conversation",
    "conversations",
    "source",
    "lang",
    "language",
    "country",
    "region",
    "created_at",
    "updated_at",
}

DEPENDENCY_SPLIT_LABELS = {"conj", "advcl", "relcl", "appos", "parataxis"}
MIN_CLAUSE_TOKENS = 3
MAX_CLAUSE_TOKENS = 18
RULE_BASED_SECONDARY_SPLIT_THRESHOLD = 18


def normalize_field(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = item.strip() if isinstance(item, str) else str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def normalize_clause_text(text: str) -> str:
    text = LEADING_PUNCT_RE.sub("", text)
    text = TRAILING_PUNCT_RE.sub("", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def split_long_clause_rule_based(text: str) -> List[str]:
    parts = [normalize_clause_text(part) for part in SUBCLAUSE_SPLIT_RE.split(text)]
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return [text]
    return dedupe_preserve_order(parts)


def fallback_split_clauses(text: str) -> List[str]:
    sentences = [normalize_clause_text(s) for s in SENTENCE_SPLIT_RE.split(text)]
    sentences = [s for s in sentences if s]

    clauses: List[str] = []
    for sentence in sentences:
        if word_count(sentence) > RULE_BASED_SECONDARY_SPLIT_THRESHOLD:
            clauses.extend(split_long_clause_rule_based(sentence))
        else:
            clauses.append(sentence)
    return dedupe_preserve_order(clauses)


class ClauseSplitter:
    def split(self, text: str) -> List[str]:
        raise NotImplementedError


class RuleBasedClauseSplitter(ClauseSplitter):
    def split(self, text: str) -> List[str]:
        return fallback_split_clauses(text)


class SpacyClauseSplitter(ClauseSplitter):
    def __init__(self, nlp) -> None:
        self._nlp = nlp

    def split(self, text: str) -> List[str]:
        doc = self._nlp(text)
        clauses: List[str] = []

        for sent in doc.sents:
            sent_clauses = self._split_sentence(sent)
            clauses.extend(sent_clauses)

        clauses = [normalize_clause_text(c) for c in clauses]
        clauses = [c for c in clauses if c]
        if not clauses:
            return fallback_split_clauses(text)
        return dedupe_preserve_order(clauses)

    def _split_sentence(self, sent) -> List[str]:
        spans = []
        covered = set()

        for token in sent:
            if token.dep_ not in DEPENDENCY_SPLIT_LABELS:
                continue
            if not self._is_good_split_candidate(token, sent):
                continue

            span_tokens = [t for t in token.subtree if t.i >= sent.start and t.i < sent.end]
            if not span_tokens:
                continue

            start = span_tokens[0].i
            end = span_tokens[-1].i + 1
            spans.append((start, end))
            covered.update(range(start, end))

        spans.sort()

        base_tokens = [t for t in sent if t.i not in covered and not t.is_punct]
        base_text = normalize_clause_text(" ".join(t.text for t in base_tokens))

        clauses: List[str] = []
        if base_text:
            clauses.extend(self._maybe_resplit(base_text))

        for start, end in spans:
            span = sent.doc[start:end]
            span_text = normalize_clause_text(span.text)
            if not span_text:
                continue
            clauses.extend(self._maybe_resplit(span_text))

        if not clauses:
            return self._maybe_resplit(normalize_clause_text(sent.text))
        return clauses

    def _is_good_split_candidate(self, token, sent) -> bool:
        if token.is_punct:
            return False
        if token.dep_ == "conj":
            return token.pos_ in {"VERB", "AUX", "ADJ", "NOUN", "PROPN"}
        if token.dep_ == "appos":
            return token.pos_ in {"NOUN", "PROPN"}
        if token.dep_ in {"advcl", "relcl", "parataxis"}:
            return True
        return False

    def _maybe_resplit(self, text: str) -> List[str]:
        if word_count(text) <= MAX_CLAUSE_TOKENS:
            return [text]
        return split_long_clause_rule_based(text)


def build_clause_splitter(spacy_model: str) -> ClauseSplitter:
    try:
        import spacy
    except ImportError:
        return RuleBasedClauseSplitter()

    try:
        nlp = spacy.load(spacy_model, disable=["ner"])
    except Exception:
        return RuleBasedClauseSplitter()

    if "parser" not in nlp.pipe_names or not nlp.has_pipe("parser"):
        return RuleBasedClauseSplitter()

    return SpacyClauseSplitter(nlp)


def split_clauses(text: str, splitter: ClauseSplitter) -> List[str]:
    clauses = splitter.split(text)
    filtered = [c for c in clauses if word_count(c) >= MIN_CLAUSE_TOKENS]
    return filtered or clauses


def _candidate_field_names(field: str) -> List[str]:
    aliases = FIELD_ALIASES.get(field)
    if aliases:
        return aliases
    if field.endswith("_list"):
        return [field, field[: -len("_list")]]
    return [field, f"{field}_list"]


def resolve_present_fields(sample_row: dict, requested_fields: Sequence[str]) -> List[str]:
    present: List[str] = []
    for field in requested_fields:
        for candidate in _candidate_field_names(field):
            if candidate in sample_row:
                present.append(candidate)
                break
    return present


def auto_detect_text_fields(sample_row: dict) -> List[str]:
    detected: List[str] = []
    for key, value in sample_row.items():
        if key in AUTO_FIELD_EXCLUDE:
            continue
        values = normalize_field(value)
        if not values:
            continue
        if any(len(v) >= 16 and (" " in v or len(v.split()) >= 2) for v in values):
            detected.append(key)
    return detected


def iter_clauses(dataset, split: str, fields: Sequence[str], limit: int, max_rows: int, splitter: ClauseSplitter):
    ds = dataset[split]
    if max_rows and max_rows > 0:
        ds = islice(ds, max_rows)

    iterator = iter(ds)
    try:
        first_row = next(iterator)
    except StopIteration:
        return

    active_fields = resolve_present_fields(first_row, fields)
    if not active_fields:
        active_fields = auto_detect_text_fields(first_row)
    if not active_fields:
        raise RuntimeError(
            "No text fields were found in the dataset row. "
            "Pass explicit fields with --fields if the schema is unusual."
        )

    def row_iter():
        yield 0, first_row
        for idx, row in enumerate(iterator, start=1):
            yield idx, row

    count = 0
    for idx, row in row_iter():
        uuid = row.get("uuid")
        for field in active_fields:
            for value in normalize_field(row.get(field)):
                for clause in split_clauses(value, splitter):
                    yield idx, uuid, field, clause
                    count += 1
                    if limit and count >= limit:
                        return


def classify_clause(client: OpenAI, model: str, clause: str, sleep_s: float, temperature) -> str:
    prompt = (
        "Output must be exactly one of: PII, PROFILE, NONE.\n"
        "Do not add reasons or extra text.\n\n"
        "Label definitions:\n"
        "- PII: Information that can identify a specific person directly, such as full names, contact details, exact identifiers, usernames, or precise addresses.\n"
        "- PROFILE: Quasi-identifiers that describe a person's background or attributes and may help narrow identity, such as occupation, hobbies, region, education, family role, or cultural background.\n"
        "- NONE: Information not useful for identifying a person, such as emotions, abstract values, generic preferences without profile value, or broad narrative filler.\n\n"
        "If uncertain, choose the label that is more useful for identification.\n\n"
        "Examples\n"
        "Phrase: My name is Olivia Chen.\n"
        "PII\n"
        "Phrase: I grew up in rural Ohio.\n"
        "PROFILE\n"
        "Phrase: I work as a pediatric nurse.\n"
        "PROFILE\n"
        "Phrase: I try to stay optimistic when life gets difficult.\n"
        "NONE\n"
        "Phrase: You can reach me at daniel.rivera92@gmail.com.\n"
        "PII\n"
        "Phrase: I spend most weekends hiking and photographing birds.\n"
        "PROFILE\n"
        "Phrase: I care deeply about being kind to other people.\n"
        "NONE\n\n"
        "Input\n"
        f"Phrase: {clause}\n"
    )
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict classifier for short English clauses."},
            {"role": "user", "content": prompt},
        ],
    }
    if temperature is not None:
        params["temperature"] = temperature
    resp = client.chat.completions.create(**params)
    label = resp.choices[0].message.content.strip()
    if sleep_s:
        time.sleep(sleep_s)
    return label


def parse_fields(fields_arg: str) -> List[str]:
    fields = [field.strip() for field in fields_arg.split(",") if field.strip()]
    return fields or list(DEFAULT_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate Nemotron USA persona clauses with PII/PROFILE/NONE."
    )
    parser.add_argument("--dataset", default="nvidia/Nemotron-Personas-USA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit (clauses)")
    parser.add_argument("--max_rows", type=int, default=2000, help="0 means no limit (rows)")
    parser.add_argument("--streaming", action="store_true", help="stream dataset to avoid full download")
    parser.add_argument("--output", default="annotations_usa.jsonl")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument(
        "--spacy-model",
        default=os.getenv("SPACY_MODEL", "en_core_web_sm"),
        help="spaCy English pipeline used for dependency-based clause splitting.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="If omitted, do not send temperature (model default).",
    )
    parser.add_argument("--sleep", type=float, default=0.1, help="sleep seconds between requests")
    args = parser.parse_args()

    fields = parse_fields(args.fields)
    splitter = build_clause_splitter(args.spacy_model)
    dataset = load_dataset(args.dataset, streaming=args.streaming)
    if args.split not in dataset:
        args.split = list(dataset.keys())[0]
    client = OpenAI()

    annotated_clauses = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as existing:
            for row in existing:
                data = json.loads(row)
                annotated_clauses.add((data["uuid"], data["clause"]))

    with open(args.output, "a", encoding="utf-8") as f:
        for idx, uuid, field, clause in tqdm(
            iter_clauses(dataset, args.split, fields, args.limit, args.max_rows, splitter)
        ):
            if (uuid, clause) in annotated_clauses:
                continue
            label = classify_clause(client, args.model, clause, args.sleep, args.temperature)
            record = {
                "row_index": idx,
                "uuid": uuid,
                "field": field,
                "clause": clause,
                "label": label,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
