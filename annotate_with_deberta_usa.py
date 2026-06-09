"""学習済み DeBERTa で nvidia/Nemotron-Personas-USA のテキストをラベル付けする。

train_qpii_bert_usa.py で全データ学習したモデル(既定
outputs/qpii_deberta_usa/final_full_data)を使い、各レコードの対象フィールドを
**学習時と同じ clause 分割**(annotate_candidates_usa.py 由来 / en_core_web_sm の
依存構造解析、無ければルールベース)で節に分け、節ごとに PII/PROFILE/NONE を予測する。

H100 想定の高速化:
  * bf16 + 大きな --batch-size、TF32、pad_to_multiple_of=8
  * 長さでソートしてバッチ内パディングを最小化(--sort-pool)
  * torch.inference_mode + autocast

出力(JSONL, 1行=1節):
  {"row_index","uuid","field","clause","label","score"[,"probs"]}

使い方:
  python annotate_with_deberta_usa.py --batch-size 512
  python annotate_with_deberta_usa.py --model-dir outputs/qpii_deberta_usa/final_full_data \
      --output annotations_pred_usa.jsonl --batch-size 512 --store-probs
  python annotate_with_deberta_usa.py --max-rows 1000   # 動作確認(先頭1000レコード)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ════════════════════════════════════════════════════════════════════════════
# clause 分割(annotate_candidates_usa.py と同一: 学習時の分布に一致させる)
# ════════════════════════════════════════════════════════════════════════════

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

FIELD_ALIASES = {
    "skills_and_expertise": ["skills_and_expertise", "skills_and_expertise_list"],
    "hobbies_and_interests": ["hobbies_and_interests", "hobbies_and_interests_list"],
}

DEPENDENCY_SPLIT_LABELS = {"conj", "advcl", "relcl", "appos", "parataxis"}
MIN_CLAUSE_TOKENS = 3
MAX_CLAUSE_TOKENS = 18
RULE_BASED_SECONDARY_SPLIT_THRESHOLD = 18

DEFAULT_FIELDS = [
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "persona",
    "cultural_background",
    "skills_and_expertise",
    "hobbies_and_interests",
]


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
    parts = [normalize_clause_text(p) for p in SUBCLAUSE_SPLIT_RE.split(text)]
    parts = [p for p in parts if p]
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
            clauses.extend(self._split_sentence(sent))
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
            if not self._is_good_split_candidate(token):
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
            span_text = normalize_clause_text(sent.doc[start:end].text)
            if span_text:
                clauses.extend(self._maybe_resplit(span_text))
        if not clauses:
            return self._maybe_resplit(normalize_clause_text(sent.text))
        return clauses

    def _is_good_split_candidate(self, token) -> bool:
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


def build_clause_splitter(spacy_model: str, use_spacy: bool) -> ClauseSplitter:
    if not use_spacy:
        return RuleBasedClauseSplitter()
    try:
        import spacy
        nlp = spacy.load(spacy_model, disable=["ner"])
    except Exception as exc:
        print(f"[splitter] spaCy 利用不可 ({exc!r}) → ルールベースにフォールバック")
        return RuleBasedClauseSplitter()
    if "parser" not in nlp.pipe_names or not nlp.has_pipe("parser"):
        print("[splitter] spaCy parser 無し → ルールベースにフォールバック")
        return RuleBasedClauseSplitter()
    print(f"[splitter] spaCy ({spacy_model}) で clause 分割")
    return SpacyClauseSplitter(nlp)


def split_clauses(text: str, splitter: ClauseSplitter) -> List[str]:
    clauses = splitter.split(text)
    filtered = [c for c in clauses if word_count(c) >= MIN_CLAUSE_TOKENS]
    return filtered or clauses


def candidate_field_names(field: str) -> List[str]:
    aliases = FIELD_ALIASES.get(field)
    if aliases:
        return aliases
    if field.endswith("_list"):
        return [field, field[: -len("_list")]]
    return [field, f"{field}_list"]


def resolve_field_value(row: dict, field: str):
    for name in candidate_field_names(field):
        if name in row and row[name] is not None:
            return row[name]
    return None


# ════════════════════════════════════════════════════════════════════════════
# 推論
# ════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def classify_clauses(
    clauses: List[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    id2label: Dict[int, str],
    use_amp: bool,
    store_probs: bool,
) -> List[Dict[str, Any]]:
    enc = tokenizer(
        clauses, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt", pad_to_multiple_of=8,
    )
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        logits = model(**enc).logits
    probs = torch.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    pred = pred.tolist()
    conf = conf.tolist()
    plist = probs.tolist() if store_probs else None
    out: List[Dict[str, Any]] = []
    for i in range(len(clauses)):
        rec: Dict[str, Any] = {"label": id2label[pred[i]], "score": round(conf[i], 5)}
        if store_probs:
            rec["probs"] = {id2label[j]: round(plist[i][j], 5) for j in range(len(id2label))}
        out.append(rec)
    return out


def flush_pool(pool, fw, model, tokenizer, device, args, id2label, use_amp) -> None:
    """pool を長さ順にソートしてバッチ推論し、元の順序で書き出す。"""
    if not pool:
        return
    order = sorted(range(len(pool)), key=lambda i: len(pool[i]["clause"]))
    preds: List[Optional[Dict[str, Any]]] = [None] * len(pool)
    for start in range(0, len(order), args.batch_size):
        idxs = order[start:start + args.batch_size]
        clauses = [pool[i]["clause"] for i in idxs]
        results = classify_clauses(
            clauses, model, tokenizer, device, args.max_length,
            id2label, use_amp, args.store_probs,
        )
        for j, i in enumerate(idxs):
            preds[i] = results[j]
    for item, pred in zip(pool, preds):
        item.update(pred)
        fw.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_id2label(model) -> Dict[int, str]:
    raw = model.config.id2label
    return {int(k): v for k, v in raw.items()}


def load_annotated_uuids(path: str) -> set:
    """既にアノテーション済みの uuid 集合を JSONL から読み込む(無ければ空集合)。"""
    uuids: set = set()
    if not path or not os.path.exists(path):
        return uuids
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading skip uuids ({os.path.basename(path)})"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = rec.get("uuid")
            if u is not None:
                uuids.add(str(u))
    return uuids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate Nemotron-Personas-USA clauses with a trained DeBERTa classifier.")
    parser.add_argument("--model-dir", default="outputs/qpii_deberta_usa/final_full_data")
    parser.add_argument("--dataset", default="nvidia/Nemotron-Personas-USA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--output", default="annotations_pred_usa.jsonl")
    parser.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="H100 想定。短い節なので 512〜1024 でも可。OOM時は下げる")
    parser.add_argument("--sort-pool-mult", type=int, default=64,
                        help="長さソート用プール = batch_size × これ。大きいほどパディング削減")
    parser.add_argument("--max-length", type=int, default=256,
                        help="節は短いので動的パディングで実効長は短い。外れ値の上限のみ")
    parser.add_argument("--store-probs", action="store_true", help="全クラス確率も出力")
    parser.add_argument("--spacy-model", default=os.getenv("SPACY_MODEL", "en_core_web_sm"))
    parser.add_argument("--no-spacy", action="store_true",
                        help="spaCy を使わずルールベース分割(高速だが学習時と分布が変わる)")
    parser.add_argument("--skip-uuids-from", default="annotations_usa2.jsonl",
                        help="このJSONLに含まれる uuid のレコードはスキップ(学習済み分の重複回避)。"
                             "空文字('')でスキップ無効")
    parser.add_argument("--max-rows", type=int, default=0, help="0で全件。>0で先頭N行のみ(動作確認用)")
    parser.add_argument("--progress-every", type=int, default=20000, help="N節ごとに進捗表示")
    args = parser.parse_args()

    # ── デバイス / 速度設定 ───────────────────────────────────────────────
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = use_cuda
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── モデル ────────────────────────────────────────────────────────────
    print(f"[load] model={args.model_dir} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if use_cuda else None,
    )
    model.to(device).eval()
    id2label = load_id2label(model)
    print(f"[load] labels={id2label}")

    splitter = build_clause_splitter(args.spacy_model, use_spacy=not args.no_spacy)
    sort_pool = max(args.batch_size, args.batch_size * args.sort_pool_mult)

    # ── 既アノテーション uuid(学習済み分)をスキップ対象に ────────────────────
    skip_uuids = load_annotated_uuids(args.skip_uuids_from)
    if skip_uuids:
        print(f"[skip] {len(skip_uuids)} 個の既アノテーション uuid をスキップ "
              f"({args.skip_uuids_from})")
    elif args.skip_uuids_from:
        print(f"[skip] {args.skip_uuids_from} が見つからない/空 → スキップ無し")

    # ── データ ────────────────────────────────────────────────────────────
    from datasets import load_dataset
    stream = load_dataset(args.dataset, split=args.split, streaming=not args.no_streaming)

    pool: List[Dict[str, Any]] = []
    n_rows = 0
    n_skipped = 0
    n_clauses = 0
    started = time.time()

    with open(args.output, "w", encoding="utf-8") as fw:
        for i, row in enumerate(tqdm(stream, desc="Records")):
            if args.max_rows and i >= args.max_rows:
                break
            uuid = row.get("uuid")
            if uuid is not None and str(uuid) in skip_uuids:
                n_skipped += 1
                continue
            n_rows += 1
            for field in args.fields:
                value = resolve_field_value(row, field)
                for text in normalize_field(value):
                    for clause in split_clauses(text, splitter):
                        pool.append({"row_index": i, "uuid": uuid, "field": field, "clause": clause})
                        n_clauses += 1
            if len(pool) >= sort_pool:
                flush_pool(pool, fw, model, tokenizer, device, args, id2label, use_amp)
                pool = []
                if args.progress_every and n_clauses % args.progress_every < sort_pool:
                    el = max(1e-9, time.time() - started)
                    print(f"[progress] rows={n_rows} clauses={n_clauses} "
                          f"{n_clauses / el:.0f} clause/s")
        flush_pool(pool, fw, model, tokenizer, device, args, id2label, use_amp)

    el = time.time() - started
    print(f"\nrows_annotated={n_rows} skipped_uuids={n_skipped} clauses={n_clauses} "
          f"elapsed={el:.1f}s ({n_clauses / max(1e-9, el):.0f} clause/s)")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
