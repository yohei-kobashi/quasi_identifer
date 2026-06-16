"""Global configuration for QPII candidate extraction experiment."""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
try:
    BASE_DIR: Path = Path(__file__).parent
except NameError:
    BASE_DIR: Path = Path.cwd()
DATA_DIR: Path = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────
SEED: int = 42
N_PERSONAS: int = 300

# ── Dataset ────────────────────────────────────────────────────────────────
HF_DATASET: str = "nvidia/Nemotron-Personas"
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")  # optional; speeds up downloads

# Narrative text fields to clause-split (exclude *_list and structured fields)
TEXT_FIELDS: list[str] = [
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "persona",
    "cultural_background",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
]

# ── Method A ───────────────────────────────────────────────────────────────
ALPHA: float = 0.05                              # significance level before correction
KEEP_POS_PREFIXES: tuple[str, ...] = ("NN",)  # nouns only (ver2)

# ── Method B (NP 展開の粒度) ────────────────────────────────────────────────
# NP チャンク文法のモード。support=1(再識別の人工的一意性)の主因＝過剰具体フレーズを
# 抑えるための切替(②)。基底NP化すると "engineer at a startup in Austin" のような
# 最長NPが {engineer, startup, austin} 相当の基底NP群に分解され、言い換え重複が
# マージされて support=1 が減る一方、属性の組合せ(本物の再識別シグナル)は温存される。
#   "base"  : 基底NP <DT>?<JJ.*>*<NN.*>+              (PP連結なし=最も一般化, 変種A)
#   "pp1"   : 基底NP + 単一PP …(<IN>…)?               ("X of Y" は温存, 3連結以上を切る, 変種B)
#   "chain" : 基底NP + PP連結(*) …(<IN>…)*            (従来=最長NP=最も具体的)
NP_GRAMMAR_MODE: str = "base"
# method-b 辞書に 1 語の基底NP も含めるか(②-full)。False だと従来通り <2語 を捨てる。
# base 文法では単独語属性(都市名・1語スキル等)が多く出るため True が前提。
METHOD_B_KEEP_SINGLE_WORD: bool = True

# ── Method C ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-mpnet-base-v2"
CATEGORY_SIM_THRESHOLD: float = 0.45
GENERAL_SIM_THRESHOLD: float = 0.55
MIN_PHRASE_FREQ: int = 3          # minimum times a phrase must appear
MAX_PHRASE_WORDS: int = 6         # maximum words in a phrase (exclude over-specific)
# QI バンドパス上限。Method B 辞書 freq がこれを超える phrase(=ありふれ過ぎて非識別な
# レジスターマーカー: enjoys/love 等)を許可語彙から除外する。0=上限なし。
# 識別価値 ≈ -log(出現率) で、ほぼ全レコードに出る語は ~0 bit。下限 MIN_PHRASE_FREQ
# (準直接識別子を除く)と合わせ、QI を「中頻度の帯」に限定する。
MAX_PHRASE_FREQ: int = 0

# ── Annotation sampling ────────────────────────────────────────────────────
ANNOTATION_SAMPLE_N: int = 100
ANNOTATION_SEED: int = 123
