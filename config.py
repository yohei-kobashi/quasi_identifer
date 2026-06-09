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

# ── Method C ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-mpnet-base-v2"
CATEGORY_SIM_THRESHOLD: float = 0.45
GENERAL_SIM_THRESHOLD: float = 0.55
MIN_PHRASE_FREQ: int = 3          # minimum times a phrase must appear
MAX_PHRASE_WORDS: int = 6         # maximum words in a phrase (exclude over-specific)

# ── Annotation sampling ────────────────────────────────────────────────────
ANNOTATION_SAMPLE_N: int = 100
ANNOTATION_SEED: int = 123
