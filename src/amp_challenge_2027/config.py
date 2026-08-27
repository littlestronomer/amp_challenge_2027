"""Central configuration: paths, constants, hyperparameters.

Single source of truth shared by training scripts and the inference entry point.
Keeping these here means ``uv run generate`` and the training scripts always agree
on where artifacts live and what the valid sequence space is.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
GENERATOR_DIR = CHECKPOINT_DIR / "generator"
RL_GENERATOR_DIR = CHECKPOINT_DIR / "generator_rl"  # RL output NEVER overwrites SFT
REWARD_DIR = CHECKPOINT_DIR / "reward"
FLOW_MATCHING_DIR = CHECKPOINT_DIR / "flow_matching"
ANTIBACTERIAL_FASTA = DATA_DIR / "antibacterial.fasta"

# Output directory for the entry point. The validator expects:
#   <entry_point>/library.fasta  (50,000 sequences)
#   <entry_point>/top.fasta      (100 ranked sequences)
GENERATE_OUTPUT_DIR = PROJECT_ROOT / "generate"

# --- Sequence constraints (must match scripts/verify_submission.py) --------

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACIDS = STANDARD_AMINO_ACIDS  # alias for readability at call sites
MIN_LENGTH = 8
MAX_LENGTH = 50

LIBRARY_SIZE = 50_000
TOP_K = 100

# Stricter novelty bound for the top-100 vs. known antibacterial peptides.
# The library only needs to avoid *exact* matches; the top-100 must additionally
# stay under this Levenshtein-ratio similarity to any reference sequence.
TOP_SIMILARITY_THRESHOLD = 0.80

DEFAULT_SEED = 42

# --- Generator hyperparameters (GPT-2-style causal decoder) ----------------
# Following Park et al. (2025): ~12 layers, 768 hidden, 12 heads.
# Tokenizer reuses ESM-2's 33-token vocabulary so embeddings/score heads stay
# compatible with the downstream ESM-2 reward surrogates.

GENERATOR_CONFIG = {
    "vocab_size": len(AMINO_ACIDS) + 4,  # <pad>,<bos>,<eos>,<mask> + 20 AAs
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "max_position_embeddings": MAX_LENGTH + 2,  # +bos/eos
    "dropout": 0.1,
}

# --- Tokenizer -------------------------------------------------------------

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
MASK_TOKEN = "<mask>"

# Special tokens first, then amino acids in canonical order.
VOCAB = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, *AMINO_ACIDS]
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {i: tok for i, tok in enumerate(VOCAB)}

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
BOS_ID = TOKEN_TO_ID[BOS_TOKEN]
EOS_ID = TOKEN_TO_ID[EOS_TOKEN]
MASK_ID = TOKEN_TO_ID[MASK_TOKEN]

# --- Reward surrogate (ESM-2 LoRA) ----------------------------------------

ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM2_LORA_RANK = 16
ESM2_LORA_ALPHA = 32
ESM2_LORA_DROPOUT = 0.05
REWARD_ENSEMBLE_SIZE = 3

# Bacterial panel (Appendix B). MDR strains marked True.
# Used to define the reward model's multi-strain output heads.
BACTERIAL_PANEL: list[tuple[str, str, bool]] = [
    # (genus/species, strain_id, is_mdr)
    # Gram-negative (15)
    ("A. baumannii", "ATCC_19606", False),
    ("A. baumannii", "ATCC_BAA_1605", True),
    ("E. cloacae", "ATCC_13047", False),
    ("E. coli", "ATCC_11775", False),
    ("E. coli", "AIC221", False),
    ("E. coli", "AIC222", True),
    ("E. coli", "ATCC_BAA_3170", True),
    ("E. coli", "K-12_BW25113", False),
    ("K. pneumoniae", "ATCC_13883", False),
    ("K. pneumoniae", "ATCC_BAA_2342", True),
    ("P. aeruginosa", "PAO1", False),
    ("P. aeruginosa", "PA14", False),
    ("P. aeruginosa", "ATCC_BAA_3197", True),
    ("S. enterica", "ATCC_9150", False),
    ("S. enterica", "Typhimurium_ATCC_700720", False),
    # Gram-positive (5)
    ("B. subtilis", "ATCC_23857", False),
    ("S. aureus", "ATCC_12600", False),
    ("S. aureus", "ATCC_BAA_1556", True),  # MRSA
    ("E. faecalis", "ATCC_700802", True),  # VRE
    ("E. faecium", "ATCC_700221", True),  # VRE
]
NUM_STRAINS = len(BACTERIAL_PANEL)

# Panel genus → Gram type. Matches the panel's documented grouping (15
# Gram-negative, 5 Gram-positive) so label builders and any future
# panel-aware ranker share one taxonomy constant.
PANEL_GENUS_TO_GRAM: dict[str, str] = {
    # Gram-negative (15 strains, 6 genera)
    "A. baumannii": "negative",
    "E. cloacae": "negative",
    "E. coli": "negative",
    "K. pneumoniae": "negative",
    "P. aeruginosa": "negative",
    "S. enterica": "negative",
    # Gram-positive (5 strains, 4 genera)
    "B. subtilis": "positive",
    "S. aureus": "positive",
    "E. faecalis": "positive",
    "E. faecium": "positive",
}

# MDR category = these 8 scored strains; keyed by strain_id from the panel.
MDR_STRAIN_IDS: frozenset[str] = frozenset(
    strain_id for _genus, strain_id, is_mdr in BACTERIAL_PANEL if is_mdr
)

# Phase-2 potency threshold (uM); "success" = MIC <= this value.
MIC_SUCCESS_THRESHOLD_UM = 16.0
MIC_CEILING_UM = 64.0
HC50_CEILING_UM = 128.0


def ensure_dirs() -> None:
    """Create the directories the pipeline writes into, if absent."""
    for p in (RAW_DATA_DIR, PROCESSED_DATA_DIR, GENERATOR_DIR, REWARD_DIR, GENERATE_OUTPUT_DIR):
        p.mkdir(parents=True, exist_ok=True)
