"""
config.py
---------
Single source of truth for the B1 sweep dimensions and plot style.

Edit MODELS, SWEEP_TEMPLATES, or DIFFICULTIES here — the change propagates
to run_b1_sweep.py and all plot scripts automatically.
"""

from pathlib import Path


# ── Sweep dimensions ──────────────────────────────────────────────────────────

MODELS = [
    "swiss-ai/Apertus-70B-Instruct-2509",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3.5-27B",
]

# Default templates run in the sweep; override at runtime with --templates
SWEEP_TEMPLATES = ["expert_rater", "llm_judge", "opro", "gepa", "opro_tree"]

# All difficulty buckets; tie pairs always have gold_label="C"
DIFFICULTIES = ["easy", "medium", "hard", "tie"]

CRITERION   = "overall"
BASE_URL    = "https://api.swissai.cscs.ch/v1"
OUTPUT_ROOT = Path("results/b1_sweep")


# ── Model helpers ─────────────────────────────────────────────────────────────

def model_key(model: str) -> str:
    name = model.split("/")[-1]
    if "Apertus"   in name: return "apertus"
    if "Llama-3.3" in name: return "llama33"
    if "Qwen3.5"   in name: return "qwen35"
    return name.lower()[:20]


MODELS_ORDER = [model_key(m) for m in MODELS]

MODEL_LABELS = {
    "apertus": "Apertus-70B",
    "llama33": "Llama-3.3-70B",
    "qwen35":  "Qwen3.5-27B",
}
MODEL_COLORS = {
    "apertus": "#b42828",
    "llama33": "#1f407a",
    "qwen35":   "#2e8b57",
}
MODEL_HATCHES = {
    "apertus": "",
    "llama33": "///",
    "qwen35":   "\\\\",
}


# ── Template helpers ──────────────────────────────────────────────────────────

# Full ordered list used by plots (includes optimised variants beyond the default sweep)
TEMPLATES_ORDER = ["expert_rater", "llm_judge", "opro", "gepa", "opro_tree"]

BASELINE_TEMPLATES          = {"expert_rater", "llm_judge"}
OPTIMIZED_TEMPLATES         = {"opro", "gepa", "opro_tree"}
GENERIC_OPTIMISED_TEMPLATES = {"gepa", "opro", "opro_tree"}

TEMPLATE_LABELS = {
    "expert_rater": "Expert Rater",
    "llm_judge":    "LLM Judge",
    "opro":         "OPRO",
    "gepa":         "GEPA",
    "opro_tree":    "OPRO Tree",
}
TEMPLATE_HATCHES = {
    "expert_rater": "",
    "llm_judge":    "///",
    "opro":         "xxx",
    "gepa":         "...",
    "opro_tree":    "\\\\",
}


# ── Accuracy breakdown ────────────────────────────────────────────────────────

ACC_TYPES = ["ab", "ba", "c"]
