from src.core.templates import SYSTEM_EXPERT_RATER, SYSTEM_LLM_JUDGE

W_CONSIST      = 0.5
W_ACCURATE     = 0.5
LENGTH_TARGET  = 2200
LENGTH_PENALTY = 1e-4
LENGTH_HARD_CAP = 3000

_FLIP = {"A": "B", "B": "A", "C": "C"}
SEED_PROMPTS = [SYSTEM_EXPERT_RATER, SYSTEM_LLM_JUDGE]


def compute_score(prompt: str, metrics: dict) -> float:
    base = W_CONSIST * metrics["position_consistency"] + W_ACCURATE * metrics["accuracy"]
    eff_len = min(len(prompt), LENGTH_HARD_CAP)
    length_cost = LENGTH_PENALTY * max(0, eff_len - LENGTH_TARGET)
    return base - length_cost
