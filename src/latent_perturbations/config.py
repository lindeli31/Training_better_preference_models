"""
config.py
---------
Configuration for the gradient-based positional bias probe experiments.

Adapts the LatentSafety (Carol-gutianle/LatentSafety) methodology to
study positional bias in LLM judges rather than safety alignment.
"""

from dataclasses import dataclass, field


@dataclass
class GradientProbeConfig:
    # --- Model ---
    model_name: str = "meta-llama/Llama-3.1-70B-Instruct"
    device: str = "cuda"
    load_in_4bit: bool = True      # required for 70B on single H100
    load_in_8bit: bool = False     # middle ground: half the noise of 4bit, 32B fits on H100
    dtype: str = "bfloat16"        # "bfloat16" or "float16"

    # --- Layer sweep (Experiment 1) ---
    # Layer indices to probe. If empty, computed at runtime as range(0, n_layers, layer_sweep_step).
    layer_indices: list[int] = field(default_factory=list)
    layer_sweep_step: int = 4      # probe every Nth layer; 4 → ~20 points for 80-layer model

    # --- Target layer (Experiments 2-4) ---
    # Set manually to the peak-sensitivity layer found in Experiment 1,
    # or pass --target-layer on CLI. -1 = final layer.
    target_layer: int = -1

    # --- Perturbation (Experiments 1 & 4) ---
    normalize: bool = False        # apply LatentSafety distribution normalization to perturbation
    alpha: float = 1.0             # default perturbation scale for Experiment 1 LASR
    # Scales swept in Experiment 4 dose-response curve.
    alpha_scales: list[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    )
    n_random_baselines: int = 5    # random direction repeats for each alpha

    # --- Dataset ---
    n_pairs: int = 200             # evaluation pairs (both orderings built automatically)
    seed: int = 42
    template_id: str = "expert_rater"
    criterion: str = "overall"

    # --- Output ---
    output_dir: str = "results/gradient_probe"

    # --- Verdict tokens ---
    verdict_tokens: list[str] = field(default_factory=lambda: ["A", "B", "C"])
