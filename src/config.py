"""Central switches for the pipeline.

Outcome of the 2026-07-21 meeting with Prof. Jalali: both datasets are in
scope, selected via one dataset switch — everything else (feature scheme,
task options) cascades from that single choice rather than being independent
settings. Black Hole's gravity weights and pruning threshold are user-facing
sliders, not hardcoded constants.
"""

from dataclasses import dataclass

DATASET_CHOICES = ("small", "large")
# "small"  -> SMILES_METAL_2000_NoPLD.csv (2,000 MOFs, precomputed PLD category, no continuous PLD)
# "large"  -> MOFCSD.csv (14,296 MOFs, continuous PLD, real metal names, supports regression too)

TASK_CHOICES = ("classification", "regression")
# "small" dataset has no continuous PLD, so it can only ever run classification.
# "large" dataset has continuous PLD, so the user gets a real choice between the two.

FEATURE_SCHEME_BY_DATASET = {
    "small": "compact",       # 7-dim descriptor style (paper-original) — matches small dataset's columns
    "large": "fingerprint",   # 1031-dim Morgan-fingerprint style (BlackHole code) — matches large dataset's columns
}


@dataclass
class PipelineConfig:
    dataset: str = "small"
    task: str = "classification"

    # Black Hole gravity score = degree_weight * norm_degree + betweenness_weight * norm_betweenness
    #                            + edge_weight_sum_weight * norm_edge_weight_sum
    # Defaults match bh_sparsification.py's own code default (0.3, 0.3, 0.4), not the Expose's
    # stated 0.33/0.33/0.33 — now moot since these are user-adjustable rather than fixed.
    gravity_degree_weight: float = 0.3
    gravity_betweenness_weight: float = 0.3
    gravity_edge_weight_sum_weight: float = 0.4

    # Black Hole pruning threshold (fraction of nodes/edges removed per community)
    pruning_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_CHOICES:
            raise ValueError(f"dataset must be one of {DATASET_CHOICES}, got {self.dataset!r}")
        if self.task not in TASK_CHOICES:
            raise ValueError(f"task must be one of {TASK_CHOICES}, got {self.task!r}")
        if self.dataset == "small" and self.task == "regression":
            # Small dataset has no continuous PLD column — regression is impossible on it.
            self.task = "classification"
        if not 0.0 <= self.pruning_threshold <= 0.9:
            raise ValueError(f"pruning_threshold must be in [0.0, 0.9], got {self.pruning_threshold}")
        for name, value in (
            ("gravity_degree_weight", self.gravity_degree_weight),
            ("gravity_betweenness_weight", self.gravity_betweenness_weight),
            ("gravity_edge_weight_sum_weight", self.gravity_edge_weight_sum_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")

    @property
    def feature_scheme(self) -> str:
        """Derived from dataset, not independently settable — the cascade the professor confirmed."""
        return FEATURE_SCHEME_BY_DATASET[self.dataset]

    @property
    def gravity_weights_normalized(self) -> tuple[float, float, float]:
        """The three gravity weight sliders don't need to sum to 1 in the UI — normalize here
        so the underlying gravity score is always a proper weighted average."""
        raw = (self.gravity_degree_weight, self.gravity_betweenness_weight, self.gravity_edge_weight_sum_weight)
        total = sum(raw)
        if total == 0:
            return (1 / 3, 1 / 3, 1 / 3)
        return tuple(w / total for w in raw)


DEFAULT_CONFIG = PipelineConfig()
