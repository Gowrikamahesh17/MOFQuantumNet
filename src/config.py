"""Central switches for the pipeline.

Both the dataset and the feature-vector scheme are open questions pending
confirmation from Prof. Jalali (see planning/MEETING_PREP.md). Everything
downstream is written against these two switches so that changing either one
is a one-line edit here, not a rewrite.
"""

from dataclasses import dataclass

DATASET_CHOICES = ("small", "large")
# "small"  -> SMILES_METAL_2000_NoPLD.csv (2,000 MOFs, precomputed PLD category, no continuous PLD)
# "large"  -> MOFCSD.csv (14,296 MOFs, continuous PLD, real metal names, supports regression too)

FEATURE_SCHEME_CHOICES = ("compact", "fingerprint")
# "compact"    -> 7-dim descriptor style (paper-original), pairs naturally with "small"
# "fingerprint" -> 1031-dim Morgan-fingerprint style (BlackHole code), pairs naturally with "large"


@dataclass
class PipelineConfig:
    dataset: str = "small"
    feature_scheme: str = "compact"

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_CHOICES:
            raise ValueError(f"dataset must be one of {DATASET_CHOICES}, got {self.dataset!r}")
        if self.feature_scheme not in FEATURE_SCHEME_CHOICES:
            raise ValueError(f"feature_scheme must be one of {FEATURE_SCHEME_CHOICES}, got {self.feature_scheme!r}")


DEFAULT_CONFIG = PipelineConfig(dataset="small", feature_scheme="compact")
