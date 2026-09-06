"""Turns one hypothetical MOF's raw description (metal + linker SMILES, ± optional pore
geometry) into the exact feature vector shape the persisted models expect — reused for
both baselines (model_store.py) and GNNs (inductive.py), since both were trained on the
same underlying feature matrix per dataset.

Two real, honest gaps found while building this against the actual data (not glossed
over — surfaced in the API response instead):

1. **Fingerprint scheme (large dataset)** was trained on each MOF's *real* pore geometry
   (Largest Cavity Diameter, Largest Free Sphere) — values that come from simulating the
   MOF's actual 3D crystal structure, not from its metal + linker identity alone. A
   genuinely new, never-synthesized MOF doesn't have these. Callers may supply them if
   known; otherwise this module imputes the training-set median and the API flags
   `used_median_geometry: true` on the response, since accuracy for that specific
   prediction is not directly comparable to the benchmarked test-set numbers.
2. **Compact scheme (small dataset)** has no metal identity at all in the source data —
   only 6 anonymous `metal_feat_0..5` numeric columns (see
   `data_ingestion.SMALL_DATASET_COLUMNS`). There is no known mapping from a metal symbol
   (e.g. "Cu") to those 6 values, so a plain "type a metal" input isn't meaningful for
   this dataset. Callers may pass `metal_feat` explicitly (6 floats); otherwise this
   module falls back to the training-set medians and flags `used_median_metal_feat: true`.
   This is a real product-design gap, not just an implementation shortcut — tracked in
   planning/DEVELOPMENT_TODO.md (Part 2) rather than hidden.
"""

import numpy as np

from feature_engineering import _molecular_weight, _morgan_fingerprint  # noqa: E402


def featurize_compact(metadata: dict, smiles: str, metal_feat: list[float] | None) -> tuple[np.ndarray, dict]:
    inf = metadata["inference"]
    caveats: dict = {}
    if metal_feat is None:
        metal_feat = inf["metal_feat_medians"]
        caveats["used_median_metal_feat"] = True
    if len(metal_feat) != 6:
        raise ValueError(f"metal_feat must have 6 values, got {len(metal_feat)}")

    mw = _molecular_weight(smiles)
    if np.isnan(mw):
        caveats["invalid_smiles"] = True
        mw = 0.0

    raw = np.array(list(metal_feat) + [mw], dtype=np.float64)
    lo = np.array(inf["scaler_min"], dtype=np.float64)
    hi = np.array(inf["scaler_max"], dtype=np.float64)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    scaled = np.clip((raw - lo) / span, 0.0, 1.0)
    return scaled.astype(np.float32).reshape(1, -1), caveats


def featurize_fingerprint(
    metadata: dict, metal: str, smiles: str,
    largest_cavity_diameter: float | None = None, largest_free_sphere: float | None = None,
) -> tuple[np.ndarray, dict]:
    inf = metadata["inference"]
    caveats: dict = {}

    fp = _morgan_fingerprint(smiles)
    if fp.sum() == 0:
        caveats["invalid_smiles"] = True

    if largest_cavity_diameter is None or largest_free_sphere is None:
        cavity_median, free_sphere_median = inf["pore_geometry_medians"]
        largest_cavity_diameter = largest_cavity_diameter if largest_cavity_diameter is not None else cavity_median
        largest_free_sphere = largest_free_sphere if largest_free_sphere is not None else free_sphere_median
        caveats["used_median_geometry"] = True
    geometry = np.array([largest_cavity_diameter, largest_free_sphere], dtype=np.float32)

    metal_to_index = inf["metal_to_index"]
    one_hot = np.zeros(len(metal_to_index), dtype=np.float32)
    if metal in metal_to_index:
        one_hot[metal_to_index[metal]] = 1.0
    else:
        caveats["unknown_metal"] = True

    features = np.concatenate([fp, geometry, one_hot]).astype(np.float32)
    return features.reshape(1, -1), caveats


def featurize_new_mof(metadata: dict, **kwargs) -> tuple[np.ndarray, dict]:
    scheme = metadata["inference"]["scheme"]
    if scheme == "compact":
        return featurize_compact(metadata, kwargs["smiles"], kwargs.get("metal_feat"))
    if scheme == "fingerprint":
        return featurize_fingerprint(
            metadata, kwargs["metal"], kwargs["smiles"],
            kwargs.get("largest_cavity_diameter"), kwargs.get("largest_free_sphere"),
        )
    raise ValueError(f"Unknown scheme {scheme!r}")
