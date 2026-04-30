"""
Linear probe training, evaluation, and layer selection utilities.
"""

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold

from src.hidden_states import get_probe_view


def eval_probe(X, y, groups, n_splits=5, max_iter=1000):
    """
    Evaluate a linear probe using stratified group k-fold cross-validation.

    Args:
        X: Feature matrix (N, D).
        y: Labels (N,).
        groups: Group IDs for stratified group splitting (N,).
        n_splits: Number of CV folds.
        max_iter: Max iterations for logistic regression.

    Returns:
        Tuple of (mean_auc, std_auc).
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs = []

    for train_idx, test_idx in cv.split(X, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=max_iter,
            )),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        probs = pipe.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))

    assert len(aucs) > 0, "No valid CV folds"
    return np.mean(aucs), np.std(aucs)


def compute_probe_margins(hidden_data, view="last", n_splits=5):
    """
    Compute signed margin distributions for each layer's probe.

    Args:
        hidden_data: Dict with 'hidden_states', 'labels', 'groups'.
        view: Representation view ('last', 'mean', 'max').
        n_splits: Number of CV folds.

    Returns:
        Dict mapping layer index -> list of margin values.
    """
    layers = list(range(len(hidden_data["hidden_states"])))
    y = hidden_data["labels"]
    groups = hidden_data["groups"]

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    layer_margins = {layer: [] for layer in layers}

    for layer in tqdm(layers, desc="Computing probe margins"):
        X = get_probe_view(hidden_data["hidden_states"][layer], view)

        for train_idx, test_idx in cv.split(X, y, groups):
            if len(np.unique(y[test_idx])) < 2:
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2", C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                )),
            ])

            pipe.fit(X[train_idx], y[train_idx])
            logits = pipe.decision_function(X[test_idx])
            y_test = y[test_idx]

            # y_test = 1 → margin = +logit
            # y_test = 0 → margin = -logit
            margins = y_test * logits - (1 - y_test) * logits
            layer_margins[layer].extend(margins.tolist())

    return layer_margins


def run_phase1A(hidden_data, views=("last", "mean", "max")):
    """
    Phase 1A: Evaluate linear probes across all layers and views.

    Args:
        hidden_data: Dict with 'hidden_states', 'labels', 'groups'.
        views: Tuple of view names to evaluate.

    Returns:
        Dict mapping layer -> view -> {mean_auc, std_auc}.
    """
    import torch

    results = {}
    for layer, layer_hs in enumerate(
        tqdm(hidden_data["hidden_states"], desc="Phase 1A")
    ):
        torch.cuda.empty_cache()
        results[layer] = {}
        for view in views:
            X = get_probe_view(layer_hs, view)
            y = hidden_data["labels"]
            groups = hidden_data["groups"]

            mean_auc, std_auc = eval_probe(X, y, groups)
            results[layer][view] = {
                "mean_auc": mean_auc,
                "std_auc": std_auc,
            }

    return results


def results_to_arrays(results, views=("last", "mean", "max")):
    """
    Convert phase 1A results dict to numpy arrays for plotting.

    Returns:
        Tuple of (layers_array, auc_data_dict).
    """
    layers = sorted(results.keys())
    out = {view: {"mean": [], "std": []} for view in views}

    for l in layers:
        for view in views:
            out[view]["mean"].append(results[l][view]["mean_auc"])
            out[view]["std"].append(results[l][view]["std_auc"])

    for view in views:
        out[view]["mean"] = np.array(out[view]["mean"])
        out[view]["std"] = np.array(out[view]["std"])

    return np.array(layers), out


def candidate_layer_mask(auc_data, auc_thresh=0.98, cv_thresh=0.01):
    """
    Identify candidate layers based on AUC threshold and stability.

    Args:
        auc_data: Dict from results_to_arrays.
        auc_thresh: Minimum mean AUC.
        cv_thresh: Maximum coefficient of variation.

    Returns:
        Tuple of (mask, support) arrays.
    """
    views = list(auc_data.keys())
    num_layers = len(auc_data[views[0]]["mean"])
    support = np.zeros(num_layers, dtype=int)

    for view in views:
        mean = auc_data[view]["mean"]
        std = auc_data[view]["std"]
        cv = std / mean
        good = (mean >= auc_thresh) & (cv <= cv_thresh)
        support += good.astype(int)

    # candidate if supported by ≥2 views
    return support >= 2, support


def contiguous_bands(mask, layers):
    """
    Find contiguous bands of True values in mask.

    Returns:
        List of (start_layer, end_layer) tuples.
    """
    bands = []
    start = None

    for i, m in enumerate(mask):
        if m and start is None:
            start = layers[i]
        elif not m and start is not None:
            bands.append((start, layers[i - 1]))
            start = None

    if start is not None:
        bands.append((start, layers[-1]))

    return bands
