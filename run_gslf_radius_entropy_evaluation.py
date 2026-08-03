"""
Evaluate already-trained GSLF hyperbolic MDS embeddings.

GSLF setting:
    labels shape: (4983, 138)
    one row = one molecule
    binary descriptor matrix
    no subject structure
    no repeated-measures structure

This script computes:
    1. Embedding quality:
       - distance Pearson
       - distance Spearman

    2. Radial entropy organization for three entropy types:
       - entropy on original binary labels
       - entropy on 138D PCA coordinates
       - entropy on pruned binary labels, where remaining labels have abs correlation <= 0.3

    3. Molecule-level permutation p-value:
       - p_mol
       
python run_gslf_radius_entropy_evaluation.py \
  --labels_path ../../HyperDimRed/data/labels/y_gslf.npy \
  --embeddings_dir results_ChemicalSenses_experiments_GSLF/original \
  --embedding_glob "*.npy" \
  --output_dir evaluation_results_gslf_entropy \
  --n_permutations 1000 \
  --keep_last_epoch_per_seed
"""

import argparse
import glob
import os
import sys
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# Imports from codebase
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from distances import distance_matrix, poincare_distance


# ---------------------------------------------------------------------
# Small utility functions
# ---------------------------------------------------------------------

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    if method == "pearson":
        res = pearsonr(x, y)
        return float(getattr(res, "statistic", res[0]))

    if method == "spearman":
        res = spearmanr(x, y)
        return float(getattr(res, "statistic", res[0]))

    raise ValueError("method must be either 'pearson' or 'spearman'")


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return float(values[0]), 0.0

    return float(np.mean(values)), float(np.std(values, ddof=1))


def upper_triangle_values(D):
    D = np.asarray(D)
    n = D.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return D[mask]


# ---------------------------------------------------------------------
# Entropy functions
# ---------------------------------------------------------------------

def entropy_from_binary_labels(labels, normalize=False, eps=1e-12):
    """
    Entropy over active binary labels.

    If a molecule has K active labels, this is log(K), because the
    probability distribution is uniform over its active labels.

    Rows with no active label are set to NaN.
    """
    labels = np.asarray(labels, dtype=float)

    if labels.ndim != 2:
        raise ValueError(f"Expected labels to have shape (n_samples, n_labels), got {labels.shape}")

    # Treat any positive value as active.
    binary = (labels > 0).astype(float)

    n_samples, n_labels = binary.shape
    row_sums = binary.sum(axis=1, keepdims=True)

    valid = row_sums[:, 0] > eps

    P = np.zeros_like(binary)
    P[valid] = binary[valid] / row_sums[valid]

    H = np.full(n_samples, np.nan, dtype=float)
    H[valid] = -np.sum(P[valid] * np.log(P[valid] + eps), axis=1)

    if normalize:
        H = H / np.log(n_labels)

    return H


def entropy_from_scores(scores, normalize=False, temperature=1.0, eps=1e-12):
    """
    Entropy after softmax, used here for PCA coordinates.
    """
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 2:
        raise ValueError(f"Expected scores to have shape (n_samples, n_features), got {scores.shape}")

    P = softmax(scores / temperature, axis=1)
    H = -np.sum(P * np.log(P + eps), axis=1)

    if normalize:
        H = H / np.log(scores.shape[1])

    return H


# ---------------------------------------------------------------------
# GSLF loading
# ---------------------------------------------------------------------

def load_gslf_labels(labels_path: str):
    labels = np.load(labels_path)
    labels = np.asarray(labels, dtype=float)

    if labels.ndim != 2:
        raise ValueError(f"Expected labels to be 2D, got shape {labels.shape}")

    descriptor_names = [f"descriptor_{i}" for i in range(labels.shape[1])]

    return labels, descriptor_names


# ---------------------------------------------------------------------
# Pruning labels by correlation
# ---------------------------------------------------------------------

def correlation_matrix_no_nan(X):
    """
    Column-wise Pearson correlation matrix, with NaNs replaced by 0.
    NaNs can occur for constant columns.
    """
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)
    return C


def greedy_prune_correlated_labels(labels, descriptor_names=None, threshold=0.3):
    """
    Greedily remove descriptors until all remaining pairwise absolute
    correlations are <= threshold.

    At each step, find the most correlated pair and remove the descriptor
    with the larger mean absolute correlation to all other remaining descriptors.
    """
    labels = np.asarray(labels, dtype=float)

    if descriptor_names is None:
        descriptor_names = [f"descriptor_{i}" for i in range(labels.shape[1])]

    keep = list(range(labels.shape[1]))
    removed = []
    pruning_steps = []

    while True:
        X_keep = labels[:, keep]

        if X_keep.shape[1] <= 1:
            break

        C = correlation_matrix_no_nan(X_keep)
        abs_C = np.abs(C)

        max_corr = np.max(abs_C)

        if max_corr <= threshold:
            break

        i_local, j_local = np.unravel_index(np.argmax(abs_C), abs_C.shape)

        mean_abs_corr = abs_C.mean(axis=0)

        if mean_abs_corr[i_local] >= mean_abs_corr[j_local]:
            remove_local = i_local
            keep_local = j_local
        else:
            remove_local = j_local
            keep_local = i_local

        remove_global = keep[remove_local]
        keep_global = keep[keep_local]

        pruning_steps.append({
            "removed_index": remove_global,
            "removed_descriptor": descriptor_names[remove_global],
            "paired_with_index": keep_global,
            "paired_with_descriptor": descriptor_names[keep_global],
            "pair_abs_correlation": float(abs_C[i_local, j_local]),
            "removed_mean_abs_correlation": float(mean_abs_corr[remove_local]),
            "kept_mean_abs_correlation": float(mean_abs_corr[keep_local]),
            "n_remaining_before_removal": len(keep),
        })

        removed.append(remove_global)
        keep.pop(remove_local)

    keep = np.array(keep, dtype=int)
    removed = np.array(removed, dtype=int)

    return keep, removed, pd.DataFrame(pruning_steps)


# ---------------------------------------------------------------------
# Poincare geometry utilities
# ---------------------------------------------------------------------

def project_inside_poincare_disk(z, eps=1e-5):
    """
    Evaluation-only safety projection if points are numerically outside the disk.
    """
    z = np.asarray(z, dtype=np.float32)

    norms = np.linalg.norm(z, axis=1, keepdims=True)
    scale = np.where(
        norms >= 1.0 - eps,
        (1.0 - eps) / np.maximum(norms, eps),
        1.0,
    )

    return z * scale


def hyperbolic_distance_matrix(z):
    z = project_inside_poincare_disk(z)
    z_torch = torch.tensor(z, dtype=torch.float32)

    with torch.no_grad():
        D = distance_matrix(z_torch, poincare_distance).cpu().numpy()

    return D


def hyperbolic_radius(z):
    z = project_inside_poincare_disk(z)
    z_torch = torch.tensor(z, dtype=torch.float32)
    zero = torch.zeros((1, z_torch.shape[1]), dtype=torch.float32)

    with torch.no_grad():
        r = poincare_distance(z_torch, zero).cpu().numpy()

    return r.reshape(-1)


# ---------------------------------------------------------------------
# Embedding file loading
# ---------------------------------------------------------------------

def parse_epoch_and_seed(path: str) -> Tuple[int, str]:
    """
    Convention: filenames start with embeddings_{epoch}_...
    and the seed is often the 8th token.

    If parsing fails, the full filename is used as seed.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")

    epoch = -1

    if len(parts) > 1 and parts[0] == "embeddings":
        try:
            epoch = int(parts[1])
        except ValueError:
            epoch = -1

    seed = stem

    if len(parts) > 7 and parts[0] == "embeddings":
        seed = parts[7]

    return epoch, seed


def select_last_epoch_per_seed(paths: List[str]) -> List[str]:
    best = {}

    for path in paths:
        epoch, seed = parse_epoch_and_seed(path)

        if seed not in best or epoch > best[seed][0]:
            best[seed] = (epoch, path)

    return [v[1] for v in sorted(best.values(), key=lambda x: x[1])]


def resolve_embedding_files(
    embeddings_dir: str,
    embedding_glob: str,
    keep_last_epoch_per_seed: bool,
):
    if os.path.isabs(embedding_glob):
        pattern = embedding_glob
    else:
        pattern = os.path.join(embeddings_dir, embedding_glob)

    paths = sorted(glob.glob(pattern, recursive=True))
    paths = [p for p in paths if p.endswith(".npy")]

    if keep_last_epoch_per_seed:
        paths = select_last_epoch_per_seed(paths)

    return paths


def load_embedding(path: str, expected_n: int):
    z = np.load(path)

    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f"Embedding {path} has shape {z.shape}, expected (N, 2).")

    if z.shape[0] != expected_n:
        raise ValueError(f"Embedding {path} has {z.shape[0]} rows, expected {expected_n}.")

    return z.astype(np.float32)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def distance_preservation_for_one_seed(training_input, z):
    D_in = cdist(training_input, training_input, metric="euclidean")
    D_emb = hyperbolic_distance_matrix(z)

    d_in = upper_triangle_values(D_in)
    d_emb = upper_triangle_values(D_emb)

    return {
        "distance_pearson": safe_corr(d_in, d_emb, "pearson"),
        "distance_spearman": safe_corr(d_in, d_emb, "spearman"),
    }


def radial_correlations_from_radius(radius, y):
    return {
        "radius_pearson": safe_corr(radius, y, "pearson"),
        "radius_spearman": safe_corr(radius, y, "spearman"),
    }


def mean_radius_pearson_from_radii(radii_by_seed, y):
    vals = []

    for radius in radii_by_seed:
        vals.append(safe_corr(radius, y, "pearson"))

    return float(np.nanmean(vals))


def permutation_p_value_radial_molecule(
    radii_by_seed,
    y,
    n_permutations,
    rng,
):
    """
    Molecule-level permutation p-value.

    The embedding is kept fixed.
    The tested variable y is shuffled across molecules.
    The statistic is the mean Pearson radius-y correlation over seeds.
    The test is two-sided, using abs(stat).
    """
    if n_permutations <= 0:
        return np.nan

    y = np.asarray(y, dtype=float)

    obs = mean_radius_pearson_from_radii(radii_by_seed, y)

    if not np.isfinite(obs):
        return np.nan

    count = 0

    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        stat = mean_radius_pearson_from_radii(radii_by_seed, y_perm)

        if np.isfinite(stat) and abs(stat) >= abs(obs):
            count += 1

    return (1 + count) / (n_permutations + 1)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to GSLF labels. Must be .npy or .csv. Expected shape: 4983 x 138.",
    )

    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Folder containing already-trained .npy embeddings.",
    )

    parser.add_argument(
        "--embedding_glob",
        type=str,
        default="*.npy",
        help=(
            "Glob for embedding files relative to embeddings_dir. "
            "Use '*.npy' if embeddings_dir is already the original folder. "
            "Use 'original/*.npy' if embeddings_dir is results_ChemicalSenses_experiments_GSLF."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results_gslf_entropy",
    )

    parser.add_argument(
        "--n_permutations",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--keep_last_epoch_per_seed",
        action="store_true",
    )

    parser.add_argument(
        "--prune_threshold",
        type=float,
        default=0.3,
        help="Remove labels until all remaining abs pairwise correlations are <= this value.",
    )

    parser.add_argument(
        "--pca_components",
        type=int,
        default=138,
        help="Number of PCA components for PCA entropy. For GSLF, default is 138.",
    )

    parser.add_argument(
        "--normalize_entropy",
        action="store_true",
        help="If set, divide entropy by log(number of descriptors/features).",
    )

    parser.add_argument(
        "--skip_distance",
        action="store_true",
        help="Skip pairwise distance-preservation metrics to save memory/time.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    # ------------------------------------------------------------
    # Load labels
    # ------------------------------------------------------------

    labels_original, descriptor_names = load_gslf_labels(args.labels_path)

    print("Loaded GSLF labels:")
    print("  labels:", labels_original.shape)
    print("  unique values:", np.unique(labels_original)[:10])

    if labels_original.shape != (4983, 138):
        print(
            "WARNING: expected GSLF labels to have shape (4983, 138), "
            f"but got {labels_original.shape}."
        )

    # ------------------------------------------------------------
    # Build entropy variants
    # ------------------------------------------------------------

    keep_idx, removed_idx, pruning_log = greedy_prune_correlated_labels(
        labels_original,
        descriptor_names=descriptor_names,
        threshold=args.prune_threshold,
    )

    labels_pruned = labels_original[:, keep_idx]

    n_components = min(args.pca_components, labels_original.shape[1])
    pca_scores = PCA(n_components=n_components).fit_transform(labels_original)

    entropy_variants = {
        "binary_entropy": entropy_from_binary_labels(
            labels_original,
            normalize=args.normalize_entropy,
        ),
        "pca138_entropy": entropy_from_scores(
            pca_scores,
            normalize=args.normalize_entropy,
        ),
        "pruned_binary_entropy": entropy_from_binary_labels(
            labels_pruned,
            normalize=args.normalize_entropy,
        ),
    }

    print("Entropy variants:")
    for name, H in entropy_variants.items():
        print(
            f"  {name}: "
            f"finite={np.isfinite(H).sum()} / {len(H)}, "
            f"min={np.nanmin(H):.4f}, max={np.nanmax(H):.4f}"
        )

    print("Pruning:")
    print("  threshold:", args.prune_threshold)
    print("  kept labels:", len(keep_idx))
    print("  removed labels:", len(removed_idx))

    pruning_log_path = os.path.join(args.output_dir, "pruned_label_selection_steps.csv")
    pruning_log.to_csv(pruning_log_path, index=False)

    kept_removed_path = os.path.join(args.output_dir, "pruned_label_summary.csv")
    pd.DataFrame({
        "descriptor_index": np.arange(labels_original.shape[1]),
        "descriptor": descriptor_names,
        "kept_after_pruning": [i in set(keep_idx) for i in range(labels_original.shape[1])],
    }).to_csv(kept_removed_path, index=False)

    # ------------------------------------------------------------
    # Load embeddings
    # ------------------------------------------------------------

    files = resolve_embedding_files(
        embeddings_dir=args.embeddings_dir,
        embedding_glob=args.embedding_glob,
        keep_last_epoch_per_seed=args.keep_last_epoch_per_seed,
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No embedding files found in {args.embeddings_dir} "
            f"with glob {args.embedding_glob}"
        )

    embeddings = [
        load_embedding(path, expected_n=labels_original.shape[0])
        for path in files
    ]

    print("Loaded embeddings:")
    print("  n files:", len(files))
    print("  first file:", files[0])

    # ------------------------------------------------------------
    # Precompute radii
    # ------------------------------------------------------------

    radii_by_seed = np.vstack([
        hyperbolic_radius(z)
        for z in embeddings
    ])

    # ------------------------------------------------------------
    # Distance preservation
    # ------------------------------------------------------------

    distance_rows = []

    if not args.skip_distance:
        for path, z in zip(files, embeddings):
            epoch, seed = parse_epoch_and_seed(path)
            d_metrics = distance_preservation_for_one_seed(labels_original, z)

            distance_rows.append({
                "embedding_file": path,
                "epoch": epoch,
                "seed": seed,
                **d_metrics,
            })

        distance_pearson_mean, distance_pearson_std = summarize(
            [r["distance_pearson"] for r in distance_rows]
        )
        distance_spearman_mean, distance_spearman_std = summarize(
            [r["distance_spearman"] for r in distance_rows]
        )

    else:
        distance_pearson_mean = np.nan
        distance_pearson_std = np.nan
        distance_spearman_mean = np.nan
        distance_spearman_std = np.nan

    # ------------------------------------------------------------
    # Radial entropy metrics and p-values
    # ------------------------------------------------------------

    summary_rows = []
    entropy_radial_rows = []

    for entropy_name, H in entropy_variants.items():
        print("\nEvaluating:", entropy_name)

        pearsons = []
        spearmans = []

        for path, radius in zip(files, radii_by_seed):
            epoch, seed = parse_epoch_and_seed(path)

            metrics = radial_correlations_from_radius(radius, H)

            pearsons.append(metrics["radius_pearson"])
            spearmans.append(metrics["radius_spearman"])

            entropy_radial_rows.append({
                "entropy_type": entropy_name,
                "embedding_file": path,
                "epoch": epoch,
                "seed": seed,
                **metrics,
            })

        radius_pearson_mean, radius_pearson_std = summarize(pearsons)
        radius_spearman_mean, radius_spearman_std = summarize(spearmans)

        p_mol = permutation_p_value_radial_molecule(
            radii_by_seed=radii_by_seed,
            y=H,
            n_permutations=args.n_permutations,
            rng=rng,
        )

        summary_rows.append({
            "config": f"original_input_{entropy_name}",
            "training_input": "original_binary_labels",
            "entropy_type": entropy_name,
            "n_molecules": labels_original.shape[0],
            "n_original_labels": labels_original.shape[1],
            "n_pruned_labels": labels_pruned.shape[1],
            "pca_components": n_components,
            "n_embedding_files": len(files),
            "distance_pearson_mean": distance_pearson_mean,
            "distance_pearson_std": distance_pearson_std,
            "distance_spearman_mean": distance_spearman_mean,
            "distance_spearman_std": distance_spearman_std,
            "radius_entropy_pearson_mean": radius_pearson_mean,
            "radius_entropy_pearson_std": radius_pearson_std,
            "radius_entropy_spearman_mean": radius_spearman_mean,
            "radius_entropy_spearman_std": radius_spearman_std,
            "p_mol_entropy_radial": p_mol,
        })

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)
    entropy_radial_df = pd.DataFrame(entropy_radial_rows)
    distance_df = pd.DataFrame(distance_rows)

    summary_path = os.path.join(args.output_dir, "summary_by_entropy_type.csv")
    entropy_radial_path = os.path.join(args.output_dir, "entropy_radial_metrics_by_seed.csv")
    distance_path = os.path.join(args.output_dir, "distance_metrics_by_seed.csv")

    summary_df.to_csv(summary_path, index=False)
    entropy_radial_df.to_csv(entropy_radial_path, index=False)
    distance_df.to_csv(distance_path, index=False)

    print("\nSaved:")
    print(" ", summary_path)
    print(" ", entropy_radial_path)
    print(" ", distance_path)
    print(" ", pruning_log_path)
    print(" ", kept_removed_path)

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()