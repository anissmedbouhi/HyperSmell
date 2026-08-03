"""
Evaluate already-trained Sagar hyperbolic MDS embeddings.

This script assumes that the embeddings have already been trained and saved as .npy files.

What it computes, for each configuration below:
    1. Embedding quality:
       - correlation between input pairwise distances and hyperbolic embedding pairwise distances
       - distance Pearson
       - distance Spearman

    2. Radial entropy organization:
       - correlation between hyperbolic radius and entropy
       - radius-entropy Pearson
       - radius-entropy Spearman
       - optional within-subject and odorant-block permutation p-values

    3. Optional descriptor tables:
       - radial correlation for each descriptor
       - angular directional R2 for each descriptor
       - optional permutation p-values

Before running, edit EXPERIMENTS below so that each embedding_glob points to the files
corresponding to the correct training input.
"""

# python run_sagar_hyperbolic_mds_evaluation.py   --base_dir /xxx/data/   --representation_name pom   --embeddings_dir results_ChemicalSenses_experiments_Sagar   --output_dir evaluation_results_sagar   --keep_last_epoch_per_seed

# python run_sagar_radius_entropy_ablation.py \
#   --base_dir /home/xxx/data/ \
#   --representation_name pom \
#   --embeddings_dir results_ChemicalSenses_experiments_Sagar \
#   --output_dir evaluation_results_sagar_radius_entropy_ablation \
#   --keep_last_epoch_per_seed

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from utils.helpers import read_embeddings, select_descriptors
from constants import sagar_tasks
from distances import distance_matrix, poincare_distance
from methods import log_map


# ---------------------------------------------------------------------
# EDIT THIS PART
# ---------------------------------------------------------------------

#correlation threshold >0.3: [11, 5, 1, 6, 7, 13]
PRUNED_DESCRIPTOR_NAMES = ["Pleasantness", "Decayed", "Musky", "Fruity", "Sweet", "Bakery"]

PCA_N_COMPONENTS = 15

# Each line is one experiment/configuration.
# IMPORTANT: edit embedding_glob so that it finds the already-trained .npy files.
# The glob is interpreted relative to --embeddings_dir unless it is an absolute path.
EXPERIMENTS = [
    {
        "name": "A_original_input_original_entropy",
        "training_input": "original",
        "entropy_type": "original",
        "embedding_glob": "*original*/*.npy",
    },
    {
        "name": "B_original_input_pruned_entropy",
        "training_input": "original",
        "entropy_type": "pruned",
        "embedding_glob": "*original*/*.npy",
    },
    {
        "name": "C_pruned_input_pruned_entropy",
        "training_input": "pruned",
        "entropy_type": "pruned",
        "embedding_glob": "*pruned*/*.npy",
    },
    {
        "name": "D_pruned_input_original_entropy",
        "training_input": "pruned",
        "entropy_type": "original",
        "embedding_glob": "*pruned*/*.npy",
    },
    {
        "name": "E_pca_input_original_entropy",
        "training_input": "pca",
        "entropy_type": "original",
        "embedding_glob": "*pca*/*.npy",
    },
    # Optional. Use only if entropy computed/interpreted on PCA coordinates.
    # {
    #     "name": "F_pca_input_pca_entropy",
    #     "training_input": "pca",
    #     "entropy_type": "pca",
    #     "embedding_glob": "*pca*/*.npy",
    # },
]


# ---------------------------------------------------------------------
# Small utility functions
# ---------------------------------------------------------------------

def ensure_base_dir_has_slash(base_dir: str) -> str:
    if base_dir.endswith(os.sep):
        return base_dir
    return base_dir + os.sep


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x)
    y = np.asarray(y)

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

    elif method == "spearman":
        res = spearmanr(x, y)
        return float(getattr(res, "statistic", res[0]))

    else:
        raise ValueError("method must be either 'pearson' or 'spearman'")


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def upper_triangle_values(D):
    D = np.asarray(D)
    n = D.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return D[mask]


def entropy_from_scores(scores):
    scores = np.asarray(scores, dtype=float)
    p = softmax(scores, axis=1)
    return -(p * np.log(p + 1e-12)).sum(axis=1)


def project_inside_poincare_disk(z, eps=1e-5):
    """Evaluation-only safety projection if points are numerically outside the disk."""
    z = np.asarray(z, dtype=np.float32)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    scale = np.where(norms >= 1.0 - eps, (1.0 - eps) / np.maximum(norms, eps), 1.0)
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


def tangent_coordinates_at_origin(z):
    z = project_inside_poincare_disk(z)
    z_torch = torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        u = log_map(z_torch).cpu().numpy()
    return u


def directional_r2(z, y):
    """Fit y approximately beta^T log_0(z) + b and return R2."""
    y = np.asarray(y, dtype=float).reshape(-1)
    u = tangent_coordinates_at_origin(z)

    valid = np.isfinite(y) & np.all(np.isfinite(u), axis=1)
    y = y[valid]
    u = u[valid]

    if len(y) < 4 or np.var(y) == 0:
        return np.nan

    A = np.column_stack([u[:, 0], u[:, 1], np.ones(len(u))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_sagar_union(base_dir: str, representation_name: str):
    base_dir = ensure_base_dir_has_slash(base_dir)
    input_embeddings = f"embeddings/{representation_name}/sagar_{representation_name}_embeddings_13_Apr17.csv"

    embeddings, labels, subjects, cids = read_embeddings(
        base_dir,
        select_descriptors("sagar"),
        input_embeddings,
        grand_avg=False,
    )

    labels = to_numpy(labels).astype(float)
    subjects = to_numpy(subjects)
    cids = to_numpy(cids)

    return labels, subjects, cids


def pruned_indices():
    missing = [name for name in PRUNED_DESCRIPTOR_NAMES if name not in sagar_tasks]
    if missing:
        raise ValueError(f"These PRUNED_DESCRIPTOR_NAMES are not in sagar_tasks: {missing}")
    remove_idx = [sagar_tasks.index(name) for name in PRUNED_DESCRIPTOR_NAMES]
    keep_idx = [i for i in range(len(sagar_tasks)) if i not in remove_idx]
    return keep_idx


def make_input_variants(labels_original: np.ndarray) -> Dict[str, np.ndarray]:
    keep_idx = pruned_indices()

    labels_pruned = labels_original[:, keep_idx]

    n_components = min(PCA_N_COMPONENTS, labels_original.shape[1])
    labels_pca = PCA(n_components=n_components).fit_transform(labels_original)

    return {
        "original": labels_original,
        "pruned": labels_pruned,
        "pca": labels_pca,
    }


def make_entropy_variants(input_variants: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "original": entropy_from_scores(input_variants["original"]),
        "pruned": entropy_from_scores(input_variants["pruned"]),
        "pca": entropy_from_scores(input_variants["pca"]),
    }


# ---------------------------------------------------------------------
# Embedding file loading
# ---------------------------------------------------------------------

def parse_epoch_and_seed(path: str) -> Tuple[int, str]:
    """
    The save_embeddings_npy filenames start with embeddings_{epoch}_...
    and, in the standard train.py pattern, the seed is the 8th token.
    If parsing fails, we fall back to grouping by the whole filename.
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


def resolve_embedding_files(embeddings_dir: str, embedding_glob: str, keep_last_epoch_per_seed: bool) -> List[str]:
    if os.path.isabs(embedding_glob):
        pattern = embedding_glob
    else:
        pattern = os.path.join(embeddings_dir, embedding_glob)

    paths = sorted(glob.glob(pattern, recursive=True))
    paths = [p for p in paths if p.endswith(".npy")]

    if keep_last_epoch_per_seed:
        paths = select_last_epoch_per_seed(paths)

    return paths


def load_embedding(path: str, expected_n: int) -> np.ndarray:
    z = np.load(path)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f"Embedding {path} has shape {z.shape}, expected (N, 2).")
    if z.shape[0] != expected_n:
        raise ValueError(f"Embedding {path} has {z.shape[0]} rows, expected {expected_n}.")
    return z.astype(np.float32)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def distance_preservation_for_one_seed(training_input: np.ndarray, z: np.ndarray):
    D_in = cdist(training_input, training_input, metric="euclidean")
    D_emb = hyperbolic_distance_matrix(z)

    d_in = upper_triangle_values(D_in)
    d_emb = upper_triangle_values(D_emb)

    return {
        "distance_pearson": safe_corr(d_in, d_emb, "pearson"),
        "distance_spearman": safe_corr(d_in, d_emb, "spearman"),
    }


def radial_correlations_for_one_seed(z: np.ndarray, y: np.ndarray):
    r = hyperbolic_radius(z)
    return {
        "radius_pearson": safe_corr(r, y, "pearson"),
        "radius_spearman": safe_corr(r, y, "spearman"),
    }


def mean_radius_pearson(embeddings: List[np.ndarray], y: np.ndarray):
    vals = [radial_correlations_for_one_seed(z, y)["radius_pearson"] for z in embeddings]
    return float(np.nanmean(vals))


def mean_directional_r2(embeddings: List[np.ndarray], y: np.ndarray):
    vals = [directional_r2(z, y) for z in embeddings]
    return float(np.nanmean(vals))


# ---------------------------------------------------------------------
# Permutations
# ---------------------------------------------------------------------

def permute_within_subject(y: np.ndarray, subjects: np.ndarray, rng: np.random.Generator):
    y_perm = np.asarray(y).copy()
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        y_perm[idx] = y_perm[rng.permutation(idx)]
    return y_perm


def permute_odorant_blocks(y: np.ndarray, subjects: np.ndarray, cids: np.ndarray, rng: np.random.Generator):
    """
    Odorant-block permutation for an incomplete CID x subject grid.

    CIDs are shuffled only among odorants that have the same set of observed subjects.
    This preserves the repeated-measures structure and the missingness pattern.
    """
    y = np.asarray(y)
    y_perm = np.asarray(y).copy()

    unique_cids = np.unique(cids)

    # For each CID, record which subjects are present.
    cid_to_subject_pattern = {}
    for cid in unique_cids:
        subj_pattern = tuple(sorted(np.unique(subjects[cids == cid]).tolist()))
        cid_to_subject_pattern[cid] = subj_pattern

    # Group CIDs by their subject-availability pattern.
    pattern_to_cids = {}
    for cid, pattern in cid_to_subject_pattern.items():
        if pattern not in pattern_to_cids:
            pattern_to_cids[pattern] = []
        pattern_to_cids[pattern].append(cid)

    # Lookup value by observed (CID, subject) pair.
    lookup = {(cids[i], subjects[i]): y[i] for i in range(len(y))}

    # Shuffle CIDs within each pattern group.
    for pattern, cids_in_group in pattern_to_cids.items():
        cids_in_group = np.array(cids_in_group)

        # If there is only one CID with this pattern, it cannot be permuted.
        if len(cids_in_group) < 2:
            continue

        shuffled_cids = rng.permutation(cids_in_group)
        cid_map = dict(zip(cids_in_group, shuffled_cids))

        for cid in cids_in_group:
            source_cid = cid_map[cid]

            for subj in pattern:
                idx = np.where((cids == cid) & (subjects == subj))[0]

                if len(idx) != 1:
                    raise ValueError(
                        "Expected exactly one row for each observed (CID, subject) pair."
                    )

                y_perm[idx[0]] = lookup[(source_cid, subj)]

    return y_perm


def permutation_p_value_radial(
    embeddings: List[np.ndarray],
    y: np.ndarray,
    subjects: np.ndarray,
    cids: np.ndarray,
    scheme: str,
    n_permutations: int,
    rng: np.random.Generator,
):
    if n_permutations <= 0:
        return np.nan

    obs = mean_radius_pearson(embeddings, y)
    count = 0

    for _ in range(n_permutations):
        if scheme == "WS":
            y_perm = permute_within_subject(y, subjects, rng)
        elif scheme == "OB":
            y_perm = permute_odorant_blocks(y, subjects, cids, rng)
        else:
            raise ValueError("scheme must be 'WS' or 'OB'")

        stat = mean_radius_pearson(embeddings, y_perm)
        if abs(stat) >= abs(obs):
            count += 1

    return (1 + count) / (n_permutations + 1)


def permutation_p_value_angular(
    embeddings: List[np.ndarray],
    y: np.ndarray,
    subjects: np.ndarray,
    cids: np.ndarray,
    scheme: str,
    n_permutations: int,
    rng: np.random.Generator,
):
    if n_permutations <= 0:
        return np.nan

    obs = mean_directional_r2(embeddings, y)
    count = 0

    for _ in range(n_permutations):
        if scheme == "WS":
            y_perm = permute_within_subject(y, subjects, rng)
        elif scheme == "OB":
            y_perm = permute_odorant_blocks(y, subjects, cids, rng)
        else:
            raise ValueError("scheme must be 'WS' or 'OB'")

        stat = mean_directional_r2(embeddings, y_perm)
        if stat >= obs:
            count += 1

    return (1 + count) / (n_permutations + 1)


# ---------------------------------------------------------------------
# Evaluation per configuration
# ---------------------------------------------------------------------

def evaluate_config(
    config: dict,
    embeddings_dir: str,
    labels_original: np.ndarray,
    input_variants: Dict[str, np.ndarray],
    entropy_variants: Dict[str, np.ndarray],
    subjects: np.ndarray,
    cids: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    keep_last_epoch_per_seed: bool,
):
    training_input_name = config["training_input"]
    entropy_type = config["entropy_type"]

    training_input = input_variants[training_input_name]
    entropy_values = entropy_variants[entropy_type]

    files = resolve_embedding_files(
        embeddings_dir=embeddings_dir,
        embedding_glob=config["embedding_glob"],
        keep_last_epoch_per_seed=keep_last_epoch_per_seed,
    )

    if len(files) == 0:
        print(f"WARNING: no embedding files found for {config['name']} with glob {config['embedding_glob']}")
        return None, [], []

    embeddings = [load_embedding(path, expected_n=len(labels_original)) for path in files]

    distance_rows = []
    radial_rows = []

    for path, z in zip(files, embeddings):
        epoch, seed = parse_epoch_and_seed(path)

        d_metrics = distance_preservation_for_one_seed(training_input, z)
        r_metrics = radial_correlations_for_one_seed(z, entropy_values)

        distance_rows.append({"embedding_file": path, "epoch": epoch, "seed": seed, **d_metrics})
        radial_rows.append({"embedding_file": path, "epoch": epoch, "seed": seed, **r_metrics})

    distance_pearson_mean, distance_pearson_std = summarize([r["distance_pearson"] for r in distance_rows])
    distance_spearman_mean, distance_spearman_std = summarize([r["distance_spearman"] for r in distance_rows])
    radius_pearson_mean, radius_pearson_std = summarize([r["radius_pearson"] for r in radial_rows])
    radius_spearman_mean, radius_spearman_std = summarize([r["radius_spearman"] for r in radial_rows])

    p_ws = permutation_p_value_radial(
        embeddings, entropy_values, subjects, cids, "WS", n_permutations, rng
    )
    p_ob = permutation_p_value_radial(
        embeddings, entropy_values, subjects, cids, "OB", n_permutations, rng
    )

    summary = {
        "config": config["name"],
        "training_input": training_input_name,
        "entropy_type": entropy_type,
        "embedding_glob": config["embedding_glob"],
        "n_embedding_files": len(files),
        "distance_pearson_mean": distance_pearson_mean,
        "distance_pearson_std": distance_pearson_std,
        "distance_spearman_mean": distance_spearman_mean,
        "distance_spearman_std": distance_spearman_std,
        "radius_entropy_pearson_mean": radius_pearson_mean,
        "radius_entropy_pearson_std": radius_pearson_std,
        "radius_entropy_spearman_mean": radius_spearman_mean,
        "radius_entropy_spearman_std": radius_spearman_std,
        "p_WS_entropy_radial": p_ws,
        "p_OB_entropy_radial": p_ob,
    }

    return summary, distance_rows, radial_rows


def descriptor_tables_for_config(
    config: dict,
    embeddings_dir: str,
    labels_original: np.ndarray,
    subjects: np.ndarray,
    cids: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    keep_last_epoch_per_seed: bool,
):
    files = resolve_embedding_files(
        embeddings_dir=embeddings_dir,
        embedding_glob=config["embedding_glob"],
        keep_last_epoch_per_seed=keep_last_epoch_per_seed,
    )
    if len(files) == 0:
        return [], []

    embeddings = [load_embedding(path, expected_n=len(labels_original)) for path in files]

    radial_rows = []
    angular_rows = []

    for k, descriptor_name in enumerate(sagar_tasks):
        y = labels_original[:, k]

        pearsons = [radial_correlations_for_one_seed(z, y)["radius_pearson"] for z in embeddings]
        spearmans = [radial_correlations_for_one_seed(z, y)["radius_spearman"] for z in embeddings]
        r2s = [directional_r2(z, y) for z in embeddings]

        pearson_mean, pearson_std = summarize(pearsons)
        spearman_mean, spearman_std = summarize(spearmans)
        r2_mean, r2_std = summarize(r2s)

        p_rad_ws = permutation_p_value_radial(embeddings, y, subjects, cids, "WS", n_permutations, rng)
        p_rad_ob = permutation_p_value_radial(embeddings, y, subjects, cids, "OB", n_permutations, rng)
        p_ang_ws = permutation_p_value_angular(embeddings, y, subjects, cids, "WS", n_permutations, rng)
        p_ang_ob = permutation_p_value_angular(embeddings, y, subjects, cids, "OB", n_permutations, rng)

        radial_rows.append({
            "config": config["name"],
            "training_input": config["training_input"],
            "descriptor_index": k,
            "descriptor": descriptor_name,
            "radius_descriptor_pearson_mean": pearson_mean,
            "radius_descriptor_pearson_std": pearson_std,
            "radius_descriptor_spearman_mean": spearman_mean,
            "radius_descriptor_spearman_std": spearman_std,
            "p_WS_radial": p_rad_ws,
            "p_OB_radial": p_rad_ob,
        })

        angular_rows.append({
            "config": config["name"],
            "training_input": config["training_input"],
            "descriptor_index": k,
            "descriptor": descriptor_name,
            "directional_R2_mean": r2_mean,
            "directional_R2_std": r2_std,
            "p_WS_angular": p_ang_ws,
            "p_OB_angular": p_ang_ob,
        })

    return radial_rows, angular_rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

### Run:
# python run_sagar_hyperbolic_mds_evaluation.py \
#   --base_dir /xxx/data/ \
#   --representation_name pom \
#   --embeddings_dir results_ChemicalSenses_experiments_Sagar \
#   --output_dir evaluation_results_sagar \
#   --keep_last_epoch_per_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./data/", help="Base data directory used by your read_embeddings function.")
    parser.add_argument("--representation_name", type=str, default="pom", help="Usually 'pom'.")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Folder containing already-trained .npy embeddings.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_sagar", help="Where CSV results are saved.")
    parser.add_argument("--n_permutations", type=int, default=1000, help="Use 0 to skip permutation p-values.")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--keep_last_epoch_per_seed", action="store_true", help="Keep only the largest saved epoch for each seed.")
    parser.add_argument("--skip_descriptor_tables", action="store_true", help="Only compute the summary table.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    labels_original, subjects, cids = load_sagar_union(args.base_dir, args.representation_name)
    input_variants = make_input_variants(labels_original)
    entropy_variants = make_entropy_variants(input_variants)

    print("Loaded Sagar union data:")
    print("  labels:", labels_original.shape)
    print("  subjects:", np.unique(subjects))
    print("  CIDs:", len(np.unique(cids)))
    print("  pruned descriptors removed:", PRUNED_DESCRIPTOR_NAMES)
    print("  PCA components:", input_variants["pca"].shape[1])

    summary_rows = []
    all_distance_rows = []
    all_entropy_radial_rows = []
    all_descriptor_radial_rows = []
    all_descriptor_angular_rows = []

    for config in EXPERIMENTS:
        print("\nEvaluating", config["name"])
        summary, distance_rows, entropy_radial_rows = evaluate_config(
            config=config,
            embeddings_dir=args.embeddings_dir,
            labels_original=labels_original,
            input_variants=input_variants,
            entropy_variants=entropy_variants,
            subjects=subjects,
            cids=cids,
            n_permutations=args.n_permutations,
            rng=rng,
            keep_last_epoch_per_seed=args.keep_last_epoch_per_seed,
        )

        if summary is None:
            continue

        summary_rows.append(summary)
        for row in distance_rows:
            row["config"] = config["name"]
            row["training_input"] = config["training_input"]
            all_distance_rows.append(row)
        for row in entropy_radial_rows:
            row["config"] = config["name"]
            row["training_input"] = config["training_input"]
            row["entropy_type"] = config["entropy_type"]
            all_entropy_radial_rows.append(row)

        if not args.skip_descriptor_tables:
            print("  Computing descriptor radial/angular tables")
            desc_radial_rows, desc_angular_rows = descriptor_tables_for_config(
                config=config,
                embeddings_dir=args.embeddings_dir,
                labels_original=labels_original,
                subjects=subjects,
                cids=cids,
                n_permutations=args.n_permutations,
                rng=rng,
                keep_last_epoch_per_seed=args.keep_last_epoch_per_seed,
            )
            all_descriptor_radial_rows.extend(desc_radial_rows)
            all_descriptor_angular_rows.extend(desc_angular_rows)

    summary_df = pd.DataFrame(summary_rows)
    distance_df = pd.DataFrame(all_distance_rows)
    entropy_radial_df = pd.DataFrame(all_entropy_radial_rows)
    descriptor_radial_df = pd.DataFrame(all_descriptor_radial_rows)
    descriptor_angular_df = pd.DataFrame(all_descriptor_angular_rows)

    summary_path = os.path.join(args.output_dir, "summary_by_configuration.csv")
    distance_path = os.path.join(args.output_dir, "distance_metrics_by_seed.csv")
    entropy_radial_path = os.path.join(args.output_dir, "entropy_radial_metrics_by_seed.csv")
    descriptor_radial_path = os.path.join(args.output_dir, "radial_descriptor_table.csv")
    descriptor_angular_path = os.path.join(args.output_dir, "angular_descriptor_table.csv")

    summary_df.to_csv(summary_path, index=False)
    distance_df.to_csv(distance_path, index=False)
    entropy_radial_df.to_csv(entropy_radial_path, index=False)
    descriptor_radial_df.to_csv(descriptor_radial_path, index=False)
    descriptor_angular_df.to_csv(descriptor_angular_path, index=False)

    print("\nSaved:")
    print(" ", summary_path)
    print(" ", distance_path)
    print(" ", entropy_radial_path)
    if not args.skip_descriptor_tables:
        print(" ", descriptor_radial_path)
        print(" ", descriptor_angular_path)

    if len(summary_df) > 0:
        print("\nSummary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
