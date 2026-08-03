"""
Subject-level analysis from union Sagar hyperbolic embeddings.

This script assumes that the embeddings have already been trained on the Sagar union
set and saved as .npy files in:

    <embeddings_dir>/original/*.npy

For each saved union embedding seed, the script subsets the rows corresponding to
each subject and computes:

1. Subject-level radial entropy analysis:
   - radius--entropy Pearson correlation
   - radius--entropy Spearman correlation
   - one subject-level permutation p-value, computed by shuffling entropy values
     within that subject and comparing the mean Pearson correlation across seeds

2. Subject-level angular descriptor analysis:
   - directional R2 for every descriptor, for every subject
   - one subject-level permutation p-value per descriptor, computed by shuffling
     descriptor values within that subject and comparing the mean R2 across seeds

Embeddings are not retrained, we analyze here each subject inside the already-trained union embedding.

python run_sagar_individual_subjects_from_union_analysis.py \
  --base_dir /xxx/data/ \
  --representation_name pom \
  --embeddings_dir results_ChemicalSenses_experiments_Sagar \
  --output_dir evaluation_results_sagar_individual_subjects_from_union \
  --keep_last_epoch_per_seed
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from utils.helpers import read_embeddings, select_descriptors
from constants import sagar_tasks
from distances import distance_matrix, poincare_distance
from methods import log_map


# ---------------------------------------------------------------------
# Utility functions
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
    if method == "spearman":
        res = spearmanr(x, y)
        return float(getattr(res, "statistic", res[0]))

    raise ValueError("method must be 'pearson' or 'spearman'")


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def entropy_from_scores(scores):
    """
    Convert descriptor scores to a softmax distribution and compute entropy.
    """
    scores = np.asarray(scores, dtype=float)
    p = softmax(scores, axis=1)
    return -(p * np.log(p + 1e-12)).sum(axis=1)


def project_inside_poincare_disk(z, eps=1e-5):
    """Evaluation-only safety projection if points are numerically outside the disk."""
    z = np.asarray(z, dtype=np.float32)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    scale = np.where(norms >= 1.0 - eps, (1.0 - eps) / np.maximum(norms, eps), 1.0)
    return z * scale


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


def radial_correlations_for_one_seed(z: np.ndarray, y: np.ndarray):
    r = hyperbolic_radius(z)
    return {
        "radius_pearson": safe_corr(r, y, "pearson"),
        "radius_spearman": safe_corr(r, y, "spearman"),
    }


def directional_r2(z, y):
    """
    Fit y approximately beta^T log_0(z) + b and return R2.
    This is the angular descriptor statistic used in our paper.
    """
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
# Loading and embedding-file handling
# ---------------------------------------------------------------------

def load_sagar_union(base_dir: str, representation_name: str):
    base_dir = ensure_base_dir_has_slash(base_dir)
    input_embeddings = f"embeddings/{representation_name}/sagar_{representation_name}_embeddings_13_Apr17.csv"

    _, labels, subjects, cids = read_embeddings(
        base_dir,
        select_descriptors("sagar"),
        input_embeddings,
        grand_avg=False,
    )

    labels = to_numpy(labels).astype(float)
    subjects = to_numpy(subjects)
    cids = to_numpy(cids)

    return labels, subjects, cids


def parse_epoch_and_seed(path: str) -> Tuple[int, str]:
    """
    The saved filenames usually start with embeddings_{epoch}_...
    and the seed is usually the 8th token. If parsing fails, we use the stem.
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


def resolve_original_embedding_files(
    embeddings_dir: str,
    keep_last_epoch_per_seed: bool,
    recursive: bool = False,
):
    if recursive:
        pattern = os.path.join(embeddings_dir, "original", "**", "*.npy")
        paths = sorted(glob.glob(pattern, recursive=True))
    else:
        pattern = os.path.join(embeddings_dir, "original", "*.npy")
        paths = sorted(glob.glob(pattern))

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
# Subject subsetting and permutation tests
# ---------------------------------------------------------------------

def subject_indices(subjects: np.ndarray, subject_id):
    return np.where(subjects == subject_id)[0]


def mean_radius_pearson(embeddings: List[np.ndarray], y: np.ndarray):
    vals = [radial_correlations_for_one_seed(z, y)["radius_pearson"] for z in embeddings]
    return float(np.nanmean(vals))


def mean_directional_r2(embeddings: List[np.ndarray], y: np.ndarray):
    vals = [directional_r2(z, y) for z in embeddings]
    return float(np.nanmean(vals))


def permutation_p_value_radial_subject(
    embeddings: List[np.ndarray],
    y: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
):
    """
    Subject-level radial p-value.
    The unit is the odorant observation inside one subject, so we simply shuffle y
    within that subject. The test is two-sided for signed Pearson correlation.
    """
    if n_permutations <= 0:
        return np.nan

    obs = mean_radius_pearson(embeddings, y)
    count = 0

    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        stat = mean_radius_pearson(embeddings, y_perm)
        if abs(stat) >= abs(obs):
            count += 1

    return (1 + count) / (n_permutations + 1)


def permutation_p_value_angular_subject(
    embeddings: List[np.ndarray],
    y: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
):
    """
    Subject-level angular p-value.
    The statistic is mean directional R2 across seeds, so the test is one-sided:
    larger R2 means stronger angular organization.
    """
    if n_permutations <= 0:
        return np.nan

    obs = mean_directional_r2(embeddings, y)
    count = 0

    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        stat = mean_directional_r2(embeddings, y_perm)
        if stat >= obs:
            count += 1

    return (1 + count) / (n_permutations + 1)


# ---------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------

def fmt_mean_std(mean, std):
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"${mean:.2f}$"
    return f"${mean:.2f} \\pm {std:.2f}$"


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<0.001$"
    return f"${p:.3f}$"


def write_entropy_latex_table(summary_df: pd.DataFrame, path: str):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Subject & $N$ & Radial Pearson & Radial Spearman & $p_{\mathrm{perm}}$ \\")
    lines.append(r"\midrule")

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['subject']} & {int(row['n_observations'])} & "
            f"{fmt_mean_std(row['radius_entropy_pearson_mean'], row['radius_entropy_pearson_std'])} & "
            f"{fmt_mean_std(row['radius_entropy_spearman_mean'], row['radius_entropy_spearman_std'])} & "
            f"{fmt_p(row['p_perm_radial_entropy'])} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Subject-level radial entropy analysis within the Sagar union embedding. The union embedding is kept fixed and radius--entropy correlations are computed separately for each subject. Values are reported as mean and standard deviation over random seeds. Permutation p-values were computed by shuffling entropy values within each subject.}")
    lines.append(r"\label{tab:sagar_subject_entropy_union}")
    lines.append(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_angular_latex_table(angular_df: pd.DataFrame, path: str, threshold: float):
    df = angular_df.copy()
    if threshold is not None:
        df = df[df["directional_R2_mean"] > threshold].copy()
    df = df.sort_values(["subject", "directional_R2_mean"], ascending=[True, False])

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"Subject & Descriptor & Directional $R^2$ & $p_{\mathrm{perm}}$ \\")
    lines.append(r"\midrule")

    previous_subject = None
    for _, row in df.iterrows():
        subject_text = str(row["subject"])
        if previous_subject is not None and subject_text != previous_subject:
            lines.append(r"\midrule")
        previous_subject = subject_text

        lines.append(
            f"{subject_text} & {row['descriptor']} & "
            f"{fmt_mean_std(row['directional_R2_mean'], row['directional_R2_std'])} & "
            f"{fmt_p(row['p_perm_angular'])} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if threshold is None:
        caption = "Full subject-level angular descriptor analysis within the Sagar union embedding."
    else:
        caption = f"Subject-level angular descriptor analysis within the Sagar union embedding for descriptors with mean directional $R^2>{threshold}$."
    caption += " Values are reported as mean and standard deviation over random seeds. Permutation p-values were computed by shuffling descriptor values within each subject."
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{tab:sagar_subject_angular_union}")
    lines.append(r"\end{table*}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./data/", help="Base data directory used by read_embeddings.")
    parser.add_argument("--representation_name", type=str, default="pom", help="Usually 'pom'.")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Folder containing original/*.npy union embeddings.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_sagar_subject_union", help="Where CSV and LaTeX results are saved.")
    parser.add_argument("--n_permutations", type=int, default=1000, help="Use 0 to skip permutation p-values.")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--keep_last_epoch_per_seed", action="store_true", help="Keep only largest saved epoch for each seed.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively inside original/.")
    parser.add_argument("--angular_r2_threshold", type=float, default=0.2, help="Threshold for the filtered angular LaTeX table.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    labels, subjects, cids = load_sagar_union(args.base_dir, args.representation_name)
    unique_subjects = np.unique(subjects)

    files = resolve_original_embedding_files(
        args.embeddings_dir,
        keep_last_epoch_per_seed=args.keep_last_epoch_per_seed,
        recursive=args.recursive,
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No .npy embeddings found in {os.path.join(args.embeddings_dir, 'original')}"
        )

    print("Loaded Sagar union data:")
    print("  labels:", labels.shape)
    print("  subjects:", unique_subjects)
    print("  CIDs:", len(np.unique(cids)))
    print("Found original union embedding files:", len(files))

    # Load all full union embeddings once.
    full_embeddings = []
    file_meta = []
    for path in files:
        epoch, seed = parse_epoch_and_seed(path)
        z = load_embedding(path, expected_n=len(labels))
        full_embeddings.append(z)
        file_meta.append({"embedding_file": path, "epoch": epoch, "seed": seed})

    # Precompute subject-specific data and subject-specific embedding subsets.
    subject_data: Dict[str, dict] = {}
    for subject_id in unique_subjects:
        idx = subject_indices(subjects, subject_id)
        subject_key = str(subject_id)
        subject_data[subject_key] = {
            "subject_id": subject_id,
            "idx": idx,
            "labels": labels[idx],
            "cids": cids[idx],
            "entropy": entropy_from_scores(labels[idx]),
            "embeddings": [z[idx] for z in full_embeddings],
            "n_observations": len(idx),
            "n_cids": len(np.unique(cids[idx])),
        }

    # --------------------------------------------------
    # Radial entropy analysis by subject
    # --------------------------------------------------
    entropy_by_seed_rows = []
    entropy_summary_rows = []

    for subject_key, data in subject_data.items():
        y = data["entropy"]
        embeddings_s = data["embeddings"]

        pearsons = []
        spearmans = []

        for meta, z_s in zip(file_meta, embeddings_s):
            metrics = radial_correlations_for_one_seed(z_s, y)
            pearsons.append(metrics["radius_pearson"])
            spearmans.append(metrics["radius_spearman"])

            entropy_by_seed_rows.append({
                "subject": subject_key,
                "n_observations": data["n_observations"],
                "n_cids": data["n_cids"],
                **meta,
                "radius_entropy_pearson": metrics["radius_pearson"],
                "radius_entropy_spearman": metrics["radius_spearman"],
            })

        pearson_mean, pearson_std = summarize(pearsons)
        spearman_mean, spearman_std = summarize(spearmans)

        p_perm = permutation_p_value_radial_subject(
            embeddings_s,
            y,
            args.n_permutations,
            rng,
        )

        entropy_summary_rows.append({
            "subject": subject_key,
            "n_observations": data["n_observations"],
            "n_cids": data["n_cids"],
            "radius_entropy_pearson_mean": pearson_mean,
            "radius_entropy_pearson_std": pearson_std,
            "radius_entropy_spearman_mean": spearman_mean,
            "radius_entropy_spearman_std": spearman_std,
            "p_perm_radial_entropy": p_perm,
        })

    entropy_by_seed_df = pd.DataFrame(entropy_by_seed_rows)
    entropy_summary_df = pd.DataFrame(entropy_summary_rows)

    # --------------------------------------------------
    # Angular descriptor analysis by subject
    # --------------------------------------------------
    angular_by_seed_rows = []
    angular_summary_rows = []

    for subject_key, data in subject_data.items():
        labels_s = data["labels"]
        embeddings_s = data["embeddings"]

        for k, descriptor_name in enumerate(sagar_tasks):
            y = labels_s[:, k]
            r2s = []

            for meta, z_s in zip(file_meta, embeddings_s):
                r2 = directional_r2(z_s, y)
                r2s.append(r2)

                angular_by_seed_rows.append({
                    "subject": subject_key,
                    "n_observations": data["n_observations"],
                    "n_cids": data["n_cids"],
                    "descriptor_index": k,
                    "descriptor": descriptor_name,
                    **meta,
                    "directional_R2": r2,
                })

            r2_mean, r2_std = summarize(r2s)

            p_perm = permutation_p_value_angular_subject(
                embeddings_s,
                y,
                args.n_permutations,
                rng,
            )

            angular_summary_rows.append({
                "subject": subject_key,
                "n_observations": data["n_observations"],
                "n_cids": data["n_cids"],
                "descriptor_index": k,
                "descriptor": descriptor_name,
                "directional_R2_mean": r2_mean,
                "directional_R2_std": r2_std,
                "p_perm_angular": p_perm,
            })

    angular_by_seed_df = pd.DataFrame(angular_by_seed_rows)
    angular_summary_df = pd.DataFrame(angular_summary_rows)
    angular_filtered_df = angular_summary_df[
        angular_summary_df["directional_R2_mean"] > args.angular_r2_threshold
    ].copy()

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    entropy_summary_path = os.path.join(args.output_dir, "subject_entropy_radial_summary.csv")
    entropy_by_seed_path = os.path.join(args.output_dir, "subject_entropy_radial_by_seed.csv")
    angular_summary_path = os.path.join(args.output_dir, "subject_angular_descriptor_summary.csv")
    angular_by_seed_path = os.path.join(args.output_dir, "subject_angular_descriptor_by_seed.csv")
    angular_filtered_path = os.path.join(args.output_dir, f"subject_angular_descriptor_R2_gt_{args.angular_r2_threshold}.csv")

    entropy_summary_df.to_csv(entropy_summary_path, index=False)
    entropy_by_seed_df.to_csv(entropy_by_seed_path, index=False)
    angular_summary_df.to_csv(angular_summary_path, index=False)
    angular_by_seed_df.to_csv(angular_by_seed_path, index=False)
    angular_filtered_df.to_csv(angular_filtered_path, index=False)

    write_entropy_latex_table(
        entropy_summary_df,
        os.path.join(args.output_dir, "subject_entropy_radial_table.tex"),
    )
    write_angular_latex_table(
        angular_filtered_df,
        os.path.join(args.output_dir, "subject_angular_descriptor_table_filtered.tex"),
        threshold=args.angular_r2_threshold,
    )
    write_angular_latex_table(
        angular_summary_df,
        os.path.join(args.output_dir, "subject_angular_descriptor_table_full.tex"),
        threshold=None,
    )

    print("\nSaved:")
    print(" ", entropy_summary_path)
    print(" ", entropy_by_seed_path)
    print(" ", angular_summary_path)
    print(" ", angular_by_seed_path)
    print(" ", angular_filtered_path)
    print(" ", os.path.join(args.output_dir, "subject_entropy_radial_table.tex"))
    print(" ", os.path.join(args.output_dir, "subject_angular_descriptor_table_filtered.tex"))
    print(" ", os.path.join(args.output_dir, "subject_angular_descriptor_table_full.tex"))

    print("\nSubject-level radial entropy summary:")
    print(entropy_summary_df.to_string(index=False))

    print(f"\nSubject-level angular descriptors with mean R2 > {args.angular_r2_threshold}:")
    if len(angular_filtered_df) == 0:
        print("None")
    else:
        cols = ["subject", "descriptor", "directional_R2_mean", "directional_R2_std", "p_perm_angular"]
        print(angular_filtered_df[cols].sort_values(["subject", "directional_R2_mean"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
