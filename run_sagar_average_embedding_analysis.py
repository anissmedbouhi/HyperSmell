"""
Evaluate Sagar embeddings trained on averaged (consensus) ratings.

This script assumes that embeddings have already been trained on an odorant-level
averaged/consensus Sagar dataset and saved as .npy files in:

    <embeddings_dir>/average_original/*.npy

It performs two analyses on the averaged-rating embeddings:

1. Radial entropy analysis
   - Computes entropy from the averaged descriptor vector of each CID.
   - Computes hyperbolic radius of each embedded CID.
   - Computes Pearson and Spearman correlations between radius and entropy.
   - Computes a CID-level permutation p-value.

2. Angular descriptor analysis
   - For each descriptor, maps the embedding to the tangent space at the origin.
   - Fits a linear model descriptor ~ tangent_x + tangent_y.
   - Reports directional R^2 per descriptor.
   - Computes a CID-level permutation p-value for each descriptor.

The script automatically builds the averaged labels from the union Sagar data.
By default, it uses all available CIDs. If the averaged embedding is trained only
on CIDs rated by all subjects, use:

    --consensus_mode all_subjects

Otherwise, use:

    --consensus_mode auto

which chooses all_available or all_subjects based on the number of rows in the
loaded .npy embeddings.

python run_sagar_average_embedding_analysis.py \
  --base_dir /xxx/data/ \
  --representation_name pom \
  --embeddings_dir results_ChemicalSenses_experiments_Sagar \
  --output_dir evaluation_results_sagar_average \
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
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


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
    """Load the union Sagar data: one row per subject--odorant observation."""
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


def make_consensus_labels(labels, subjects, cids, consensus_mode="all_available"):
    """
    Average descriptor ratings per CID.

    consensus_mode:
        - 'all_available': keep every CID; some averages may use 1, 2, or 3 subjects.
        - 'all_subjects': keep only CIDs observed for all subjects.
    """
    labels = np.asarray(labels, dtype=float)
    subjects = np.asarray(subjects)
    cids = np.asarray(cids)

    unique_subjects = np.unique(subjects)
    n_subjects_total = len(unique_subjects)

    rows = []
    for cid in np.unique(cids):
        idx = np.where(cids == cid)[0]
        subj_here = np.unique(subjects[idx])

        if consensus_mode == "all_subjects" and len(subj_here) < n_subjects_total:
            continue

        rows.append({
            "CID": cid,
            "n_subjects": len(subj_here),
            "labels": labels[idx].mean(axis=0),
        })

    if len(rows) == 0:
        raise ValueError("No CIDs left after applying consensus_mode.")

    consensus_cids = np.asarray([r["CID"] for r in rows])
    n_subjects_per_cid = np.asarray([r["n_subjects"] for r in rows])
    consensus_labels = np.vstack([r["labels"] for r in rows])

    return consensus_labels, consensus_cids, n_subjects_per_cid


# ---------------------------------------------------------------------
# Embedding file loading
# ---------------------------------------------------------------------

def parse_epoch_and_seed(path: str) -> Tuple[int, str]:
    """
    Convention: save_embeddings_npy filenames start with embeddings_{epoch}_...
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


def resolve_embedding_files(embeddings_dir: str, embedding_subdir: str, keep_last_epoch_per_seed: bool, recursive: bool):
    if recursive:
        pattern = os.path.join(embeddings_dir, embedding_subdir, "**", "*.npy")
        paths = sorted(glob.glob(pattern, recursive=True))
    else:
        pattern = os.path.join(embeddings_dir, embedding_subdir, "*.npy")
        paths = sorted(glob.glob(pattern))

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


def infer_consensus_mode_from_embedding_shape(paths, labels, subjects, cids):
    if len(paths) == 0:
        raise ValueError("No embedding files found.")

    z0 = np.load(paths[0])
    n_rows = z0.shape[0]

    labels_all, _, _ = make_consensus_labels(labels, subjects, cids, consensus_mode="all_available")
    labels_all_subjects, _, _ = make_consensus_labels(labels, subjects, cids, consensus_mode="all_subjects")

    if n_rows == len(labels_all):
        return "all_available"
    if n_rows == len(labels_all_subjects):
        return "all_subjects"

    raise ValueError(
        f"Could not infer consensus mode from embedding shape {z0.shape}. "
        f"all_available has {len(labels_all)} rows; all_subjects has {len(labels_all_subjects)} rows."
    )


# ---------------------------------------------------------------------
# Metrics and p-values
# ---------------------------------------------------------------------

def radial_entropy_for_one_seed(z, entropy_values):
    r = hyperbolic_radius(z)
    return {
        "radius_entropy_pearson": safe_corr(r, entropy_values, "pearson"),
        "radius_entropy_spearman": safe_corr(r, entropy_values, "spearman"),
    }


def mean_radius_entropy_pearson(embeddings, entropy_values):
    vals = [radial_entropy_for_one_seed(z, entropy_values)["radius_entropy_pearson"] for z in embeddings]
    return float(np.nanmean(vals))


def mean_directional_r2(embeddings, y):
    vals = [directional_r2(z, y) for z in embeddings]
    return float(np.nanmean(vals))


def permutation_p_value_radial_entropy(embeddings, entropy_values, n_permutations, rng):
    if n_permutations <= 0:
        return np.nan

    obs = mean_radius_entropy_pearson(embeddings, entropy_values)
    count = 0

    for _ in range(n_permutations):
        entropy_perm = rng.permutation(entropy_values)
        stat = mean_radius_entropy_pearson(embeddings, entropy_perm)
        if abs(stat) >= abs(obs):
            count += 1

    return (1 + count) / (n_permutations + 1)


def permutation_p_value_angular(embeddings, y, n_permutations, rng):
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
# LaTeX helpers
# ---------------------------------------------------------------------

def format_mean_std(mean, std):
    if not np.isfinite(mean) or not np.isfinite(std):
        return "--"
    return f"${mean:.2f} \\pm {std:.2f}$"


def format_p(p):
    if not np.isfinite(p):
        return "--"
    if p <= 0.001:
        return "$<0.001$"
    return f"${p:.3f}$"


def save_radial_latex(summary_row, output_path):
    pearson = format_mean_std(
        summary_row["radius_entropy_pearson_mean"],
        summary_row["radius_entropy_pearson_std"],
    )
    spearman = format_mean_std(
        summary_row["radius_entropy_spearman_mean"],
        summary_row["radius_entropy_spearman_std"],
    )
    p = format_p(summary_row["p_entropy_radial"])

    tex = rf"""\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{lccc}}
\toprule
Analysis & Radial Pearson & Radial Spearman & Permutation $p$ \\
\midrule
Averaged ratings & {pearson} & {spearman} & {p} \\
\bottomrule
\end{{tabular}}
\caption{{Radial entropy analysis for the Sagar averaged-rating embedding. Values are reported as mean and standard deviation over random seeds.}}
\label{{tab:sagar_average_radial_entropy}}
\end{{table}}
"""
    with open(output_path, "w") as f:
        f.write(tex)


def save_angular_latex(df, output_path, filtered=False):
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{row['descriptor']} & "
            f"{format_mean_std(row['directional_R2_mean'], row['directional_R2_std'])} & "
            f"{format_p(row['p_angular'])} \\\\" 
        )
    body = "\n".join(rows)

    caption = "Angular descriptor analysis for the Sagar averaged-rating embedding."
    if filtered:
        caption = "Angular descriptor analysis for descriptors with mean directional $R^2>0.2$ in the Sagar averaged-rating embedding."

    tex = rf"""\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{lcc}}
\toprule
Descriptor & Directional $R^2$ & Permutation $p$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{{caption} Values are reported as mean and standard deviation over random seeds.}}
\label{{tab:sagar_average_angular_descriptors{'_filtered' if filtered else '_all'}}}
\end{{table}}
"""
    with open(output_path, "w") as f:
        f.write(tex)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./data/", help="Base data directory used by read_embeddings.")
    parser.add_argument("--representation_name", type=str, default="pom", help="Usually 'pom'.")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Folder containing embedding subfolders.")
    parser.add_argument("--embedding_subdir", type=str, default="average_original", help="Subfolder containing averaged-rating embeddings.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_sagar_average_original", help="Where outputs are saved.")
    parser.add_argument("--consensus_mode", choices=["auto", "all_available", "all_subjects"], default="auto")
    parser.add_argument("--n_permutations", type=int, default=1000, help="Use 0 to skip permutation p-values.")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--keep_last_epoch_per_seed", action="store_true")
    parser.add_argument("--recursive", action="store_true", help="Search recursively inside embedding_subdir.")
    parser.add_argument("--r2_threshold", type=float, default=0.2, help="Threshold for filtered angular table.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    labels_union, subjects, cids = load_sagar_union(args.base_dir, args.representation_name)

    files = resolve_embedding_files(
        embeddings_dir=args.embeddings_dir,
        embedding_subdir=args.embedding_subdir,
        keep_last_epoch_per_seed=args.keep_last_epoch_per_seed,
        recursive=args.recursive,
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No .npy files found in {os.path.join(args.embeddings_dir, args.embedding_subdir)}"
        )

    consensus_mode = args.consensus_mode
    if consensus_mode == "auto":
        consensus_mode = infer_consensus_mode_from_embedding_shape(files, labels_union, subjects, cids)

    labels_avg, consensus_cids, n_subjects_per_cid = make_consensus_labels(
        labels_union,
        subjects,
        cids,
        consensus_mode=consensus_mode,
    )

    embeddings = [load_embedding(path, expected_n=len(labels_avg)) for path in files]
    entropy_values = entropy_from_scores(labels_avg)

    print("Loaded averaged-rating analysis data:")
    print("  consensus_mode:", consensus_mode)
    print("  averaged labels:", labels_avg.shape)
    print("  CIDs:", len(consensus_cids))
    print("  n_embedding_files:", len(files))
    print("  embedding_subdir:", args.embedding_subdir)
    print("  subjects per CID counts:", dict(zip(*np.unique(n_subjects_per_cid, return_counts=True))))

    # Radial entropy by seed
    radial_by_seed = []
    for path, z in zip(files, embeddings):
        epoch, seed = parse_epoch_and_seed(path)
        metrics = radial_entropy_for_one_seed(z, entropy_values)
        radial_by_seed.append({
            "embedding_file": path,
            "epoch": epoch,
            "seed": seed,
            **metrics,
        })

    pearson_mean, pearson_std = summarize([r["radius_entropy_pearson"] for r in radial_by_seed])
    spearman_mean, spearman_std = summarize([r["radius_entropy_spearman"] for r in radial_by_seed])
    p_entropy = permutation_p_value_radial_entropy(
        embeddings,
        entropy_values,
        args.n_permutations,
        rng,
    )

    radial_summary = {
        "analysis": "average_original",
        "embedding_subdir": args.embedding_subdir,
        "consensus_mode": consensus_mode,
        "n_cids": len(consensus_cids),
        "n_embedding_files": len(files),
        "radius_entropy_pearson_mean": pearson_mean,
        "radius_entropy_pearson_std": pearson_std,
        "radius_entropy_spearman_mean": spearman_mean,
        "radius_entropy_spearman_std": spearman_std,
        "p_entropy_radial": p_entropy,
    }

    # Angular descriptor analysis
    angular_by_seed = []
    angular_summary_rows = []
    for k, descriptor_name in enumerate(sagar_tasks):
        y = labels_avg[:, k]
        seed_vals = []

        for path, z in zip(files, embeddings):
            epoch, seed = parse_epoch_and_seed(path)
            r2 = directional_r2(z, y)
            seed_vals.append(r2)
            angular_by_seed.append({
                "embedding_file": path,
                "epoch": epoch,
                "seed": seed,
                "descriptor_index": k,
                "descriptor": descriptor_name,
                "directional_R2": r2,
            })

        r2_mean, r2_std = summarize(seed_vals)
        p_ang = permutation_p_value_angular(
            embeddings,
            y,
            args.n_permutations,
            rng,
        )

        angular_summary_rows.append({
            "analysis": "average_original",
            "embedding_subdir": args.embedding_subdir,
            "consensus_mode": consensus_mode,
            "n_cids": len(consensus_cids),
            "descriptor_index": k,
            "descriptor": descriptor_name,
            "directional_R2_mean": r2_mean,
            "directional_R2_std": r2_std,
            "p_angular": p_ang,
        })

    radial_summary_df = pd.DataFrame([radial_summary])
    radial_by_seed_df = pd.DataFrame(radial_by_seed)
    angular_summary_df = pd.DataFrame(angular_summary_rows)
    angular_by_seed_df = pd.DataFrame(angular_by_seed)
    angular_filtered_df = angular_summary_df[
        angular_summary_df["directional_R2_mean"] > args.r2_threshold
    ].sort_values("directional_R2_mean", ascending=False)
    angular_summary_df = angular_summary_df.sort_values("directional_R2_mean", ascending=False)

    # Rounded CSVs for quick viewing
    radial_summary_rounded = radial_summary_df.copy()
    angular_summary_rounded = angular_summary_df.copy()
    for df in [radial_summary_rounded, angular_summary_rounded]:
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].round(3)

    # Save CSV outputs
    radial_summary_df.to_csv(os.path.join(args.output_dir, "average_entropy_radial_summary.csv"), index=False)
    radial_summary_rounded.to_csv(os.path.join(args.output_dir, "average_entropy_radial_summary_rounded.csv"), index=False)
    radial_by_seed_df.to_csv(os.path.join(args.output_dir, "average_entropy_radial_by_seed.csv"), index=False)
    angular_summary_df.to_csv(os.path.join(args.output_dir, "average_angular_descriptor_summary.csv"), index=False)
    angular_summary_rounded.to_csv(os.path.join(args.output_dir, "average_angular_descriptor_summary_rounded.csv"), index=False)
    angular_by_seed_df.to_csv(os.path.join(args.output_dir, "average_angular_descriptor_by_seed.csv"), index=False)
    angular_filtered_df.to_csv(os.path.join(args.output_dir, "average_angular_descriptor_R2_gt_0.2.csv"), index=False)

    # Save LaTeX outputs
    save_radial_latex(
        radial_summary,
        os.path.join(args.output_dir, "average_entropy_radial_table.tex"),
    )
    save_angular_latex(
        angular_filtered_df,
        os.path.join(args.output_dir, "average_angular_descriptor_table_filtered.tex"),
        filtered=True,
    )
    save_angular_latex(
        angular_summary_df,
        os.path.join(args.output_dir, "average_angular_descriptor_table_full.tex"),
        filtered=False,
    )

    print("\nSaved outputs in:", args.output_dir)
    print("  average_entropy_radial_summary.csv")
    print("  average_entropy_radial_by_seed.csv")
    print("  average_angular_descriptor_summary.csv")
    print("  average_angular_descriptor_by_seed.csv")
    print("  average_angular_descriptor_R2_gt_0.2.csv")
    print("  average_entropy_radial_table.tex")
    print("  average_angular_descriptor_table_filtered.tex")
    print("  average_angular_descriptor_table_full.tex")
    print("\nRadial entropy summary:")
    print(radial_summary_rounded.to_string(index=False))
    print("\nAngular descriptors with mean R2 >", args.r2_threshold)
    print(angular_filtered_df.to_string(index=False))


if __name__ == "__main__":
    main()
