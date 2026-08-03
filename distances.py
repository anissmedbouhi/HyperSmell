import torch
from torch.linalg import vector_norm
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
import scipy
from typing import Callable, Union

EPS = 1e-5

def distance_matrix(data: torch.Tensor, distance_func: Callable) -> torch.Tensor:
    """Builds a distance matrix given a vectorized distance function."""
    preliminary = distance_func(data.unsqueeze(0), data.unsqueeze(1))
    result = preliminary + torch.eye(data.shape[0], device=data.device)
    return result.float()

def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance between x and y."""
    return vector_norm(x - y, dim=-1)

def poincare_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Poincare distance between x and y."""
    euc_dist = ((x - y)**2).sum(-1)
#     norm_x = 1 - (x**2).sum(-1)
#     norm_y = 1 - (y**2).sum(-1)
    norm_x = torch.clamp(1 - (x**2).sum(-1), min=EPS)
    norm_y = torch.clamp(1 - (y**2).sum(-1), min=EPS)
    x = torch.clamp(1 + 2 * torch.div(euc_dist, (norm_x * norm_y)), min=1 + EPS)
    return torch.acosh(x)

# def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
#     """Compute Hamming distance between x and y."""
#     return (x.astype(np.int32) ^ y.astype(np.int32)).sum()
def hamming_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Count of differing coordinates (pairwise, last dim)."""
    return (x.to(torch.int32) ^ y.to(torch.int32)).sum(dim=-1).float()

def knn_geodesic_distance_matrix(data: np.ndarray, n_neighbors: int = 3) -> torch.Tensor:
    """Compute geodesic distance matrix using k-nearest neighbors."""
    data_nn_matrix = kneighbors_graph(data, n_neighbors, mode='connectivity', include_self=False)
    data_dist_matrix = data_nn_matrix.toarray()
    # data_dist_matrix = dijkstra(data_nn_matrix)

    data_dist_matrix = torch.FloatTensor(data_dist_matrix)
    # data_dist_matrix = torch.where(data_dist_matrix == torch.inf, 1000 * torch.ones_like(data_dist_matrix), data_dist_matrix)
    return data_dist_matrix

def knn_graph_weighted_adjacency_matrix(data: np.ndarray, n_neighbors: int = 3, metric: str = 'minkowski') -> np.ndarray:
    """Compute k-nearest neighbors graph weighted adjacency matrix."""
    data_nn_matrix = kneighbors_graph(data, n_neighbors, mode='distance', include_self=False, metric=metric)
    return data_nn_matrix.toarray()

# def hamming_distance_matrix(data: np.ndarray) -> np.ndarray:
#     """Compute Hamming distance matrix for binary data."""
#     return scipy.spatial.distance.cdist(data, data, metric='hamming') * data.shape[-1]
#

### new helpers for islands visualization ###

def project_to_poincare_disk_for_viz(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Visualization-only projection.
    Ensures points are strictly inside the Poincaré unit disk.
    """
    r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    scale = torch.where(
        r >= 1.0 - eps,
        (1.0 - eps) / torch.clamp(r, min=eps),
        torch.ones_like(r)
    )

    return x * scale

def cross_distance_matrix_for_viz(
    x: torch.Tensor,
    y: torch.Tensor,
    distance_func
) -> torch.Tensor:
    """
    Pairwise distances between two different point sets. Convvention different than distance_matrix since no identity matrix is added here.

    x: shape (N, 2)
    y: shape (M, 2)

    Returns:
        shape (N, M)
    """
    return distance_func(x.unsqueeze(1), y.unsqueeze(0)).float()

def hyperbolic_kde_for_viz(
    grid_points,
    data_points,
    bandwidth,
    distance_func,
    chunk_size=10000,
    device=None
):
    """
    Hyperbolic KDE for visualization.
    This replaces scipy.stats.gaussian_kde for Poincaré plots.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not torch.is_tensor(grid_points):
        grid_points = torch.tensor(grid_points, dtype=torch.float32, device=device)
    else:
        grid_points = grid_points.to(device=device, dtype=torch.float32)

    if not torch.is_tensor(data_points):
        data_points = torch.tensor(data_points, dtype=torch.float32, device=device)
    else:
        data_points = data_points.to(device=device, dtype=torch.float32)

    grid_points = project_to_poincare_disk_for_viz(grid_points)
    data_points = project_to_poincare_disk_for_viz(data_points)

    densities = []

    with torch.no_grad():
        for start in range(0, grid_points.shape[0], chunk_size):
            end = start + chunk_size

            d = cross_distance_matrix_for_viz(
                grid_points[start:end],
                data_points,
                distance_func
            )

            k = torch.exp(-(d ** 2) / (2.0 * bandwidth ** 2))
            densities.append(k.mean(dim=1).cpu())

    return torch.cat(densities).numpy()

def estimate_hyperbolic_bandwidth_for_viz(
    points,
    distance_func,
    min_bandwidth=0.05,
    device=None
):
    """
    Bandwidth heuristic for visualization.
    Uses median nearest-neighbor hyperbolic distance.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not torch.is_tensor(points):
        points = torch.tensor(points, dtype=torch.float32, device=device)
    else:
        points = points.to(device=device, dtype=torch.float32)

    points = project_to_poincare_disk_for_viz(points)

    if points.shape[0] < 3:
        return min_bandwidth

    with torch.no_grad():
        d = distance_matrix(points, distance_func)

        # The distance_matrix function adds identity,
        # so we explicitly remove the diagonal from NN search.
        d.fill_diagonal_(float("inf"))

        nearest = torch.min(d, dim=1).values
        h = torch.median(nearest).item()

    return max(h, min_bandwidth)

def hyperbolic_area_element_for_viz(x_grid, y_grid, eps=EPS):
    """
    Hyperbolic area element in the Poincaré disk.
    dA_H = 4 / (1 - r^2)^2 dx dy
    """
    r2 = x_grid ** 2 + y_grid ** 2
    return 4.0 / np.maximum((1.0 - r2) ** 2, eps)

