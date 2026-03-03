#!/usr/bin/env python3
"""
kmeans_part1.py

CSE 575 K-Means Strategy Project — Part 1 (Strategy 1: random initial centers)

This script provides:
  1) A fully-commented K-Means implementation (2D points, but works for any d)
  2) Strategy-1 initialization compatible with the assignment's precode behavior
  3) A CLI entrypoint that can run k=2..10 on AllSamples.npy and print:
       - Final centroids for each k (one line per k)
       - Final loss for each k (one line per k)
  4) Optional plot generation (loss vs k, and clustering visualization)
  5) Unit tests (run with: python -m unittest kmeans_part1.py)

Notes for GitHub:
- This file is self-contained; it does NOT require precode.py to run.
- If you do have the course's precode.py, you can still use it by importing and
  swapping the initializer, but the default initializer here mimics the one shown
  in the assignment prompt.
"""

# Standard library imports
import argparse  # Parse command-line arguments cleanly
import os        # File existence checks and directory creation
import sys       # Exit codes and stderr printing
import unittest  # Built-in unit testing framework

# Third-party imports (standard in ML Python stacks)
import numpy as np  # Numerical computing and vectorized operations


# -----------------------------
# Strategy 1 initialization
# -----------------------------

def initial_point_idx(last4_int: int, k: int, n_samples: int) -> np.ndarray:
    """
    Return k unique indices sampled from [0, n_samples) using a deterministic seed.

    This mirrors the pattern commonly used in the course prompt:
      np.random.RandomState(seed=(id + k)).permutation(N)[:k]

    Parameters
    ----------
    last4_int : int
        Last four digits of student id as an integer (or any deterministic integer).
    k : int
        Number of clusters / indices to select.
    n_samples : int
        Number of data points.

    Returns
    -------
    np.ndarray
        Array of shape (k,) containing unique indices.
    """
    # Create a deterministic RNG local to this function (does not affect global RNG)
    rng = np.random.RandomState(seed=(last4_int + k))
    # Permute [0..N-1] and take the first k indices (guaranteed unique)
    return rng.permutation(n_samples)[:k]


def initial_S1(student_last4: str, k: int, data: np.ndarray) -> np.ndarray:
    """
    Strategy 1 initializer:
      - Convert student id string to an integer "i" (mod 150 in the prompt)
      - Generate k indices via initial_point_idx(i, k, N)
      - Return the corresponding data points as initial centers

    Parameters
    ----------
    student_last4 : str
        Last four digits of student id (e.g., "0111").
    k : int
        Number of clusters.
    data : np.ndarray
        Dataset of shape (N, d).

    Returns
    -------
    np.ndarray
        Initial centers of shape (k, d).
    """
    # Convert id string to an int safely (fallback to 0 if invalid)
    try:
        sid = int(student_last4)
    except ValueError:
        sid = 0

    # Apply the same modulus as the assignment snippet (keeps seed range consistent)
    i = sid % 150

    # Compute the indices of initial centers deterministically
    init_idx = initial_point_idx(i, k, data.shape[0])

    # Select those rows from the dataset as the initial centers
    return data[init_idx, :].astype(float)


# -----------------------------
# Core K-Means implementation
# -----------------------------

def compute_squared_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distances between each data point and each center.

    Returns an (N, k) matrix where entry (n, j) is ||x_n - mu_j||^2.

    Parameters
    ----------
    data : np.ndarray
        Dataset of shape (N, d).
    centers : np.ndarray
        Centers of shape (k, d).

    Returns
    -------
    np.ndarray
        Squared distances of shape (N, k).
    """
    # Broadcasting: (N, 1, d) - (1, k, d) -> (N, k, d)
    diff = data[:, None, :] - centers[None, :, :]
    # Sum squares across the feature dimension -> (N, k)
    return np.sum(diff * diff, axis=2)


def kmeans_strategy1(
    data: np.ndarray,
    init_centers: np.ndarray,
    max_iter: int = 1000
) -> tuple[np.ndarray, float]:
    """
    Run K-Means until convergence using assignment-stability as stopping criterion.

    Steps per iteration:
      1) Assign each point to nearest centroid (by squared distance)
      2) Recompute each centroid as mean of assigned points
      3) Stop when labels no longer change (or when max_iter reached)

    Empty clusters:
      - If a cluster gets zero points, keep its centroid unchanged (deterministic).

    Parameters
    ----------
    data : np.ndarray
        Dataset of shape (N, d).
    init_centers : np.ndarray
        Initial centers of shape (k, d).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    (centers, loss) : (np.ndarray, float)
        centers is shape (k, d) and loss is SSE objective value.
    """
    # Make a float copy so we can compute means without modifying caller's array
    centers = np.array(init_centers, dtype=float).copy()
    # Store k and N for convenience
    k = centers.shape[0]
    n = data.shape[0]

    # Previous labels (None on the first iteration)
    prev_labels = None

    # Main loop bounded by max_iter
    for _ in range(max_iter):
        # Compute squared distances from each point to each center
        d2 = compute_squared_distances(data, centers)
        # Assign each point to the nearest center
        labels = np.argmin(d2, axis=1)

        # If labels are unchanged, we have converged
        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break

        # Update previous labels for next iteration's convergence check
        prev_labels = labels

        # Prepare a new centers array (start with old centers for empty-cluster stability)
        new_centers = centers.copy()

        # Update each centroid
        for j in range(k):
            # Extract all points assigned to cluster j
            pts = data[labels == j]
            # If cluster is non-empty, compute the mean
            if pts.shape[0] > 0:
                new_centers[j] = np.mean(pts, axis=0)
            # Else: keep the old center (already in new_centers)

        # Replace centers with updated values
        centers = new_centers

    # After loop, recompute labels using the final centers for correct loss calculation
    d2 = compute_squared_distances(data, centers)
    labels = np.argmin(d2, axis=1)

    # Compute loss (sum of squared distances to assigned center)
    loss = float(np.sum(d2[np.arange(n), labels]))

    # Return final centers and loss
    return centers, loss


# -----------------------------
# Experiment runner (k=2..10)
# -----------------------------

def run_part1(
    data: np.ndarray,
    student_last4: str,
    k_min: int = 2,
    k_max: int = 10,
    max_iter: int = 1000
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """
    Run Strategy-1 K-Means for k in [k_min, k_max] and return results.

    Parameters
    ----------
    data : np.ndarray
        Dataset of shape (N, d).
    student_last4 : str
        Last 4 digits of student id (used only for deterministic initialization).
    k_min : int
        Minimum k (inclusive).
    k_max : int
        Maximum k (inclusive).
    max_iter : int
        Maximum iterations for k-means convergence.

    Returns
    -------
    (final_centers, final_losses) : (dict, dict)
        final_centers[k] is (k,d), final_losses[k] is float
    """
    # Dictionaries to collect results
    final_centers: dict[int, np.ndarray] = {}
    final_losses: dict[int, float] = {}

    # Iterate over k values
    for k in range(k_min, k_max + 1):
        # Build initialization for this k (Strategy 1)
        init_c = initial_S1(student_last4, k, data)
        # Run K-Means to convergence
        centers_k, loss_k = kmeans_strategy1(data, init_c, max_iter=max_iter)
        # Store results
        final_centers[k] = centers_k
        final_losses[k] = loss_k

    # Return results
    return final_centers, final_losses


# -----------------------------
# Plotting helpers (optional)
# -----------------------------

def save_loss_plot(final_losses: dict[int, float], out_path: str) -> None:
    """
    Save a loss-vs-k line plot to disk.

    Parameters
    ----------
    final_losses : dict[int, float]
        Loss values keyed by k.
    out_path : str
        Output path for PNG image.
    """
    # Import matplotlib lazily so tests can run without GUI backends
    import matplotlib.pyplot as plt

    # Sort keys to ensure increasing k on x-axis
    ks = sorted(final_losses.keys())
    # Gather loss values aligned with ks
    losses = [final_losses[k] for k in ks]

    # Create a figure
    plt.figure()
    # Plot
    plt.plot(ks, losses, marker="o")
    # Label axes
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Loss (SSE objective)")
    plt.title("K-Means Strategy 1: Loss vs k")
    plt.xticks(ks)
    # Save figure
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    # Close to free memory
    plt.close()


def save_clustering_plot(data: np.ndarray, centers: np.ndarray, out_path: str) -> None:
    """
    Save a scatter plot of clustered points and centroids for a single k.

    Parameters
    ----------
    data : np.ndarray
        Dataset (N,d). Only first two dims are plotted.
    centers : np.ndarray
        Centroids (k,d). Only first two dims are plotted.
    out_path : str
        Output path for PNG image.
    """
    import matplotlib.pyplot as plt

    # Compute labels for final centers (assignment step)
    d2 = compute_squared_distances(data, centers)
    labels = np.argmin(d2, axis=1)

    # Create a figure
    plt.figure()
    # Plot points colored by label
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=10)
    # Plot centroids
    plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=200)
    # Add title and labels
    plt.title(f"K-Means Strategy 1: Final Clustering (k={centers.shape[0]})")
    plt.xlabel("x")
    plt.ylabel("y")
    # Save to disk
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


# -----------------------------
# CLI / main
# -----------------------------

def main(argv: list[str]) -> int:
    """
    CLI entrypoint.

    Examples:
      python kmeans_part1.py --data AllSamples.npy --student 0111
      python kmeans_part1.py --data AllSamples.npy --student 0111 --plots

    Returns process exit code (0 success).
    """
    # Build argument parser
    parser = argparse.ArgumentParser(description="CSE575 K-Means Part 1 (Strategy 1)")
    # Path to .npy file
    parser.add_argument("--data", type=str, default="AllSamples.npy", help="Path to .npy data file")
    # Student last 4 digits
    parser.add_argument("--student", type=str, required=True, help="Last 4 digits of student ID (e.g., 0111)")
    # Optional plots
    parser.add_argument("--plots", action="store_true", help="Save plots to ./plots/")
    # Optional max_iter override
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for K-Means (default 1000)")
    # Optional cluster plot k
    parser.add_argument("--plot_k", type=int, default=5, help="k to use for clustering plot (default 5)")

    # Parse args
    args = parser.parse_args(argv)

    # Load data array
    if not os.path.exists(args.data):
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        return 2

    # Load .npy (expects shape (N,2) for this assignment, but works for any d)
    data = np.load(args.data)

    # Run experiments for k=2..10
    final_centers, final_losses = run_part1(
        data=data,
        student_last4=args.student,
        k_min=2,
        k_max=10,
        max_iter=args.max_iter
    )

    # Print centroids (k=2..10), one per line
    for k in range(2, 11):
        print(final_centers[k].tolist())

    # Print losses (k=2..10), one per line
    for k in range(2, 11):
        print(final_losses[k])

    # Save plots if requested
    if args.plots:
        # Create plots directory if needed
        os.makedirs("plots", exist_ok=True)
        # Loss plot
        save_loss_plot(final_losses, os.path.join("plots", "part1_loss_vs_k.png"))
        # Clustering plot for a selected k (default 5)
        k_plot = args.plot_k
        if k_plot in final_centers:
            save_clustering_plot(
                data,
                final_centers[k_plot],
                os.path.join("plots", f"part1_clustering_k{k_plot}.png")
            )

    return 0


# -----------------------------
# Unit tests
# -----------------------------

class TestKMeansPart1(unittest.TestCase):
    """
    Unit tests to validate correctness on small synthetic datasets.

    These are not the course autograder tests; they ensure the implementation
    behaves correctly and robustly for GitHub use.
    """

    def test_squared_distance_shape(self):
        # Simple 3 points in 2D
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        # Two centers
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        # Compute distance matrix
        d2 = compute_squared_distances(data, centers)
        # Expect (N,k) = (3,2)
        self.assertEqual(d2.shape, (3, 2))

    def test_kmeans_converges_on_two_blobs(self):
        # Create two tight clusters
        blob1 = np.array([[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1]])
        blob2 = np.array([[5.0, 5.0], [5.1, 5.0], [4.9, 5.1]])
        data = np.vstack([blob1, blob2])

        # Deterministic init: pick one point from each blob
        init_centers = np.array([[0.0, 0.0], [5.0, 5.0]])

        # Run k-means
        centers, loss = kmeans_strategy1(data, init_centers, max_iter=1000)

        # Check shape
        self.assertEqual(centers.shape, (2, 2))
        # Loss should be small for tight blobs
        self.assertTrue(loss < 0.2)

    def test_loss_monotonic_nonincreasing_with_more_clusters_on_simple_data(self):
        # Create a dataset with 3 obvious groups in 2D
        g1 = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        g2 = np.array([[3.0, 3.0], [3.1, 3.0], [3.0, 3.1]])
        g3 = np.array([[6.0, 0.0], [6.1, 0.0], [6.0, 0.1]])
        data = np.vstack([g1, g2, g3])

        # Run part1 with deterministic student id
        _, losses_dict = run_part1(data, student_last4="0111", k_min=2, k_max=4, max_iter=1000)

        # Extract losses in order
        losses = [losses_dict[k] for k in range(2, 5)]

        # Loss should not increase as k increases
        self.assertTrue(losses[1] <= losses[0] + 1e-9)
        self.assertTrue(losses[2] <= losses[1] + 1e-9)

    def test_initializer_returns_k_points(self):
        # Random dataset (deterministic RNG for repeatable test)
        data = np.random.RandomState(0).randn(50, 2)
        # Init for k=7
        init_c = initial_S1("0111", 7, data)
        # Should be shape (7,2)
        self.assertEqual(init_c.shape, (7, 2))


# If executed directly: run CLI. If executed via unittest: tests run instead.
if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
