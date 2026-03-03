#!/usr/bin/env python3
"""
kmeans_part2.py

CSE 575 K-Means Strategy Project — Part 2 (Strategy 2: distance-based initialization)

This script provides:
  1) A fully-commented K-Means implementation (works for any dimensionality d)
  2) Strategy-2 initialization:
        - Pick the first center "randomly" (deterministic seed derived from student id + k)
        - For each subsequent center, choose the sample that maximizes the
          AVERAGE Euclidean distance to all previously selected centers
  3) A CLI entrypoint to run k=2..10 on AllSamples.npy and print:
        - Final centroids for each k (one line per k)
        - Final loss for each k (one line per k)
  4) Optional plot generation (loss vs k, and clustering visualization)
  5) Unit tests (run with: python -m unittest kmeans_part2.py)

Notes for GitHub:
- This file is self-contained; it does NOT require precode.py to run.
- If you have the course's precode.py for Part 2, you can swap in its initial_S2
  to pick the first center, then call build_strategy2_centers(...) to generate
  the full k-center initialization deterministically.
"""

import argparse
import os
import sys
import unittest

import numpy as np


# -----------------------------
# Utilities: distances + loss
# -----------------------------

def compute_squared_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distances between each point and each center.

    Returns d2 of shape (N, k) where:
      d2[i, j] = ||data[i] - centers[j]||^2
    """
    diff = data[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def compute_labels(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Assign each point to the nearest centroid by squared Euclidean distance.
    """
    d2 = compute_squared_distances(data, centers)
    return np.argmin(d2, axis=1)


def compute_sse_loss(data: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute SSE objective:
      L = sum_i ||x_i - mu_{label_i}||^2
    """
    diff = data - centers[labels]
    return float(np.sum(diff * diff))


# -----------------------------
# Strategy 2 initialization
# -----------------------------

def deterministic_first_center_index(student_last4: str, k: int, n_samples: int) -> int:
    """
    Deterministically choose the index of the first center.

    Uses a local RNG seeded by ( (int(student_last4) % 150) + 800 + k ).
    """
    try:
        sid = int(student_last4)
    except ValueError:
        sid = 0

    i = sid % 150
    rng = np.random.RandomState(seed=(i + 800 + k))
    return int(rng.randint(0, n_samples))


def initial_S2(student_last4: str, k: int, data: np.ndarray) -> np.ndarray:
    """
    Strategy 2 "first center" initializer (self-contained).

    Returns a SINGLE first center as a sample from data.
    """
    idx = deterministic_first_center_index(student_last4, k, data.shape[0])
    return data[idx].astype(float)


def build_strategy2_centers(data: np.ndarray, first_center: np.ndarray, k: int) -> np.ndarray:
    """
    Build k initial centers for Strategy 2:

      centers[0] = first_center
      for t = 1..k-1:
        pick x that maximizes average Euclidean distance to all previous centers

    IMPORTANT: Uses Euclidean distance (sqrt) per the prompt wording "average distance".
    """
    first_center = np.array(first_center, dtype=float).reshape(-1)
    centers = [first_center]

    chosen = np.zeros(data.shape[0], dtype=bool)

    matches = np.where(np.all(data == first_center, axis=1))[0]
    if matches.size > 0:
        chosen[int(matches[0])] = True

    for _ in range(1, k):
        avg_dist = np.zeros(data.shape[0], dtype=float)

        for j in range(data.shape[0]):
            dsum = 0.0
            for c in centers:
                dsum += float(np.linalg.norm(data[j] - c))
            avg_dist[j] = dsum / len(centers)

        avg_dist[chosen] = -np.inf
        next_idx = int(np.argmax(avg_dist))
        chosen[next_idx] = True
        centers.append(data[next_idx].astype(float))

    return np.vstack(centers)


# -----------------------------
# Core K-Means loop
# -----------------------------

def kmeans(data: np.ndarray, init_centers: np.ndarray, max_iter: int = 1000) -> tuple[np.ndarray, float]:
    """
    Run standard K-Means with assignment-stability convergence.

    Empty clusters:
      - Keep centroid unchanged (deterministic).
    """
    centers = np.array(init_centers, dtype=float).copy()
    k = centers.shape[0]

    prev_labels = None

    for _ in range(max_iter):
        labels = compute_labels(data, centers)

        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break
        prev_labels = labels

        new_centers = centers.copy()
        for j in range(k):
            pts = data[labels == j]
            if pts.shape[0] > 0:
                new_centers[j] = np.mean(pts, axis=0)

        centers = new_centers

    labels = compute_labels(data, centers)
    loss = compute_sse_loss(data, centers, labels)
    return centers, loss


# -----------------------------
# Experiment runner (k=2..10)
# -----------------------------

def run_part2(
    data: np.ndarray,
    student_last4: str,
    k_min: int = 2,
    k_max: int = 10,
    max_iter: int = 1000
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """
    Run Strategy-2 K-Means for k in [k_min, k_max] and return dict results.
    """
    final_centers: dict[int, np.ndarray] = {}
    final_losses: dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        first = initial_S2(student_last4, k, data)
        init_c = build_strategy2_centers(data, first_center=first, k=k)
        centers_k, loss_k = kmeans(data, init_c, max_iter=max_iter)

        final_centers[k] = centers_k
        final_losses[k] = loss_k

    return final_centers, final_losses


# -----------------------------
# Plotting helpers (optional)
# -----------------------------

def save_loss_plot(final_losses: dict[int, float], out_path: str) -> None:
    """Save loss-vs-k plot."""
    import matplotlib.pyplot as plt

    ks = sorted(final_losses.keys())
    losses = [final_losses[k] for k in ks]

    plt.figure()
    plt.plot(ks, losses, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Loss (SSE objective)")
    plt.title("K-Means Strategy 2: Loss vs k")
    plt.xticks(ks)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


def save_clustering_plot(data: np.ndarray, centers: np.ndarray, out_path: str) -> None:
    """Save a scatter plot of data colored by cluster with centroids overlaid (2D only)."""
    import matplotlib.pyplot as plt

    labels = compute_labels(data, centers)

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=10)
    plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=200)
    plt.title(f"K-Means Strategy 2: Final Clustering (k={centers.shape[0]})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


# -----------------------------
# CLI / main
# -----------------------------

def main(argv: list[str]) -> int:
    """
    CLI entrypoint.

    Prints:
      - Centroids for k=2..10 (one per line)
      - Losses for k=2..10 (one per line)
    """
    parser = argparse.ArgumentParser(description="CSE575 K-Means Part 2 (Strategy 2)")
    parser.add_argument("--data", type=str, default="AllSamples.npy", help="Path to .npy data file")
    parser.add_argument("--student", type=str, required=True, help="Last 4 digits of student ID (e.g., 0111)")
    parser.add_argument("--plots", action="store_true", help="Save plots to ./plots/")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for K-Means (default 1000)")
    parser.add_argument("--plot_k", type=int, default=5, help="k to visualize clustering (default 5)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.data):
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        return 2

    data = np.load(args.data)

    final_centers, final_losses = run_part2(
        data=data,
        student_last4=args.student,
        k_min=2,
        k_max=10,
        max_iter=args.max_iter
    )

    for k in range(2, 11):
        print(final_centers[k].tolist())

    for k in range(2, 11):
        print(final_losses[k])

    if args.plots:
        os.makedirs("plots", exist_ok=True)
        save_loss_plot(final_losses, os.path.join("plots", "part2_loss_vs_k.png"))

        k_plot = args.plot_k
        if k_plot in final_centers and data.shape[1] >= 2:
            save_clustering_plot(
                data,
                final_centers[k_plot],
                os.path.join("plots", f"part2_clustering_k{k_plot}.png")
            )

    return 0


# -----------------------------
# Unit tests
# -----------------------------

class TestKMeansPart2(unittest.TestCase):
    """Unit tests for Strategy 2 initialization and K-Means behavior."""

    def test_strategy2_initialization_unique(self):
        data = np.array([[0., 0.], [1., 0.], [0., 1.], [2., 2.], [3., 3.]])
        first = data[0]
        init_c = build_strategy2_centers(data, first, k=3)
        self.assertEqual(init_c.shape, (3, 2))
        self.assertEqual(len({tuple(r) for r in init_c}), 3)

    def test_kmeans_converges_on_two_blobs(self):
        blob1 = np.array([[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1]])
        blob2 = np.array([[5.0, 5.0], [5.1, 5.0], [4.9, 5.1]])
        data = np.vstack([blob1, blob2])

        init_c = build_strategy2_centers(data, blob1[0], k=2)
        centers, loss = kmeans(data, init_c, max_iter=1000)

        self.assertEqual(centers.shape, (2, 2))
        self.assertTrue(loss < 0.2)

    def test_loss_nonincreasing_with_more_clusters(self):
        g1 = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        g2 = np.array([[3.0, 3.0], [3.1, 3.0], [3.0, 3.1]])
        g3 = np.array([[6.0, 0.0], [6.1, 0.0], [6.0, 0.1]])
        data = np.vstack([g1, g2, g3])

        _, losses_dict = run_part2(data, student_last4="0111", k_min=2, k_max=4, max_iter=1000)
        losses = [losses_dict[k] for k in range(2, 5)]

        self.assertTrue(losses[1] <= losses[0] + 1e-9)
        self.assertTrue(losses[2] <= losses[1] + 1e-9)

    def test_first_center_deterministic(self):
        data = np.random.RandomState(0).randn(100, 2)
        idx1 = deterministic_first_center_index("0111", 7, data.shape[0])
        idx2 = deterministic_first_center_index("0111", 7, data.shape[0])
        self.assertEqual(idx1, idx2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
