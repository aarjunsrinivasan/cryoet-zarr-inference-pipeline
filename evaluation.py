"""
Implements the competition matching convention:
  - TP: each GT particle covered by ≥1 prediction within the matching radius
  - FP: predictions with no GT within radius
  - FN: GT particles with no prediction within radius
  - Matching radius = EVAL_RADIUS_MULT × particle_radius (default 0.5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

BETA = 4  # F_β weighting (competition default — recall weighted 4× over precision)


def compute_tp_fp_fn_dist(
    gt_pts: np.ndarray,
    pred_pts: np.ndarray,
    radius: float,
) -> tuple[int, int, int, float]:
    n_gt, n_pred = len(gt_pts), len(pred_pts)
    if n_gt == 0:
        return 0, n_pred, 0, float("nan")
    if n_pred == 0:
        return 0, 0, n_gt, float("nan")
    ref_tree = KDTree(gt_pts)
    matched_gt: set[int] = set()
    for m in KDTree(pred_pts).query_ball_tree(ref_tree, r=radius):
        matched_gt.update(m)
    tp = len(matched_gt)
    fp = max(n_pred - tp, 0)
    fn = n_gt - tp
    if tp == 0:
        return tp, fp, fn, float("nan")
    idx = np.array(sorted(matched_gt), dtype=int)
    dists, _ = KDTree(pred_pts).query(gt_pts[idx], k=1)
    return tp, fp, fn, float(np.mean(dists))


def fbeta(precision: float, recall: float, beta: int = BETA) -> float:
    denom = beta**2 * precision + recall
    return (1 + beta**2) * precision * recall / denom if denom > 0 else 0.0


def evaluate_at_threshold(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    threshold: float,
    particle_type: str,
    radius: float,
) -> dict:
    gt_sub   = gt_df[gt_df["particle_type"] == particle_type]
    pred_sub = pred_df[
        (pred_df["particle_type"] == particle_type) & (pred_df["conf"] > threshold)
    ]
    gt_pts   = gt_sub[["x", "y", "z"]].values
    pred_pts = pred_sub[["x", "y", "z"]].values
    tp, fp, fn, avg_dist = compute_tp_fp_fn_dist(gt_pts, pred_pts, radius)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "particle_type": particle_type,
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn,
        "n_gt": len(gt_pts), "n_pred": len(pred_pts),
        "precision": prec, "recall": rec,
        f"f{BETA}": fbeta(prec, rec, BETA),
        "avg_tp_dist_A": avg_dist,
    }


def precision_recall_curve(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    particle_type: str,
    radius: float,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame([
        evaluate_at_threshold(pred_df, gt_df, th, particle_type, radius)
        for th in thresholds
    ])


def average_precision(pr_df: pd.DataFrame) -> float:
    pr = pr_df.sort_values("recall")
    return float(np.trapezoid(pr["precision"], pr["recall"]))
