"""Phase 2 tooling: train and validate AU models against FACS-coded datasets.

Workflow (e.g. DISFA / BP4D, which must be obtained under their own licences):

1. ``cogsense extract-features video.avi --subject SN001 -o SN001.csv``
   runs MediaPipe on every frame and writes pose-normalized features.
2. Join per-frame AU labels (columns ``au01, au02, au04, au07, au14``,
   DISFA 0–5 intensity scale) onto those CSVs.
3. ``cogsense validate data/*.csv --save-model au_model.json`` trains
   per-AU logistic models with leave-one-subject-out cross-validation and
   reports F1 (acceptance target: F1 ≥ 0.82).

Features are made subject-relative by subtracting each subject's median
over frames where all labelled AUs are inactive (the dataset analogue of
the 15 s neutral calibration).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cogsense.features import FEATURE_NAMES

LABEL_AUS = ("au01", "au02", "au04", "au07", "au14")


def extract_video_features(video: str | Path, model_path: str | Path, subject: str, out_csv: str | Path) -> int:
    import cv2

    from cogsense.features import extract
    from cogsense.geometry import normalize, to_pixel_space
    from cogsense.runtime import MediaPipeTracker

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = MediaPipeTracker(model_path)
    n = 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "frame", *FEATURE_NAMES])
        frame = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lm, _ = tracker(rgb, frame / fps)
            if lm is not None:
                f = extract(normalize(to_pixel_space(lm, img.shape[1], img.shape[0])).points)
                w.writerow([subject, frame, *(f"{f[k]:.6f}" for k in FEATURE_NAMES)])
                n += 1
            frame += 1
    tracker.close()
    cap.release()
    return n


def load_labeled(paths: list[str | Path], threshold: float = 2.0):
    """Returns (X subject-relative features, Y binary labels dict, subjects array)."""
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        raise ValueError("no rows")
    present = [a for a in LABEL_AUS if a in rows[0]]
    subj = np.array([r["subject"] for r in rows])
    X = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows])
    Yraw = {a: np.array([float(r[a]) for r in rows]) for a in present}
    for s in np.unique(subj):
        m = subj == s
        neutral = m & np.all([Yraw[a] == 0 for a in present], axis=0)
        ref = np.median(X[neutral] if neutral.sum() >= 10 else X[m], axis=0)
        X[m] -= ref
    return X, {a: (v >= threshold).astype(int) for a, v in Yraw.items()}, subj


def train_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 600, lr: float = 0.5) -> dict:
    mean, scale = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / scale
    pos = max(y.mean(), 1e-3)
    sw = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-3))  # class balancing
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = sw * (p - y)
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return {"coef": w.tolist(), "intercept": float(b), "mean": mean.tolist(), "scale": scale.tolist()}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    z = ((X - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-z))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def cross_validate(X: np.ndarray, Y: dict[str, np.ndarray], subjects: np.ndarray) -> dict[str, float]:
    """Leave-one-subject-out F1 per AU (predictions pooled across folds)."""
    results = {}
    for au, y in Y.items():
        pred = np.zeros_like(y)
        for s in np.unique(subjects):
            test = subjects == s
            if y[~test].sum() == 0:
                continue
            m = train_logistic(X[~test], y[~test])
            pred[test] = (predict(m, X[test]) >= 0.5).astype(int)
        results[au] = round(f1_score(y, pred), 4)
    return results


def fit_all(X: np.ndarray, Y: dict[str, np.ndarray]) -> dict[str, dict]:
    return {au: train_logistic(X, y) for au, y in Y.items() if y.sum() > 0}


def save_model(models: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(models))
