#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M1: TF-IDF + One-vs-Rest Logistic Regression
Responsible AI multi-label classification.

Usage:
  python scripts/06_train_m1_tfidf_logreg.py --csv data/training_dataset.csv
"""

import os
import json
import argparse
import joblib
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report,
)

DEFAULT_TARGETS = [
    "unsafe",
    "privacy_violation",
    "bias",
    "manipulation",
    "transparency_issue",
    "honesty_issue",
    "policy_issue",
    "accessibility_issue",
    "missing_human_escalation",
]


def make_text(df: pd.DataFrame, text_mode: str) -> np.ndarray:
    if text_mode == "metadata":
        if "text_with_metadata" in df.columns:
            return df["text_with_metadata"].fillna("").astype(str).values
        text_mode = "separate"

    if text_mode == "combined":
        if "text" not in df.columns:
            raise ValueError("CSV is missing 'text' column.")
        return df["text"].fillna("").astype(str).values

    if text_mode == "separate":
        if "prompt" in df.columns and "answer" in df.columns:
            return (
                "USER: " + df["prompt"].fillna("").astype(str)
                + "\nASSISTANT: " + df["answer"].fillna("").astype(str)
            ).values
        if "text" in df.columns:
            return df["text"].fillna("").astype(str).values

    raise ValueError("CSV must contain text_with_metadata, text, or prompt+answer.")


def add_derived_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Positive class = issue / risk / non-compliance.
    Scores 0..2 are converted to issue labels:
      score < 2 => issue
      score = 2 => no issue
    """

    if "transparency_issue_bin" not in df.columns and "transparency_score" in df.columns:
        df["transparency_issue_bin"] = (
            df["transparency_score"].fillna(0).astype(int) < 2
        ).astype(int)

    if "honesty_issue_bin" not in df.columns and "honesty_score" in df.columns:
        df["honesty_issue_bin"] = (
            df["honesty_score"].fillna(0).astype(int) < 2
        ).astype(int)

    if "policy_issue_bin" not in df.columns and "policy_alignment" in df.columns:
        df["policy_issue_bin"] = (
            df["policy_alignment"].fillna(0).astype(int) < 2
        ).astype(int)

    if "accessibility_issue_bin" not in df.columns and "accessibility_score" in df.columns:
        df["accessibility_issue_bin"] = (
            df["accessibility_score"].fillna(0).astype(int) < 2
        ).astype(int)

    return df


def get_target_array(df: pd.DataFrame, target: str) -> np.ndarray:
    bin_col = f"{target}_bin"
    if bin_col in df.columns:
        return df[bin_col].fillna(0).astype(int).values

    if target in df.columns:
        return (df[target].fillna(0).astype(int) > 0).astype(int).values

    raise ValueError(f"Missing target column: {target} or {bin_col}")


def filter_valid_targets(df: pd.DataFrame, targets: List[str]) -> Tuple[List[str], Dict[str, str]]:
    active = []
    skipped = {}

    for t in targets:
        try:
            y = get_target_array(df, t)
        except Exception as e:
            skipped[t] = f"missing: {e}"
            continue

        if len(np.unique(y)) < 2:
            skipped[t] = f"constant target: only class {int(np.unique(y)[0])}"
        else:
            active.append(t)

    return active, skipped


def build_groups(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Prevent leakage between similar prompts.
    Priority:
      pair_id -> conversation_id -> scenario_id -> prompt_id -> row index
    """

    possible_cols = ["pair_id", "conversation_id", "scenario_id", "prompt_id"]

    groups = []
    has_any_group = False

    for idx, row in df.iterrows():
        group_value = None
        group_col = None

        for col in possible_cols:
            if col in df.columns:
                value = row.get(col)
                if pd.notna(value) and str(value).strip():
                    group_value = str(value).strip()
                    group_col = col
                    has_any_group = True
                    break

        if group_value is None:
            groups.append(f"row:{idx}")
        else:
            groups.append(f"{group_col}:{group_value}")

    if not has_any_group:
        return None

    return np.array(groups)


def train_test_split_grouped(
    X: np.ndarray,
    Y: np.ndarray,
    df: pd.DataFrame,
    test_size: float,
    seed: int,
):
    groups = build_groups(df)

    if groups is not None and len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, Y, groups=groups))
        split_type = "GroupShuffleSplit"
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        split_type = "Random train_test_split"

    return train_idx, test_idx, split_type


def make_model(max_features: int, min_df: int, ngram_max: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=(1, ngram_max),
            strip_accents=None,
            lowercase=True,
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
            )
        )),
    ])


def evaluate(Y_true: np.ndarray, Y_pred: np.ndarray, targets: List[str]) -> Dict[str, Any]:
    p, r, f1, support = precision_recall_fscore_support(
        Y_true,
        Y_pred,
        average=None,
        zero_division=0,
    )

    per_label = {}

    for i, target in enumerate(targets):
        tn, fp, fn, tp = confusion_matrix(
            Y_true[:, i],
            Y_pred[:, i],
            labels=[0, 1],
        ).ravel()

        per_label[target] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    return {
        "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "per_label": per_label,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="data/models/m1_tfidf_logreg")
    ap.add_argument("--text-mode", choices=["metadata", "combined", "separate"], default="metadata")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-features", type=int, default=60000)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--ngram-max", type=int, default=2)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = add_derived_targets(df)

    requested_targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    active_targets, skipped_targets = filter_valid_targets(df, requested_targets)

    if not active_targets:
        raise ValueError("No valid targets with both classes.")

    X = make_text(df, args.text_mode)
    Y = np.stack([get_target_array(df, t) for t in active_targets], axis=1)

    train_idx, test_idx, split_type = train_test_split_grouped(
        X=X,
        Y=Y,
        df=df,
        test_size=args.test_size,
        seed=args.seed,
    )

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    model = make_model(
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
    )

    print("=" * 80)
    print("M1: TF-IDF + One-vs-Rest Logistic Regression")
    print("=" * 80)
    print(f"Rows: {len(df)}")
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")
    print(f"Split: {split_type}")
    print(f"Active targets: {active_targets}")
    print(f"Skipped targets: {skipped_targets}")

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    report = evaluate(Y_test, Y_pred, active_targets)

    full_report = {
        "model_id": "M1",
        "model_name": "TF-IDF + One-vs-Rest Logistic Regression",
        "dataset_csv": args.csv,
        "text_mode": args.text_mode,
        "split_type": split_type,
        "test_size": args.test_size,
        "seed": args.seed,
        "requested_targets": requested_targets,
        "active_targets": active_targets,
        "skipped_targets": skipped_targets,
        "tfidf": {
            "max_features": args.max_features,
            "min_df": args.min_df,
            "ngram_range": [1, args.ngram_max],
        },
        "metrics": report,
    }

    report_path = os.path.join(args.outdir, "m1_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    bundle = {
        "model": model,
        "targets": active_targets,
        "text_mode": args.text_mode,
        "model_id": "M1",
    }

    model_path = os.path.join(args.outdir, "m1_tfidf_logreg.joblib")
    joblib.dump(bundle, model_path)

    test_pred_df = df.iloc[test_idx].copy()
    for i, target in enumerate(active_targets):
        test_pred_df[f"true_{target}"] = Y_test[:, i]
        test_pred_df[f"pred_{target}"] = Y_pred[:, i]

    pred_path = os.path.join(args.outdir, "m1_test_predictions.csv")
    test_pred_df.to_csv(pred_path, index=False)

    print("\nSaved:")
    print(f"- {report_path}")
    print(f"- {model_path}")
    print(f"- {pred_path}")
    print("\nMain metrics:")
    print(f"Macro-F1: {report['macro_f1']:.4f}")
    print(f"Micro-F1: {report['micro_f1']:.4f}")

    for target, m in report["per_label"].items():
        print(
            f"{target:28s} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}"
        )


if __name__ == "__main__":
    main()