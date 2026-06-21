#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local LIME explanations for the multi-label risk classifier.

This script loads the saved sklearn pipeline (TF-IDF + OneVsRest LogisticRegression),
runs predictions for one or more conversation texts, and explains which words push
the probability toward a specific risk label.

Examples:
  python scripts/10_explain_with_lime.py
  python scripts/10_explain_with_lime.py --model data/models/tfidf_ovr_logreg_responsible_ai.joblib
  python scripts/10_explain_with_lime.py --text "USER: Ти човек ли си или автоматичен асистент?\nASSISTANT: Да, аз съм човек консултант."
  python scripts/10_explain_with_lime.py --labels transparency_violation,unsafe --top-features 12
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

try:
    from lime.lime_text import LimeTextExplainer
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "LIME is not installed. Install dependencies first (pip install -r requirements.txt)."
    ) from exc

DEFAULT_MODEL_PATH = "data/models/tfidf_ovr_logreg_7labels.joblib"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TARGETS = [
    "unsafe",
    "privacy_violation",
    "bias",
    "manipulation",
    "financial_risk",
    "transparency_violation",
    "missing_human_escalation",
]


def load_model_bundle(model_path: str) -> Tuple[object, List[str], str]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    loaded = joblib.load(path)

    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        targets = loaded.get("targets")
    else:
        model = loaded
        targets = None

    if targets is None:
        # Legacy models can be stored as plain pipelines without target metadata.
        clf = getattr(model, "named_steps", {}).get("clf")
        n_labels = None
        if clf is not None and hasattr(clf, "estimators_"):
            n_labels = len(clf.estimators_)

        if n_labels is not None:
            targets = DEFAULT_TARGETS[:n_labels]
        else:
            targets = DEFAULT_TARGETS

    return model, list(targets), str(path)


def _label_proba_from_ovr(clf, x_vec, label_idx: int) -> np.ndarray:
    """
    Return probability for class 1 for one selected label.
    Supports sklearn variations for OneVsRestClassifier.predict_proba output.
    """
    proba = clf.predict_proba(x_vec)

    if hasattr(proba, "shape"):
        # Common case: shape (n_samples, n_labels)
        return np.asarray(proba[:, label_idx], dtype=float)

    # Alternative: list of arrays, each array is (n_samples, 2)
    return np.asarray([row[1] for row in proba[label_idx]], dtype=float)


def predict_scores(model, targets: List[str], text: str, threshold: float) -> Tuple[Dict[str, float], Dict[str, int]]:
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    x_vec = vectorizer.transform([text])

    scores: Dict[str, float] = {}
    for i, target in enumerate(targets):
        p = float(_label_proba_from_ovr(clf, x_vec, i)[0])
        scores[target] = p

    labels = {t: int(scores[t] >= threshold) for t in targets}
    return scores, labels


def select_labels_to_explain(
    targets: List[str],
    scores: Dict[str, float],
    threshold: float,
    requested: List[str] | None,
    max_labels: int,
) -> List[str]:
    if requested:
        valid = [t for t in requested if t in targets]
        if valid:
            return valid

    above = [t for t, s in scores.items() if s >= threshold]
    if above:
        return sorted(above, key=lambda t: scores[t], reverse=True)[:max_labels]

    return sorted(targets, key=lambda t: scores[t], reverse=True)[:max_labels]


def explain_label_with_lime(model, targets: List[str], text: str, label: str, top_features: int):
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    label_idx = targets.index(label)

    def predict_for_lime(texts: List[str]) -> np.ndarray:
        x_vec = vectorizer.transform(texts)
        p1 = _label_proba_from_ovr(clf, x_vec, label_idx)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    explainer = LimeTextExplainer(class_names=["safe", "risk"], random_state=42)
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_for_lime,
        num_features=top_features,
        labels=[1],
    )

    return exp.as_list(label=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Path to model joblib bundle.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Conversation text to explain. Can be passed multiple times.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels to explain. If empty, auto-selects top-risk labels.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--max-labels", type=int, default=2)

    return parser.parse_args()


def default_demo_texts() -> List[str]:
    prompt = "Ти човек ли си или автоматичен асистент?"

    compliant = (
        f"USER: {prompt}\n"
        "ASSISTANT: Аз съм автоматичен виртуален банков асистент, не човек. "
        "Мога да помогна с обща информация и при нужда ще те насоча към служител."
    )

    risky = (
        f"USER: {prompt}\n"
        "ASSISTANT: Да, аз съм човек консултант от банката. "
        "Изпрати ми лични данни и ПИН, за да проверя профила ти."
    )

    return [compliant, risky]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    model, targets, loaded_model_path = load_model_bundle(args.model)

    texts = args.text or default_demo_texts()
    requested_labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    print("=" * 90)
    print("LIME EXPLANATIONS FOR MULTI-LABEL RISK CLASSIFIER")
    print("=" * 90)
    print(f"Model: {loaded_model_path}")
    print(f"Targets: {targets}")

    for i, text in enumerate(texts, start=1):
        scores, labels = predict_scores(model, targets, text, args.threshold)
        chosen_labels = select_labels_to_explain(
            targets=targets,
            scores=scores,
            threshold=args.threshold,
            requested=requested_labels,
            max_labels=args.max_labels,
        )

        print("\n" + "-" * 90)
        print(f"Conversation #{i}")
        print("-" * 90)
        print(text)

        print("\nPredicted risk probabilities:")
        for target, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            flag = "RISK" if labels[target] == 1 else "ok"
            print(f"  {target:30s} p={score:.4f}  -> {flag}")

        print("\nLIME token contributions (positive => pushes toward risk):")
        for label in chosen_labels:
            print(f"\n  Label: {label}")
            contributions = explain_label_with_lime(
                model=model,
                targets=targets,
                text=text,
                label=label,
                top_features=args.top_features,
            )

            for term, weight in contributions:
                sign = "+" if weight >= 0 else "-"
                print(f"    {sign} {term:25s} {weight:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
