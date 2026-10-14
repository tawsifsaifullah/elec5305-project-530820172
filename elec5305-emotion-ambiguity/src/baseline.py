"""
Speaker-independent classifier on any utterance-level feature matrix.
Same lightweight classifier is reused later for WavLM / UniSpeech-SAT layers,
so differences come from the representation, not the classifier.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from splits import actor_folds


def make_classifier():
    return make_pipeline(SimpleImputer(strategy="median"),
                         StandardScaler(),
                         LogisticRegression(C=0.5, max_iter=3000))


def run_loao(meta, X, label_col="emotion"):
    """Leave-one-actor-out. Returns per-utterance predictions incl. confidence."""
    y = meta[label_col].values
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    for tr, te in actor_folds(meta):
        clf = make_classifier().fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        # map in case a class is missing from a training fold
        proba[np.ix_(te, np.searchsorted(classes, clf.classes_))] = p

    pred = classes[proba.argmax(1)]
    true_idx = np.searchsorted(classes, y)
    sorted_p = np.sort(proba, axis=1)
    out = meta.copy()
    out["pred"] = pred
    out["correct"] = (pred == y).astype(int)
    out["p_true"] = proba[np.arange(len(y)), true_idx]          # prob. of intended label
    out["confidence"] = sorted_p[:, -1]                          # max prob.
    out["margin"] = sorted_p[:, -1] - sorted_p[:, -2]            # top-1 minus top-2
    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i]
    return out


def summarise(pred_df, label_col="emotion"):
    y, p = pred_df[label_col], pred_df["pred"]
    # Human study counted neutral<->calm as correct; report the equivalent too.
    merge = lambda s: s.replace({"calm": "neutral/calm", "neutral": "neutral/calm"})
    per_actor = pred_df.groupby("actor").apply(
        lambda d: f1_score(d[label_col], d["pred"], average="macro"), include_groups=False)
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "macro_f1_actor_sd": per_actor.std(),
        "accuracy_neutral_calm_merged": accuracy_score(merge(y), merge(p)),
        "chance": 1 / y.nunique(),
    }
