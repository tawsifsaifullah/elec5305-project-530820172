# importing the libraries
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd

# metadata (labels), folds (train/test splits) and the lab acoustic features from MATLAB
meta  = pd.read_csv("results/metadata.csv")
folds = pd.read_csv("results/folds.csv")
acou  = pd.read_csv("results/acoustic_features.csv")

# SSL layer features from extract_ssl.py, shape (1440, 13, 768)            # NEW
wavlm  = np.load("results/wavlm_layers.npy")                                 # NEW
unisat = np.load("results/unispeechsat_layers.npy")                          # NEW

# target: the intended emotion
y = meta["emotion"].values

# list of every feature set to test: (system name, layer, feature matrix)  # NEW
systems = [("acoustic", -1, acou.drop(columns=["filename"]).values)]        # NEW
for layer in range(13):                                                      # NEW
    systems.append(("wavlm", layer, wavlm[:, layer, :]))                     # NEW
    systems.append(("unispeechsat", layer, unisat[:, layer, :]))             # NEW

rows = []
fold_scores = []

for system, layer, X in systems:                                             # NEW
    print("Running", system, "layer", layer)                                 # NEW
    for fold in range(1, 25):                       # (everything below is indented one level more)
        train = (folds["fold"] != fold).values
        test  = (folds["fold"] == fold).values

        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C = 0.01, max_iter=2000, class_weight="balanced"))
        model.fit(X[train], y[train])               # learn on 23 actors
        probs = model.predict_proba(X[test])        # confidence for the held-out actor
        pred  = model.predict(X[test])

        # score this fold (one actor)
        fold_scores.append({"system": system, "layer": layer,               # NEW
                            "fold": fold,
                            "macro_f1": f1_score(y[test], pred, average="macro"),
                            "accuracy": accuracy_score(y[test], pred)})

        # keep every test clip's prediction and confidence
        classes = list(model.classes_)
        for i, idx in enumerate(np.where(test)[0]):
            rows.append({"system": system, "layer": layer,                  # NEW
                         "filename": meta["filename"][idx],
                         "emotion": y[idx],
                         "pred": pred[i],
                         "p_true": probs[i, classes.index(y[idx])],   # confidence in the right answer
                         "correct": int(pred[i] == y[idx])})

# summary: mean and std across the 24 actors for every system/layer       # NEW
scores = pd.DataFrame(fold_scores)
summary = scores.groupby(["system", "layer"])[["macro_f1", "accuracy"]].agg(["mean", "std"])
print(summary.round(3))
summary.to_csv("results/classifier_summary.csv")
pd.DataFrame(rows).to_csv("results/predictions.csv", index=False)