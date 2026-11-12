"""
Create leave-one-actor-out (LOAO) cross-validation folds.

This is basically just for data splitting, to ensure that any actors in the 
training set are never included in the test set to prevent actor identiy in our emotion 
recognition task

Output: results/folds.csv (filename, actor, fold), shared by ALL classifiers.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths (relative to the repo root)
# ---------------------------------------------------------------------------
METADATA_FILE = Path("results/metadata.csv")
OUTPUT_FILE = Path("results/folds.csv")


# ---------------------------------------------------------------------------
# Step 1: load the metadata table
# ---------------------------------------------------------------------------
meta = pd.read_csv(METADATA_FILE)
print(f"Loaded {len(meta)} rows from {METADATA_FILE}")


# ---------------------------------------------------------------------------
# Step 2: leave-one-actor-out folds
# The fold number is simply the actor number (1-24). In fold k, actor k is
# the test set and the other 23 actors are the training set, so the model
# is always tested on a speaker it has never heard.
# ---------------------------------------------------------------------------
folds = pd.DataFrame()
folds["filename"] = meta["filename"]
folds["actor"] = meta["actor"]
folds["fold"] = meta["actor"]


# ---------------------------------------------------------------------------
# Step 3: keep only filename, actor, fold (same row order as metadata.csv)
# ---------------------------------------------------------------------------
folds = folds[["filename", "actor", "fold"]]


# ---------------------------------------------------------------------------
# Step 4: sanity checks
# ---------------------------------------------------------------------------
# check the folds look right
print("Total clips:", len(folds))                       # should be 1440
print("Number of folds:", folds["fold"].nunique())      # should be 24
print("Clips per fold:", folds["fold"].value_counts().unique())   # should be [60]

# each actor should only be in one fold (no speaker leakage)
actors_in_many_folds = (folds.groupby("actor")["fold"].nunique() > 1).sum()
print("Actors in more than one fold:", actors_in_many_folds)       # should be 0

# example: fold 1 should give 1380 training clips and 60 test clips
print("Fold 1 train/test:", (folds["fold"] != 1).sum(), "/", (folds["fold"] == 1).sum())


# ---------------------------------------------------------------------------
# Step 5: save the folds table
# ---------------------------------------------------------------------------
folds.to_csv(OUTPUT_FILE, index=False)
print()
print(f"Saved {len(folds)} rows to {OUTPUT_FILE}")
