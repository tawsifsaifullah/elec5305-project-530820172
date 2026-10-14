"""
Speaker-independent evaluation: leave-one-actor-out (LOAO).
All recordings from an actor (both statements, both repetitions, all emotions,
both intensities) stay on the same side of the train/test boundary.
"""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut


def actor_folds(meta):
    logo = LeaveOneGroupOut()
    groups = meta["actor"].values
    for train_idx, test_idx in logo.split(np.zeros(len(meta)), groups=groups):
        check_no_leakage(meta, train_idx, test_idx)
        yield train_idx, test_idx


def check_no_leakage(meta, train_idx, test_idx):
    train_actors = set(meta["actor"].iloc[train_idx])
    test_actors = set(meta["actor"].iloc[test_idx])
    overlap = train_actors & test_actors
    assert not overlap, f"Actor leakage between train and test: {overlap}"
