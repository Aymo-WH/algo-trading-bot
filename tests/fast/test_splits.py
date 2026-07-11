"""Proofs that purge/embargo actually work (mission Phase 0 deliverable)."""
from math import comb

import numpy as np
import pytest

from validation.splits import cpcv_splits, walk_forward_splits, _contiguous_blocks

T, PURGE, EMBARGO = 1000, 5, 10


def test_cpcv_split_count():
    splits = list(cpcv_splits(T, n_groups=8, k_test=2, purge=PURGE, embargo=EMBARGO))
    assert len(splits) == comb(8, 2) == 28


def test_cpcv_no_train_test_overlap():
    for train, test in cpcv_splits(T, 8, 2, PURGE, EMBARGO):
        assert len(np.intersect1d(train, test)) == 0


def test_cpcv_purge_before_each_test_block():
    for train, test in cpcv_splits(T, 8, 2, PURGE, EMBARGO):
        for start, _ in _contiguous_blocks(test):
            forbidden = np.arange(max(0, start - PURGE), start)
            assert len(np.intersect1d(train, forbidden)) == 0, \
                f"train obs inside purge window before test block at {start}"


def test_cpcv_embargo_after_each_test_block():
    for train, test in cpcv_splits(T, 8, 2, PURGE, EMBARGO):
        for _, end in _contiguous_blocks(test):
            forbidden = np.arange(end + 1, min(T, end + 1 + PURGE + EMBARGO))
            assert len(np.intersect1d(train, forbidden)) == 0, \
                f"train obs inside embargo window after test block at {end}"


def test_cpcv_every_group_tested_equally():
    counts = np.zeros(T)
    for _, test in cpcv_splits(T, 8, 2, PURGE, EMBARGO):
        counts[test] += 1
    # each group appears in C(7,1)=7 test combinations
    assert (counts == 7).all()


def test_cpcv_rejects_too_small_panel():
    with pytest.raises(ValueError):
        list(cpcv_splits(50, n_groups=8, k_test=2, purge=5, embargo=10))


def test_walk_forward_is_strictly_causal():
    folds = list(walk_forward_splits(T, n_folds=8, min_train=252, purge=PURGE))
    assert len(folds) == 8
    for train, test in folds:
        assert train.max() < test.min() - PURGE + 1  # gap of >= purge
        assert train.min() == 0                      # anchored
    # test blocks tile the post-min_train region without overlap
    all_test = np.concatenate([t for _, t in folds])
    assert len(all_test) == len(np.unique(all_test)) == T - 252


def test_walk_forward_no_lookahead_between_folds():
    folds = list(walk_forward_splits(T, n_folds=4, min_train=100, purge=PURGE))
    for (tr1, te1), (tr2, te2) in zip(folds, folds[1:]):
        assert te1.max() < te2.min()          # chronological test order
        assert tr2.max() > tr1.max()          # expanding train
