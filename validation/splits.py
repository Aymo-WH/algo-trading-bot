"""Purged, embargoed cross-validation splits (López de Prado, AFML Ch. 7).

Positions are integer indices 0..T-1 over a chronologically sorted panel of
rebalance dates. Label semantics: the observation at position i uses the forward
return over (i, i+h] — i.e. a label horizon of ``h`` positions.

Leakage rules implemented:
- A train observation strictly BEFORE a test block leaks if its label window
  reaches into the block: drop the last ``purge`` train positions before each
  test block start.
- A train observation AFTER a test block leaks through the test labels (which
  extend ``purge`` beyond the block) and through serial correlation: drop
  ``purge + embargo`` train positions after each test block end.
"""
from itertools import combinations

import numpy as np


def _drop_around_block(train_mask: np.ndarray, start: int, end: int,
                       purge: int, embargo: int) -> None:
    """Remove leak-prone train positions around test block [start, end]."""
    T = len(train_mask)
    train_mask[max(0, start - purge):start] = False           # purge before
    train_mask[end + 1:min(T, end + 1 + purge + embargo)] = False  # purge+embargo after


def _contiguous_blocks(positions: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end)] inclusive runs of consecutive integers."""
    if len(positions) == 0:
        return []
    splits = np.where(np.diff(positions) > 1)[0]
    starts = np.concatenate([[0], splits + 1])
    ends = np.concatenate([splits, [len(positions) - 1]])
    return [(int(positions[s]), int(positions[e])) for s, e in zip(starts, ends)]


def cpcv_splits(n_obs: int, n_groups: int = 8, k_test: int = 2,
                purge: int = 5, embargo: int = 10):
    """Combinatorial purged CV: all C(n_groups, k_test) (train, test) index pairs.

    Yields (train_idx, test_idx) as int arrays. Groups are contiguous in time.
    """
    if n_obs < n_groups * (purge + embargo + 2):
        raise ValueError("n_obs too small for the requested groups/purge/embargo")
    groups = np.array_split(np.arange(n_obs), n_groups)
    for combo in combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[g] for g in combo])
        train_mask = np.ones(n_obs, dtype=bool)
        train_mask[test_idx] = False
        for start, end in _contiguous_blocks(np.sort(test_idx)):
            _drop_around_block(train_mask, start, end, purge, embargo)
        yield np.where(train_mask)[0], np.sort(test_idx)


def walk_forward_splits(n_obs: int, n_folds: int = 8, min_train: int = 252,
                        purge: int = 5):
    """Anchored (expanding) walk-forward: train = [0, test_start - purge).

    The post-min_train region is split into n_folds contiguous test blocks.
    Yields (train_idx, test_idx). No embargo needed: train never follows test.
    """
    if n_obs <= min_train + n_folds:
        raise ValueError("n_obs too small for min_train/n_folds")
    test_region = np.arange(min_train, n_obs)
    for block in np.array_split(test_region, n_folds):
        train_end = int(block[0]) - purge
        if train_end <= 0:
            continue
        yield np.arange(0, train_end), block.astype(int)
