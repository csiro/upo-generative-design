"""Utilities"""

import typing as T
from itertools import product

import numpy as np
import torch
from polyleven import levenshtein

#
# Script helper functions
#

AMINO_ALPHA = "ARNDCQEGHILKMFPSTWYV"
AA_TO_INT = {a: i for i, a in enumerate(AMINO_ALPHA)}
INT_TO_AA = {i: a for i, a in enumerate(AMINO_ALPHA)}


def seq2arr(S: T.Sequence[str]) -> np.ndarray:
    X = np.vstack([np.array(list(s)).view("U1") for s in S])
    return X


def arr2seq(X: np.ndarray) -> T.Sequence[str]:
    S = ["".join(list(x)) for x in X]
    return S


def seq2intarr(S: T.Sequence[str]) -> torch.IntTensor:
    Xc = seq2arr(S)
    c2int = AA_TO_INT.__getitem__
    Xi = np.vectorize(c2int)(Xc)
    return torch.tensor(Xi).long()


def intarr2seq(X: torch.IntTensor) -> T.Sequence[str]:
    int2c = INT_TO_AA.__getitem__
    Xc = np.vectorize(int2c)(X.detach().numpy())
    S = arr2seq(Xc)
    return S


def to_probability(logp: torch.Tensor) -> np.ndarray:
    return np.exp(logp.detach().to("cpu").numpy().squeeze())


def diversity(S: T.Sequence[str]) -> float:
    num_seqs = len(S)
    if num_seqs <= 1:
        return 0.0
    total_dist = 0
    for si, sj in product(S, S):
        if si == sj:
            continue
        total_dist += levenshtein(si, sj)
    return total_dist / (num_seqs * (num_seqs - 1))
