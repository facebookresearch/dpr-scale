# Codes are adapted from https://github.com/luyug/COIL/tree/main/retriever/retriever_ext
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


# cython: wraparound = False
# cython: boundscheck = False
# cython: language_level=3
import numpy as np
cimport numpy as np
import cython
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
def scatter_max(
        np.ndarray[float, ndim=1] src,
        np.ndarray[long, ndim=1] indices,
        np.ndarray[float, ndim=1] tgt,
):
    cdef long i, n=src.shape[0]
    if n < 16384:
            for i in range(n):
                if src[i] > 0:
                    tgt[indices[i]] = max(tgt[indices[i]], src[i])
                
    else:
        for i in prange(n, nogil=True, schedule='static'):
            if src[i] > 0:
                tgt[indices[i]] = max(tgt[indices[i]], src[i])


@cython.boundscheck(False)
@cython.wraparound(False)
def scatter_add(
        np.ndarray[float, ndim=1] src,
        np.ndarray[long, ndim=1] indices,
        np.ndarray[float, ndim=1] tgt,
):
    cdef long i, n=indices.shape[0]
    if n < 16384:
            for i in range(n):
                tgt[indices[i]] = tgt[indices[i]] + src[indices[i]]
                
    else:
        for i in prange(n, nogil=True, schedule='static'):
            tgt[indices[i]] = tgt[indices[i]] + src[indices[i]]


@cython.boundscheck(False)
@cython.wraparound(False)
def scatter_index(
        np.ndarray[float, ndim=2] src,
        np.ndarray[long, ndim=1] indices,
        np.ndarray[float, ndim=1] weights,
        np.ndarray[float, ndim=2] tgt,
):
    cdef long i, n=tgt.shape[0]
    cdef long j, m=tgt.shape[1]
    if n < 16384:
            for i in range(n):
                for j in range(m):
                    tgt[i, j] = weights[i] * src[indices[i], j]
                
    else:
        for i in prange(n, nogil=True, schedule='static'):
            for j in range(m):
                tgt[i, j] = weights[i] * src[indices[i], j]
