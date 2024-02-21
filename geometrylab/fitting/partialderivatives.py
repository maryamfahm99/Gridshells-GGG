# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from scipy import sparse

from scipy.sparse import coo_matrix

from scipy import spatial

from pypardiso import spsolve

# -----------------------------------------------------------------------------

from geometrylab import utilities

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


def uv_derivatives(mesh, vertex_function=None):
    V = mesh.V
    v0, vj = mesh.vertex_ring_vertices_iterators(sort=True)
    U0 = mesh._uv[v0,0]
    Uj = mesh._uv[vj,0]
    V0 = mesh._uv[v0,1]
    Vj = mesh._uv[vj,1]
    K = v0.shape[0]
    f0 = vertex_function[v0]
    fj = vertex_function[vj]
    dfu = v0
    dfv = V + v0
    i = np.arange(K)
    i = np.hstack((i, i))
    j = np.hstack((dfu, dfv))
    data = np.hstack((Uj-U0, Vj-V0))
    r = fj - f0
    H = sparse.coo_matrix((data, (i,j)), shape=(K, 2*V))
    D = sparse.linalg.lsqr(H,r)[0]
    dfu = D[0:V]
    dfv = D[V:2*V]
    return dfu, dfv




















