# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from scipy.sparse import coo_matrix

from scipy import spatial

from pypardiso import spsolve

# -----------------------------------------------------------------------------

from geometrylab import utilities

# -----------------------------------------------------------------------------

__author__ = 'Davide Pellis'


def tutte_embedding(mesh):
    mesh.make_simply_connected()
    bc = mesh.boundary_curves()
    if len(bc) != 1:
        return False
    curve = bc[0]
    N = len(curve)
    phi = np.linspace(0, 2*np.pi, N+1)[:-1]
    x = np.sin(phi)
    y = np.cos(phi)
    V = mesh.V
    v, vj, L = mesh.vertex_ring_vertices_iterators(sort=True, return_lengths=True)
    mask = np.invert(np.in1d(v, curve))
    mask0 = np.full(V, True)
    mask0[curve] = False
    v = v[mask]
    vj = vj[mask]
    L = L[mask0]
    v0 = np.arange(V)[mask0]
    i = np.hstack((v, v0, curve))
    j = np.hstack((vj, v0, curve))
    data = np.hstack((np.ones(vj.shape[0]), -L, np.ones(curve.shape[0])))
    M = coo_matrix((data, (i, j)), shape=(V, V))
    M = M.tocsr()
    a = np.zeros(V)
    a[curve] = x
    X = spsolve(M, a)
    a[curve] = y
    Y = spsolve(M, a)
    uv = np.array([X, Y]).T
    return uv




















