# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from geometrylab.geometry import Frame


__author__ = 'Davide Pellis'


def mesh_plane_projection(mesh, frame=None, light_direction=[1,-1,1], offset=None):
    '''
    see:
    https://www.scratchapixel.com/lessons/3d-basic-rendering/
    ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    '''
    if frame is None:
        frame = Frame()
        if offset is None:
            frame.origin[:,2] = np.min(mesh.vertices[:,2])
        else:
            frame.origin =- frame.e3*offset
    A = np.tile(frame.origin, (mesh.V,1))
    O = mesh.vertices
    E_1 = np.tile(frame.e1, (mesh.V,1))
    E_2 = np.tile(frame.e2, (mesh.V,1))
    D = np.array(light_direction)
    T = O - A
    P = np.cross(D, E_2, axisb=1)
    Q = np.cross(T, E_1, axisb=1)
    f = (np.einsum('ij,ij -> i', P, E_1) + 1e-20)**-1
    #t = f * np.einsum('ij,ij -> i', Q, E_2)
    u = f * np.einsum('ij,ij -> i', P, T)
    v = f * np.einsum('ij,j -> i', Q, D)
    i = (np.einsum('i,ij -> ij', u, E_1) + np.einsum('i,ij -> ij', v, E_2)) + A
    return i

def mesh_plane_shadow(mesh, frame=None, light_direction=[1,-1,1], offset=None):
    P = mesh_plane_projection(mesh, frame=frame,
                              light_direction=light_direction, offset=offset)
    M = mesh.copy_mesh()
    M.vertices = P
    return M


if __name__ == '__main__':

    A = np.array([[1,2],[1,2],[1,2]])
    B = np.array([1,2,3])
    print(np.einsum('i,ij -> ij', B, A))
    pass





