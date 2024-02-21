# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np


__author__ = 'Davide Pellis'


def ray_triangle_intersection(ray_origin, ray_direction, V1, V2, V3):
    '''
    see:
    https://www.scratchapixel.com/lessons/3d-basic-rendering/
    ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    '''
    E_1 = V2 - V1
    E_2 = V3 - V1
    O = np.array(ray_origin)
    D = np.array(ray_direction)
    T = O[[0,1,2]] - V1[:,[0,1,2]]
    P = np.cross(D, E_2, axisb=1)
    Q = np.cross(T, E_1, axisb=1)
    f = (np.einsum('ij,ij -> i', P, E_1) + 1e-10)**-1
    t = f * np.einsum('ij,ij -> i', Q, E_2)
    u = f * np.einsum('ij,ij -> i', P, T)
    v = f * np.einsum('ij,j -> i', Q, D)
    intersection = np.column_stack((u, v, t))
    return intersection

def is_ray_triangle_intersection(ray_origin, ray_direction, V1, V2, V3):
    P = ray_triangle_intersection(ray_origin, ray_direction, V1, V2, V3)
    a = np.logical_and(P[:,0] >= 0, P[:,1] >= 0)
    intersection = np.logical_and(a, P[:,0] + P[:,1] <= 1)
    return intersection

def closest_ray_triangle_intersection(ray_origin, ray_direction, V1, V2, V3):
    eps = 1e-5
    P = ray_triangle_intersection(ray_origin, ray_direction, V1, V2, V3)
    a = np.logical_and(P[:,0]  >= -eps, P[:,1]  >= -eps)
    intersection = np.logical_and(a, P[:,0] + P[:,1]  <= 1 + eps)
    indices = np.where(intersection)[0]
    if len(indices) > 0:
        index = indices[np.argmin(np.abs(P[indices, 2]))]
    else:
        index = -1
    return index

def closest_mesh_face_ray_intersection(mesh, ray_origin, ray_direction):
    T, f = mesh.face_triangles()
    A = mesh.vertices[T[:,0]]
    B = mesh.vertices[T[:,1]]
    C = mesh.vertices[T[:,2]]
    t = closest_ray_triangle_intersection(ray_origin, ray_direction, A, B, C)
    if t == -1:
        return -1
    else:
        return f[t]



if __name__ == '__main__':


    ro = np.array([-1,-1,-1])
    d = np.array([1,1,1])
    v1 = np.random.random((1,3)) * 2
    v2 = np.random.random((1,3)) * 2
    v3 = np.random.random((1,3)) * 2
    i = closest_ray_triangle_intersection(ro, d, v1, v2, v3)
    print(i)





