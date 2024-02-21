#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

# -----------------------------------------------------------------------------

from geometrylab import utilities

from geometrylab.geometry.meshpy import Mesh

# -----------------------------------------------------------------------------

'''_'''

__author__ = 'Davide Pellis'


def make_beams(mesh, height=0.1, width=0.05, unweld=False,
              epsilon=1e-5, offset=0):
    H = mesh.halfedges
    vertices = []
    faces = []
    index = 0
    edge_normals = mesh.edge_normals()
    V1, V2 = mesh.edge_vertices()
    edge_axis = mesh.edge_versors()
    vertex_normals = mesh.vertex_normals()
    edge_tan = np.cross(edge_axis, edge_normals)
    #edge_tan = edge_tan / np.linalg.norm(edge_tan, axis=1, keepdims=True)
    boundary_edges = mesh.are_boundary_edges()
    H1, H2 = mesh.edge_half_edges()
    for i in range(mesh.E):
        boundary = boundary_edges[i]
        n = edge_normals[i]
        t = edge_tan[i]
        ev = edge_axis[i]
        v1 = mesh.vertices[V1[i],:]
        v2 = mesh.vertices[V2[i],:]
        h1 = H1[i]
        h2 = H2[i]
        n_v1 = vertex_normals[V1[i],:]
        n_v2 = vertex_normals[V2[i],:]
        e1 = H[H[h1,3],5]
        e2 = H[H[h2,2],5]
        e3 = H[H[h1,2],5]
        e4 = H[H[h2,3],5]
        boundary_e1 = False
        boundary_e2 = False
        boundary_e3 = False
        boundary_e4 = False
        if boundary:
            boundary_e1 = boundary_edges[e1]
            boundary_e2 = boundary_edges[e2]
            boundary_e3 = boundary_edges[e3]
            boundary_e4 = boundary_edges[e4]
        v1top = v1 + (height/2 + offset) * n
        v1bot = v1 - (height/2 - offset) * n
        v2top = v2 + (height/2 + offset) * n
        v2bot = v2 - (height/2 - offset) * n
        v_r = v1 - (width/2) * t
        v_l = v1 + (width/2) * t
        if V1[e1] == V1[i]:
            t1 = edge_tan[e1]
            ev1 = -edge_axis[e1]
        else:
            t1 = -edge_tan[e1]
            ev1 = edge_axis[e1]
        if V1[e2] == V1[i]:
            t2 = -edge_tan[e2]
            ev2 = -edge_axis[e2]
        else:
            t2 = edge_tan[e2]
            ev2 = edge_axis[e2]
        if V1[e3] == V2[i]:
            t3 = -edge_tan[e3]
            ev3 = edge_axis[e3]
        else:
            t3 = edge_tan[e3]
            ev3 = -edge_axis[e3]
        if V1[e4] == V2[i]:
            t4 = edge_tan[e4]
            ev4 = edge_axis[e4]
        else:
            t4 = -edge_tan[e4]
            ev4 = -edge_axis[e4]
        o1 = v1 + (width/2) * t1
        o2 = v1 + (width/2) * t2
        o3 = v2 + (width/2) * t3
        o4 = v2 + (width/2) * t4

        if np.linalg.norm(np.cross(t,t1)) > epsilon and not boundary_e1:
            A = np.array([n,t,t1])
            b1 = np.array([np.dot(n,v1bot), np.dot(t,v_r), np.dot(t1,o1)])
            b2 = np.array([np.dot(n,v1top), np.dot(t,v_r), np.dot(t1,o1)])
            p5 = np.linalg.solve(A,b1)
            p6 = np.linalg.solve(A,b2)
        else:
            t1 = ev + ev1
            A = np.array([n,t,t1])
            b1 = np.array([np.dot(n,v1bot), np.dot(t,v_r), np.dot(t1,v1)])
            b2 = np.array([np.dot(n,v1top), np.dot(t,v_r), np.dot(t1,v1)])
            p5 = np.linalg.solve(A,b1)
            p6 = np.linalg.solve(A,b2)

        if np.linalg.norm(np.cross(t,t2)) > epsilon and not boundary_e2:
            A = np.array([n,t,t2])
            b1 = np.array([np.dot(n,v1bot), np.dot(t,v_l), np.dot(t2,o2)])
            b2 = np.array([np.dot(n,v1top), np.dot(t,v_l), np.dot(t2,o2)])
            p7 = np.linalg.solve(A,b1)
            p8 = np.linalg.solve(A,b2)
        else:
            t2 = ev + ev2
            A = np.array([n,t,t2])
            b1 = np.array([np.dot(n,v1bot), np.dot(t,v_l), np.dot(t2,v1)])
            b2 = np.array([np.dot(n,v1top), np.dot(t,v_l), np.dot(t2,v1)])
            p7 = np.linalg.solve(A,b1)
            p8 = np.linalg.solve(A,b2)

        if np.linalg.norm(np.cross(t,t3)) > epsilon and not boundary_e3:
            A = np.array([n,t,t3])
            b1 = np.array([np.dot(n,v1bot), np.dot(t,v_r), np.dot(t3,o3)])
            b2 = np.array([np.dot(n,v1top), np.dot(t,v_r), np.dot(t3,o3)])
            p9 = np.linalg.solve(A,b1)
            p10 = np.linalg.solve(A,b2)
        else:
            t3 = ev + ev3
            A = np.array([n,t,t3])
            b1 = np.array([np.dot(n,v1bot), np.dot(t,v_r), np.dot(t3,v2)])
            b2 = np.array([np.dot(n,v1top), np.dot(t,v_r), np.dot(t3,v2)])
            p9 = np.linalg.solve(A,b1)
            p10 = np.linalg.solve(A,b2)

        if np.linalg.norm(np.cross(t,t4)) > epsilon and not boundary_e4:
            A = np.array([n,t,t4])
            b1 = np.array([np.dot(n,v1bot), np.dot(t,v_l), np.dot(t4,o4)])
            b2 = np.array([np.dot(n,v1top), np.dot(t,v_l), np.dot(t4,o4)])
            p11 = np.linalg.solve(A,b1)
            p12 = np.linalg.solve(A,b2)
        else:
            t4 = ev + ev4
            A = np.array([n,t,t4])
            b1 = np.array([np.dot(n,v1bot), np.dot(t,v_l), np.dot(t4,v2)])
            b2 = np.array([np.dot(n,v1top), np.dot(t,v_l), np.dot(t4,v2)])
            p11 = np.linalg.solve(A,b1)
            p12 = np.linalg.solve(A,b2)

        r1v1 = utilities.orthogonal_vector(n_v1)
        r2v1 = np.cross(n_v1,r1v1)
        r1v2 = utilities.orthogonal_vector(n_v2)
        r2v2 = np.cross(n_v2,r1v2)

        A = np.array([r1v1, r2v1, n])
        b1 = np.array([np.dot(v1,r1v1), np.dot(v1,r2v1), np.dot(n,v1bot)])
        b2 = np.array([np.dot(v1,r1v1), np.dot(v1,r2v1), np.dot(n,v1top)])
        p1 = np.linalg.solve(A,b1)
        p3 = np.linalg.solve(A,b2)

        A = np.array([r1v2, r2v2, n])
        b1 = np.array([np.dot(v2,r1v2), np.dot(v2,r2v2), np.dot(n,v2bot)])
        b2 = np.array([np.dot(v2,r1v2), np.dot(v2,r2v2), np.dot(n,v2top)])
        p2 = np.linalg.solve(A,b1)
        p4 = np.linalg.solve(A,b2)

        if not unweld:
            vertices.extend((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12))
            faces.append([index,index+6,index+10,index+1,index+8,index+4])
            faces.append([index+2,index+5,index+9,index+3,index+11,index+7])
            faces.append([index+4,index+8,index+9,index+5])
            faces.append([index+6,index+7,index+11,index+10])
            index += 12
        else:
            vertices.extend((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p5,p6,p7,
                             p8,p9,p10,p11,p12))
            faces.append([index+12,index+16,index+17,index+13])
            faces.append([index+14,index+15,index+19,index+18])
            faces.append([index+4,index,index+1,index+8])
            faces.append([index,index+6,index+10,index+1])
            faces.append([index+2,index+5,index+9,index+3])
            faces.append([index+2,index+3,index+11,index+7])
            index += 20
    M = Mesh()
    M.make_mesh(vertices, faces)
    return M




