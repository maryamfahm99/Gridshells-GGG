# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:35:11 2021

@author: wangh0m
"""
__author__ = 'Hui'
#------------------------------------------------------------------------------
import numpy as np

from scipy import sparse
#------------------------------------------------------------------------------
"""
from constraints_basic import 
    column3D,con_edge,con_unit,con_bigger_than,\
    con_constl,con_equal_length,\
    con_planarity,con_planarity_constraints,con_unit_normal,con_orient
"""
# -------------------------------------------------------------------------
#                           general / basic
# -------------------------------------------------------------------------

def column3D(arr, num1, num2):
    """
    Parameters
    ----------
    array : array([1,4,7]).
    num1 : starting num.=100
    num2 : interval num.= 10

    Returns
    -------
    a : array(100+[1,4,7, 10,14,17, 20,24,27]).
    """
    a = num1 + np.r_[arr, num2+arr, 2*num2+arr]
    return a

def con_edge(X,c_v1,c_v3,c_ld1,c_ud1):
    "(v1-v3) = ld1*ud1"
    num = len(c_ld1)
    ld1 = X[c_ld1]
    ud1 = X[c_ud1]
    a3 = np.ones(3*num)
    row = np.tile(np.arange(3*num),4)
    col = np.r_[c_v1,c_v3,np.tile(c_ld1,3),c_ud1]
    data = np.r_[a3,-a3,-ud1,-np.tile(ld1,3)]
    r = -np.tile(ld1,3)*ud1
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, len(X)))
    return H,r

def con_unit_normal(X,c_e1,c_e2,c_e3,c_e4,c_n):
    "n^2=1; n*(e1-e3)=0; n*(e2-e4);"
    "Hui: better than (l*n=(e1-e3)x(e2-e4), but no orientation"
    H1,r1 = con_unit(X,c_n)
    H2,r2 = con_planarity(X,c_e1,c_e3,c_n)
    H3,r3 = con_planarity(X,c_e2,c_e4,c_n)
    H = sparse.vstack((H1,H2,H3))
    r = np.r_[r1,r2,r3]
    return H,r

def con_unit(X,c_ud1,w=100):
    "ud1**2=1"
    num = int(len(c_ud1)/3)
    arr = np.arange(num)
    row = np.tile(arr,3)
    col = c_ud1
    data = 2*X[col]
    r =  np.linalg.norm(X[col].reshape(-1,3,order='F'),axis=1)**2 + np.ones(num)
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    f = np.linalg.norm(X[col].reshape(-1,3,order='F'),axis=1)**2 - np.ones(num)
    print("NORM OF f ", np.linalg.norm(f))
    return H*w,r*w

def con_constl(c_ld1,init_l1,N):
    "ld1 == const."
    num = len(c_ld1)
    row = np.arange(num,dtype=int)
    col = c_ld1
    data = np.ones(num,dtype=int)
    r = init_l1
    H = sparse.coo_matrix((data,(row,col)), shape=(num, N))
    return H,r

def con_bigger_than(X,minl,c_vi,c_vj,c_ai,num):
    "(vi-vj)^2-ai^2=minl"
    col = np.r_[c_vi,c_vj,c_ai]
    row = np.tile(np.arange(num),7)
    data = 2*np.r_[X[c_vi]-X[c_vj], -X[c_vi]+X[c_vj], -X[c_ai]]
    r = np.linalg.norm((X[c_vi]-X[c_vj]).reshape(-1,3,order='F'),axis=1)**2
    r = r - X[c_ai]**2 + np.ones(num)*minl
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_planarity(X,c_v1,c_v2,c_n): 
    "n*(v1-v2)=0"
    num = int(len(c_n)/3)
    col = np.r_[c_n,c_v1,c_v2]
    row = np.tile(np.arange(num),9)
    data = np.r_[X[c_v1]-X[c_v2],X[c_n],-X[c_n]]
    r = np.einsum('ij,ij->i',X[c_n].reshape(-1,3, order='F'),(X[c_v1]-X[c_v2]).reshape(-1,3, order='F')) 
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    #print("length ")
    #print(np.shape(H), np.shape(r))
    #print(r[0],np.dot(X[c_n].reshape(-1,3, order='F')[0],(X[c_v1]-X[c_v2]).reshape(-1,3, order='F')[0])) these are the same
    print("NORM OF f ", np.linalg.norm(r))
    return H,r

def con_equal_length(X,c1,c2,c3,c4):
    "(v1-v3)^2=(v2-v4)^2"
    num = int(len(c1)/3)
    row = np.tile(np.arange(num),12)
    col = np.r_[c1,c2,c3,c4]
    data = 2*np.r_[X[c1]-X[c3],X[c4]-X[c2],X[c3]-X[c1],X[c2]-X[c4]]
    r = np.linalg.norm((X[c1]-X[c3]).reshape(-1,3, order='F'),axis=1)**2
    r = r-np.linalg.norm((X[c2]-X[c4]).reshape(-1,3, order='F'),axis=1)**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num,len(X)))
    return H,r

def con_orient(X,Nv,c_vN,c_a,neg=False):
    "vN*Nv = a^2; if neg: vN*Nv = -a^2; variables: vN, a; Nv is given"
    if neg:
        sign = -1
    else:
        sign = 1    
    num = int(len(c_a))
    row = np.tile(np.arange(num),4)
    col = np.r_[c_vN,c_a]
    data = np.r_[Nv.flatten('F'),-sign*2*X[c_a]]
    r = -sign*X[c_a]**2
    H = sparse.coo_matrix((data,(row,col)), shape=(num,len(X))) 
    return H,r


    # -------------------------------------------------------------------------
    #                          Geometric Constraints (from Davide)
    # -------------------------------------------------------------------------

def con_normal_constraints(**kwargs):
    "represent unit normal: n^2=1"
    #w = kwargs.get('normal') * kwargs.get('geometric')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    V = mesh.V
    F = mesh.F
    f = 3*V + np.arange(F)
    i = np.arange(F)
    i = np.hstack((i, i, i)) #row ==np.tile(i,3) == np.r_[i,i,i]
    j = np.hstack((f, F+f, 2*F+f)) #col ==np.r_[f,F+f,2*F+f]
    data = 2 * np.hstack((X[f], X[F+f], X[2*F+f])) #* w
    H = sparse.coo_matrix((data,(i,j)), shape=(F,len(X)))
    r = ((X[f]**2 + X[F+f]**2 + X[2*F+f]**2) + 1) #* w
    return H,r


def con_planarity_constraints(is_unit_edge=False,**kwargs):
    "n*(vi-vj) = 0; Note: making sure normals is always next to V in X[V,N]"
    w = kwargs.get('planarity')
    mesh = kwargs.get('mesh')
    X = kwargs.get('X')
    V = mesh.V
    F = mesh.F
    f, v1, v2 = mesh.face_edge_vertices_iterators(order=True)
    if is_unit_edge:
        "f*(v1-v2)/length = 0, to avoid shrinkage of edges"
        num = len(v1)
        col_v1 = column3D(v1,0,V)
        col_v2 = column3D(v2,0,V)
        col_f = column3D(f,3*V,F)
        Ver = mesh.vertices
        edge_length = np.linalg.norm(Ver[v1]-Ver[v2],axis=1)
        row = np.tile(np.arange(num), 9)
        col = np.r_[col_f,col_v1,col_v2]
        l = np.tile(edge_length,3)
        data = np.r_[(X[col_v1]-X[col_v2])/l, X[col_f]/l, -X[col_f]/l]
        H = sparse.coo_matrix((data,(row, col)), shape=(num, len(X)))
        r = np.einsum('ij,ij->i',X[col_f].reshape(-1,3,order='F'),(X[col_v1]-X[col_v2]).reshape(-1,3,order='F'))
        r /= edge_length
        Hn,rn = con_unit(X,col_f,10*w)
        H = sparse.vstack((H*w,Hn))
        r = np.r_[r*w,rn]
    else:
        K = f.shape[0]
        f = 3*V + f
        r = ((X[v2] - X[v1]) * X[f] + (X[V+v2] - X[V+v1]) * X[F+f]
             + (X[2*V+v2] - X[2*V+v1]) * X[2*F+f] ) * w
        v1 = np.hstack((v1, V+v1, 2*V+v1))
        v2 = np.hstack((v2, V+v2, 2*V+v2))
        f = np.hstack((f, F+f, 2*F+f))
        i = np.arange(K)
        i = np.hstack((i, i, i, i, i, i, i, i, i))
        j = np.hstack((f, v2, v1))
        data = 2 * np.hstack((X[v2] - X[v1], X[f], -X[f])) * w
        H = sparse.coo_matrix((data,(i,j)), shape=(K, len(X)))
        Hn,rn = con_normal_constraints(**kwargs)
        H = sparse.vstack((H*w,Hn*w*10))
        r = np.r_[r*w,rn*w*10]
    return H,r
    # -------------------------------------------------------------------------
    #                          AAG-net constraint (from Maryam)
    # -------------------------------------------------------------------------

def G_diag(X, c_n, c_m):
    "n*m=0"
    # print("G_diag c_n shape, ", c_m, c_n, X.shape)
    num = int(len(c_n)/3)
    col = np.r_[c_n, c_m]
    row  = np.tile(np.arange(num), 6)
    #print("X dim again")
    #print(np.shape(X))
    data = np.r_[X[c_m], X[c_n]]
    r = np.einsum('ij,ij->i',X[c_n].reshape(-1,3, order='F'),X[c_m].reshape(-1,3, order='F')) # reshape(-1,3, order='F') provides us with the x, y, z, coordinates in one  vector
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    #print("length of r, gdiag")
    #print(np.shape(r), np.shape(H))
    print("NORM OF f from G_diag: ", np.linalg.norm(r))
    r = H*X-r # not sure
    return H,r

    # -------------------------------------------------------------------------
    #                          AGG constraint (from Maryam)
    # -------------------------------------------------------------------------

def con_equal_angles(X,c_v1, c_v2, c_v3, c_v4, c_v):
    "g1*g2 - g3*g4 = 0"

    # where gi = (vi-v)/norm(vi-v)
    X1 = (X[c_v1]-X[c_v]).reshape(-1,3, order='F')
    X2 = (X[c_v2]-X[c_v]).reshape(-1,3, order='F')
    X3 = (X[c_v3]-X[c_v]).reshape(-1,3, order='F')
    X4 = (X[c_v4]-X[c_v]).reshape(-1,3, order='F')
    #print(X[c_v1],X[c_v2],X[c_v3],X[c_v4])
    l1 =  np.linalg.norm(X1, axis = 1)
    l2 =  np.linalg.norm(X2, axis = 1)
    l3 =  np.linalg.norm(X3, axis = 1)
    l4 =  np.linalg.norm(X4, axis = 1)
    #print(l1,l2,l3,l4)
    num = int(len(c_v)/3)
    col = np.r_[c_v, c_v1, c_v2, c_v3, c_v4]
    row = np.tile(np.arange(num), 15)
    data = []
    f = []
    delta_theta = []
    #print((-X[c_v3]-X[c_v4]+2*X[c_v]).reshape(-1,3, order='F'))
    for i in range(num):
        data = np.r_[data, (-X[c_v1]-X[c_v2]+2*X[c_v]).reshape(-1,3, order='F')[i]/(l1[i]*l2[i]) - (-X[c_v3]-X[c_v4]+2*X[c_v]).reshape(-1,3, order='F')[i]/(l3[i]*l4[i]),
                     X2[i]/(l1[i]*l2[i]), X1[i]/(l1[i]*l2[i]), -X4[i]/(l3[i]*l4[i]), -X3[i]/(l3[i]*l4[i]) ]
        f = np.r_[f, np.dot(X1[i]/l1[i], X2[i]/l2[i]) - np.dot(X3[i]/l3[i], X4[i]/l4[i])]
        print("thetas: ")
        print(" ")
        delta_theta = np.r_[delta_theta, np.abs(np.arccos(np.dot(X1[i]/l1[i], X2[i]/l2[i])) - np.arccos(np.dot(X3[i]/l3[i], X4[i]/l4[i])))]
        print(np.arccos(np.dot(X1[i]/l1[i], X2[i]/l2[i])))
        print(np.arccos(np.dot(X3[i]/l3[i], X4[i]/l4[i])))
        print(" ")
        #print(r)
    #r = np.einsum('ij,ij->i',X[c_n].reshape(-1,3, order='F'),X[c_m].reshape(-1,3, order='F'))
    data = data.reshape(num, 15).T.flatten() 
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    print(" ")
    print("SUM ERROR theta", np.sum(delta_theta))
    print("NORM OF f ", np.linalg.norm(f))
    print("SUM ERROR  cos theta ", np.sum(np.abs(f)))
    print()
    r = H*X-f
    return H,r


def con_parallel(X,c_n, c_v,c_v1,c_v3):
    """
    "nX(g1+g2) = (0,0,0)"
    "n x m = (ny * (g1z + g3z) - nz * (g1y + g3y)) i + (nz * (g1x + g3x) - nx * (g1z + g3z)) j 
                             + (nx * (g1y + g3y) - ny * (g1x + g3x)) k"
    """

    # Set up the indicies
    num = int(len(c_n)/3)
    c_nx = c_n[:num]
    c_ny = c_n[num:2*num]
    c_nz = c_n[2*num:]
    c_vx = c_v[:num]
    c_vy = c_v[num:2*num]
    c_vz = c_v[2*num:]
    c_v1x = c_v1[:num]
    c_v1y = c_v1[num:2*num]
    c_v1z = c_v1[2*num:]
    c_v3x = c_v3[:num]
    c_v3y = c_v3[num:2*num]
    c_v3z = c_v3[2*num:]
    X1 = (X[c_v1]-X[c_v]).reshape(-1,3, order='F')
    X3 = (X[c_v3]-X[c_v]).reshape(-1,3, order='F')
    l1 =  np.linalg.norm(X1, axis = 1)
    l3 =  np.linalg.norm(X3, axis = 1)
    
    # for r = (ny * (g1z + g3z) - nz * (g1y + g3y))
    col1 = np.r_[c_ny, c_nz, c_vz, c_v1z, c_v3z, c_vy, c_v1y, c_v3y]
    row1 = np.tile(np.arange(num),8)
    print("shapes")
    print(len(c_ny), num)
    #print((X[c_v1z]-X[c_vz])/l1)
    #print((X[c_v1z]-X[c_vz])/l1[:None])
    data1 = np.r_[(X[c_v1z]-X[c_vz])/l1 + (X[c_v3z]-X[c_vz])/l3, -(X[c_v1y]-X[c_vy])/l1 -(X[c_v3y]-X[c_vy])/l3,
             -X[c_ny]/l1 -X[c_ny]/l3, X[c_ny]/l1, X[c_ny]/l3, X[c_nz]/l1 + X[c_nz]/l3, -X[c_nz]/l1, -X[c_nz]/l3 ]
    r1 = np.r_[X[c_ny]*((X[c_v1z]-X[c_vz])/l1 + (X[c_v3z]-X[c_vz])/l3) - X[c_nz]*((X[c_v1y]-X[c_vy])/l1 + (X[c_v3y]-X[c_vy])/l3)]
    print(len(data1), len(row1), len(col1))
    H1 = sparse.coo_matrix((data1,(row1,col1)), shape=(num, len(X)))

    # for r = (nz * (g1x + g3x) - nx * (g1z + g3z))
    col2 = np.r_[c_nz, c_nx, c_vx, c_v1x, c_v3x, c_vz, c_v1z, c_v3z]
    row2 = np.tile(np.arange(num),8)
    
    data2 = np.r_[(X[c_v1x]-X[c_vx])/l1 + (X[c_v3x]-X[c_vx])/l3, -(X[c_v1z]-X[c_vz])/l1 -(X[c_v3z]-X[c_vz])/l3,
             -X[c_nz]/l1 -X[c_nz]/l3, X[c_nz]/l1, X[c_nz]/l3, X[c_nx]/l1 + X[c_nx]/l3, -X[c_nx]/l1, -X[c_nx]/l3 ]
    r2 = np.r_[X[c_nz]*((X[c_v1x]-X[c_vx])/l1 + (X[c_v3x]-X[c_vx])/l3) - X[c_nx]*((X[c_v1z]-X[c_vz])/l1 + (X[c_v3z]-X[c_vz])/l3)]

    H2 = sparse.coo_matrix((data2,(row2,col2)), shape=(num, len(X)))

    # for r = (nx * (g1y + g3y) - ny * (g1x + g3x))
    col3 = np.r_[c_nx, c_ny, c_vy, c_v1y, c_v3y, c_vx, c_v1x, c_v3x]
    row3 = np.tile(np.arange(num),8)
    
    data3 = np.r_[(X[c_v1y]-X[c_vy])/l1 + (X[c_v3y]-X[c_vy])/l3, -(X[c_v1x]-X[c_vx])/l1 -(X[c_v3x]-X[c_vx])/l3,
             -X[c_nx]/l1 -X[c_nx]/l3, X[c_nx]/l1, X[c_nx]/l3, X[c_ny]/l1 + X[c_ny]/l3, -X[c_ny]/l1, -X[c_ny]/l3 ]
    r3 = np.r_[X[c_nx]*((X[c_v1y]-X[c_vy])/l1 + (X[c_v3y]-X[c_vy])/l3) - X[c_ny]*((X[c_v1x]-X[c_vx])/l1 + (X[c_v3x]-X[c_vx])/l3)]

    H3 = sparse.coo_matrix((data3,(row3,col3)), shape=(num, len(X)))

    # Combine the constraints
    H = sparse.vstack((H1,H2,H3))
    r = np.r_[r1,r2,r3]
    
    return H,r

def con_vertex_control_1st_polyline(X,X_1st_polyline,V):

    #  X_1st_polyline is the vertices of the 1st polyline first row Vx, second row Vy, third row Vz
    #  c_v_1st_polyline is the indices of the optimized vertices of the 1st polyline
    #   v_original - v_new = 0 where v_original is constant 

    #print("X first polyline ", X[np.r_[np.arange(11), np.arange(mesh.V+1,mesh.V+11), np.arange(2*mesh.V+1,2*mesh.V+11)]]) # Apply vertex control on these. 

    # Set up the indicies
    # print("X_1st_polyline shape ", X_1st_polyline.shape)
    # print("X_1st_polyline ", X_1st_polyline)
    num = int(len(X_1st_polyline))
    X_1st_polyline = X_1st_polyline.T.flatten()
    # print("X_1st_polyline shape ", X_1st_polyline.shape)
    # print("X_1st_polyline ", X_1st_polyline)

    c_vx = np.arange(num)
    c_vy = np.arange(V,V+num)
    c_vz = np.arange(2*V,2*V+num)
    # print("c_vx,y,z, ", c_vx, c_vy, c_vz)
    X_vx = X_1st_polyline[np.arange(num)]
    X_vy = X_1st_polyline[np.arange(num,2*num)]
    X_vz = X_1st_polyline[np.arange(2*num,3*num)]
    # print("x_vx shape and others ", X_vx,  X_vy, X_vz) # these are the original.  
    # print("X[c_vx,y,z], ", X[c_vx], X[c_vy], X[c_vz])
    # print("X, ", X)
    # print("X Shape: ", X.shape)


    col = np.r_[c_vx, c_vy, c_vz]
    row =  np.r_[np.arange(3*num)]
    data = np.r_[np.tile(-1,3*num)]
    f = np.r_[X_vx-X[c_vx], X_vy-X[c_vy], X_vz-X[c_vz]]
    # print("r value ", X_vx, X[c_vx])
    # print(data, row, col)
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, len(X)))
    # print("H and r, ", H, r)
    print("NORM OF f ", np.linalg.norm(f))
    r = H*X - f
    return H,r

def con_vertex_control(X, v_ids, X_copy,mesh):
    print("con_vertex_control")
    # X_orig = mesh._copy._vertices[v_ids]
    X_orig = mesh._vertices[v_ids]

    # X_orig = X_copy[v_ids]
    
    num = int(len(X_orig))
    # print("X_orig:", num)
    X_orig = X_orig.T.flatten()
    # print("X_orig:", X_orig)
    V =  mesh.V
    c_vx = v_ids
    # print("c_vx:", c_vx)
    c_vy = v_ids+V
    # print("c_vy:", c_vy)
    c_vz = v_ids+2*V
    X_vx = X_orig[np.arange(num)]
    X_vy = X_orig[np.arange(num,2*num)]
    X_vz = X_orig[np.arange(2*num,3*num)]

    col = np.r_[c_vx, c_vy, c_vz]
    # print("col:", col)
    row =  np.r_[np.arange(3*num)]
    # print("row:", row)
    data = np.r_[np.tile(-1,3*num)]
    # print("data:", data)
    f = np.r_[X_vx-X[c_vx], X_vy-X[c_vy], X_vz-X[c_vz]]
    # print("f:", f)
    # print("r value ", X_vx, X[c_vx])
    # print(data, row, col)
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, len(X)))
    # print("H and r, ", H, r)
    print("NORM OF f ", np.linalg.norm(f))
    r = H*X - f
    return H,r

def con_vertex_control2(X, v_ids, X_copy,mesh):
    print("con_vertex_control")
    X_orig = mesh._copy._vertices[v_ids]
    # X_orig = X_copy[v_ids]
    
    num = int(len(X_orig))
    print("X_orig:", num)
    X_orig = X_orig.T.flatten()
    print("X_orig:", X_orig)
    V =  mesh.V
    c_vx = v_ids
    # print("c_vx:", c_vx)
    c_vy = v_ids+V
    # print("c_vy:", c_vy)
    c_vz = v_ids+2*V
    X_vx = X_orig[np.arange(num)]
    X_vy = X_orig[np.arange(num,2*num)]
    X_vz = X_orig[np.arange(2*num,3*num)]
    # X_vy = X_orig[np.arange(V,V+num)]
    # X_vz = X_orig[np.arange(2*V,2*V+num)]

    col = np.r_[c_vx, c_vy, c_vz]
    # print("col:", col)
    row =  np.r_[np.arange(3*num)]
    # print("row:", row)
    data = np.r_[np.tile(-1,3*num)]
    # print("data:", data)
    f = np.r_[X_vx-X[c_vx], X_vy-X[c_vy], X_vz-X[c_vz]]
    # print("f:", f)
    # print("r value ", X_vx, X[c_vx])
    # print(data, row, col)
    H = sparse.coo_matrix((data,(row,col)), shape=(3*num, len(X)))
    print("NORM OF f ", np.linalg.norm(f))
    r = H*X - f
    print("H and r, ", H, r)
    return H,r


def con_isometric2(X, mesh):
    "(va-vd)^2=(va'-vd')^2"
    "(vc-vb)^2=(vc'-vb')^2"
    print("con_isometric2")
    # print("X: ", X)
    X_orig = mesh._copy._vertices
    # print("X reference: ", X_orig)

    V =  mesh.V
    # n = 20   # number of vertices for each polyline
    n = 32
    nc = int(V/n) # number of rows
    va = []
    vb = []
    vc = []
    vd = []

    ### Create the indices

    for i in range(n-1):      #  i = 0, ..., 18 
        for j in range(nc-1): #  j = 0, ..., 10    12 is number of rows 
            va.append(j*n+i)
            vc.append((j+1)*n+i)
            vd.append((j+1)*n+i+1)
            vb.append(j*n+i+1)
    # print("va, vb,..: ", nc)
    # print("va, vb,..: ", va)
    # print("va, vb,..: ", vb)
    # print("va, vb,..: ", vc)
    # print("va, vb,..: ", vd)

    c_va = column3D(va, 0, V)
    c_vb = column3D(vb, 0, V)
    c_vc = column3D(vc, 0, V)
    c_vd = column3D(vd, 0, V)
    # print("c_vs: ", c_va)
    # print("c_vs: ", c_vb)
    # print("c_vs: ", c_vc)
    # print("c_vs: ", c_vd)

    # #### The vertices
            
    
    
    X_va = X_orig[va]
    X_vb = X_orig[vb]
    X_vc = X_orig[vc]
    X_vd = X_orig[vd]

    num = int(len(X_va))


    # ####  "(va-vd)^2=(va'-vd')^2"

    col1 = np.r_[c_va, c_vd]
    row1 = np.tile(np.arange(num),6)
    data1= 2*np.r_[X[c_va]-X[c_vd],X[c_vd]-X[c_va]]
    f1 = np.linalg.norm((X[c_va]-X[c_vd]).reshape(-1,3, order='F'),axis=1)**2 - np.linalg.norm(X_va-X_vd, axis = 1)**2
    H1 = sparse.coo_matrix((data1,(row1,col1)), shape=(num,len(X)))
    # print("shapes: ", H.shape, X.shape, f.shape)
    print("NORM OF f ", np.linalg.norm(f1))
    r1 = H1*X - f1

    # ####   "(vc-vb)^2=(vc'-vb')^2"

    col2 = np.r_[c_vc, c_vb]
    row2 = np.tile(np.arange(num),6)
    data2= 2*np.r_[X[c_vc]-X[c_vb],X[c_vb]-X[c_vc]]
    f2 = np.linalg.norm((X[c_vc]-X[c_vb]).reshape(-1,3, order='F'),axis=1)**2 - np.linalg.norm(X_vc-X_vb, axis = 1)**2
    H2 = sparse.coo_matrix((data2,(row2,col2)), shape=(num,len(X)))
    # print("shapes: ", H.shape, X.shape, f.shape)
    print("NORM OF f ", np.linalg.norm(f2))
    r2 = H2*X - f2

    # ####   "(va-vd) . (vb-vc) =  (va'-vd') . (vb'-vc')"

    col3 = np.r_[c_va,c_vb, c_vc, c_vd]
    row3 = np.tile(np.arange(num),12)
    data3 = np.r_[X[c_vb]-X[c_vc],X[c_va]-X[c_vd],X[c_vd]-X[c_va],X[c_vc]-X[c_vb]]
    f3 = np.einsum('ij,ij->i',(X[c_va]-X[c_vd]).reshape(-1,3, order='F'),(X[c_vb]-X[c_vc]).reshape(-1,3, order='F')) - np.einsum('ij,ij->i',(X_va-X_vd), (X_vb-X_vc))
    H3 = sparse.coo_matrix((data3,(row3,col3)), shape=(num,len(X)))
    print("NORM OF f ", np.linalg.norm(f3))
    # print("shapes: ", H.shape, X.shape, f.shape)
    r3 = H3*X - f3

    H = sparse.vstack((H1,H2, H3))
    r = np.r_[r1,r2,r3]

    # # print("shape r: ", r.shape)
    return H,r
    # return 0 


def _con_fairness(X,c_v,c_v1,c_v3):
    """
    "g1+g3 = 0"
    """
    
    X1 = (X[c_v1]-X[c_v]).reshape(-1,3, order='F')
    X3 = (X[c_v]-X[c_v3]).reshape(-1,3, order='F')
    l1 =  np.linalg.norm(X1, axis = 1)
    l3 =  np.linalg.norm(X3, axis = 1)

    num = int(len(c_v)/3)
    c_vx = c_v[:num]
    c_vy = c_v[num:2*num]
    c_vz = c_v[2*num:]
    c_v1x = c_v1[:num]
    c_v1y = c_v1[num:2*num]
    c_v1z = c_v1[2*num:]
    c_v3x = c_v3[:num]
    c_v3y = c_v3[num:2*num]
    c_v3z = c_v3[2*num:]

    # for r = g1x-g3x 
    col1 = np.r_[c_vx, c_v1x, c_v3x]
    row1 = np.tile(np.arange(num),3)
    data1 = np.r_[(-1/l1)+(-1/l3), 1/l1,1/l3]
    r1 = np.r_[(X[c_v1x]-X[c_vx])/l1 + (X[c_v3x]-X[c_vx])/l3]
    print(len(data1), len(row1), len(col1))
    H1 = sparse.coo_matrix((data1,(row1,col1)), shape=(num, len(X)))

    # for r = g1y-g3y 
    col2 = np.r_[c_vy, c_v1y, c_v3y]
    row2 = np.tile(np.arange(num),3)
    data2 = np.r_[(-1/l1)+(-1/l3), 1/l1,1/l3]
    r2 = np.r_[(X[c_v1y]-X[c_vy])/l1 + (X[c_v3y]-X[c_vy])/l3]
    print(len(data2), len(row2), len(col2))
    H2 = sparse.coo_matrix((data2,(row2,col2)), shape=(num, len(X)))

    # for r = g1z-g3z 
    col3 = np.r_[c_vz, c_v1z, c_v3z]
    row3 = np.tile(np.arange(num),3)
    data3 = np.r_[(-1/l1)+(-1/l3), 1/l1,1/l3]
    r3 = np.r_[(X[c_v1z]-X[c_vz])/l1 + (X[c_v3z]-X[c_vz])/l3]
    print(len(data3), len(row3), len(col3))
    H3 = sparse.coo_matrix((data3,(row3,col3)), shape=(num, len(X)))

    # Combine the constraints
    H = sparse.vstack((H1,H2,H3))
    r = np.r_[r1,r2,r3]

    return H,r

#trial and error functions:
"""
def con_gnet_planarity(X, c_va, c_v, c_v1, c_v3):
    "n*(v_a - v)=0"
    # or
    "n*(v_c - v)=0"
    #  where n = (g1+g3)

    X1 =  (X[c_v1]-X[c_v]).reshape(-1,3, order='F') # reshapes to x y z x y z x y z
    Xa =  (X[c_va]-X[c_v]).reshape(-1,3, order='F')
    X3 =  (X[c_v3]-X[c_v]).reshape(-1,3, order='F')
    l1 = np.linalg.norm(X1,axis=1) 
    l3 = np.linalg.norm(X3,axis=1)
    X1av  = (-X[c_va]-X[c_v1]+2*X[c_v]).reshape(-1,3, order='F')
    X3av  = (-X[c_va]-X[c_v3]+2*X[c_v]).reshape(-1,3, order='F')
    num = int(len(c_v)/3)
    col = np.r_[c_v, c_va, c_v1,c_v3]
    row = np.tile(np.arange(num), 12)
    data = []
    r = []
    
    for i in range(num):
        data = np.r_[data, (X1av[i]/l1[i]+X3av[i]/l3[i]),
                           (X1[i]/l1[i]+X3[i]/l3[i]), Xa[i]/(l1[i]), Xa[i]/(l3[i]) ]
        #r = np.r_[r, (np.dot(X1[i],Xa[i])/l1[i] +  np.dot(X3[i], Xa[i])/l3[i])]
        r  = np.r_[r, np.dot(X1[i]/l1[i]+X3[i]/l3[i], Xa[i])]
    data = data.reshape(num, 12).T.flatten()
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r

def con_parallel(X, c_v, c_v1, c_v3, c_n):
    "nX(g1+g3)=0 or ( n.(g1+g3)-|n||g1+g3|=0 if they are parallal & same direction or n.(g1+g3)+|n||g1+g3|=0 if parallel and opposite direction)"
    X1 = (X[c_v1]-X[c_v]).reshape(-1,3, order='F')
    X3 = (X[c_v3]-X[c_v]).reshape(-1,3, order='F')
    Xn = (X[c_n]).reshape(-1,3, order='F')
    l1 =  np.linalg.norm(X1, axis = 1)
    l3 =  np.linalg.norm(X3, axis = 1)
    ln =  np.linalg.norm(Xn, axis = 1)

    num = int(len(c_v)/3)
    col = np.r_[c_n, c_v, c_v1, c_v3]
    row = np.tile(np.arange(num), 12)
    data = []
    r = []
    l13 = []
    for i in range(num):
        l13 = np.r_[l13,np.linalg.norm(X1[i]/l1[i]+X3[i]/l3[i])]
    for i in range(num):
        data = np.r_[data, (X1[i]/l1[i])+(X3[i]/l3[i]), (-Xn[i]/l1[i])+(-Xn[i]/l3[i]), Xn[i]/l1[i], Xn[i]/l3[i]]
        dot_product = np.dot(Xn[i], (X1[i]/l1[i])+(X3[i]/l3[i]))
        magnitude_product = (ln[i]*l13[i])
        r = np.r_[r, dot_product + magnitude_product]
    #r = np.einsum('ij,ij->i',X[c_n].reshape(-1,3, order='F'),X[c_m].reshape(-1,3, order='F'))
    data = data.reshape(num, 12).T.flatten() 
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r
def con_parallel2(X,c_n, c_m):
    "nXm = (0,0,0) => (nXm).(nXm) = 0"
    num = int(len(c_n)/3)
    c_nx = c_n[:num]
    c_ny = c_n[num:2*num]
    c_nz = c_n[2*num:]
    c_mx = c_m[:num]
    c_my = c_m[num:2*num]
    c_mz = c_m[2*num:]

    col = np.r_[c_nx, c_ny, c_nz, c_mx, c_my, c_mz]
    row = np.tile(np.arange(num), 6)
                      # I want to multiply element wise
    #data = np.r_[ -2*(X[c_nz]*X[c_mx]-X[c_nx]X[c_mz])*X[c_mz] + 2*(X[c_nx]*X[c_my]-X[c_ny]X[c_mx])*X[c_my], ]
    
    data = []
    r = []
    for i in range(num):
        nx= X[c_nx[i]]
        ny= X[c_ny[i]]
        nz= X[c_nz[i]]
        mx= X[c_mx[i]]
        my= X[c_my[i]]
        mz= X[c_mz[i]]
        data = np.r_[data, -2*(nz*mx-nx*mz)*mz + 2*(nx*my-ny*mx)*my, 2*(ny*mz-nz*my)*mz - 2*(nx*my-ny*mx)*mx, -2*(ny*mz-nz*my)*my + 2*(nz*mx-nx*mz)*mx,
                        2*(nz*mx-nx*mz)*nz - 2*(nx*my-ny*mx)*ny, -2*(ny*mz-nz*my)*nz + 2*(nx*my-ny*mx)*nx, 2*(ny*mz-nz*my)*ny - 2*(nz*mx-nx*mz)*nx]
        r = np.r_[r, (ny*mz-nz*my)**2 + (nz*mx-nx*mz)**2 + (nx*my-ny*mx)**2]
    data = data.reshape(num, 6).T.flatten()
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    
    return H,r

def g1g3(X, c_v, c_v1, c_v3, c_m):
    "m-(g1+g3) = (0,0,0) => (m-(g1+g3)).(m-(g1+g3)) = 0"

    num = int(len(c_v)/3)
    col = np.r_[c_m, c_v, c_v1, c_v3]
    row = np.tile(np.arange(num), 12)

    X1 = (X[c_v1]-X[c_v]).reshape(-1,3, order='F')
    X3 = (X[c_v3]-X[c_v]).reshape(-1,3, order='F')
    Xm = (X[c_m]).reshape(-1,3, order='F')
    l1 =  np.linalg.norm(X1, axis = 1)
    l3 =  np.linalg.norm(X3, axis = 1)

    data = []
    r =  []
    
    for i in range(num):
        E = (Xm[i] - ((X1[i]/l1[i])+(X3[i]/l3[i])))
        data = np.r_[data, 2*E, 2*E*(1/l1[i]+1/l3[i]), 2*(-1/l1[i])*E,2*(-1/l3[i])*E]
        r = np.r_[r, np.dot(E, E)]

    data = data.reshape(num,12).T.flatten()
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r 

def new(X, c_v, c_v1, c_v3, c_n):

    X1 = (X[c_v1]-X[c_v]).reshape(-1,3, order='F')
    X3 = (X[c_v3]-X[c_v]).reshape(-1,3, order='F')
    l1 =  np.linalg.norm(X1, axis = 1)
    l3 =  np.linalg.norm(X3, axis = 1)

    num = int(len(c_v)/3)
    col = np.r_[c_n, c_v, c_v1, c_v3]
    row = np.tile(np.arange(num), 12)
    data = []
    r = []

    for i in range(num):
        data = np.r_[data, np.ones(3), np.ones(3)/l1[i] + np.ones(3)/l3[i], -np.ones(3)/l1[i], -np.ones(3)/l3[i] ]
        r  = np.r_[r, X[c_v]]
    data = data.reshape(num, 12).T.flatten()
    H = sparse.coo_matrix((data,(row,col)), shape=(num, len(X)))
    return H,r
"""