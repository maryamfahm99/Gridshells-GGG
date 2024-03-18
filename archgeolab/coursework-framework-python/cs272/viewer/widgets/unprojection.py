import imgui
import numpy as np
from viewer.ViewerWidget import ViewerWidget, AnchorType
from viewer.opengl.MeshGL import DirtyFlags
from geometry.Mesh import Mesh, NoiseDirection
#### Maryam
# import igl
from utils.camera import project, unproject
from viewer.opengl.MeshGL import DirtyFlags, MeshGL
#### 

def unproject_onto_mesh(pos,model,proj,viewport,V,F):
    s, dir = unproject_ray(pos,model,proj,viewport)
    bc, fid, hit = shoot_ray(s,dir, V, F)
    # print("shooting direction",dir / np.linalg.norm(dir))
    return bc, fid, hit

def unproject_ray(pos,model,proj,viewport):
    win_s = np.array([pos[0], pos[1], 0])
    win_d = np.array([pos[0], pos[1], 1])

    # Source and direction in world coordinates
    s = _unproject(win_s, model, proj, viewport)
    d = _unproject(win_d, model, proj, viewport)
    dir = d - s
    print("ray direction,",  dir/np.linalg.norm(dir))
    return s, dir
    
def shoot_ray(s, dir, V, F):
    # Loop over all triangles
    min_t = (np.inf)
    id = -1
    for f in range(F.shape[0]):
        v0 = V[F[f,0]]
        v1 = V[F[f,1]]
        v2 = V[F[f,2]]
        hit,parallel, t, u, v= ray_triangle_intersect(s, dir, v0,v1,v2)
        w = 1 - u - v
        if hit:
            # point =  s + dir*t
            # hits.append(point)
            if min_t > t:
                min_t = t
                id = f
    
    # Sort hits based on distance     ///  need to  work on it
    # hits.sort(key=lambda x: x.t)
    if id == -1:
        return [], id, False
    v0 = V[F[id,0]]
    v1 = V[F[id,1]]
    v2 = V[F[id,2]]
    hit,parallel, t, u, v= ray_triangle_intersect(s, dir, v0,v1,v2)
    w = 1.0 - u - v
    bc  = (np.array([w,u,v])).astype(float)

    return bc, id, True

def _unproject(win, model, proj, viewport):
    scene = (np.zeros((3,1))).astype(float)
    obj = (np.zeros((4,1))).astype(float)
    Inverse = np.linalg.inv(np.dot(proj.astype(float), model.astype(float)))
    tmp = (np.hstack((win[0],win[1],win[2], 1))).astype(float)
    # print("temp0: ", tmp)
    tmp[0] = (tmp[0] - viewport[0]) / float(viewport[2])
    tmp[1] = (tmp[1] - viewport[1]) / float(viewport[3])

    # print("temp1: ", tmp)
    # print("vp: ", viewport[0],viewport[1])
    tmp = tmp * 2.0 - np.array([1,1,1,1])
    # print("temp2: ", tmp)

    obj = (np.dot(Inverse, tmp)).astype(float)
    obj /= obj[3]
    scene = (obj[:3].T)
    return scene

def ray_triangle_intersect(O, D, V0, V1, V2, epsilon=1e-6):
    # Calculate edge vectors
    edge1 = V1 - V0
    edge2 = V2 - V0
    # print("edge 1 2", edge1, edge2)
    # Compute determinant and check for parallelism
    pvec = np.cross(D, edge2)
    det = np.dot(edge1, pvec)
    if det > -epsilon and det < epsilon:
        parallel = True
        return False, parallel, 0, 0, 0
    parallel = False

    inv_det = 1.0 / det
    tvec =  O - V0
    # print("tvec", tvec)
    u = np.dot(tvec, pvec) * inv_det
    if u < 0.0 - epsilon or u > 1.0 + epsilon:
        return False, parallel, 0, 0, 0

    qvec = np.cross(tvec, edge1)
    v = np.dot(D, qvec) * inv_det
    if v < 0.0 - epsilon or u + v > 1.0 + epsilon:
        return False, parallel, 0, 0, 0

    t = np.dot(edge2, qvec) * inv_det
    return True, parallel, t, u, v

def proj(obj,model,proj,viewport):
    tmp = np.zeros(4).astype(float)
    # print("tmp1, ", tmp)

    tmp = np.array([obj[0], obj[1], obj[2], 1.0], dtype=float)
    # print("tmp2, ", tmp)
    tmp = model@ tmp
    # print("tmp3, ", tmp)
    tmp = proj@ tmp
    # print("tmp4, ", tmp)
    tmp /= tmp[3]
    # print("tmp5, ", tmp)
    tmp = tmp *  0.5 + np.array([0.5,0.5,0.5,0.5])
    # print("tmp6, ", tmp)
    tmp[0] = tmp[0] * viewport[2] + viewport[0]
    tmp[1] = tmp[1] * viewport[3] + viewport[1]
    # print("tmp7, ", tmp)
    # print("tmp8, ", tmp[:3])
    return tmp[:3]
