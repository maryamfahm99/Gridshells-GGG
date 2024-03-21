import copy
import os
import math
import numpy
import openmesh
from enum import IntEnum
from utils.colormap import ColorMapType
from utils.maths import gaussian, random_vector
from geometry.Geometry import Geometry
import numpy as np

#######################  Maryam import
import sys
# Add the parent directory of geometry-lab-main to the Python path
# path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path = '/Users/memo2/Desktop/2022-2023/Summer2023/WebImplementation/geometry-lab-main'
sys.path.append(path)
# print("first path: ", path)
# # Now you can import from the geometrylab.geometry module
from geometrylab.geometry.meshpy import Mesh as quad_mesh
from archgeolab.archgeometry.quadrings import MMesh
from archgeolab.archgeometry.gridshell import Gridshell
# from meshpy import mesh as quad_mesh
# tran = quad_mesh()
from geometrylab.geometry import Polyline
from archgeolab.proj_orthonet.guidedprojection_orthonet import GP_OrthoNet
# from archgeolab.proj_orthonet.guidedprojection_orthonet import GP_OrthoNet
######################


class NoiseDirection(IntEnum):
    NORMAL = 0
    RANDOM = 1


class Mesh(Geometry):

    def __init__(self):
        super().__init__()
        self.render_edges = True
        self.render_flat_faces = False
        self.edge_color = numpy.array([0.0, 0.0, 0.0, 1.0])
        self.colormap = ColorMapType.NUM_COLOR_MAP_TYPES
        self.mesh = None
        self.mesh_original = None
        ###### Maryam
        self.optimizer = GP_OrthoNet() # Maryam
        self.quad_mesh = None          # Maryam
        self.meshFairness = 0.01
        self.boundFairness = 0.01
        self.GGG_weight = 0.01
        self.vertexControl = 1.0
        self.iso_weight = 1.0
        self.aprox_weight = 0.0  ### APROX
        GGG = False
        aprox = False ### APROX
        p1 = None
        p2= None
        # self.v_ids = list(range(20))
        # self.v_ids = list(range(20))
        self.v_ids = list(range(20))
        self.numVerPoly = 20
        self.weights = {
        'GGG' : 0, # Maryam
        'iso' : 0,
        'aprox': 0,  ### APROX
        'vertex_control': None
        }
        self.optimizer.add_weights(self.weights)
        #####

    def load(self, filename):
        print("load in Mesh.py")
        # Load a mesh
        try:
            self.mesh = openmesh.read_polymesh(
                filename, binary=False, msb=False, lsb=False, swap=False,
                vertex_normal=False, vertex_color=False, vertex_tex_coord=False, halfedge_tex_coord=False,
                edge_color=False, face_normal=False, face_color=False, face_texture_index=False,
                color_alpha=False, color_float=False)
        except RuntimeError as error:
            print("Error:", error)
            return False
        if self.mesh is None:
            print("Error: Error loading mesh from file ", filename)
            return False
        self.mesh.request_vertex_texcoords2D()

        # Mesh name
        self.name = os.path.splitext(os.path.basename(filename))[0]
        print("self.name: ", self.name)

        # We need normals
        self.mesh.request_face_normals()
        self.mesh.request_vertex_normals()
        self.mesh.update_normals()

        # Save original mesh
        self.mesh_original = copy.deepcopy(self.mesh)

        ############# Maryam

        # Load the file as a quad mesh using the old framework such that we can optimize later through it
        self.quad_mesh = Gridshell()
        self.quad_mesh.read_obj_file(filename)
        self.optimizer.mesh = self.quad_mesh 
        # MMesh.make_mesh()

        self.optimizer.mesh._copy = self.optimizer.mesh.copy_mesh()   
        
        # print("self.quad_mesh._faces: ", self.quad_mesh._faces)
        print("how many: ", (self.quad_mesh._faces).shape)
        #############

        # Success!
        return True

    def save(self, filename):
        # Save a mesh
        openmesh.write_mesh(self.mesh, filename)
        # if False:
        #     print("Error: Error saving mesh to file ", filename)
        #     return False
        # Success!
        return True

    def update_viewer_data(self, data):
        print("update_viewer_data, mesh")
        # Clear viewer
        data.clear()
        data.clear_edges()

        # Convert mesh to viewer format
        tmp_v, tmp_f, tmp_f_to_f, tmp_n, tmp_uv, p1, p2 = Mesh.to_render_data(self.mesh)
        # qd_mesh = Mesh.optimize(self.mesh) ## Maryam

        # Plot the mesh
        data.set_mesh(tmp_v, tmp_f)
        data.FtoF = tmp_f_to_f
        data.set_normals(tmp_n)
        data.set_uv(tmp_uv)
        if self.render_flat_faces:
            data.compute_normals()
        else:
            data.face_based = False
        if self.render_edges:
            data.add_edges(p1, p2, numpy.array([self.edge_color]))
        data.line_width = 1.0
        data.show_lines = False
        # show_texture = True
        data.show_texture = False

        # Colors
        data.uniform_colors([51.0 / 255.0, 43.0 / 255.0, 33.3 / 255.0, 255.0],
                            [255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0, 255.0],
                            [255.0 / 255.0, 235.0 / 255.0, 80.0 / 255.0, 255.0])

    @staticmethod
    def to_render_data(mesh: openmesh.PolyMesh):
        print("to_render_data")
        # Triangulate
        face_map = dict()
        normals_orig = dict()
        tri_mesh = copy.deepcopy(mesh)
        
        for fh in tri_mesh.faces():
            try:
                face_map[fh.idx()]
                
            except KeyError:
                face_map[fh.idx()] = fh.idx()
                # print("tri_mesh.faces(): ",face_map[fh.idx()])

            n = tri_mesh.normal(fh)
            try:
                normals_orig[fh.idx()]
            except KeyError:
                normals_orig[fh.idx()] = n

            base_heh = tri_mesh.halfedge_handle(fh)
            start_vh = tri_mesh.from_vertex_handle(base_heh)
            prev_heh = tri_mesh.prev_halfedge_handle(base_heh)
            next_heh = tri_mesh.next_halfedge_handle(base_heh)

            while tri_mesh.to_vertex_handle(tri_mesh.next_halfedge_handle(next_heh)) != start_vh:

                next_next_heh = tri_mesh.next_halfedge_handle(next_heh)

                new_fh = tri_mesh.new_face()
                tri_mesh.set_halfedge_handle(new_fh, base_heh)

                face_map[new_fh.idx()] = fh.idx()

                normals_orig[new_fh.idx()] = n

                new_heh = tri_mesh.new_edge(tri_mesh.to_vertex_handle(next_heh), start_vh)

                tri_mesh.set_next_halfedge_handle(base_heh, next_heh)
                tri_mesh.set_next_halfedge_handle(next_heh, new_heh)
                tri_mesh.set_next_halfedge_handle(new_heh, base_heh)

                tri_mesh.set_face_handle(base_heh, new_fh)
                tri_mesh.set_face_handle(next_heh, new_fh)
                tri_mesh.set_face_handle(new_heh, new_fh)

                tri_mesh.copy_all_properties(prev_heh, new_heh, True)
                tri_mesh.copy_all_properties(prev_heh, tri_mesh.opposite_halfedge_handle(new_heh), True)
                tri_mesh.copy_all_properties(fh, new_fh, True)

                base_heh = tri_mesh.opposite_halfedge_handle(new_heh)
                next_heh = next_next_heh

            tri_mesh.set_halfedge_handle(fh, base_heh)  # the last face takes the handle _fh
            tri_mesh.set_next_halfedge_handle(base_heh, next_heh)
            tri_mesh.set_next_halfedge_handle(tri_mesh.next_halfedge_handle(next_heh), base_heh)
            tri_mesh.set_face_handle(base_heh, fh)

        # Resize arrays
        verts = numpy.empty((tri_mesh.n_vertices(), 3))
        faces = numpy.empty((tri_mesh.n_faces(), 3), dtype=numpy.uint32)
        f_to_f = numpy.empty((tri_mesh.n_faces(), 1), dtype=numpy.uint32)
        norms = numpy.empty((tri_mesh.n_faces(), 3))
        if mesh.has_vertex_texcoords2D():
            texs = numpy.empty((tri_mesh.n_vertices(), 2))
        else:
            texs = None

        # Vertices
        for vh in tri_mesh.vertices():
            # print("vh is  what: ", vh.idx)
            p = tri_mesh.point(vh)
            verts[vh.idx(), 0] = p[0]
            verts[vh.idx(), 1] = p[1]
            verts[vh.idx(), 2] = p[2]

        # Faces
        for fh in tri_mesh.faces():
            vi = 0
            for fvi in tri_mesh.fv(fh):
                faces[fh.idx(), vi] = fvi.idx()
                vi += 1
        print("faces: ", faces.shape)
        # print("faces: ", faces)
        # Face map
        for key, value in face_map.items():
            f_to_f[key, 0] = value
            # print("f_to_f[key, 0]: ",  key, value)
        print("f_to_f[key, 0]: ",  f_to_f.shape)

        # Normals
        for key, value in normals_orig.items():
            n = value
            norms[key, 0] = n[0]
            norms[key, 1] = n[1]
            norms[key, 2] = n[2]

        # TexCoords
        if mesh.has_vertex_texcoords2D():
            for vh in tri_mesh.vertices():
                tex = tri_mesh.texcoord2D(vh)
                texs[vh.idx(), 0] = tex[0]
                texs[vh.idx(), 1] = tex[1]

        # Edges
        edges1 = numpy.empty((mesh.n_edges(), 3))
        edges2 = numpy.empty((mesh.n_edges(), 3))
        for eh in mesh.edges():
            vh1 = mesh.to_vertex_handle(mesh.halfedge_handle(eh, 0))
            vh2 = mesh.from_vertex_handle(mesh.halfedge_handle(eh, 0))
            v1 = mesh.point(mesh.vertex_handle(vh1.idx()))
            v2 = mesh.point(mesh.vertex_handle(vh2.idx()))

            edges1[eh.idx(), 0] = v1[0]
            edges1[eh.idx(), 1] = v1[1]
            edges1[eh.idx(), 2] = v1[2]
            edges2[eh.idx(), 0] = v2[0]
            edges2[eh.idx(), 1] = v2[1]
            edges2[eh.idx(), 2] = v2[2]
        # print("tri mesh vertices: ", verts)
        return verts, faces, f_to_f, norms, texs, edges1, edges2
    ############### Maryam

    def quad_to_tri(self):  # Something wrong with the faces
        # This takes  a mesh object from geometry-lab-main and returns a Mesh object

        vertices = self.optimizer.mesh._vertices
        faces = self.optimizer.mesh._faces

        # Create a new PolyMesh
        mesh = openmesh.PolyMesh()

        # Add vertices to the mesh
        for vertex in vertices:
            mesh.add_vertex(vertex)

        # Add faces to the mesh
        for face in faces:
           
            # Create a new face with the vertex indices
            face_handles = [mesh.vertex_handle(vertex_index) for vertex_index in face]
            # print("face handles: ", face_handles)
            mesh.add_face(face_handles)

        # Assign the created tri mesh to self.mesh
        self.mesh = mesh
        self.mesh.request_vertex_texcoords2D()

        # We need normals
        self.mesh.request_face_normals()
        self.mesh.request_vertex_normals()
        self.mesh.update_normals()

        # Save original mesh
        # self.mesh_original = copy.deepcopy(self.mesh)

        return True
    
    def optimize(self, GGG, ISO):
        print("optimize")
        ### use optimize from ortho
        
        self.set_settings(GGG,ISO)
        self.optimizer.optimize()
        self.quad_to_tri()
        return 0 
    
    def set_settings(self, GGG, ISO):
        print("set_setting in Mesh.py")
        self.optimizer.threshold = 1e-20
    
        self.optimizer.add_weight('mesh_fairness', self.meshFairness)
        self.optimizer.add_weight('boundary_fairness', self.boundFairness)
        self.optimizer.add_weight('GGG_weight', self.GGG_weight) # Maryam 
        # print("set_setting, G weight: ", self.GGG_weight)
        self.optimizer.add_weight('vertex_control_weight', self.vertexControl) # Maryam 
        self.optimizer.add_weight('iso_weight', self.iso_weight )
        self.optimizer.set_weight('GGG' , GGG)
        self.optimizer.set_weight('iso' , ISO)

        ############ APROX
        self.optimizer.add_weight('aprox', self.aprox_weight)
        print("self.v_ids in Mesh Class: ",self.v_ids)
        self.optimizer.set_weight('vertex_control', self.v_ids)

        # self.optimizer.add_weight('self_closeness', self.self_closeness)

        # ---------------------------------------------------------------------

        # self.optimizer.set_weight('AAG',  self.AAG) # Maryam
        # self.optimizer.set_weight('AGG',  self.AGG) # Maryam
        # self.optimizer.set_weight('GGG',  self.GGG) # Maryam
        # self.optimizer.set_weight('planar_ply1', self.opt_planar_polyline1)
        # self.optimizer.set_weight('planar_ply2', self.opt_planar_polyline2)
    
    def diagonal(self):
        self.optimizer.mesh._rrv4f4 = None
        self.optimizer.mesh._rr_star = None
        self.optimizer.mesh._ind_rr_star_v4f4 = None
        self.optimizer.mesh._num_rrv4f4 = 0
        self.optimizer.mesh._ver_rrv4f4  = None
        self.optimizer.mesh._rr_star_corner = None

        V = self.optimizer.mesh.vertices
        # print('V shape',V.shape)
        v,va,vb,vc,vd = self.optimizer.mesh.rr_star_corner

        return va, v, vc
        # pl1, pl2, pl3, pl4 = self.quad_mesh.get_quad_diagonals()
        # print("pl1: ", pl1)
    
    def propogate(self):
        # print("vertices: ", self.mesh._vertices)
        # print("faces: ", self.mesh._faces)
        # print("type of  mesh: ", type(self.mesh))
        # print(type(self.mesh._vertices))
        print("propogate")
        self.optimizer.mesh._mesh_propogate = True
        # Store the initial strip.
        # if (self.mesh._copy is None):
        #     self.mesh._copy = self.mesh.copy_mesh()

        # Number of vertices belonging to the first polyline   
        numVperPoly  = self.numVerPoly
        # numVperPoly = 32
        print(numVperPoly)
        # numVperPoly = 20

        # Number of all the vertices
        V = self.optimizer.mesh._V # last index = V - 1 here V = 23
        vertices = self.optimizer.mesh._vertices

        # Define a list of indices for v*, v1, v2, v3, v4, va, vb, vc, vd

        #  va  v1  vd  #
        #  v2  v*  v4  #
        #  vb  v3  vc  #

        v_star = np.arange(V-numVperPoly+1,V-1)
        v1 = v_star - 1
        v2 = v_star - numVperPoly
        v3 = v_star + 1
        v4 = v_star + numVperPoly  # This one is not defined yet
        va = v2 - 1
        vb = v2 + 1
        vc = v4 + 1                # This one is not defined yet
        vd = v4 - 1                # This one is not defined yet

        # initiate bisectors for each v_star and call them normals // check if vectors same dire, norm of cross prod is small
        norm1 = np.linalg.norm((vertices[v1]-vertices[v_star]), axis = 1)[:, np.newaxis]
        norm2 = np.linalg.norm((vertices[v3]-vertices[v_star]), axis = 1)[:, np.newaxis]
        delta = (vertices[v1]-vertices[v_star])/norm1+(vertices[v3]-vertices[v_star])/norm2 # (a-b)
        norm3 = np.linalg.norm((vertices[v2]-vertices[v_star]), axis = 1)[:, np.newaxis]
        delta = np.cross(((vertices[v1]-vertices[v_star])/norm1), ((vertices[v2]-vertices[v_star])/norm3))
        norms = np.linalg.norm(delta, axis = 1) 
        # print(norms)
        normals = delta / norms[:, np.newaxis]
        # print(normals)

        # # find the lengths of new sides of the inner triangles
            
        # 1) define the angles A, B, C
        A = []
        B = []
        C = []
        c = []
        for i in range(len(v_star)):
            # print("i: ", i)
            # A = arccos( (v1-v_star)*(va-v_star) / l1 * la)
            a_vec = (vertices[v1[i]]-vertices[v_star[i]])
            b_vec = (vertices[va[i]]-vertices[v_star[i]])
            la = np.linalg.norm(a_vec)
            lb = np.linalg.norm(b_vec)
            A_temp = np.arccos(np.dot(a_vec,b_vec)/(la*lb))
            A = np.r_[A,A_temp]  
            
            if(i != len(v_star) - 1):  # makes sure we are still in the inner triangles
                # B = arccos( (v_star-v3)*(v_star-v2))
                a_vec = (vertices[v3[i+1]]-vertices[v_star[i+1]])
                # print("v3 should be 7: ", v3[i+1], v_star[i+1])
                b_vec = (vertices[v2[i+1]]-vertices[v_star[i+1]])
                la = np.linalg.norm(a_vec)
                lb = np.linalg.norm(b_vec)
                B_temp = np.arccos(np.dot(a_vec,b_vec)/(la*lb))
                B = np.r_[B,B_temp] 

                # C = 180 - A - B
                C = np.r_[C,np.pi - A_temp - B_temp] 

                # c = || v_star[i]-v_star[i+1] ||
                c = np.r_[c,np.linalg.norm(vertices[v_star[i]] - vertices[v_star[i+1]])]

        # print("A: ", np.degrees(A))
        # print("B: ", np.degrees(B))
        # print("C: ", np.degrees(C))
            
        # print(A)
        a = np.abs(c*np.sin(A[1:])/np.sin(C))
        b = np.abs(c*np.sin(B)/np.sin(C))
        # print("a: ", a)
        # print("b: ", b)
        # print("c: ", c)

        # find the direction of the new sides of the inner triangles
        va_vec = (vertices[va] - vertices[v_star])
        va_norms = np.linalg.norm(va_vec, axis = 1)  
        va_normalized = va_vec / va_norms[:, np.newaxis]
        e1_dir =  2 * np.einsum('ij,ij->i', normals, va_normalized)[:, np.newaxis]* normals - va_normalized
        # print("calculate manually, ", normals, va_normalized, e1_dir)
        # print(e1_dir)
        # print(va_normalized)
        # print(normals)

        # print("va dot n: ", np.einsum('ij,ij->i', normals, va_normalized))
        # print("n dot vc: ", np.einsum('ij,ij->i', normals, e1_dir))

        v2_vec = (vertices[v2] - vertices[v_star])
        v2_norms = np.linalg.norm(v2_vec, axis = 1) 
        v2_normalized = v2_vec / v2_norms[:, np.newaxis]
        e2_dir =  2 * np.einsum('ij,ij->i', normals, v2_normalized)[:, np.newaxis]* normals - v2_normalized
        # print(e2_dir)

        # print("v2 dot n: ", np.einsum('ij,ij->i', normals, v2_normalized))
        # print("n dot v4: ", np.einsum('ij,ij->i', normals, e2_dir))

        # print("brod: ",  e1_dir[:len(v_star)-1], b[:, np.newaxis])
        mid_points1  = vertices[v_star][:len(v_star)-1] + e1_dir[:len(v_star)-1]*b[:, np.newaxis]
        # print("mid points: ", mid_points1)

        mid_points2  = vertices[v_star][1:] + e2_dir[1:]*a[:, np.newaxis]
        # print("mid points: ", mid_points2)

        mid_points = (mid_points1+ mid_points2)/2
        # print(mid_points)


        # find the outer three vertices.    

        # case 1
        c = np.r_[c, np.abs(np.linalg.norm(vertices[V-1]-vertices[V-1-numVperPoly]))]
        vec = (vertices[vb[-1]] - vertices[v3[-1]])
        vec_normalized = vec / np.linalg.norm(vec)
        dir1 = 2 * np.dot(normals[-1],vec_normalized)* normals[-1] - vec_normalized
        last_point = vertices[V-1] + dir1 * c[-1]


        # case 2
        length = np.abs(np.linalg.norm(vertices[v_star[0]]-vertices[v2[0]]))
        vec2 = (-vertices[v_star[0]] + vertices[v2[0]])

        vec2_normalized = vec2 / np.linalg.norm(vec2)
        dir2 = 2 * np.dot(normals[0],vec2_normalized)* normals[0] - vec2_normalized
        second_point = vertices[V-numVperPoly+1] + dir2 * length
        # print("case2: v2 * normal vs e* normal", np.dot(normals[0], vec2_normalized), np.dot(normals[0], dir2))

        # case 3
        first_point = vertices[V-numVperPoly] + (vertices[V-numVperPoly]-vertices[V-2*numVperPoly])

        new_v = np.vstack((first_point, second_point, mid_points, last_point))
        # new_points = np.vstack((first_point, second_point, mid_points, last_point))

        # print(new_v)

        # Add the new faces
        new_f = np.zeros((numVperPoly-1,4))
        last_poly = np.arange(V-numVperPoly,V)
        new_poly = np.arange(V,V+numVperPoly)
        for i in range(numVperPoly-1):
            new_f[i] =  np.array([last_poly[i+1],new_poly[i+1],new_poly[i],last_poly[i]])

        #print(self.mesh._vertices)
        #print(self.mesh._faces)
        #print(self.mesh._V,self.mesh._F,self.mesh._E)

        # Update the new mesh info
        self.optimizer.mesh._V += len(new_v)
        self.optimizer.mesh._F += len(new_f)
        self.optimizer.mesh._E += 2*len(new_v)-1
        # print(self.mesh._V,self.mesh._F,self.mesh._E)
        self.optimizer.mesh._vertices = np.vstack((self.optimizer.mesh._vertices, new_v))
        self.optimizer.mesh._faces = np.vstack((self.optimizer.mesh._faces, new_f))
        self.optimizer.mesh.make_mesh(self.optimizer.mesh._vertices, self.optimizer.mesh._faces)

      
        ############################################### update the variables:
        ########### X = [Vx, Vy, Vz, 
                        #  Nx, Ny, Nz,
                        #  B1x, B1y, B1z, 
                        #  B2x, B2y, B2z,
                        #  B3x, B3y, B3z]
        ########### for each new added v_star (numVperPoly - 2), we add 4 vectors to the auxiliary variables
        print("done  with mesh propogation")
        self.optimizer.mesh._copy = self.optimizer.mesh.copy_mesh()  # not tested
        self.quad_to_tri()
          
    ############## Maryam
    def save_obj(self):
        vertices = self.optimizer.mesh._vertices
        faces = self.optimizer.mesh._faces

        # Create and write to OBJ file
        with open('stripp.obj', 'w') as obj_file:
            for v in vertices:
                obj_file.write(f'v {v[0]} {v[1]} {v[2]}\n')

            for f in faces:
                obj_file.write(f'f {" ".join(map(str, f))}\n')
        with open('file.obj', 'w') as f:
            for vertex in vertices:
                f.write("v " + " ".join([str(coord) for coord in vertex]) + "\n")
            for face in faces:
                f.write("f " + " ".join([str(int(idx)+1) for idx in face]) + "\n")

    def mesh(self):
        return self.mesh

    def clean(self):
        pass

    def reset_mesh(self):
        self.mesh = copy.deepcopy(self.mesh_original)
        self.clean()

    def set_mesh(self, mesh):
        self.mesh = copy.deepcopy(mesh)
        self.clean()

    def render_flat_faces(self):
        return self.render_flat_faces

    def render_edges(self):
        return self.render_edges

    def edge_color(self):
        return self.edge_color

    def colormap(self):
        return self.colormap

    def num_vertices(self):
        return self.mesh.n_vertices()

    def vertex(self, index):
        p = self.mesh.point(self.mesh.vertex_handle(index))
        return p

    def set_vertex(self, index, p):
        self.mesh.set_point(self.mesh.vertex_handle(index), p)

    def face_center(self, index):
        fh = self.mesh.face_handle(index)
        p = numpy.array([0.0, 0.0, 0.0])
        for vh in self.mesh.fv(fh):
            p += self.mesh.point(vh)
        p /= self.mesh.valence(fh)
        return p

    def set_face_center(self, index, v):
        t = v - self.face_center(index)
        fh = self.mesh.face_handle(index)
        for vh in self.mesh.fv(fh):
            self.mesh.set_point(vh, self.mesh.point(vh) + t)

    def normalize(self):
        total_area = 0.0
        barycenter = [numpy.array([0.0, 0.0, 0.0])] * self.mesh.n_faces()
        area = [0.0] * self.mesh.n_faces()

        # loop over faces
        for fh in self.mesh.faces():

            # compute barycenter of face
            center = numpy.array([0.0, 0.0, 0.0])
            valence = 0
            vertices = []
            for vh in self.mesh.fv(fh):
                center += self.mesh.point(vh)
                valence += 1
                vertices.append(vh)
            barycenter[fh.idx()] = center / valence

            # compute area of face
            if valence == 3:
                v0 = self.mesh.point(vertices[0])
                v1 = self.mesh.point(vertices[1])
                v2 = self.mesh.point(vertices[2])

                # A = 0.5 * || (v0 - v1) x (v2 - v1) ||
                a = 0.5 * numpy.linalg.norm(numpy.cross((v0 - v1), (v2 - v1)))
                area[fh.idx()] = a
                total_area += area[fh.idx()]

            elif valence == 4:
                v0 = self.mesh.point(vertices[0])
                v1 = self.mesh.point(vertices[1])
                v2 = self.mesh.point(vertices[2])
                v3 = self.mesh.point(vertices[3])

                # A = 0.5 * || (v0 - v1) x (v2 - v1) ||
                a012 = numpy.linalg.norm(numpy.cross((v0 - v1), (v2 - v1)))
                a023 = numpy.linalg.norm(numpy.cross((v0 - v2), (v3 - v2)))
                a013 = numpy.linalg.norm(numpy.cross((v0 - v1), (v3 - v1)))
                a123 = numpy.linalg.norm(numpy.cross((v1 - v2), (v3 - v2)))
                area[fh.idx()] = (a012 + a023 + a013 + a123) * 0.25
                total_area += area[fh.idx()]

            else:
                print("Error: Arbitrary polygonal faces not supported")
                return

        # compute mesh centroid
        centroid = numpy.array([0.0, 0.0, 0.0])
        for i in range(self.mesh.n_faces()):
            centroid += area[i] / total_area * barycenter[i]

        # normalize mesh
        for vh in self.mesh.vertices():
            p = self.mesh.point(vh)
            p -= centroid  # subtract centroid (important for numerics)
            p /= math.sqrt(total_area)  # normalize to unit surface area (important for numerics)
            self.mesh.set_point(vh, p)

    def average_edge_length(self):
        sum_edge_length = 0.0
        for eh in self.mesh.edges():
            sum_edge_length += self.mesh.calc_edge_length(eh)
        return sum_edge_length / self.mesh.n_edges()

    def average_dihedral_angle(self):
        sum_dihedral_angle = 0.0
        for eh in self.mesh.edges():
            if self.mesh.is_boundary(eh):
                sum_dihedral_angle += self.mesh.calc_dihedral_angle(eh)
        return sum_dihedral_angle / self.mesh.n_edges()

    def noise(self, standard_deviation, noise_direction):

        average_length = self.average_edge_length()

        if noise_direction == NoiseDirection.NORMAL:

            for vh in self.mesh.vertices():
                n = self.mesh.normal(vh)
                g = gaussian(0, average_length * standard_deviation)
                p = self.mesh.point(vh) + n * g
                self.mesh.set_point(vh, p)

        elif noise_direction == NoiseDirection.RANDOM:

            for vh in self.mesh.vertices():
                n = random_vector()
                g = gaussian(0, average_length * standard_deviation)
                p = self.mesh.point(vh) + n * g
                self.mesh.set_point(vh, p)

    def curvatures(self, k=6):
        # TODO: compute curvatures
        gauss = numpy.random.rand(self.mesh.n_vertices(), 1)
        mean = numpy.random.rand(self.mesh.n_vertices(), 1)
        return gauss, mean

    def asymptotic_directions(self):
        # TODO: compute asymptotic directions
        indices = numpy.arange(self.mesh.n_vertices())
        normals = self.mesh.vertex_normals()
        return indices, normals
    
