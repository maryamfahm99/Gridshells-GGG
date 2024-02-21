import copy
import os
import math
import numpy
import openmesh
from enum import IntEnum
from utils.colormap import ColorMapType
from utils.maths import gaussian, random_vector
from geometry.Geometry import Geometry

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
        self.optimizer = GP_OrthoNet() # Maryam

    def load(self, filename):
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

        # We need normals
        self.mesh.request_face_normals()
        self.mesh.request_vertex_normals()
        self.mesh.update_normals()

        # Save original mesh
        self.mesh_original = copy.deepcopy(self.mesh)

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
        # Face map
        for key, value in face_map.items():
            f_to_f[key, 0] = value

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
        
        return verts, faces, f_to_f, norms, texs, edges1, edges2
    ############### Maryam

    def tri_to_quad(self):
        print("tri_to_quad")
        # This takes a Mesh object and returns a mesh object from geometry-lab-main
        # transformed_mesh = quad_mesh()
        transformed_mesh = MMesh()  # MMesh already makes a Mesh object so i can use both's functions
        verts, faces,  _, _, _, _, _ = Mesh.to_render_data(self.mesh)
        extracted_faces = faces[:, :3]
        print("self.mesh.vertices(): ",self.mesh.vertices())
        print("verts", verts)
        # print(extracted_faces)
        #### make mesh should be initialized first, i need to follow the other framework way
        transformed_mesh.make_mesh(verts, extracted_faces)
        return transformed_mesh
    def quad_to_tri(self):
        # This takes  a mesh object from geometry-lab-main and returns a Mesh object
        return 0
    
    def optimize(self):
        print("optimize")
        quad = Mesh.tri_to_quad(self)
        ### use optimize from ortho
        print("quad type: ", type(quad))
        print("opt mesh: ",self.optimizer.mesh) 
        ### update q.V, F, and such to set the dimentions like mesh_propogate
        # self.mesh._V += len(new_v)
        # self.mesh._F += len(new_f)
        # self.mesh._E += 2*len(new_v)-1
        # # print(self.mesh._V,self.mesh._F,self.mesh._E)
        # self.mesh._vertices = np.vstack((self.mesh._vertices, new_v))
        # self.mesh._faces = np.vstack((self.mesh._faces, new_f))
        self.optimizer.mesh = quad 
        print("opt mesh: ",self.optimizer.mesh) 
        # self.optimizer.optimize()
        return 0 

    ############### Maryam

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
