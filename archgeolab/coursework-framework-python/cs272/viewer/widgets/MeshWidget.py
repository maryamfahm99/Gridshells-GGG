import imgui
import numpy
from viewer.ViewerWidget import ViewerWidget, AnchorType
from viewer.opengl.MeshGL import DirtyFlags
from geometry.Mesh import Mesh, NoiseDirection
#### Maryam
import igl
from utils.camera import project, unproject
from viewer.opengl.MeshGL import DirtyFlags, MeshGL
import openmesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#### 
from viewer.widgets import unprojection as upj


def write_paraboloid_obj(a, b, x1min, x1max, x2min, x2max, nx1, nx2, filename):
    vnbr = nx1*nx2
    fnbr = (nx1-1)*(nx2-1)*2
    V = numpy.zeros((vnbr, 3))
    F = numpy.zeros((fnbr, 3))
    verline = 0
    x1itv = (x1max-x1min)/(nx1-1)
    x2itv = (x2max-x2min)/(nx2-1)
    for i in range(nx1):
        for j in range(nx2):
            x1 = x1min+i*x1itv
            x2 = x2min+j*x2itv
            x3 = (a*x1*x1+b*x2*x2)/2
            V[verline][0] = x1
            V[verline][1] = x2
            V[verline][2] = x3
            verline = verline+1
    fline = 0
    for i in range(nx1-1):
        for j in range(nx2-1):
            id0 = nx2*i+j+1
            id1 = nx2 * (i + 1) + j+1
            id2 = nx2 * (i + 1) + j + 2
            id3 = nx2 * i + j + 2

            F[fline] = numpy.array([id0, id1, id2])
            F[fline+1] = numpy.array([id0, id2, id3])
            fline = fline+2
    f = open(filename, "a")
    for i in range(vnbr):
        vinfo = "v "+str(V[i][0])+" "+str(V[i][1])+" "+str(V[i][2])+"\n"
        f.write(vinfo)
    for i in range(fnbr):
        finfo = "f "+str(F[i][0])+" " + str(F[i][1])+" " + str(F[i][2])+"\n"
        f.write(finfo)

    f.close()


class MeshWidget(ViewerWidget):

    class BreakoutException(Exception):
        pass

    noiseStandardDeviation = 0.1
    noiseDirection = NoiseDirection.NORMAL
    curvatureNeighbours = 6
    ### Maryam
    meshFairness = 0.01
    boundFairness = 0.01
    GGG_weight = 0.01
    vertexControl = 1.0
    iso_weight = 1.0
    aprox_weight = 0.0
    iter_number = 6
    consider_new_vids = False
    show_vertices_controlled = False 
    v_id = None
    
    ###


    def __init__(self, expanded, loader):
        super().__init__(expanded)
        self.anchor = AnchorType.TopRight
        self.loader = loader
        s = None  # Maraym
        dir = None  # Maryam
        p1 = None  # Maryam 
        p2 = None  # Maryam
        # v_id = None  # Maryam

    def post_load(self):
        return False

    def pre_draw(self, first):
        return False

    def draw(self, first, scaling, x_size, y_size, x_pos, y_pos):

        imgui.begin("Mesh", True, imgui.WINDOW_NO_SAVED_SETTINGS)

        if not first:
            imgui.set_window_position(
                (x_size - x_pos) * scaling, y_pos * scaling, imgui.ONCE)
        imgui.set_window_collapsed(self.collapsed, imgui.ONCE)

        imgui.push_item_width(imgui.get_window_width() * 0.5)

        if len(self.viewer.data_list) == 0:
            imgui.text("No mesh loaded!")
        elif self.viewer.selected_data_index < 0:
            imgui.text("No mesh selected!")
        else:

            try:
                index = self.viewer.selected_data_index
                mesh = self.loader.model(index)
                if not isinstance(mesh, Mesh):
                    imgui.text("Object selected is not a mesh!")
                else:
                    # Reset
                    if imgui.button("Reset##Mesh", -1, 0):
                        mesh.reset_mesh()
                        mesh.update_viewer_data(self.viewer.data(index))

                    # Save
                    if imgui.button("Save##Mesh", -1, 0):
                        # self.viewer.open_dialog_save_mesh()
                        mesh.save_obj()

                    # Remove
                    if imgui.button("Remove##Mesh", -1, 0):
                        self.viewer.unload()
                        raise MeshWidget.BreakoutException

                    imgui.spacing()

                    # Draw options
                    if imgui.collapsing_header("Draw Options", imgui.TREE_NODE_DEFAULT_OPEN):

                        if imgui.button("Center view to mesh", -1, 0):
                            self.viewer.core().align_camera_center(
                                self.viewer.data().V, self.viewer.data().F)

                        if imgui.button("Normalize mesh", -1, 0):
                            mesh.normalize()
                            mesh.update_viewer_data(self.viewer.data(index))

                        changed, value = imgui.checkbox(
                            "Visible", self.viewer.core().is_set(self.viewer.data().is_visible))
                        if changed:
                            self.viewer.data().is_visible = \
                                self.viewer.core().set(self.viewer.data().is_visible, value)

                        changed, self.viewer.data().face_based = imgui.checkbox(
                            "Face-based", self.viewer.data().face_based)
                        if changed:
                            self.viewer.data().dirty = DirtyFlags.DIRTY_ALL

                        changed, mesh.render_flat_faces = imgui.checkbox(
                            "Render as flat triangles", mesh.render_flat_faces)
                        if changed:
                            mesh.update_viewer_data(self.viewer.data(index))

                        changed, value = imgui.checkbox(
                            "Show texture", self.viewer.core().is_set(self.viewer.data().show_texture))
                        if changed:
                            self.viewer.data().show_texture = \
                                self.viewer.core().set(self.viewer.data().show_texture, value)

                        changed, self.viewer.data().invert_normals = imgui.checkbox(
                            "Invert normals", self.viewer.data().invert_normals)
                        if changed:
                            self.viewer.data().dirty |= DirtyFlags.DIRTY_NORMAL

                        # changed, value = imgui.checkbox(
                        #     "Show overlay", self.viewer.core().is_set(self.viewer.data().show_overlay))
                        # if changed:
                        #     self.viewer.data().show_overlay = \
                        #         self.viewer.core().set(self.viewer.data().show_overlay, value)

                        # changed, value = imgui.checkbox(
                        #     "Show overlay depth", self.viewer.core().is_set(self.viewer.data().show_overlay_depth))
                        # if changed:
                        #     self.viewer.data().show_overlay_depth = \
                        #         self.viewer.core().set(self.viewer.data().show_overlay_depth, value)

                        changed, color = imgui.color_edit4(
                            "Line color",
                            mesh.edge_color[0],
                            mesh.edge_color[1],
                            mesh.edge_color[2],
                            mesh.edge_color[3],
                            True)
                        if changed:
                            mesh.edge_color = numpy.array(color)
                            mesh.update_viewer_data(self.viewer.data(index))

                        imgui.push_item_width(imgui.get_window_width() * 0.3)
                        _, self.viewer.data().shininess = imgui.drag_float(
                            "Shininess", self.viewer.data().shininess, 0.05, 0.0, 100.0)
                        imgui.pop_item_width()

                    imgui.spacing()

                    # Noise
                    if imgui.collapsing_header("Noise", imgui.TREE_NODE_DEFAULT_OPEN):

                        _, MeshWidget.noiseStandardDeviation = imgui.drag_float(
                            "sigma", MeshWidget.noiseStandardDeviation, 0.01, 0.0, 5.0, "%.2f")

                        _, MeshWidget.noiseDirection = imgui.combo(
                            "Direction", int(MeshWidget.noiseDirection), ["Normals", "Random"])

                        if imgui.button("Add Gaussian noise", -1, 0):
                            mesh.noise(MeshWidget.noiseStandardDeviation,
                                       MeshWidget.noiseDirection)
                            mesh.update_viewer_data(self.viewer.data(index))

                    imgui.spacing()

                    ################## Maryam

                    # Optimization
                    if imgui.collapsing_header("Optimization", imgui.TREE_NODE_DEFAULT_OPEN):

                        _, MeshWidget.meshFairness = imgui.drag_float(
                            "meshFairness", MeshWidget.meshFairness, 0.001, 0.0, 1.0, format="%.3f")
                        # _, MeshWidget.meshFairness = imgui.input_float(
                        #     "Fairness", MeshWidget.meshFairness, 0.001, 0.0, 1.0, format="%.3f")
                        _, MeshWidget.boundFairness = imgui.drag_float(
                            "boundFairness", MeshWidget.boundFairness, 0.001, 0.0, 1.0, format="%.3f")
                        
                        _, MeshWidget.GGG_weight = imgui.drag_float(
                            "GGG weight", MeshWidget.GGG_weight, 0.001, 0.0, 1.0, format="%.3f")
                        
                        _, MeshWidget.iso_weight = imgui.drag_float(
                            "iso weight", MeshWidget.iso_weight, 0.001, 0.0, 1.0, format="%.3f")
                        
                        _, MeshWidget.vertexControl = imgui.drag_float(
                            "vertex Control", MeshWidget.vertexControl, 0.001, 0.0, 1.0, format="%.3f")
                        _, MeshWidget.aprox_weight = imgui.drag_float(
                            "Aproximity", MeshWidget.aprox_weight, 0.001, 0.0, 1.0, format="%.3f")

                        if imgui.button("GGG", -1, 0):
                            mesh.GGG = True
                            print("opt")
                            mesh.meshFairness = self.meshFairness
                            mesh.boundFairness = self.boundFairness
                            mesh.GGG_weight = self.GGG_weight
                            mesh.vertexControl = self.vertexControl



                            for i in range(self.iter_number):
                                mesh.optimize(1,0)
                                print("before update_viewer")
                                mesh.update_viewer_data(self.viewer.data(index))
                                print("after update_viewer")
                            
                            if (MeshWidget.show_vertices_controlled):
                                color = numpy.array([0, 1, 0])
                                colors = numpy.repeat([color], 1, axis=0)
                                V = (self.viewer.data().V).astype(numpy.float64)
                                self.viewer.data(self.viewer.selected_data_index).set_points(numpy.array(V[mesh.v_ids]).reshape((-1,3)), colors)

                        if imgui.button("Iso", -1, 0):
                            mesh.meshFairness = self.meshFairness
                            mesh.boundFairness = self.boundFairness
                            mesh.iso_weight = self.iso_weight
                            mesh.vertexControl = self.vertexControl
                            mesh.GGG_weight = self.GGG_weight
                            # print("ISO weights: ", mesh.meshFairness, mesh.boundFairness, mesh.iso_weight, mesh.vertexControl)

                            # if (self.consider_new_vids == False): # Do not consider any points chosen
                            #     print("dont consider any new")
                            #     if len(mesh.v_ids) > 20:  # Check if the list has the point chosen from previous selections
                            #         print("there is that needs to be popped")
                            #         mesh.v_ids.pop()
                            # else:  
                            #     print("consider new ")                               # Consider the point
                            #     if len(mesh.v_ids) == 20  and self.v_id is not None:     # if it was popped
                            #         mesh.v_ids.append(self.v_id)

                            for i in range(self.iter_number):
                            # for i in range(1):
                                print("iso optimize from widget")
                                mesh.optimize(0,1)
                                mesh.update_viewer_data(self.viewer.data(index))
                            if (MeshWidget.show_vertices_controlled):
                                color = numpy.array([0, 1, 0])
                                colors = numpy.repeat([color], 1, axis=0)
                                V = (self.viewer.data().V).astype(numpy.float64)
                                self.viewer.data(self.viewer.selected_data_index).set_points(numpy.array(V[mesh.v_ids]).reshape((-1,3)), colors)

                        if imgui.button("Propogate", -1, 0):
                            mesh.propogate()
                            mesh.update_viewer_data(self.viewer.data(index))

                        changed, MeshWidget.show_vertices_controlled = imgui.checkbox("Show vertices to be controlled", MeshWidget.show_vertices_controlled)
                        if changed:
                            if (MeshWidget.show_vertices_controlled):
                                color = numpy.array([0, 1, 0])
                                colors = numpy.repeat([color], 1, axis=0)
                                V = (self.viewer.data().V).astype(numpy.float64)
                                self.viewer.data(self.viewer.selected_data_index).set_points(numpy.array(V[mesh.v_ids]).reshape((-1,3)), colors)
                            else:
                                mesh.update_viewer_data(self.viewer.data(index))

                        # consider_new_vids
                        changed, MeshWidget.consider_new_vids = imgui.checkbox("Keep last vertex", MeshWidget.consider_new_vids)
                        if changed:
                            print("Checkbox state changed:", MeshWidget.consider_new_vids)
                            # self.consider_new_vids  = not (self.consider_new_vids)

                        if imgui.button("Aprox", -1, 0):
                            mesh.meshFairness = self.meshFairness
                            mesh.boundFairness = self.boundFairness
                            mesh.iso_weight = self.iso_weight
                            mesh.vertexControl = self.vertexControl
                            mesh.GGG_weight = self.GGG_weight
                            mesh.aprox_weight = self.aprox_weight
                            mesh.optimizer.set_weight('aprox' , 1)
                            print("Aprox weights: ", mesh.meshFairness, mesh.boundFairness, mesh.iso_weight, mesh.vertexControl, mesh.aprox_weight)

                            for i in range(self.iter_number):
                            # for i in range(1):
                                print("aprox optimize from widget")
                                mesh.optimize(0,0) ### maybe needs fixing
                                mesh.update_viewer_data(self.viewer.data(index))
                        
                    
                    if imgui.collapsing_header("Diagonal", imgui.TREE_NODE_DEFAULT_OPEN):
                        ## why does it automatically show them? 
                        va, v, vc = mesh.diagonal()
                        num = len(va)
                        src1 = self.viewer.data(
                            index).V[va, :]
                        dst1 = self.viewer.data(
                            index).V[v, :] 
                        src2 = self.viewer.data(
                            index).V[v, :]
                        dst2 = self.viewer.data(
                            index).V[vc, :]
                        color = numpy.array([1, 0, 0])
                        colors = numpy.repeat(
                            [color], num, axis=0)
                        self.viewer.data(index).add_edges(src1, dst1, colors)
                        self.viewer.data(index).add_edges(src2, dst2, colors)

                    ##################

                    # Overlays
                    if imgui.collapsing_header("Overlays", imgui.TREE_NODE_DEFAULT_OPEN):

                        changed, mesh.render_edges = imgui.checkbox(
                            "Wireframe", mesh.render_edges)
                        if changed:
                            mesh.update_viewer_data(self.viewer.data(index))

                        changed, value = imgui.checkbox(
                            "Fill", self.viewer.core().is_set(self.viewer.data().show_faces))
                        if changed:
                            self.viewer.data().show_faces = \
                                self.viewer.core().set(self.viewer.data().show_faces, value)

                        changed, value = imgui.checkbox(
                            "Show vertex labels", self.viewer.core().is_set(self.viewer.data().show_vertex_labels))
                        if changed:
                            self.viewer.data().show_vertex_labels = \
                                self.viewer.core().set(self.viewer.data().show_vertex_labels, value)

                        changed, value = imgui.checkbox(
                            "Show faces labels", self.viewer.core().is_set(self.viewer.data().show_face_labels))
                        if changed:
                            self.viewer.data().show_face_labels = \
                                self.viewer.core().set(self.viewer.data().show_face_labels, value)

                        changed, value = imgui.checkbox(
                            "Show extra labels", self.viewer.core().is_set(self.viewer.data().show_custom_labels))
                        if changed:
                            self.viewer.data().show_custom_labels = \
                                self.viewer.core().set(self.viewer.data().show_custom_labels, value)

                    imgui.spacing()

                    # Subdivision
                    if imgui.collapsing_header("Subdivision", imgui.TREE_NODE_DEFAULT_OPEN):

                        if imgui.button("Catmull Clark##Mesh", -1, 0):
                            # Initialize subdivider
                            # OpenMesh::Subdivider::Uniform::CatmullClarkT<OpenMesh::Mesh> catmull;
                            # Execute 3 subdivision steps
                            # catmull.attach(mLoader->mesh(index)->mesh())
                            # catmull(3)
                            # catmull.detach()
                            mesh.update_viewer_data(self.viewer.data(index))

                    # Curvatures
                    if imgui.collapsing_header("Project 1", imgui.TREE_NODE_DEFAULT_OPEN):

                        _, MeshWidget.curvatureNeighbours = imgui.drag_int(
                            "Num. neighbours", MeshWidget.curvatureNeighbours, 1.0, 6, 25, "%d")

                        if imgui.button("Paint Gaussian Curvatures##Mesh", -1, 0):
                            curvatures, _ = mesh.curvatures(
                                k=MeshWidget.curvatureNeighbours)
                            self.viewer.data(index).set_data(curvatures)

                        if imgui.button("Paint Mean Curvatures##Mesh", -1, 0):
                            _, curvatures = mesh.curvatures(
                                k=MeshWidget.curvatureNeighbours)
                            self.viewer.data(index).set_data(curvatures)

                        if imgui.button("Paint Asymptotic Directions##Mesh", -1, 0):
                            indices, directions = mesh.asymptotic_directions()
                            num_directions = len(indices)
                            src = self.viewer.data(
                                index).V[indices, :] - 0.5 * directions
                            dst = self.viewer.data(
                                index).V[indices, :] + 0.5 * directions
                            color = numpy.array([1, 0, 0])
                            colors = numpy.repeat(
                                [color], num_directions, axis=0)
                            self.viewer.data(index).add_edges(src, dst, colors)
                        if imgui.button("write mesh", -1, 0):
                           write_paraboloid_obj(
                               1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 10, 10, "test.obj")

            except MeshWidget.BreakoutException:
                pass

        imgui.set_window_size(250.0 * scaling, 0.0)

        dim = imgui.Vec4(
            imgui.get_window_position().x,
            imgui.get_window_position().y,
            imgui.get_window_size().x,
            imgui.get_window_size().y)

        imgui.pop_item_width()

        imgui.end()

        return dim

    def post_draw(self, first):
        return False

    def post_resize(self, width, height):
        return False

    def mouse_down(self, button, modifier):
        

        return False

    def mouse_up(self, button, modifier):
        return False

    def mouse_move(self, mouse_x, mouse_y):
        return False

    def mouse_scroll(self, delta_y):
        return False

    def key_pressed(self, key, modifiers):
        color = numpy.array([1, 0, 0])
        colors = numpy.repeat([color], 1, axis=0)
        color = numpy.array([0, 1, 0])
        colorb = numpy.repeat([color], 1, axis=0)
        print("mesh wedget press, X ",imgui.KEY_X)
        print("mesh wedget press, real ",key)
        if key == 65 or key == '65' or key == 97 or key == '97':                         # Press A 

            # ########## Maryam 
            print("\n\n\nA pressed: ")
            mouse_x = self.viewer.current_mouse_x
            mouse_y = self.viewer.current_mouse_y

            s = [mouse_x, self.viewer.core().viewport[3] - mouse_y]
            # Set up the model, proj, viewport, V, F
            model = (self.viewer.core().view).astype(numpy.float64)
            proj = self.viewer.core().proj
            viewport = numpy.array(self.viewer.core().viewport).astype(numpy.float64)
            V = (self.viewer.data().V).astype(numpy.float64)
            F = (self.viewer.data().F).astype(numpy.int32)
            
            bc, fid, hit = upj.unproject_onto_mesh(s,model.transpose(),proj.transpose(),viewport.transpose(),V,F)
            # print("hit info",hit,fid,bc)
            # print("xy: ", s)
            
            # print("infos,",s,model.transpose(),proj.transpose(),viewport.transpose())
            
            if hit:
                # whichpoint = numpy.argmax(bc)
                # # print("which point",whichpoint)
                # # print("face",F[fid,0],F[fid,1],F[fid,2])
                # s1, dir1 = upj.unproject_ray(s,model,proj,viewport)
                # # self.s = s1
                # self.s = s1 
                # self.dir = dir1
                # self.v_id = F[fid,whichpoint]
                # index = self.viewer.selected_data_index
                # mesh = self.loader.model(index)
                # if len(mesh.v_ids) > 20:  # Check if the list is not empty
                #     mesh.v_ids.pop()
                # mesh.v_ids.append(self.v_id)
                # print("vertices that should be fixed: ", mesh.v_ids)
                # # print("vertices that should be fixed: ", mesh.v_ids)
                # # self.p1 = V[F[fid,0]] * bc[0]+V[F[fid,1]] * bc[1]+V[F[fid,2]] * bc[2]
                # self.p1 = V[F[fid,whichpoint]]
                # print("p1  in press A: ", self.p1)
                # # index = self.viewer.selected_data_index
                # # mesh = self.loader.model(index)
                # print("v from v_ids in quad and tri: ", mesh.optimizer.mesh._vertices[self.v_id])
                # self.viewer.data(self.viewer.selected_data_index).set_points(numpy.array(V[F[fid,whichpoint]]).reshape((1,3)), colors)
                # # self.viewer.data(self.viewer.selected_data_index).add_points(numpy.array(self.p1).reshape((1,3)), colors)

                whichpoint = numpy.argmax(bc)
                s1, dir1 = upj.unproject_ray(s,model,proj,viewport)
                self.s = s1 
                self.dir = dir1
                v_id = F[fid,whichpoint]
                index = self.viewer.selected_data_index
                mesh = self.loader.model(index)
                if(self.v_id is not None):   # pressed before
                    if(self.v_id in mesh.v_ids):
                        mesh.v_ids.remove(self.v_id)  # remove the previous
                self.v_id = v_id             # assign to the new point
                if(v_id not in mesh.v_ids):  # if the new chosen point not there append it
                    mesh.v_ids.append(self.v_id)
                self.p1 = V[F[fid,whichpoint]]
                print("p1  in press A: ", self.p1)
                print("v from v_ids in quad and tri: ", mesh.optimizer.mesh._vertices[self.v_id])
                self.viewer.data(self.viewer.selected_data_index).set_points(numpy.array(V[F[fid,whichpoint]]).reshape((1,3)), colors)
                
            # if (MeshWidget.show_vertices_controlled):
            #     color = numpy.array([0, 1, 0])
            #     colors = numpy.repeat([color], 1, axis=0)
            #     V = (self.viewer.data().V).astype(numpy.float64)
            #     self.viewer.data(self.viewer.selected_data_index).set_points(numpy.array(V[mesh.v_ids]).reshape((-1,3)), colors)
            # else:
            #     self.viewer.data(self.viewer.selected_data_index).set_points(numpy.array(V[F[fid,whichpoint]]).reshape((1,3)), colors)



        if key == 67 or key == '67' or key == 99 or key == '99':                    # Press C
            print("\n\n\nC pressed: ")
            mouse_x = self.viewer.current_mouse_x
            mouse_y = self.viewer.current_mouse_y

            s = [mouse_x, self.viewer.core().viewport[3] - mouse_y]
            # Set up the model, proj, viewport, V, F
            model = (self.viewer.core().view).astype(numpy.float64)
            proj = self.viewer.core().proj
            viewport = numpy.array(self.viewer.core().viewport).astype(numpy.float64)
            V = (self.viewer.data().V).astype(numpy.float64)
            F = (self.viewer.data().F).astype(numpy.int32)
            
            bc, fid, hit = upj.unproject_onto_mesh(s,model.transpose(),proj.transpose(),viewport.transpose(),V,F)

            color = numpy.array([0, 1, 0])
            colors = numpy.repeat([color], 1, axis=0)
            if hit:
                whichpoint = numpy.argmax(bc)
                v_id = F[fid,whichpoint]
                index = self.viewer.selected_data_index
                mesh = self.loader.model(index)
                
                if (MeshWidget.show_vertices_controlled):
                    if(v_id not in mesh.v_ids):
                        mesh.v_ids.append(v_id)
                        self.viewer.data(index).set_points(V[mesh.v_ids].reshape((-1,3)), colors)
                    else:
                        print("it is there already")
                        mesh.v_ids.remove(v_id)
                        self.viewer.data(index).set_points(V[mesh.v_ids].reshape((-1,3)), colors)
            # if (MeshWidget.show_vertices_controlled)

        if key == 66 or key == '66' or key == 98 or key == '98':                    # Press B
            print("\n\n\nB pressed: ")
            mouse_x = self.viewer.current_mouse_x
            mouse_y = self.viewer.current_mouse_y

            s = [mouse_x, self.viewer.core().viewport[3] - mouse_y]
            model = (self.viewer.core().view).astype(numpy.float64)
            proj = self.viewer.core().proj
            viewport = numpy.array(self.viewer.core().viewport).astype(numpy.float64)
            V = (self.viewer.data().V).astype(numpy.float64)
            F = (self.viewer.data().F).astype(numpy.int32)

            p1_proj = upj.proj(self.p1,model.transpose(),proj.transpose(),viewport.transpose())
            # print("xy: ", s)
            # print("p1_qproj: ", p1_proj)
            print("p1: ", self.p1)
            p1_proj[0] = s[0]
            p1_proj[1] = s[1]
            self.p2 = upj._unproject(p1_proj, model.transpose(),proj.transpose(),viewport.transpose())
            print("p2: ", self.p2)
            # self.viewer.data(self.viewer.selected_data_index).add_points(numpy.array(self.p2).reshape((1,3)), colorb)
            
            
            #### 1) Update the viweing mesh
            index = self.viewer.selected_data_index
            mesh = self.loader.model(index)
            v = openmesh.VertexHandle(self.v_id)  

            #### 1.1) Optimize after each small change
            num_times = 3.0
            step =  numpy.linalg.norm(self.p2-self.p1)/num_times
            dir = ((self.p2-self.p1)/numpy.linalg.norm(self.p2-self.p1))
            step_fairness = (0.01-0.1)/num_times
            step_GGG = (0.-0.001)/num_times

            #### 1.2) Perserve the variables
            p1 = self.p1
            p2 = self.p2
            # if len(mesh.v_ids) > 11:  # Check if the list is not empty
            #     mesh.v_ids.pop()
            # mesh.v_ids.append(self.v_id)
            # print("vertices that should be fixed: ", mesh.v_ids)

            print("mesh.v_ids: ", mesh.v_ids)
            parasBegin = [[0.1, 1], [0.01, 1], [0.001, 1]]
            print("p1: ", self.p1)
            p1s = [self.p1]
            mesh.meshFairness = 0.1
            mesh.boundFairness = 0.1
            mesh.GGG_weight = 0.01
            mesh.iso_weight = 0.01
            g = 1
            s = 0
            for i in range(int(num_times)):
                p2 = p1 + dir * step 
                print("p2 in press B: ", self.p2)
                mesh.optimizer.mesh._vertices[self.v_id] = p2
                mesh.mesh.set_point(v, self.p2)
                # mesh.meshFairness = mesh.meshFairness + step_fairness
                # mesh.boundFairness = mesh.boundFairness + step_fairness
                # print("mesh.meshFairness: ", mesh.meshFairness)
                # mesh.GGG_weight = mesh.GGG_weight + step_GGG
                mesh.meshFairness = parasBegin[0][0]
                mesh.boundFairness = parasBegin[0][0]
                mesh.GGG_weight = parasBegin[0][1]
                mesh.iso_weight = parasBegin[0][1]
                mesh.optimize(g,s)
                mesh.meshFairness = parasBegin[1][0]
                mesh.boundFairness = parasBegin[1][0]
                mesh.GGG_weight = parasBegin[1][1]
                mesh.iso_weight = parasBegin[1][1]
                mesh.optimize(g,s)
                mesh.meshFairness = parasBegin[2][0]
                mesh.boundFairness = parasBegin[2][0]
                mesh.GGG_weight = parasBegin[2][1]
                mesh.iso_weight = parasBegin[2][1]
                mesh.optimize(g,s)
                mesh.optimize(g,s)
                mesh.optimize(g,s)
                mesh.update_viewer_data(self.viewer.data(index))
                p1 = p2
                p1s.append(p1)
            # mesh.optimizer.mesh._copy = mesh.optimizer.mesh.copy_mesh()
            
            # # Create a 3D plot
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # # Plot points
            # # Extract x, y, and z coordinates from the list of points
            # x = [point[0] for point in p1s]
            # y = [point[1] for point in p1s]
            # z = [point[2] for point in p1s]

            # # Plot points
            # ax.scatter(x, y, z, color='r')

            # # Add labels
            # for i, point in enumerate(p1s):
            #     ax.text(point[0], point[1], point[2], str(i), color='black')

            # # Set labels
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')

            # Show plot
            # plt.show()


                # mesh.mesh.set_point(v, self.p2)

            # if(self.p1 != None and self.p2 !=  None):

            #     step = (self.p2-self.p1)/3
            #     for i in range(3):
            #         mesh.p1 = self.p1 + i*(step)
            #         mesh.p2 = self.p1  + *step


            # else:

            ### Changes the openmesh: mesh variable in the Mesh Class 
            # mesh.mesh.set_point(v, self.p2)
            # #### Update the quad mesh  # NOT TESTED
            # mesh.quad_mesh._vertices[self.v_id] = self.p2
            # mesh.optimize()
            # mesh.update_viewer_data(self.viewer.data(index))
        

            
            

        return False

    def key_down(self, key, modifiers):
        return False

    def key_up(self, key, modifiers):
        return False

    def key_repeat(self, key, modifiers):
        return False
