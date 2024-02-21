import os
import numpy
from viewer import Viewer
from viewer.plugins import LoaderPlugin, WidgetsPlugin
from viewer.widgets import MainWidget, MeshWidget, PointCloudWidget


####################### Imports from Maryam

import igl
import sys
# Add the parent directory of geometry-lab-main to the Python path
# path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path = '/Users/memo2/Desktop/2022-2023/Summer2023/WebImplementation/geometry-lab-main'
sys.path.append(path)
print("first path: ", path)
# Now you can import from the geometrylab.geometry module
from geometrylab.geometry import Polyline
from archgeolab.proj_orthonet.guidedprojection_orthonet import GP_OrthoNet
# from geometrylab.geometry.meshpy import Mesh
## the following works:  
# optimizer = GP_OrthoNet()
# optimizer.optimize()
#######################

def main():

    ################################################ Maryam
    # '''Instantiate the sample component'''
    # print("Before OrthoNet")
    # from opt_gui_orthonet import OrthoNet
    # print("After OrthoNet")
    # component = OrthoNet()
    # #Bolun change this
    # # component.optimization_step()  # this only
    # component.optimizer.optimize()
    ###############################################
    viewer = Viewer()

    # Change default path
    viewer.path = os.getcwd()

    # Attach menu plugins
    loader = LoaderPlugin()
    viewer.plugins.append(loader)
    menus = WidgetsPlugin()
    viewer.plugins.append(menus)
    menus.add(MainWidget(True, loader))
    menus.add(MeshWidget(True, loader))
    menus.add(PointCloudWidget(True, loader))

    # General drawing settings
    viewer.core().is_animating = False
    viewer.core().animation_max_fps = 30.0
    viewer.core().background_color = numpy.array([0.6, 0.6, 0.6, 1.0])
    print("viewer.core().view", viewer.core().view)
    print("viewer.core().proj", viewer.core().proj)
    print("viewer.core().viewport", viewer.core().viewport)

    # Initialize viewer
    if not viewer.launch_init(True, False, True, "viewer", 0, 0):
        viewer.launch_shut()
        return

    # Example of loading a model programmatically
    path = os.path.join(os.getcwd(), 'models', 'bunny10k.obj')
    # a = path + '\archgeolab\objs'
    local = '/Users/memo2/Desktop/2022-2023/Summer2023/WebImplementation/geometry-lab-main/archgeolab/objs'
    # path = local + '/obj_anet' + '/mesh_initialization/stripp17.obj' ##M1_eq TC2_eq.
    path = local + '/obj_equilibrium/tri_dome.obj' #+ '/mesh_initialization/stripp17.obj' ##M1_eq TC2_eq.

    print("model path",path)
    # print("current path", path)
    # switch to "True" to enable "only_vertices"
    viewer.load(path, False)
    
    # Rendering
    viewer.launch_rendering(True)
    viewer.launch_shut()


if __name__ == "__main__":
    main()
