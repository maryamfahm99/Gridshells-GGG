import os
import numpy
from viewer import Viewer
from viewer.plugins import LoaderPlugin, WidgetsPlugin
from viewer.widgets import MainWidget, MeshWidget, PointCloudWidget

def main():
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
    path = local + '/obj_anet' + '/mesh_initialization/stripp17.obj' ##M1_eq TC2_eq.
    print("model path",path)
    # print("current path", path)
    # switch to "True" to enable "only_vertices"
    viewer.load(path, False)

    # Rendering
    viewer.launch_rendering(True)
    viewer.launch_shut()


if __name__ == "__main__":
    main()
