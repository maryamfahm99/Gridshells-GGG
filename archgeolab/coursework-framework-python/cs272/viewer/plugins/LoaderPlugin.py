from viewer.ViewerPlugin import ViewerPlugin
from geometry.Mesh import Mesh
from geometry.PointCloud import PointCloud


class LoaderPlugin(ViewerPlugin):

    def __init__(self):
        super().__init__()
        self.model_list = []

    def init(self, viewer):
        super().init(viewer)

    def shutdown(self):
        super().shutdown()

    def load(self, filename, only_vertices):
        if only_vertices:
            model = PointCloud()
        else:
            model = Mesh()
        if model.load(filename):
            print("load1")
            self.viewer.append_data(True)
            print("load2")
            self.model_list.append(model)
            print("load3")
            assert (len(self.model_list) == len(self.viewer.data_list))
            print("load4: ", self.viewer.data_list[-1].id)
            model.update_viewer_data(self.viewer.data_list[-1])
            print("load5")
            return True
        return False

    def unload(self):
        if (self.viewer.selected_data_index < 0 or
                self.viewer.selected_data_index >= len(self.viewer.data_list)):
            return True
        self.model_list.pop(self.viewer.selected_data_index)
        self.viewer.data(self.viewer.selected_data_index).clear()
        self.viewer.data(self.viewer.selected_data_index).clear_edges()
        self.viewer.erase_data(self.viewer.selected_data_index)
        self.viewer.selected_data_index = -1
        return True

    def save(self, filename, only_vertices):
        if (self.viewer.selected_data_index < 0 or
                self.viewer.selected_data_index >= len(self.viewer.data_list)):
            return True
        if self.model_list[self.viewer.selected_data_index] is not None:
            return self.model_list[self.viewer.selected_data_index].save(filename)
        return False

    def serialize(self, buffer):
        return False

    def deserialize(self, buffer):
        return False

    def post_load(self):
        return False

    def pre_draw(self, first):
        return False

    def post_draw(self, first):
        return False

    def post_resize(self, w, h):
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
        return False

    def key_down(self, key, modifiers):
        return False

    def key_up(self, key, modifiers):
        return False

    def key_repeat(self, key, modifiers):
        return False

    def model(self, index):
        if index < 0 or index >= len(self.model_list):
            return None
        return self.model_list[index]
