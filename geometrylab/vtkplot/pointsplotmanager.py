#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from tvtk.api import tvtk

import numpy as np

#------------------------------------------------------------------------------

from geometrylab.vtkplot.plotmanager import PlotManager

from geometrylab.vtkplot.pointsource import Points

from geometrylab.vtkplot.vectorsource import Vectors

#------------------------------------------------------------------------------

'''-'''

__author__ = 'Davide Pellis'


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                               Vertices PlotManager
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class PointsPlotManager(PlotManager):
    print("PointsPlotManager") # Maryam

    def __init__(self, scene_model=None):
        PlotManager.__init__(self, scene_model)

        self.scale = 1

        self.glyph_scale = 1

        self.vector_scale = 1

        self._vertex_data = None

        self.selected_vertices = []

        self.vertex_color = 'cornflower'

        self.selection_color = 'yellow'

        self.view_mode = 'solid'

        self._r = None

        self._g = None

        self._points = None

        self._updating = True

        self._record = False

        self.__Vertex_data = None

        self.__dimensions = (0)

        self.__counter = 0

    # -------------------------------------------------------------------------
    #                              Properties
    # -------------------------------------------------------------------------

    @property
    def type(self):
        return 'Points_plot_manager'

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if points.type != 'Points' and type(points) != np.ndarray:
            raise ValueError('*Vertices* attribute must be geometry.Vertices!')
        self._points = points
        self._set_r()
        self._set_g()
        self.vertex_data = np.zeros(points.V)

    @property
    def geometry(self):
        return self._points

    @geometry.setter
    def geometry(self, points):
        self.points = points

    @property
    def r(self):
        return self._r * self.scale

    @property
    def g(self):
        return self._r* self._g * self.glyph_scale

    @property
    def v(self):
        return 12 * self._r * self.vector_scale

    @property
    def updating(self):
        return self._updating

    @updating.setter
    def updating(self, updating):
        if type(updating) == bool:
            self._updating = updating
        else:
            raise ValueError('the updating attribute must be a boolean!')

    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, bool):
        self._record = bool

    @property
    def vertex_data(self):
        if self._vertex_data is None:
            self._vertex_data = np.zeros(self.mesh.V)
        return self._vertex_data

    @vertex_data.setter
    def vertex_data(self, vertex_data):
        self._vertex_data = vertex_data

    @property
    def object_selection_actors(self):
        return self.get_actor('virtual-points').actors

    # -------------------------------------------------------------------------
    #                              Size set
    # -------------------------------------------------------------------------

    def check_updates(self):
        updated = False
        if self.__dimensions != self.geometry.V:
            updated = True
        if updated:
            self._set_r()
            self._set_g()
            self._vertex_data = None
        self.__dimensions = (self.geometry.V)
        return updated

    def _set_r(self):
        if self.geometry is None or self.geometry.V == 0:
            self._r = None
        else:
            p_max = np.max(self.geometry, axis=0)
            p_min = np.min(self.geometry, axis=0)
            r = np.linalg.norm(p_max - p_min) / 70
            r = r / self.geometry.V**0.1
            if r == 0:
                r = np.linalg.norm(p_max) / 10 + 0.1
            self._r = r

    def _set_g(self):
        if self.geometry is None or self.geometry.V == 0:
            self._r = None
        else:
            p_max = np.max(self.geometry, axis=1)
            p_min = np.min(self.geometry, axis=1)
            g = np.linalg.norm(p_max - p_min) / 100
            if g == 0:
                g = np.linalg.norm(p_max) / 10 + 0.1
            self._g = g

    # -------------------------------------------------------------------------
    #                                Clear
    # -------------------------------------------------------------------------

    def clear(self, delete=True):
        PlotManager.clear(self)
        self.picker_off()
        self.selected_vertices = []
        self.remove_widgets()

    # -------------------------------------------------------------------------
    #                            Plot functions
    # -------------------------------------------------------------------------

    def initialize_plot(self):
        pass
        #self.plot_edges()

    def update_plot(self, **kwargs):
        print("update_plot 1")
        self.check_updates()
        PlotManager.update_plot(self, **kwargs)
        self.plot_selected_vertices()

    def plot_vertices(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'vertices'
        if 'radius' not in kwargs:
            kwargs['radius'] = self.r
        if 'color' not in kwargs:
            kwargs['color'] = self.vertex_color
        elif 'vertex_data' in kwargs:
            if self.record:
                kwargs['vertex_data'] = self.vertex_data
            else:
                self.vertex_data = kwargs['vertex_data']
        if kwargs['name'] not in self.sources or not self.updating:
            P = Points(self.geometry, **kwargs)
            self.add(P)
        else:
            self.update(**kwargs)

    def plot_glyph(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'glyph'
        if 'Vertices' not in kwargs:
            kwargs['vertices'] = self.mesh
        if 'radius' not in kwargs:
            kwargs['radius'] = self.r
        if 'color' not in kwargs:
            kwargs['color'] = self.vertex_color
        if kwargs['name'] not in self.sources or not self.updating:
            P = Points(**kwargs)
            self.add(P)
        else:
            self.update(**kwargs)

    def plot_vectors(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'point-vectors'
        if 'anchor' not in kwargs:
            kwargs['anchor'] = self.geometry
        if 'color' not in kwargs:
            kwargs['color'] = self.vertex_color
        if kwargs['name'] not in self.sources or not self.updating:
            P = Vectors(**kwargs)
            self.add(P)
            if 'scale_factor' not in kwargs:
                kwargs['scale_factor'] = self.v
                self.update(**kwargs)
        else:
            if 'scale_factor' not in kwargs:
                kwargs['scale_factor'] = self.v
            self.update(**kwargs)

    def hide_vertices(self):
        self.hide('vertices')

    def remove_vertices(self):
        self.remove('vertices')

    # -------------------------------------------------------------------------
    #                             Virtual plot
    # -------------------------------------------------------------------------

    def selection_on(self):
        self.virtual_vertices_on()
        return self.geometry.V * 8

    def selection_off(self):
        self.virtual_vertices_off()

    def virtual_vertices_on(self):
        if 'virtual-points' not in self.sources or not self.updating:
            M = Points(self.geometry,
                       name = 'virtual-points',
                       glyph_type = 'cube',
                       radius = self.r * 1.1,
                       opacity = 0.001)
            self.add(M, pickable=True)
        else:
            self.update('virtual-points',
                        radius = self.r * 1.1)

    def virtual_vertices_off(self):
        self.hide('virtual-points')

    #--------------------------------------------------------------------------
    #                            Selection plot
    #--------------------------------------------------------------------------

    def highlight(self):
        radius = self.r * 1.1
        P = Points(self.geometry,
                   radius = radius,
                   color = self.selection_color,
                   shading = False,
                   name = 'highlight')
        self.add(P)

    def highlight_off(self):
        self.remove('highlight')

    def plot_selected_vertices(self):
        if len(self.selected_vertices) > 0:
            if 'selected-vertices' not in self.sources or not self.updating:
                if self.view_mode == 'wireframe' or self.view_mode == '3d':
                    glyph = 'wirefrane'
                    radius = None
                elif self.view_mode == 'solid':
                    radius = self.r * 1.1
                    glyph = 'sphere'
                P = Points(self.geometry[self.selected_vertices],
                           color = self.selection_color,
                           radius = radius,
                           shading = False,
                           glyph_type = glyph,
                           name = 'selected-vertices',
                           resolution = 18)
                self.add(P)
            else:
                self.update('selected-vertices',
                            points = self.geometry[self.selected_vertices],
                            radius = self.r * 1.1)
        else:
            self.hide('selected-vertices')

    def hide_selected_vertices(self):
        self.hide('selected-vertices')

    # -------------------------------------------------------------------------
    #                               Selection
    # -------------------------------------------------------------------------

    def select_vertices(self):
        self.select_off()
        self.virtual_vertices_on()
        self.selected_vertices = []
        def callback(p_id):

            if p_id == -1:
                return
            v = p_id // 6
            if v not in self.selected_vertices:
                self.selected_vertices.append(v)
            else:
                self.selected_vertices.remove(v)
            self.plot_selected_vertices()
        self.picker_callback(callback, mode='cell', name='virtual-points')

    def on_vertex_selection(self, vertex_callback):
        self.select_off()
        self.virtual_vertices_on()
        self.selected_vertices = []
        def callback(p_id):
            if p_id == -1:
                return
            v = p_id // 6
            self.selected_vertices = [v]
            if vertex_callback is not None:
                vertex_callback(v)
            self.virtual_vertices_on()
        self.picker_callback(callback, mode='cell', name='virtual-points')

    def select_vertices_off(self):
        self.virtual_vertices_off()
        self.hide_selected_vertices()
        self.picker_off()
        self.selected_vertices = []

    def clear_selection(self):
        self.hide_selected_vertices()
        self.selected_vertices = []
        self.virtual_vertices_off()
        self.remove_widgets()

    def select_off(self):
        self.select_vertices_off()
        self.move_vertex_off()
        self.move_vertices_off()

    #--------------------------------------------------------------------------
    #                               Moving
    #--------------------------------------------------------------------------

    def move_vertex(self, vertex_index, interaction_callback=None,
                    end_callback=None):
        self.remove_widgets()
        S = tvtk.SphereWidget()
        point = self.geometry[vertex_index]
        S.center = point
        S.radius = self.r * 1.05
        S.representation = 'surface'
        P = Points(points = np.array([point]),
                   radius = self.r * 1.1,
                   color = self.selection_color,
                   shading = False,
                   name = 'widget-point')
        self.add(P)
        self.virtual_vertices_on()
        def i_callback(obj, event):
            #self.selected_vertices = []
            self.virtual_vertices_off()
            c = obj.GetCenter()
            center = np.array([c[0],c[1],c[2]])
            self.geometry[vertex_index,:] = center
            self.update('widget-point', points=center)
            if interaction_callback is not None:
                interaction_callback()
        S.add_observer("InteractionEvent", i_callback)
        def e_callback(obj, event):
            if end_callback is not None:
                end_callback()
            self.virtual_vertices_on()
            point = self.geometry[vertex_index]
            S.center = point
            self.update('widget-point', points=point)
        S.add_observer("EndInteractionEvent", e_callback)
        self.add_widget(S, name='vertex_handler')

    def move_vertex_off(self):
        self.remove_widgets()
        self.remove('widget-point')
        self.virtual_vertices_off()

    def move_vertices(self, interaction_callback=None, start_callback=None,
                     end_callback=None):
        print("move_vertices 5") # Maryam
        self.select_off()
        self.virtual_vertices_on()
        def v_callback(p_id):
            self.hide_selected_vertices()
            if p_id == -1:
                return
            v = p_id//6
            self.selected_vertices = [v]
            if start_callback is not None:
                try:
                    start_callback()
                except:
                    pass
            self.move_vertex(v, interaction_callback, end_callback)
        self.picker_callback(v_callback, mode='cell', name='virtual-points')

    def move_vertices_off(self):
        print("move_vertices_off 2") #Maryam
        self.selected_vertices = []
        self.move_vertex_off()
        self.picker_off()
