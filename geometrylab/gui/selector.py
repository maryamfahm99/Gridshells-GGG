#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from traits.api import Enum, Button, Str, Float, Bool, Int, Array, Range

from traits.api import on_trait_change

from traitsui.api import View, Item, HGroup, Group, Tabbed, VGroup

from traitsui.api import ArrayEditor, EnumEditor, CheckListEditor

import numpy as np

#------------------------------------------------------------------------------

from geometrylab.gui.geolabcomponent import GeolabComponent

from geometrylab.gui.geolabgui import GeolabGUI

#------------------------------------------------------------------------------

__author__ = 'Davide Pellis'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                                CHECK COMPONENT
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class Check(GeolabComponent):

    name = 'Check'

    #--------------------------------------------------------------------------
    #                                 Traits
    #--------------------------------------------------------------------------

    halfedges_check = Bool(False)

    dual_mesh = Button()

    edge_index = Str('_')

    halfedge_index = Str('_')

    #--------------------------------------------------------------------------
    #                              Component View
    #--------------------------------------------------------------------------


    view = View(
                VGroup(
                       Item('halfedges_check'),
                       Item('dual_mesh'),
                       show_border=True,
                       ),
                resizable=True,
                )

    #--------------------------------------------------------------------------
    #                                Attributes
    #--------------------------------------------------------------------------

    counter = 0

    #--------------------------------------------------------------------------
    #                            Standard Methods
    #--------------------------------------------------------------------------

    def geolab_settings(self):
        self.geolab.height = 800
        self.geolab.width = 900

    def object_open(self, file_name, geometry):
        name = ('mesh_{}').format(self.counter)
        self.geolab.add_object(geometry, name=name)
        self.counter += 1

    def object_change(self):
        self.halfedges_check = False

    def object_changed(self):
        pass

    def object_save(self, file_name):
        pass

    def set_state(self, state):
        if state != 'halfedges_check':
            self.halfedges_check = False

    #--------------------------------------------------------------------------
    #                              Plot Functions
    #--------------------------------------------------------------------------

    def initialize_mesh_plot_functions(self, obj):
        pass

    #--------------------------------------------------------------------------
    #                              Traits Change
    #--------------------------------------------------------------------------

    @on_trait_change('dual_mesh')
    def fire_dual_mesh(self):
        obj = self.geolab.current_object
        #obj.geometry._connectivity_check()
        if self.halfedges_check:
            self.set_state('halfedges_check')
            obj.on_edge_selection(self.plot_halfedges)
        else:
            obj.remove('points')
            obj.remove('hhe_e')
            obj.remove('hhe_f')
            obj.select_edges_off()

    @on_trait_change('halfedges_check')
    def start_halfedges_check(self):
        obj = self.geolab.current_object
        #obj.geometry._connectivity_check()
        if self.halfedges_check:
            self.set_state('halfedges_check')
            obj.on_edge_selection(self.plot_halfedges)
        else:
            obj.remove('points')
            obj.remove('hhe_e')
            obj.remove('hhe_f')
            obj.select_edges_off()

    def plot_halfedges(self, edge_index):
        obj = self.geolab.current_object
        H = obj.mesh.halfedges
        e = np.where(H[:,5] == edge_index)[0]
        h1 = e[0]; h2 = e[1]
        t1 = H[H[h1,4],5]
        t2 = H[H[h2,4],5]
        edges = np.unique(np.array([edge_index, t1, t2]))
        if len(edges) > 1:
            edge_color = 'r'
        elif len(e) != 2:
            edge_color = 'r'
        elif H[h1,4] != h2:
            edge_color = 'r'
        elif H[h2,4] != h1:
            edge_color = 'r'
        else:
            edge_color = 'w'
        f1 = H[h1,1]
        f2 = H[h2,1]
        if f1 == -1:
            if f2 != -1:
                faces = np.array([f2])
        elif f2 == -1:
            if f1 != -1:
                faces = np.array([f1])
        else:
            faces = np.array([f1,f2])
        if f1 == f2:
            face_color = 'r'
        else:
            face_color = 'w'
        nex1 = H[H[h1,2],5]
        nex2 = H[H[h2,2],5]
        nex = np.array([nex1,nex2])
        pre1 = H[H[h1,3],5]
        pre2 = H[H[h2,3],5]
        pre = np.array([pre1,pre2])
        v1 = H[h1,0]
        v2 = H[h2,0]
        vertices = np.array([v1,v2])
        edge_data = np.zeros(obj.geometry.E)
        edge_data[edges] = 1
        edge_data[pre] = 2
        edge_data[nex] = 3
        face_data = np.ones(obj.geometry.F)
        face_data[faces] = 0
        obj.plot_edges(edge_data = edge_data,
                       color = [(160,160,160), edge_color, 'b', 'r'],
                       lut_range = '-:+',
                       name = 'hhe_e')
        obj.plot_faces(face_data = face_data,
                       color = [face_color, (160,160,160)],
                       lut_range = '-:+',
                       name = 'hhe_f')
        obj.plot_vertices(vertex_indices = vertices,
                          color = 'yellow',
                          name = 'points')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                                   Check
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def check(file_name=None):
    GUI = GeolabGUI()
    GUI.add_component(Check())
    GUI.open_obj_file(file_name)
    GUI.start()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                               COMPONENT TEST
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':

    import os

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    file_name = path + '/test/tri_dome.obj'

    check(file_name)


