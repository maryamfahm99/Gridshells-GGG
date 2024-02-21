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


class Editor(GeolabComponent):

    name = 'Check'

    #--------------------------------------------------------------------------
    #                                 Traits
    #--------------------------------------------------------------------------

    delete_edges = Bool(False)

    dual_mesh = Button()

    edge_index = Str('_')

    halfedge_index = Str('_')

    #--------------------------------------------------------------------------
    #                              Component View
    #--------------------------------------------------------------------------


    view = View(
                VGroup(
                       Item('delete_edges'),
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
        if state != 'delete_edges':
            self.delete_edges = False

    #--------------------------------------------------------------------------
    #                              Plot Functions
    #--------------------------------------------------------------------------

    def initialize_mesh_plot_functions(self, obj):
        pass

    #--------------------------------------------------------------------------
    #                              Traits Change
    #--------------------------------------------------------------------------



    @on_trait_change('delete_edges')
    def delete_edges_check(self):
        if self.delete_edges:
            self.handler.set_state('delete_edges')
            def callback(e):
                self.geolab.current_object.geometry.delete_edge(e)
                self.geolab.current_object.update_plot()
            self.geolab.current_object.on_edge_selection(callback)
        else:
            self.geolab.current_object.select_edges_off()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                                   Check
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def edit(obj):
    GUI = GeolabGUI()
    GUI.add_component(Editor())
    GUI.open_geometry(obj)
    GUI.start()




