#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division


from traits.api import HasTraits, Instance, Property, Enum, Button,String,\
                       on_trait_change, Float, Bool, Int, Array, Range

from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor, HGroup,\
                         Group, Tabbed, VGroup, ArrayEditor

from pyface.image_resource import ImageResource

import numpy as np

#------------------------------------------------------------------------------
################ we add this
import sys
import os


path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(path)
################ we add this
from geometrylab.optimization.guidedprojection import GuidedProjection

from geometrylab.optimization.scaligner import StressCurvatureAligner

from geometrylab.vtkplot.polylinesource import Polyline

#------------------------------------------------------------------------------

from geometrylab.gui.geolabcomponent import GeolabComponent

#------------------------------------------------------------------------------

'''_'''

__author__ = 'Davide Pellis'

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                        InteractiveGuidedProjection
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class InteractiveGuidedProjection(GeolabComponent):

    name = String('Guided Projection')

    iterations = Int(1)

    epsilon = Float(0.001, label='dumping')

    step = Float(1)

    fairness_reduction = Float(0)

    planarity = Float(0.5)

    circularity = Float(0)

    equilibrium = Float(1)

    alignment = Float(0)

    geometric = Float(1)

    mesh_fairness = Float(0.1)

    tangential_fairness = Float(0.1)

    boundary_fairness = Float(0.1)

    reference_closeness = Float(0)

    boundary_closeness = Float(0)

    self_closeness = Float(0)

    reinitialize = Button(label='Initialize')

    gliding = Bool(True)

    compression = Bool(False)

    tension = Bool(False)

    fix_corners = Bool(True)

    reset = Button(label='Reset')

    optimize = Button(label='Optimize')

    equilibrium_error = String('_', label='Equilibrium')

    planarity_error = String('_', label='Planarity')

    geometric_error = String('_', label='Geometric')

    alignment_error = String('_', label='S/C alignment')

    interactive = Bool(False)

    #--------------------------------------------------------------------------
    view = View(
                VGroup(
                       Group('iterations',
                             'epsilon',
                             'step',
                             'mesh_fairness',
                             'tangential_fairness',
                             'boundary_fairness',
                             'fairness_reduction',
                             'planarity',
                             'circularity',
                             'equilibrium',
                             'alignment',
                             'geometric',
                             'reference_closeness',
                             'boundary_closeness',
                             'self_closeness',
                             '_',
                             HGroup('compression',
                                    'tension'),
                             show_border=True,
                             label='settings'),
                       VGroup('geometric_error',
                              'equilibrium_error',
                              'planarity_error',
                              'alignment_error',
                              style='readonly',
                              label='errors [ mean | max ]',
                              show_border=True),
                       HGroup(Item('interactive',
                                   tooltip='Interactive',
                                   show_label=False,),
                              Item('_'),
                              'optimize',
                              'reinitialize',
                              show_labels=False,
                              show_border=True),
                       show_border=False,
                       show_labels=True,
                       ),
               resizable=False,
               width = 0.1,
               title = 'Guided Projection',
               )

    # -------------------------------------------------------------------------
    #                                Initialize
    # -------------------------------------------------------------------------

    def __init__(self):
        GeolabComponent.__init__(self)

        self.optimizer = StressCurvatureAligner()#Guided_projection()


    # -------------------------------------------------------------------------
    #                                Properties
    # -------------------------------------------------------------------------

    @property
    def mesh(self):
        return self.geolab.current_object.mesh

    @property
    def settings(self):
        return self.optimizer.settings

    @property
    def meshmanager(self):
        print(self.geolab.current_object)
        return self.geolab.current_object

    def object_changed(self):
        if self.meshmanager.type == 'Mesh_plot_manager':
            self.optimizer.mesh = self.mesh

    # -------------------------------------------------------------------------
    #                                Handler
    # -------------------------------------------------------------------------

    def set_state(self, state):
        if state != 'gp_interactive':
            self.interactive = False

    # -------------------------------------------------------------------------
    #                              Optimization
    # -------------------------------------------------------------------------

    def optimization_step(self):
        if not self.interactive:
            self.handler.set_state(None)
        self.set_settings()
        self.optimizer.optimize()
        self.print_error()
        self.mesh_fairness = self.mesh_fairness/(10**(self.fairness_reduction))
        self.tangential_fairness = self.tangential_fairness/(10**
                                    (self.fairness_reduction))
        self.boundary_fairness = self.boundary_fairness/(10**
                                (self.fairness_reduction))
        self.meshmanager.update_plot()

    def print_error(self):
        self.geometric_error = self.optimizer.geometric_error()
        self.equilibrium_error = self.optimizer.equilibrium_error()
        self.planarity_error = self.optimizer.planarity_error()
        self.alignment_error = self.optimizer.alignment_error()

    @on_trait_change('optimize')
    def optimize_mesh(self):
        self.meshmanager.iterate(self.optimization_step, self.iterations)

    @on_trait_change('interactive')
    def interactive_optimize_mesh(self):
        self.handler.set_state('gp_interactive')
        if self.interactive:
            def start():
                self.mesh.handle = self.meshmanager.selected_vertices
            def interact():
                self.meshmanager.iterate(self.optimization_step,1)
            def end():
                self.meshmanager.iterate(self.optimization_step,5)
            self.meshmanager.move_vertices(interact,start,end)
        else:
            self.mesh.handle = None
            self.meshmanager.move_vertices_off()

    # -------------------------------------------------------------------------
    #                              Settings
    # -------------------------------------------------------------------------

    def set_settings(self):
        self.settings.threshold = 1e-20
        self.settings.iterations = 1
        self.settings.epsilon = self.epsilon
        self.settings.step = self.step
        self.settings.fairness_reduction = self.fairness_reduction
        self.settings.mesh_fairness = self.mesh_fairness
        self.settings.tangential_fairness = self.tangential_fairness
        self.settings.boundary_fairness = self.boundary_fairness
        self.settings.reference_closeness = self.reference_closeness
        self.settings.boundary_closeness = self.boundary_closeness
        self.settings.self_closeness = self.self_closeness
        self.settings.planarity = self.planarity
        self.settings.circularity = self.circularity
        self.settings.equilibrium = self.equilibrium
        self.settings.alignment = self.alignment
        self.settings.geometric = self.geometric
        self.settings.compression = self.compression
        self.settings.tension = self.tension

    @on_trait_change('compression')
    def set_compression(self):
        if self.compression:
            self.tension = False

    @on_trait_change('tension')
    def set_tension(self):
        if self.tension:
            self.compression = False

    # -------------------------------------------------------------------------
    #                                  Reset
    # -------------------------------------------------------------------------

    @on_trait_change('reinitialize')
    def initialize_x(self):
        self.set_settings()
        self.optimizer.initialize()
        self.print_error()
        self.meshmanager.update_plot()

