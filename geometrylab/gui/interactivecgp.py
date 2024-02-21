#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division


from traits.api import HasTraits, Instance, Property, Enum, Button,String,\
                       on_trait_change, Float, Bool, Int, Array, Range

from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor, HGroup,\
                         Group, Tabbed, VGroup, ArrayEditor, EnumEditor

from pyface.image_resource import ImageResource

import numpy as np

#------------------------------------------------------------------------------

from geometrylab.optimization.curvaturegp import CurvatureGuidedProjection

from geometrylab.gui.geolabcomponent import GeolabComponent

from geometrylab.geometry import meshutilities

#------------------------------------------------------------------------------

'''_'''

__author__ = 'Davide Pellis'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                        InteractiveGuidedProjection
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class InteractiveCGP(GeolabComponent):

    name = String('GuidedP')

    iterations = Int(1)

    epsilon = Float(0.001, label='dumping')

    step = Float(1)

    fairness_reduction = Float(0)

    min_tac = Float(0.1)

    geometric = Float(1)

    mesh_fairness = Float(0.3)

    tangential_fairness = Float(0.3)

    boundary_fairness = Float(0.3)

    reference_closeness = Float(0)

    boundary_closeness = Float(0)

    self_closeness = Float(0)

    reinitialize = Button(label='Initialize')

    fix_corners = Bool(True)

    reset = Button(label='Reset')

    optimize = Button(label='Optimize')

    geometric_error = String('_', label='Geometric')

    curvature_error = String('_', label='Curvature')

    interactive = Bool(False, label='Interactive')

    step_control = Bool(False)

    fix_boundary_normals = Bool(False, label='Fix b-normals')

    #--------------------------------------------------------------------------
    view = View(
                VGroup(
                       Group('iterations',
                             'epsilon',
                             'step',
                             'mesh_fairness',
                             'tangential_fairness',
                             'boundary_fairness',
                             #'fairness_reduction',
                             'min_tac',
                             'geometric',
                             'reference_closeness',
                             'boundary_closeness',
                             'self_closeness',
                             'fix_boundary_normals',
                             show_border=True,
                             label='settings'),
                       VGroup('geometric_error',
                              'curvature_error',
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
               )

    #--------------------------------------------------------------------------
    #                                Initialize
    #--------------------------------------------------------------------------

    def __init__(self):
        GeolabComponent.__init__(self)

        self.optimizer = CurvatureGuidedProjection()

        self.counter = 0

    #--------------------------------------------------------------------------
    #                                Properties
    #--------------------------------------------------------------------------

    @property
    def mesh(self):
        return self.geolab.current_object.geometry

    @property
    def meshmanager(self):
        return self.geolab.current_object

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------

    def geolab_settings(self):
        pass

    def object_open(self, file_name, geometry):
        name = ('mesh_{}').format(self.counter)
        self.geolab.add_object(geometry, name=name)
        self.counter += 1

    def object_change(self):
        pass

    def object_changed(self):
        self.optimizer.mesh = self.geolab.current_object.geometry

    def object_save(self, file_name):
        self.optimizer.save_report(file_name)

    def set_state(self, state):
        if state != 'kr_interactive':
            self.interactive = False
        if state != 'mask_target':
            self.mask_target = False

    #--------------------------------------------------------------------------
    #                              Optimization
    #--------------------------------------------------------------------------

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
        self.geometric_error = self.optimizer.geometric_error_string()
        self.curvature_error = self.optimizer.curvature_error_string()

    @on_trait_change('optimize')
    def optimize_mesh(self):
        self.meshmanager.iterate(self.optimization_step, self.iterations)

    @on_trait_change('interactive')
    def interactive_optimize_mesh(self):
        self.handler.set_state('kr_interactive')
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

    #--------------------------------------------------------------------------
    #                              Settings
    #--------------------------------------------------------------------------

    def set_settings(self):
        self.optimizer.threshold = 1e-20
        self.optimizer.iterations = 1
        self.optimizer.epsilon = self.epsilon
        self.optimizer.step = self.step
        self.optimizer.fairness_reduction = self.fairness_reduction
        self.optimizer.set_weight('mesh_fairness', self.mesh_fairness)
        self.optimizer.set_weight('tangential_fairness', self.tangential_fairness)
        self.optimizer.set_weight('boundary_fairness', self.boundary_fairness)
        self.optimizer.set_weight('reference_closeness', self.reference_closeness)
        self.optimizer.set_weight('boundary_closeness', self.boundary_closeness)
        self.optimizer.set_weight('self_closeness', self.self_closeness)
        self.optimizer.set_weight('geometric', self.geometric)
        self.optimizer.set_weight('min_abs_curvature', self.min_tac)

    #--------------------------------------------------------------------------
    #                                  Reset
    #--------------------------------------------------------------------------

    @on_trait_change('fix_boundary_normals')
    def fix_boundary_normals_fired(self):
        if self.fix_boundary_normals:
            self.mesh.fix_double_boundary()
        else:
            self.mesh.reset_fixed()
        self.meshmanager.update_plot()

    @on_trait_change('reinitialize')
    def reinitialize_optimizer(self):
        self.set_settings()
        self.optimizer.reinitialize()
        self.print_error()
        self.meshmanager.update_plot()

#------------------------------------------------------------------------------
#                                      Test
#------------------------------------------------------------------------------

if __name__ == '__main__':

    import os

    from geometrylab.gui.geolabgui import GeolabGUI

    path = os.path.dirname(os.path.abspath(__file__))

    file_name = path + '/tri_dome.obj'

    component = InteractiveCGP()

    GUI = GeolabGUI()

    '''Add the component to geolab'''
    GUI.add_component(component)

    '''Open an obj file'''
    GUI.open_obj_file(file_name)

    '''Start geolab main loop'''
    GUI.start()
