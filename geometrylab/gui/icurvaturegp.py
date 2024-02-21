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

from geometrylab.optimization.curvaturegp import CurvatureGP

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


class InteractiveCurvatureGP(GeolabComponent):

    name = String('Optimize')

    iterations = Int(1)

    epsilon = Float(0.001, label='dumping')

    step = Float(1)

    fairness_reduction = Float(0)

    eq_curv_ratio = Float(0)

    geometric = Float(1)

    curvature_ratio = Float(-1)

    mesh_fairness = Float(0.3)

    tangential_fairness = Float(0.3)

    boundary_fairness = Float(0.3)

    reference_closeness = Float(0)

    boundary_closeness = Float(0)

    willmore = Float(0)

    curvature = Float(0)

    willmore_energy = String('_', label='Willmore')

    curvature_energy = String('_', label='Curvature')

    self_closeness = Float(0)

    reinitialize = Button(label='Initialize')

    fix_corners = Bool(True)

    reset = Button(label='Reset')

    optimize = Button(label='Optimize')

    curvature_ratios_error = String('_', label='Curv. ratio')

    geometric_error = String('_', label='Geometric')

    tot_abs_curv = String('_', label='Tot abs curv')

    tot_abs_stress = String('_', label='Tot abs stress')

    interactive = Bool(False, label='Interactive')

    mask_target = Bool(False, label='Mask target')

    show_curvature_directions = Bool(False, label='Curv. dir.')

    mask_depth = Int(5, label='Depth')

    step_control = Bool(False)

    delta = Float(0.2)

    linear_ratio = Bool(True)

    abs_curvature = Float(0)

    comb_directions = Bool(False)

    fix_boundary_normals = Bool(False, label='Fix b-normals')

    add_noise = Button()

    noise_factor = Float(0.1, label='Factor')

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
                             'curvature_ratio',
                             'eq_curv_ratio',
                             #'linear_ratio',
                             'abs_curvature',
                             'willmore',
                             'curvature',
                             'geometric',
                             'reference_closeness',
                             'boundary_closeness',
                             'self_closeness',
                             HGroup(Item('add_noise', show_label=False),
                                    'noise_factor',),
                             HGroup('comb_directions',
                                    'show_curvature_directions',),
                             'fix_boundary_normals',
                             show_border=True,
                             label='settings'),
                       HGroup('mask_target',
                              Item('mask_depth',
                                   width=1,
                                   resizable=False),
                              show_border=True),
                       VGroup('geometric_error',
                              'curvature_ratios_error',
                              'tot_abs_curv',
                              'willmore_energy',
                              'curvature_energy',
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

    # -------------------------------------------------------------------------
    #                                Initialize
    # -------------------------------------------------------------------------

    def __init__(self):
        GeolabComponent.__init__(self)

        self.optimizer = CurvatureGP()

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
        return self.geolab.current_object

    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------

    def object_changed(self):
        if self.meshmanager.type is 'Mesh_plot_manager':
            self.optimizer.mesh = self.mesh
            self.meshmanager.hide('curvature-d1')
            self.meshmanager.hide('curvature-d2')
            self.meshmanager.add_face_callback(self.plot_curvature_ratio,
                                             name='Curvature ratio')
            self.meshmanager.add_plot_callback(self.plot_curvature_directions)

    def object_save(self, file_name):
        self.optimizer.save_report(file_name)

    def set_state(self, state):
        if state != 'kr_interactive':
            self.interactive = False
        if state != 'mask_target':
            self.mask_target = False

    # -------------------------------------------------------------------------
    #                              Optimization
    # -------------------------------------------------------------------------

    def return_ratio(self):
        return self.curvature_ratio

    # -------------------------------------------------------------------------
    #                              Optimization
    # -------------------------------------------------------------------------

    def optimization_step(self):
        if not self.interactive and not self.mask_target:
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
        self.curvature_ratios_error = self.optimizer.curvature_ratios_error()
        self.tot_abs_curv = self.optimizer.total_absolute_curvature()
        self.willmore_energy = self.optimizer.willmore_energy()
        self.curvature_energy = self.optimizer.curvature_energy()

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

    @on_trait_change('mask_depth')
    def mask_depth(self):
        if self.mask_target:
            v = self.meshmanager.selected_vertices[0]
            mask = self.mesh.vertex_ring_expansion(v, depth=self.mask_depth)
            self.settings.curvature_mask = mask

    def set_mask(self, v):
        mask = self.mesh.vertex_multiple_ring_vertices(v, self.mask_depth)
        self.settings.curvature_mask = mask
        self.meshmanager.plot_selected_vertices()

    @on_trait_change('mask_target')
    def mask_target(self):
        if self.mask_target:
            self.handler.set_state('mask_target')
            self.meshmanager.on_vertex_selection(self.set_mask)
        else:
            self.meshmanager.select_vertices_off()
            self.settings.curvature_mask = None

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
        self.settings.geometric = self.geometric
        self.settings.curvature_ratio = self.eq_curv_ratio
        self.settings.target_curvature_ratio = self.curvature_ratio
        self.settings.min_abs_curvature = self.abs_curvature
        self.settings.step_control = self.step_control
        self.settings.linear_abs_curvature_constraint = self.linear_ratio
        self.settings.willmore_energy = self.willmore
        self.settings.curvature_energy = self.curvature



    # -------------------------------------------------------------------------
    #                                  Reset
    # -------------------------------------------------------------------------

    @on_trait_change('fix_boundary_normals')
    def fix_boundary_normals(self):
        if self.fix_boundary_normals:
            self.mesh.fix_double_boundary()
        else:
            self.mesh.reset_fixed()
        self.meshmanager.update_plot()

    @on_trait_change('curvature_ratio')
    def curvature_ratio_update(self):
        self.meshmanager.update_plot()

    @on_trait_change('comb_directions')
    def comb_directions(self):
        if self.comb_directions:
            def callback(v):
                self.settings.comb_curvature_directions(v)
                self.meshmanager.select_vertices_off()
                self.comb_directions = False
            self.meshmanager.on_vertex_selection(callback)
        self.meshmanager.update_plot()

    @on_trait_change('reinitialize')
    def reinitialize_optimizer(self):
        self.set_settings()
        self.optimizer.reinitialize()
        self.print_error()
        self.meshmanager.update_plot()

    def plot_curvature_ratio(self):
        ref = self.curvature_ratio
        vertex_data = self.mesh.curvature_ratios()
        delta = 0.1
        lut_range = [ref - delta, ref + delta]
        self.meshmanager.plot_faces(vertex_data = vertex_data,
                                   color = 'blue-red',
                                   lut_range = lut_range)

    @on_trait_change('show_curvature_directions')
    def plot_curvature_directions(self):
        try:
            V1, V2 = self.optimizer.curvature_directions()
            if self.show_curvature_directions and V1 is not None:
                self.meshmanager.plot_vectors(vectors=V1,
                                 position = 'center',
                                 glyph_type = 'line',
                                 line_width = 2,
                                 color = 'b',
                                 name = 'curvature-d1')
                self.meshmanager.plot_vectors(vectors=V2,
                                 position = 'center',
                                 glyph_type = 'line',
                                 line_width = 2,
                                 color = 'r',
                                 name = 'curvature-d2')
            else:
                self.meshmanager.hide('curvature-d1')
                self.meshmanager.hide('curvature-d2')
        except:
            self.meshmanager.hide('curvature-d1')
            self.meshmanager.hide('curvature-d2')

    @on_trait_change('add_noise')
    def add_noise(self):
        meshutilities.add_random_noise(self.mesh, self.noise_factor)
        self.meshmanager.update_plot()

