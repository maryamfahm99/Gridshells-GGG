#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division


'''_'''

__author__ = 'Davide Pellis'

#------------------------------------------------------------------------------

from traits.api import HasTraits, Instance, Property, Enum, Button,String,\
                       on_trait_change, Float, Bool, Int, Array, Range

from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor, HGroup,\
                         Group, Tabbed, VGroup, ArrayEditor

#from pyface.image_resource import ImageResource

import numpy as np

#------------------------------------------------------------------------------

from geometrylab.optimization.scaligner import StressCurvatureAligner

from geometrylab.vtkplot.vectorsource import Vectors

from geometrylab.vtkplot.polylinesource import Polyline


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                     InteractiveStressCurvatureAligner
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class InteractiveStressCurvatureAligner(HasTraits):

    name = String('SCAligner')

    iterations = Int(1)

    epsilon = Float(0.01, label='dumping')

    step = Float(1)

    fairness_reduction = Float(0)

    geometric = Float(1)

    curvature = Float(1)

    stress = Float(1)

    equilibrium = Float(1)

    stress_alignment = Float(1)

    curvat_alignment = Float(1)

    mesh_fairness = Float(0.1)

    tangential_fairness = Float(0.1)

    boundary_fairness = Float(0.1)

    reference_closeness = Float(0.0)

    reinitialize = Button(label='Initialize')

    plot_forces = Bool(True, label='forces')

    gliding = Bool(True)

    fix_corners = Bool(False)

    reset = Button(label='Reset')

    align = Button(label='Align')

    equilibrium_error = String('_')

    geometric_error = String('_')

    stress_error = String('_')

    curvature_error = String('_')

    plot_dir = Bool(True, label='directions')

    plot_stress_dir = Bool(True, label='stress')

    plot_curvature_dir = Bool(True, label='curvature')

    plot_faces = Bool(False, label='mesh')

    scale = Float(1)

    interactive = Bool(False,label='Interactive')

    #--------------------------------------------------------------------------
    view = View(
                VSplit(
                       VGroup('iterations',
                             'epsilon',
                             'step',
                             'mesh_fairness',
                             'tangential_fairness',
                             'boundary_fairness',
                             'fairness_reduction',
                             'geometric',
                             'curvature',
                             'stress',
                             'equilibrium',
                             'stress_alignment',
                             'curvat_alignment',
                             'reference_closeness',
                             show_border=True),
                       Group(HGroup(
                                    'plot_dir',
                                    'plot_stress_dir',
                                    'plot_curvature_dir',),
                             HGroup(
                                    'scale',
                                    'plot_faces',),
                             show_border=True),
                       Group('geometric_error',
                             'equilibrium_error',
                             'stress_error',
                             'curvature_error',
                             style='readonly',
                             show_border=True),
                       HGroup('align',
                              'reinitialize',
                              'reset',
                              show_labels=False,
                              show_border=True),
                       HGroup(Item('interactive'),
                              Item('_'),
                              show_border=True),
                       show_border=False),
               resizable=False,
               width = 0.1,
               )

    #--------------------------------------------------------------------------

    def __init__(self, scene=None):
        HasTraits.__init__(self)

        self.optimizer = StressCurvatureAligner()

        self.selected = []

        self._mesh = None

        self._settings = None

        self._meshmanager = None

        self._handler = None

        self._movie_manager = None

    #--------------------------------------------------------------------------
    #                            Set attributes
    #--------------------------------------------------------------------------

    @property
    def mesh(self):
        return self._mesh

    @property
    def settings(self):
        return self._settings

    @property
    def meshmanager(self):
        return self._meshmanager

    @property
    def handler(self):
        return self._handler

    @property
    def movie_manager(self):
        return self._movie_manager

    @meshmanager.setter
    def meshmanager(self, plot_manager):
        self._meshmanager = plot_manager
        self._mesh = self._meshmanager.mesh
        self._settings = self.optimizer.settings
        self.optimizer.mesh = self.mesh

    @handler.setter
    def handler(self, handler):
        handler.add(self)
        self._handler = handler

    @movie_manager.setter
    def movie_manager(self, movie_manager):
        self._movie_manager = movie_manager
        #self._movie_manager.plot_callback = self.plot_results

    #--------------------------------------------------------------------------

    def _set_state(self, state):
        if state != 'sca_interactive':
            self.interactive = False

    def _initialize_plot(self):
        pass

    def _update_plot(self):
        print("update_plot 5")
        self.meshmanager.remove(['d1','d2','d3','d4','d5','d6'])

    #--------------------------------------------------------------------------

    def plot_results(self):
        self.handler._update_plot()
        self.plot_axial_force()
        self.plot_faces()
        self.plot_directions()
        self.plot_curvature_directions()
        self.plot_stress_directions()

    def optimization_step(self):
        if not self.interactive:
            self.handler._set_state(None)
        self.set_settings()
        self.optimizer.align()
        self.print_error()
        self.plot_results()

    def print_error(self):
        error = self.optimizer.global_error()
        self.geometric_error = error[0]
        self.equilibrium_error = error[1]
        self.stress_error = error[2]
        self.curvature_error = error[3]

    def set_settings(self):
        if self.mesh.last_iter != 'alignment':
            self.optimizer.reinitialize()
            self.mesh.last_iter = 'alignment'
        if self.mesh.reinitialize:
            self.optimizer.reinitialize()
            self.mesh.reinitialize = False
        self.settings.reference_mesh = self.mesh.reference_mesh
        self.settings.epsilon = self.epsilon
        self.settings.step = self.step
        self.settings.fairness_reduction = self.fairness_reduction
        self.settings.geometric = self.geometric
        self.settings.curvature = self.curvature
        self.settings.equilibrium = self.equilibrium
        self.settings.stress = self.stress
        self.settings.stress_alignment = self.stress_alignment
        self.settings.curvature_alignment = self.curvat_alignment
        self.settings.area_rho = 1
        self.settings.beam_rho = 0
        self.settings.mesh_fairness = self.mesh_fairness
        self.settings.tangential_fairness = self.tangential_fairness
        self.settings.boundary_fairness = self.boundary_fairness
        self.settings.reference_closeness = self.reference_closeness
        self.settings.glid_constrained = self.gliding
        self.settings.boundary_interpolation_N = 8
        self.settings.constrained_vertices = np.copy(
                       self.mesh.constrained_vertices)
        fix = np.copy(self.mesh.fixed_vertices)
        fix = np.unique(np.hstack((fix,np.array(self.selected))))
        self.settings.fixed_vertices = fix
        if self.fix_corners:
            corners = self.mesh.mesh_corners()
            fix = np.hstack((self.settings.fixed_vertices,corners))
            self.settings.fixed_vertices = np.array(np.unique(fix),dtype=np.int)
        if self.gliding:
            self.settings.gliding_vertices = np.copy(
                       self.mesh.constrained_vertices)
        else:
            fix = np.hstack((self.settings.fixed_vertices,
                             self.mesh.constrained_vertices))
            self.settings.fixed_vertices = np.array(np.unique(fix),dtype=np.int)
            self.settings.gliding_vertices = np.array([])

    #--------------------------------------------------------------------------

    @on_trait_change('reinitialize')
    def initialize_x(self):
        self.meshmanager.remove_edges()
        self.set_settings()
        self.optimizer.reinitialize()
        self.optimizer.initialize()
        self.print_error()
        self.plot_results()
        self.mesh.last_iter = 'alignment'
        self.mesh.reinitialize = False

    @on_trait_change('reset')
    def reset_mesh(self):
        self.mesh.reset()
        self.handler._initialize_plot()
        self.handler._set_state(None)
        self.set_settings()
        self.optimizer.reinitialize()
        self.print_error()
        self.plot_results()
        self.mesh.last_iter = 'alignment'
        self.mesh.reinitialize = False

    #--------------------------------------------------------------------------

    @on_trait_change('align')
    def align_mesh(self):
        self.meshmanager.iterate(self.optimization_step,self.iterations)

    @on_trait_change('interactive')
    def interactive_align_mesh(self):
        self.handler._set_state('sca_interactive')
        if self.interactive:
            def start():
                self.selected = self.meshmanager.selected_vertices
            def interact():
                self.meshmanager.iterate(self.optimization_step,1)
            def end():
                self.meshmanager.iterate(self.optimization_step,5)
            self.meshmanager.move_vertices(interact,start,end)
        else:
            self.selected = []
            self.meshmanager.move_vertices_off()

    #--------------------------------------------------------------------------

    @on_trait_change('plot_dir')
    def plot_directions(self):
        self.meshmanager.remove(['d1','d2'])
        if self.plot_dir:
            r = self.meshmanager.r
            factor = r * 6 * (self.mesh.F)**0.25
            D1,D2 = self.optimizer.return_directions()
            V1 = Vectors(D1, self.mesh,
                         position = 'center',
                         glyph_type = 'line',
                         scale_factor = factor*self.scale,
                         line_width = 2,
                         color = 'r',
                         name = 'd1')
            V2 = Vectors(D2, self.mesh,
                         position = 'center',
                         glyph_type = 'line',
                         scale_factor = factor*self.scale,
                         line_width = 2,
                         color = 'r',
                         name = 'd2')
            self.meshmanager.add([V1,V2])

    @on_trait_change('plot_curvature_dir')
    def plot_curvature_directions(self):
        self.meshmanager.remove(['d3','d4'])
        if self.plot_curvature_dir:
            r = self.meshmanager.r
            factor = r * 6 * (self.mesh.F)**0.25
            D1,D2 = self.optimizer.return_curvature_directions()
            V1 = Vectors(D1, self.mesh,
                         position = 'center',
                         glyph_type = 'line',
                         scale_factor = factor*self.scale,
                         line_width = 2,
                         color = 'b',
                         name = 'd3')
            V2 = Vectors(D2, self.mesh,
                         position = 'center',
                         glyph_type = 'line',
                         scale_factor = factor*self.scale,
                         line_width = 2,
                         color = 'b',
                         name = 'd4')
            self.meshmanager.add([V1,V2])

    @on_trait_change('plot_stress_dir')
    def plot_stress_directions(self):
        self.meshmanager.remove(['d5','d6'])
        if self.plot_stress_dir:
            r = self.meshmanager.r
            factor = r * 6 * (self.mesh.F)**0.25
            D1,D2 = self.optimizer.return_stress_directions()
            V1 = Vectors(D1, self.mesh,
                         position='center',
                         glyph_type = 'line',
                         scale_factor = factor*self.scale,
                         line_width = 2,
                         color = 'g',
                         name = 'd5')
            V2 = Vectors(D2, self.mesh,
                         position='center',
                         glyph_type = 'line',
                         scale_factor = factor*self.scale,
                         line_width = 2,
                         color = 'g',
                         name = 'd6')
            self.meshmanager.add([V1,V2])

    @on_trait_change('plot_faces')
    def plot_faces(self):
        self.meshmanager.remove('faces')
        if self.plot_faces:
            self.meshmanager.plot_faces()

    @on_trait_change('plot_forces')
    def plot_axial_force(self):
        if self.plot_forces:
            N = self.optimizer.return_forces()
        else:
            N = np.zeros(self.mesh.E)
        self.meshmanager.plot_edges(edge_data=-N,
                                   color='bwr_e',
                                   lut_range='-:0:+')

