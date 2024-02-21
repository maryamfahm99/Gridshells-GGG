#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

from tvtk.api import tvtk

from mayavi.sources.vtk_data_source import VTKDataSource

from mayavi.modules.glyph import Glyph

import numpy as np

# -----------------------------------------------------------------------------

from geometrylab.vtkplot import plotutilities

# -----------------------------------------------------------------------------

'''pointsource.py: Point plot source class, for meshes and arrays'''

__author__ = 'Davide Pellis'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                                    Axes
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class Axes(object):

    def __init__(self, **kwargs):

        self._basis      = kwargs.get('basis',  np.eye(3))

        self._origin     = kwargs.get('origin', np.zeros(self._basis.shape[1]))

        self._x_range     = kwargs.get('x_range', [-10, 10])

        self._y_range     = kwargs.get('y_range', [-10, 10])

        self._z_range     = kwargs.get('z_range', [-10, 10])

        self._color      = kwargs.get('color', 'white')

        self._opacity    = kwargs.get('opacity', 1)

        self._line_width  = kwargs.get('line_width', 1)

        self.name        = kwargs.get('name', 'axes')

        self._sources = {}

        self._make_sources()


    @property
    def type(self):
        return 'Axes-source'

    #--------------------------------------------------------------------------
    #                               Data Structure
    #--------------------------------------------------------------------------

    def make_sources(self):
        pass

    #--------------------------------------------------------------------------
    #                                Surface
    #--------------------------------------------------------------------------

    def make_surface(self):
        self.surface = Surface()
        if self.line_width != None:
            self.surface.actor.property.line_width = self.line_width
        if type(self.opacity) == int or type(self.opacity) == float:
            self.surface.actor.property.opacity = self.opacity
        if not self.shading:
            self.surface.actor.actor.property.lighting = False
        if self.glossy != 0:
            self.surface.actor.actor.property.specular = 0.7 * self.glossy
            self.surface.actor.actor.property.specular_power = 11 * self.glossy

    #--------------------------------------------------------------------------
    #                                 Tube
    #--------------------------------------------------------------------------

    def make_tube(self):
        self.tube = None
        if self.tube_radius == 'adaptive':
            self.surface.actor.actor.property.representation = 'wireframe'
            self.surface.actor.actor.property.render_lines_as_tubes = True
        elif self.tube_radius is not None:
            self.tube = Tube()
            self.tube.filter.radius = self.tube_radius
            self.tube.filter.number_of_sides = self.tube_sides
            self.tube

    #--------------------------------------------------------------------------
    #                               Pipeline
    #--------------------------------------------------------------------------

    def assemble_pipeline(self):
        src = VTKDataSource(data=self.data)
        self.module.add_child(self.surface)
        if self.tube is None:
            src.add_child(self.module)
        else:
            self.tube.add_child(self.module)
            src.add_child(self.tube)
        self.source = src