#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import os

import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(path)

import numpy as np

import geometrylab as geo


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                               COMPONENT TEST
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                     File selection from the file folder
#------------------------------------------------------------------------------

path = os.path.dirname(os.path.abspath(__file__))

file_name = path + '/quad_dome.obj'

#------------------------------------------------------------------------------
#                     Main code to run the application
#------------------------------------------------------------------------------

if __name__ == '__main__':

    M = geo.geometry.Mesh(file_name)
    f = geo.intersect.closest_mesh_face_ray_intersection(M, [8,8.1,0], [0,0,1])
    print(f)



