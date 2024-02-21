# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:47:19 2019

@author: Davide
"""

import os

import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(path)

import geometrylab as geo