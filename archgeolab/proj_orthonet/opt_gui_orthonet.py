# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 22:16:21 2022

@author: WANGH0M
"""
__author__ = 'Hui'
#------------------------------------------------------------------------------
import os

import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(path)
#print(path)

# from traits.api import Button,String,on_trait_change, Float, Bool, Range,Int

# from traitsui.api import View, Item, HGroup, Group, VGroup

import numpy as np
#------------------------------------------------------------------------------


from geometrylab.gui.geolabcomponent import GeolabComponent
# from geometrylab.vtkplot.edgesource import Edges
# from geometrylab.vtkplot.facesource import Faces
from geometrylab.geometry import Polyline

from guidedprojection_orthonet import GP_OrthoNet
from archgeolab.archgeometry.conicSection import get_sphere_packing,\
    get_vs_interpolated_sphere
#from archgeolab.archgeometry.Multinets_Mesh import Get_Multinets_Mesh
#------------------------------------------------------------------------------

''' build:  
    show:   
'''

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                        InteractiveGuidedProjection
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class OrthoNet(GeolabComponent):

    # name = String('Orthogonal-Net')
    
    # itera_run = Int(5)

    # epsilon = Float(0.001, label='dumping')

    # step = Float(1)
    # # change step

    # fairness_reduction = Range(low=0,high=5,value=0,label='F-reduce')

    # mesh_fairness = Float(0.0000,label='meshF')

    # GGG_weight = Float(0.0000,label='GGG_w') # Maryam 
    # vertex_control_weight = Float(0.0000,label='vcntr_w') # Maryam 

    # tangential_fairness = Float(0.0000,label='tangF')

    # boundary_fairness = Float(0.0000,label='bondF')

    # spring_fairness = Float(0.0000,label='springF')
    
    # corner_fairness = Float(0,label='cornerF')
    
    # fairness_diagmesh = Float(0,label='diagF')

    # reference_closeness = Float(0,label='refC')
    
    # fairness_4diff = Float(0,label='Fair4diff')
    # fairness_diag_4diff = Float(0,label='FairDiag4diff')

    # boundary_glide = Float(0,label='bdryGlide') ##weight for all
    # i_boundary_glide = Float(0,label='iBdryGlide') ##weight for i-th
    # glide_1st_bdry = Bool(label='1st')
    # glide_2nd_bdry = Bool(label='2nd')
    # glide_3rd_bdry = Bool(label='3rd')
    # glide_4th_bdry = Bool(label='4th')
    # glide_5th_bdry = Bool(label='5th')
    # glide_6th_bdry = Bool(label='6th')
    # glide_7th_bdry = Bool(label='7th')
    # glide_8th_bdry = Bool(label='8th')

    # sharp_corner = Bool(label='SharpCor')
    
    # self_closeness = Float(0,label='selfC')

    # avoid_shrinkage = Float(0,label='NoSmaller')
    
    # set_refer_mesh = Bool(label='SetRefer')
    # show_refer_mesh = Bool(label='ShowRefer')
    # show_ref_mesh_boundary = Bool(label='ShowReferBdry')
    
    # fair0 = Button(label='0')
    # fair1 = Button(label='0.1')
    # fair01 = Button(label='0.01')
    # fair001 = Button(label='0.001')
    # fair0001 = Button(label='0.0001')

    # close0 = Button(label='0')
    # close005 = Button(label='0.005')
    # close01 = Button(label='0.01')
    # close05 = Button(label='0.05')
    # close1 = Button(label='0.1')
    # close5 = Button(label='0.5')

    # weight_fix = Float(10)
    # fix_all = Bool(label='Fix')
    # fix_boundary = Bool(label='FixB')
    # fix_boundary_i = Bool(label='FixBi')
    # fix_corner = Bool(label='FixC')
    # fix_p_weight = Float(0,label='Fix_p')
    # fix_button = Button(label='Fix')
    # unfix_button = Button(label='Unfix')
    # clearfix_button = Button(label='Clear')

    # boundary_z0 = Bool(label='BZ0')
    # selected_z0 = Bool(label='S_Z0')
    # selected_y0 = Bool(label='S_Y0')
    # z0 = Float(0)#Bool(label='Z=0')
    
    # reinitialize = Button(label='ini')
    # optimize = Button(label='Opt')
    # mesh_propogate = Button(label='Mesh Propogate') # Maryam
    # reset_mesh_propogation = Button(label='Reset') # Maryam
    # interactive = Bool(False, label='Interactive')
    # hide_face = Bool(label='HideF')
    # hide_edge = Bool(label='HideE')    
    # ####----------------------------------------------------------------------- 
    # #--------------Optimization: -----------------------------
    # button_clear_constraint = Button(label='Clear')

    # orthogonal = Bool(label='Orthogonal')
    
    # button_minimal_mesh = Button(label='Minimal')
    # Anet = Bool(0)  
    # Anet_diagnet = Bool(label='AnetDiag')

    # AAG = Bool(0) # Maryam
    # AGG = Bool(0) # Maryam
    # GGG = Bool(0) # Maryam

    # button_CMC_mesh = Button(label='CMC')
    # Snet = Bool(label='Snet')
    # Snet_diagnet = Bool(label='SnetDiag')
    # Snet_orient = Bool(True,label='Orient') ##only under Snet/Snet_diagnet
    # Snet_constR = Bool(False,label='constR') ##only under Snet/Snet_diagnet
    # if_uniqR = Bool(False) 
    # Snet_constR_assigned = Float(label='const.R')

    # button_principal_mesh = Button(label='PrincipalMesh')
    # planarity = Bool(label='PQ')
    # circular_mesh = Bool(label='CircularM')
    # conical_mesh = Bool(label='ConicalM') #TODO
    
    # button_opt_PPO = Button(label='OrthoPP')
    # set_another_poly = Range(low=0, high=1, value=0,label='_1st|2nd_Poly')
    # opt_planar_polyline1 = Bool(label='PlanarPly1')
    # opt_planar_polyline2 = Bool(label='PlanarPly2')
    
    # button_funicularity = Button(label='OrthoFunicular')
    # equilibrium = Bool(label='Equilibrium') #equilibrium with vertical load
    
    # #button_principal_stress = Button(label='PrincipalStress') #TODO
    # button_Multinets_Orthogonal = Button(label='Multinets-Orthogonal')
    # multinets_orthogonal = Bool(label='multinets-orthogonal')
    # # if_set_weight = Bool(False)
    # weigth_multinets_orthogonal = Float(label='weight.MO')
    
    # button_ortho_planarPolyline_funicularity = Button(label='O+P+F')
    
    # #--------------Plotting: -----------------------------
    # show_isogonal_face_based_vector = Bool(label='F-Vec')
    # show_midpoint_edge1 = Bool(label='E1')
    # show_midpoint_edge2 = Bool(label='E2')
    # show_midpoint_polyline1 = Bool(label='Ply1')
    # show_midpoint_polyline2 = Bool(label='Ply2')
    # show_quad_diagonal1 = Bool(label='Quad-Diagonal1') #Maryam
    # show_quad_diagonal2 = Bool(label='Quad-Diagonal2') #Maryam
    # show_vertex_normals = Bool(label='normals') #Maryam
    # numVperPoly = 0
    # show_midline_mesh = Bool(label='ReMesh')

    # show_planar_poly1_normal = Bool(label='Ply1-N')
    # show_planar_poly1_plane= Bool(label='Ply1-Pln')
    # show_planar_poly2_normal = Bool(label='Ply2-N')
    # show_planar_poly2_plane = Bool(label='Ply2-Pln')   

    # show_vs_sphere = Bool(label='VS-Sphere')
    # show_snet_center = Bool(label='Snet-C')
    # show_snet_normal = Bool(label='Snet-N')
    # show_snet_tangent = Bool(label='Snet-T')

    # show_multinets_diagonals = Bool(label='Multinets-Diagonals') 
    
    # show_circumcircle = Bool(label='Circumcircle')
    
    # print_error = Button(label='Error')
    # #--------------Save: --------------------
    # save_button = Button(label='Save')
    # label = String('obj')
    # save_new_mesh = None
    
    # #--------------Print: -----------------------------
    # print_orthogonal = Button(label='Check')
    # print_computation = Button(label='Computation')
    #--------------------------------------------------------------------------
   
    # -------------------------------------------------------------------------
    #                                Initialize
    # -------------------------------------------------------------------------

    def __init__(self):
        GeolabComponent.__init__(self)

        self.optimizer = GP_OrthoNet()
        
        self.counter = 0
        
        self._fixed_vertex = []
        
        self.ref_glide_bdry_polyline = None

        self.snet_normal = self.snet_diagG_binormal = None
    # -------------------------------------------------------------------------
    #                                Properties
    # -------------------------------------------------------------------------
    @property
    def mesh(self):
        return self.geolab.current_object.geometry
    
    @property
    def meshmanager(self):
        print("meshmanager")
        return self.geolab.current_object
    
    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------
    
 

   

    def optimization_step(self):
        # if not self.interactive:
        #     self.handler.set_state(None)
        # self.set_settings()
        self.optimizer.optimize()
        # print("Opt Step")
        # #self.print_error()
        # # self.updating_plot()
        # print("before mesh manager in opt")
        # # self.meshmanager.update_plot()
        # # if self.show_quad_diagonal1:
        # #     name = 'quad_diagonals'
        # #     self.meshmanager.remove([name+'1', name+'2'])
        # #     self.plot_diagonals_polylines()
        # print("after mesh manager in opt")
        # if self.fairness_reduction !=0:
        #     self.mesh_fairness = self.mesh_fairness/(10**(self.fairness_reduction))
        #     self.tangential_fairness = self.tangential_fairness/(10**
        #                                 (self.fairness_reduction))
        #     self.boundary_fairness = self.boundary_fairness/(10**
        #                             (self.fairness_reduction))
        #     self.spring_fairness = self.spring_fairness/(10**
        #                             (self.fairness_reduction)) 
        # print("faces in opt")

    def optimize_mesh(self):
        
        itera = 1
        self.meshmanager.iterate(self.optimization_step, itera) # note:iterations from gpbase.py
        # self.meshmanager.update_plot()
