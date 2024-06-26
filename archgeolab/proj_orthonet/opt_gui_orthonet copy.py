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

from traits.api import Button,String,on_trait_change, Float, Bool, Range,Int

from traitsui.api import View, Item, HGroup, Group, VGroup

import numpy as np
#------------------------------------------------------------------------------


from geometrylab.gui.geolabcomponent import GeolabComponent
from geometrylab.vtkplot.edgesource import Edges
from geometrylab.vtkplot.facesource import Faces
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

    name = String('Orthogonal-Net')
    
    itera_run = Int(5)

    epsilon = Float(0.001, label='dumping')

    step = Float(1)
    # change step

    fairness_reduction = Range(low=0,high=5,value=0,label='F-reduce')

    mesh_fairness = Float(0.0000,label='meshF')

    GGG_weight = Float(0.0000,label='GGG_w') # Maryam 
    vertex_control_weight = Float(0.0000,label='vcntr_w') # Maryam 

    tangential_fairness = Float(0.0000,label='tangF')

    boundary_fairness = Float(0.0000,label='bondF')

    spring_fairness = Float(0.0000,label='springF')
    
    corner_fairness = Float(0,label='cornerF')
    
    fairness_diagmesh = Float(0,label='diagF')

    reference_closeness = Float(0,label='refC')
    
    fairness_4diff = Float(0,label='Fair4diff')
    fairness_diag_4diff = Float(0,label='FairDiag4diff')

    boundary_glide = Float(0,label='bdryGlide') ##weight for all
    i_boundary_glide = Float(0,label='iBdryGlide') ##weight for i-th
    glide_1st_bdry = Bool(label='1st')
    glide_2nd_bdry = Bool(label='2nd')
    glide_3rd_bdry = Bool(label='3rd')
    glide_4th_bdry = Bool(label='4th')
    glide_5th_bdry = Bool(label='5th')
    glide_6th_bdry = Bool(label='6th')
    glide_7th_bdry = Bool(label='7th')
    glide_8th_bdry = Bool(label='8th')

    sharp_corner = Bool(label='SharpCor')
    
    self_closeness = Float(0,label='selfC')

    avoid_shrinkage = Float(0,label='NoSmaller')
    
    set_refer_mesh = Bool(label='SetRefer')
    show_refer_mesh = Bool(label='ShowRefer')
    show_ref_mesh_boundary = Bool(label='ShowReferBdry')
    
    fair0 = Button(label='0')
    fair1 = Button(label='0.1')
    fair01 = Button(label='0.01')
    fair001 = Button(label='0.001')
    fair0001 = Button(label='0.0001')

    close0 = Button(label='0')
    close005 = Button(label='0.005')
    close01 = Button(label='0.01')
    close05 = Button(label='0.05')
    close1 = Button(label='0.1')
    close5 = Button(label='0.5')

    weight_fix = Float(10)
    fix_all = Bool(label='Fix')
    fix_boundary = Bool(label='FixB')
    fix_boundary_i = Bool(label='FixBi')
    fix_corner = Bool(label='FixC')
    fix_p_weight = Float(0,label='Fix_p')
    fix_button = Button(label='Fix')
    unfix_button = Button(label='Unfix')
    clearfix_button = Button(label='Clear')

    boundary_z0 = Bool(label='BZ0')
    selected_z0 = Bool(label='S_Z0')
    selected_y0 = Bool(label='S_Y0')
    z0 = Float(0)#Bool(label='Z=0')
    
    reinitialize = Button(label='ini')
    optimize = Button(label='Opt')
    mesh_propogate = Button(label='Mesh Propogate') # Maryam
    reset_mesh_propogation = Button(label='Reset') # Maryam
    interactive = Bool(False, label='Interactive')
    hide_face = Bool(label='HideF')
    hide_edge = Bool(label='HideE')    
    ####----------------------------------------------------------------------- 
    #--------------Optimization: -----------------------------
    button_clear_constraint = Button(label='Clear')

    orthogonal = Bool(label='Orthogonal')
    
    button_minimal_mesh = Button(label='Minimal')
    Anet = Bool(0)  
    Anet_diagnet = Bool(label='AnetDiag')

    AAG = Bool(0) # Maryam
    AGG = Bool(0) # Maryam
    GGG = Bool(0) # Maryam

    button_CMC_mesh = Button(label='CMC')
    Snet = Bool(label='Snet')
    Snet_diagnet = Bool(label='SnetDiag')
    Snet_orient = Bool(True,label='Orient') ##only under Snet/Snet_diagnet
    Snet_constR = Bool(False,label='constR') ##only under Snet/Snet_diagnet
    if_uniqR = Bool(False) 
    Snet_constR_assigned = Float(label='const.R')

    button_principal_mesh = Button(label='PrincipalMesh')
    planarity = Bool(label='PQ')
    circular_mesh = Bool(label='CircularM')
    conical_mesh = Bool(label='ConicalM') #TODO
    
    button_opt_PPO = Button(label='OrthoPP')
    set_another_poly = Range(low=0, high=1, value=0,label='_1st|2nd_Poly')
    opt_planar_polyline1 = Bool(label='PlanarPly1')
    opt_planar_polyline2 = Bool(label='PlanarPly2')
    
    button_funicularity = Button(label='OrthoFunicular')
    equilibrium = Bool(label='Equilibrium') #equilibrium with vertical load
    
    #button_principal_stress = Button(label='PrincipalStress') #TODO
    button_Multinets_Orthogonal = Button(label='Multinets-Orthogonal')
    multinets_orthogonal = Bool(label='multinets-orthogonal')
    # if_set_weight = Bool(False)
    weigth_multinets_orthogonal = Float(label='weight.MO')
    
    button_ortho_planarPolyline_funicularity = Button(label='O+P+F')
    
    #--------------Plotting: -----------------------------
    show_isogonal_face_based_vector = Bool(label='F-Vec')
    show_midpoint_edge1 = Bool(label='E1')
    show_midpoint_edge2 = Bool(label='E2')
    show_midpoint_polyline1 = Bool(label='Ply1')
    show_midpoint_polyline2 = Bool(label='Ply2')
    show_quad_diagonal1 = Bool(label='Quad-Diagonal1') #Maryam
    show_quad_diagonal2 = Bool(label='Quad-Diagonal2') #Maryam
    show_vertex_normals = Bool(label='normals') #Maryam
    numVperPoly = 0
    show_midline_mesh = Bool(label='ReMesh')

    show_planar_poly1_normal = Bool(label='Ply1-N')
    show_planar_poly1_plane= Bool(label='Ply1-Pln')
    show_planar_poly2_normal = Bool(label='Ply2-N')
    show_planar_poly2_plane = Bool(label='Ply2-Pln')   

    show_vs_sphere = Bool(label='VS-Sphere')
    show_snet_center = Bool(label='Snet-C')
    show_snet_normal = Bool(label='Snet-N')
    show_snet_tangent = Bool(label='Snet-T')

    show_multinets_diagonals = Bool(label='Multinets-Diagonals') 
    
    show_circumcircle = Bool(label='Circumcircle')
    
    print_error = Button(label='Error')
    #--------------Save: --------------------
    save_button = Button(label='Save')
    label = String('obj')
    save_new_mesh = None
    
    #--------------Print: -----------------------------
    print_orthogonal = Button(label='Check')
    print_computation = Button(label='Computation')
    #--------------------------------------------------------------------------
    view = View(VGroup(Group(
    #---------------------------------------------------------
    Group(## 1st-panel
        VGroup(
              #HGroup(Item('set_another_poly'),),    
              HGroup('orthogonal',),
              HGroup('planarity',
                     'circular_mesh',
                     'conical_mesh',
                     ),
              HGroup('opt_planar_polyline1',
                     'opt_planar_polyline2',),
              HGroup('Anet',
                     'Anet_diagnet',
                     'AAG', 'AGG', 'GGG', 'GGG_weight','vertex_control_weight',), # Maryam 
              HGroup('Snet',
                     'Snet_diagnet',
                     #'Snet_orient',
                     'Snet_constR',
                     Item('if_uniqR',show_label=False),
                     'Snet_constR_assigned'),
              HGroup('equilibrium',
                     'multinets_orthogonal',
                     'weigth_multinets_orthogonal'),
              HGroup(Item('button_principal_mesh',show_label=False),
                     Item('button_opt_PPO',show_label=False),
                     Item('button_minimal_mesh',show_label=False),
                     Item('button_CMC_mesh',show_label=False),
                     ),
              HGroup(Item('button_funicularity',show_label=False),
                     Item('button_ortho_planarPolyline_funicularity',show_label=False),
                     Item('button_Multinets_Orthogonal',show_label=False),
                     Item('button_clear_constraint',show_label=False)), 
        label='Opt',show_border=True),
        #------------------------------------------------  
        VGroup(HGroup('show_isogonal_face_based_vector',
                      'show_midpoint_edge1',
                      'show_midpoint_edge2',
                      'show_midpoint_polyline1',
                      'show_midpoint_polyline2',
                      'show_midline_mesh'),
              HGroup('show_planar_poly1_normal',
                     'show_planar_poly2_normal',
                     'show_planar_poly1_plane',
                     'show_planar_poly2_plane',
                     'show_quad_diagonal1',
                     'show_quad_diagonal2',
                     'show_vertex_normals'),
              HGroup('show_vs_sphere',
                     'show_snet_center',
                     'show_snet_tangent',
                     'show_snet_normal',),
              HGroup('show_circumcircle'),
              HGroup('show_multinets_diagonals'),
        label='Plotting',show_border=True),
        #------------------------------------------------  
        VGroup(HGroup(Item('print_error',show_label=False)),),
        
        #------------------------------------------------  
        label='Opt',show_border=True),
    #---------------------------------------------------------
    #---------------------------------------------------------
    Group(## 2nd-panel
        HGroup('itera_run',#'epsilon','step'
               ),
        # VGroup(HGroup(Item('fair1',show_label=False),
        #               Item('fair01',show_label=False),
        #               Item('fair001',show_label=False),
        #               Item('fair0001',show_label=False),
        #               Item('fair0',show_label=False)),
                HGroup('mesh_fairness',
                       'boundary_fairness',
                       'corner_fairness', 'avoid_shrinkage'),
                HGroup('tangential_fairness',
                       'spring_fairness'),
        #        HGroup('fairness_4diff',
        #               'fairness_diag_4diff'),
        #               'fairness_reduction',
        #               #'avoid_shrinkage',
        #               'fairness_diagmesh',
        #        show_border=True,label='Fairness'),
        
        # VGroup(HGroup(Item('close5',show_label=False),
        #               Item('close1',show_label=False),
        #               Item('close05',show_label=False),
        #               Item('close01',show_label=False),
        #               Item('close0',show_label=False)),
                HGroup('self_closeness',
                       'reference_closeness'),
        #               'boundary_glide',
        #        HGroup('set_refer_mesh',
        #               'show_refer_mesh',
        #               'show_ref_mesh_boundary',),
        #        HGroup('i_boundary_glide',
        #               'glide_1st_bdry',
        #               'glide_2nd_bdry',
        #               'glide_3rd_bdry',
        #               'glide_4th_bdry',
        #               'glide_5th_bdry',
        #               'glide_6th_bdry',
        #               'glide_7th_bdry',
        #               'glide_8th_bdry',),
        #        HGroup('sharp_corner'),
        #        show_border=True,label='Closeness'),
                
         VGroup(HGroup('weight_fix'),
                HGroup(#'hide_face','hide_edge',
                       'fix_all',
                       'fix_boundary',
                       'fix_boundary_i',
                       'fix_corner',),
                HGroup('boundary_z0',
                       'z0'),
                HGroup('selected_z0','selected_y0',
                       Item('fix_p_weight',show_label=False),),
                HGroup(Item('fix_button',show_label=False),
                       Item('unfix_button',show_label=False),
                       Item('clearfix_button',show_label=False)),    
             show_border=True,label='select'),   
                                      
          ###-------------------------------------
           VGroup(VGroup(
                   HGroup(Item('print_orthogonal',show_label=False),
                          Item('print_computation',show_label=False),),
                       ),
                  label='Check',
                  show_border=True),
           #--------------------------------------
          HGroup('label',
                 Item('save_button',show_label=False),
                 label='Saving',show_border=True),
        HGroup(Item('interactive',
                         tooltip='Interactive',
                         show_label=False,),
                    Item('_'),
                    'optimize','reinitialize',
                    'hide_face','hide_edge',
                    'mesh_propogate',  'reset_mesh_propogation', # Maryam 
                    show_labels=False,
                    show_border=True),
       #----------------------------------------------------------------------
       
       show_border=False,
              label='GP'),                        
             #-----------------
             show_border=False,
             layout='tabbed'),  
             #----------------
                  
         #----------------    
         show_labels=False,
         show_border=False),                
    resizable=True,
    width = 0.04,
    )
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
    def geolab_settings(self):
        self.geolab.height = 800
        self.geolab.width = 850
        
    def object_open(self, file_name, geometry):
        name = ('mesh_{}').format(self.counter)
        self.geolab.add_object(geometry, name=name)
        self.counter += 1

    def object_change(self):
        pass

    def object_changed(self): #Huinote should not comment
        self.optimizer.mesh = self.geolab.current_object.geometry
        self.meshmanager.update_plot() # Hui add

    def object_save(self, file_name):
        self.optimizer.save_report(file_name)

    def set_state(self, state):
        print("set_state 1") # Maryam
        if state != 'kr_interactive':
            self.interactive = False
        if state != 'mask_target':
            self.mask_target = False
     
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    #                     GP-ALGORITHM: Common used:
    # -------------------------------------------------------------------------
    @on_trait_change('hide_face')
    def plot_hide_faces(self):
        if self.hide_face:
            self.meshmanager.hide_faces()
        else:
            self.meshmanager.plot_faces(color='white',#(100, 193, 151),#'grammarly',#,,
                                        glossy=1,
                                        opacity=1,
                                        )#color=(192,174,136)turquoise

    @on_trait_change('hide_edge')
    def plot_hide_edges(self):
        if self.hide_edge:
            self.meshmanager.hide_edges()
        else:
            self.meshmanager.plot_edges(color=(157,157,157),##(77,83,87),##
                                        tube_radius=0.4*self.meshmanager.r)
            #geo:red:(240,114,114);asy:blue:(98,113,180)
            #self.meshmanager.plot_edges(color =(98,113,180),tube_radius=1.2*self.meshmanager.r)

    @on_trait_change('fix_all')
    def plot_and_fix_all_vertices(self):
        name = 'fixall'
        v = np.arange(self.mesh.V)
        #v = self.mesh.patch_matrix[:-2,:-2].flatten()
        r = self.meshmanager.r
        if self.fix_all:
            self.mesh.fix(v)
            V = self.mesh.vertices[v]
            self.meshmanager.plot_glyph(points=V,radius=2*r,color='r',name=name)
        else:
            self.mesh.unfix(v)
            self.meshmanager.remove(name)

    @on_trait_change('fix_boundary')
    def plot_and_fix_boundary(self):
        "fixed_vertices='boundary"
        name = 'fixb'
        v,B = self.mesh.get_all_boundary_vertices()
        r = self.meshmanager.r
        if self.fix_boundary:
            self.mesh.fix(v)
            self.meshmanager.plot_glyph(points=B,radius=2*r,color='r',name=name)
        else:
            self.mesh.unfix(v)
            self.meshmanager.remove(name)  
            
    @on_trait_change('fix_boundary_i')
    def plot_and_fix_boundary_i(self):
        name = 'fixbi'
        if len(self.meshmanager.selected_edges)!=0:
            e = self.meshmanager.selected_edges 
        else:
            print('\n Please selecte an edge first!')        
        v,B = self.mesh.get_i_boundary_vertices(e[0],by_corner2=True) # need to choose
        r = self.meshmanager.r
        if self.fix_boundary_i:
            self.mesh.fix(v)
            self.meshmanager.plot_glyph(points=B,radius=2*r,color='r',name=name)
        else:
            self.mesh.unfix(v)
            self.meshmanager.remove(name)   

    @on_trait_change('fix_corner')
    def plot_and_fix_corner(self):
        name = 'fixc'
        v = self.mesh.corner #get_all_corner_vertices()
        C = self.mesh.vertices[v]
        if self.fix_corner:
            self.mesh.fix(v)
            r = self.meshmanager.r
            self.meshmanager.plot_glyph(points=C,radius=2*r,color='r',name=name)
        else:
            self.mesh.unfix(v)
            self.meshmanager.remove(name)

    @on_trait_change('fix_button')
    def fix_current_handler_selected_vertex(self):
        name = 'fixv'
        v =  self.meshmanager.selected_vertices
        print(v)
        #v = np.array([14, 15, 16, 572, 573, 574])
        self._fixed_vertex.extend(v)
        self.mesh.fix(self._fixed_vertex)
        C = self.mesh.vertices[self._fixed_vertex]
        r = self.meshmanager.r
        self.meshmanager.plot_glyph(points=C,radius=2.5*r,
                                    shading = True,glossy=1,
                                    color=(255,144,64), ##red(210,92,106) #geo:red:(240,114,114);asy:blue:(98,113,180)
                                    name=name)#(210,92,106) #()(98,113,180)

    @on_trait_change('unfix_button')
    def unfix_current_handler_selected_vertex(self):
        name = 'fixv'
        v = self.meshmanager.selected_vertices
        self._fixed_vertex.remove(v[0])
        self.mesh.unfix(v)
        #self.meshmanager.remove(name)
        C = self.mesh.vertices[self._fixed_vertex]
        r = self.meshmanager.r
        self.meshmanager.plot_glyph(points=C,radius=2*r,color='r',name=name)

    @on_trait_change('clearfix_button')
    def clearfix_current_handler_selected_vertex(self):
        name = 'fixv'
        self.mesh.unfix(self._fixed_vertex)
        self._fixed_vertex = []
        self.meshmanager.remove(name)

    @on_trait_change('boundary_z0')
    def plot_and_bdry_z0(self):
        name = 'bdpt'
        v = np.array([119,68],dtype=int)
        C = self.mesh.vertices[v]
        if self.boundary_z0:
            r = self.meshmanager.r
            self.meshmanager.plot_glyph(points=C,radius=2*r,
                                        shading = True,glossy=1,
                                        color='g',name=name)
        else:
            self.meshmanager.remove(name)


    @on_trait_change('selected_z0')
    def set_selected_vertices_xy_plane(self):
        name='s_z0'
        if self.selected_z0:
            self.optimizer.set_weight('selected_z0', 0.005)
            #print(self.meshmanager.selected_vertices)
            #ind = self.meshmanager.selected_vertices
            ind = np.array([985, 341, 426, 846, 629, 702, 264])
            self.optimizer.selected_v = ind
            vv = self.mesh.vertices[ind]
            self.meshmanager.plot_glyph(points=vv,glossy=1,
                                        radius=2*self.meshmanager.r,
                                        color = 'yellow',name=name)   
        else:
            self.meshmanager.remove(name)
            self.optimizer.set_weight('selected_z0', 0)
       
    @on_trait_change('selected_y0')
    def set_selected_vertices_xz_plane(self):
        name='s_y0'
        if self.selected_y0:
            self.optimizer.set_weight('selected_y0', 0.005)
            self.optimizer.selected_v = self.meshmanager.selected_vertices
            vv = self.mesh.vertices[self.meshmanager.selected_vertices]
            self.meshmanager.plot_glyph(points=vv,glossy=1,
                                        radius=2*self.meshmanager.r,
                                        color = 'yellow',name=name)   
        else:
            self.meshmanager.remove(name)
            self.optimizer.set_weight('selected_y0', 0)
  
        # ---------------------------------------------------------------------
        #                     Fairness weights:
        # ---------------------------------------------------------------------
    @on_trait_change('fair1')
    def set_fairness_1(self):
        self.mesh_fairness = self.boundary_fairness = 0.1
        self.tangential_fairness = self.spring_fairness = 0.1
    @on_trait_change('fair01')
    def set_fairness_01(self):
        self.mesh_fairness = self.boundary_fairness = 0.01
        self.corner_fairness = 0.01
        self.tangential_fairness = self.spring_fairness = 0.01
    @on_trait_change('fair001')
    def set_fairness_001(self):
        self.mesh_fairness = self.boundary_fairness = 0.005
        self.corner_fairness = 0.008
        self.tangential_fairness = self.spring_fairness = 0.0001
        self.fairness_diagmesh = 0.005
    @on_trait_change('fair0001')
    def set_fairness_0001(self):
        self.mesh_fairness = self.boundary_fairness = 0.0005
        self.corner_fairness = 0.0008
        self.tangential_fairness = self.spring_fairness = 0.0001
        self.fairness_diagmesh = 0.0005
    @on_trait_change('fair0')
    def set_fairness_0(self):
        self.mesh_fairness = self.boundary_fairness = 0
        self.corner_fairness = 0
        self.tangential_fairness = self.spring_fairness =  0
        self.fairness_diagmesh = 0

    @on_trait_change('fairness_4diff')
    def set_fairness_4differential(self):
        if self.fairness_4diff:
            self.mesh_fairness = self.boundary_fairness = 0
            self.fairness_diagmesh = 0  
        #self.corner_fairness = 0
        #self.tangential_fairness = self.spring_fairness =  0

    @on_trait_change('close5')
    def set_close_5(self):
        self.reference_closeness = 0.5
        self.self_closeness = 0.5
    @on_trait_change('close1')
    def set_close_1(self):
        self.reference_closeness = 0.1
        self.self_closeness = 0.1
    @on_trait_change('close05')
    def set_close_05(self):
        self.reference_closeness = 0.05
        self.self_closeness = 0.05
    @on_trait_change('close01')
    def set_close_01(self):
        self.reference_closeness = 0.01
        self.self_closeness = 0.01
    @on_trait_change('close0')
    def set_close_0(self):
        self.reference_closeness = 0
        self.self_closeness = 0

    # -------------------------------------------------------------------------
    #                      Change topology (partial net/web)
    # ------------------------------------------------------------------------- 
    @on_trait_change('set_refer_mesh')
    def set_refermesh(self):
        if self.set_refer_mesh:
            if self.geolab.last_object is not None:
                #self.geolab.last_object.geometry.set_reference()
                self.mesh.reference_mesh = self.geolab.last_object.geometry
        else:
            self.mesh.reference_mesh = self.mesh.copy_mesh()#self.mesh
        rm = self.mesh.reference_mesh
        "get the index of boundary vertices:"
        ind = rm.boundary_curves(corner_split=False)[0]    
        "refine the boundary vertices"
        Ver = rm.vertices[ind]
        poly = Polyline(Ver,closed=True)  
        N = 5
        poly.refine(N)
        self.ref_glide_bdry_polyline = poly
        self.optimizer.glide_reference_polyline = self.ref_glide_bdry_polyline
        
    @on_trait_change('show_refer_mesh')
    def plot_reference_mesh(self):
        self.geolab.last_object.hide_faces()
        self.geolab.last_object.hide_edges()
        name = 'refer_mesh'
        if self.show_refer_mesh:
            try:
                rm = self.mesh.reference_mesh
            except:
                self.set_refermesh()
                rm = self.mesh.reference_mesh
            showe = Edges(rm,color ='black',tube_radius=0.5*self.meshmanager.r,
                          #color=(157,157,157),tube_radius=0.3*self.meshmanager.r,
                          name=name+'e')
            # showf = Faces(rm,color = (77,77,77),
            #               opacity=0.1,
            #               name=name+'f')
            self.meshmanager.add([showe])
        else:
            self.meshmanager.remove(name+'e')
            self.meshmanager.remove(name+'f')
            
    @on_trait_change('show_ref_mesh_boundary')
    def plot_reference_mesh_boundary(self):
        name = 'ref_mesh_pl'
        if self.show_ref_mesh_boundary:
            self.set_refermesh()
            rm = self.mesh.reference_mesh
            if True:
                poly = self.ref_glide_bdry_polyline
            else:
                poly = rm.boundary_polylines() ##Note: has problem!
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=0.7*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)

    def update_ith_boundary(self,N=3):
        self.optimizer.i_glide_bdry_crv = []
        self.optimizer.i_glide_bdry_ver = []
        if self.glide_1st_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=0)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
        if self.glide_2nd_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=1)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
        if self.glide_3rd_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=2)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
        if self.glide_4th_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=3)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)            
        if self.glide_5th_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=4)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)   
        if self.glide_6th_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=5)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)   
        if self.glide_7th_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=6)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)   
        if self.glide_8th_bdry:
            v,B = self.mesh.get_i_boundary_vertex_indices(i=7)
            poly = Polyline(B,closed=False)  
            poly.refine(steps=N)
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)               
    @on_trait_change('glide_1st_bdry')
    def plot_1st_boundary(self):
        name = '1stB'  
        if self.glide_1st_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=0)
            print(v)
            x = self.mesh.vertices[v][:,0]
            y = self.mesh.vertices[v][:,1]
            z = self.mesh.vertices[v][:,2]
            print('x:',np.mean(x),np.min(x),np.max(x))
            print('y:',np.mean(y),np.min(y),np.max(y))
            print('z:',np.mean(z),np.min(z),np.max(z))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)  
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
            ##print(self.mesh.boundary_curves(True),self.mesh.get_a_closed_boundary())
        else:
            self.meshmanager.remove(name)
      
    @on_trait_change('glide_2nd_bdry')
    def plot_2nd_boundary(self):
        name = '2ndB' 
        if self.glide_2nd_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=1)
            print(v)
            z = self.mesh.vertices[v][:,2]
            print(np.mean(z),np.min(z),np.max(z))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)  
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)

    @on_trait_change('glide_3rd_bdry')
    def plot_3rd_boundary(self):
        name = '3rdB'   
        if self.glide_3rd_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=2)
            print(v)
            z = self.mesh.vertices[v][:,2]
            print(np.mean(z),np.min(z),np.max(z))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)  
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)
    @on_trait_change('glide_4th_bdry')
    def plot_4th_boundary(self):
        name = '4thB'  
        if self.glide_4th_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=3)
            print(v)
            x = self.mesh.vertices[v][:,0]
            y = self.mesh.vertices[v][:,1]
            print('4th-x:',np.mean(x),np.min(x),np.max(x))
            print('4th-y:',np.mean(y),np.min(y),np.max(y))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)   
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)
    @on_trait_change('glide_5th_bdry')
    def plot_5th_boundary(self):
        name = '5thB'  
        if self.glide_5th_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=4)
            print(v)
            y = self.mesh.vertices[v][:,1]
            print(np.mean(y),np.min(y),np.max(y))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)   
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)
    @on_trait_change('glide_6th_bdry')
    def plot_6th_boundary(self):
        name = '6thB'  
        if self.glide_6th_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=5)
            print(v)
            y = self.mesh.vertices[v][:,1]
            print(np.mean(y),np.min(y),np.max(y))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)   
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)
    @on_trait_change('glide_7th_bdry')
    def plot_7th_boundary(self):
        name = '7thB'  
        if self.glide_7th_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=6)
            print(v)
            y = self.mesh.vertices[v][:,1]
            print(np.mean(y),np.min(y),np.max(y))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)   
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)   
    @on_trait_change('glide_8th_bdry')
    def plot_8th_boundary(self):
        name = '8thB'  
        if self.glide_8th_bdry:
            self.update_ith_boundary(N=3)
            v,B = self.mesh.get_i_boundary_vertex_indices(i=7)
            print(v)
            y = self.mesh.vertices[v][:,1]
            print(np.mean(y),np.min(y),np.max(y))
            poly = Polyline(B,closed=False)  
            poly.refine(steps=3)   
            self.optimizer.i_glide_bdry_crv.append(poly)
            self.optimizer.i_glide_bdry_ver.append(v)
            self.meshmanager.plot_polyline(polyline=poly,glossy=1,
                                           tube_radius=1.5*self.meshmanager.r,
                                           color = 'r',name=name)
        else:
            self.meshmanager.remove(name)   
            
    # -------------------------------------------------------------------------
    #                      Orthogonal NET: Weights + Plotting:
    # -------------------------------------------------------------------------

    @on_trait_change('button_clear_constraint')
    def set_clear_webs(self):
        self.planarity = False
        self.orthogonal = False
        self.equilibrium = False
        self.circular_mesh = False
        self.conical_mesh = False

        self.Anet = False
        self.Anet_diagnet = False

        self.AAG = False # Maryam
        self.AGG = False # Maryam
        self.GGG = False # Maryam

        self.Snet = False
        self.Snet_diagnet = False
        self.Snet_constR = False
        self.if_uniqR = False
        
        self.opt_planar_polyline1 = False
        self.opt_planar_polyline2 = False
        
        self.multinets_orthogonal = False
        self.weigth_multinets_orthogonal = 0.0
        # self.if_set_weight = False
        
        self.optimizer.set_weight('unit_edge_vec', 0)
        self.optimizer.set_weight('unit_diag_edge_vec', 0)


    @on_trait_change('button_opt_PPO')
    def set_orthogonal_planar_ply12(self): 
        self.orthogonal = True
        self.opt_planar_polyline1,self.opt_planar_polyline2 = True, True
        
    @on_trait_change('button_minimal_mesh')
    def set_orthogonal_Anet(self): 
        self.orthogonal = True  
        self.Anet = True

    @on_trait_change('button_CMC_mesh')
    def set_CMC_mesh(self): 
        self.orthogonal = True  
        self.Snet = True
        self.Snet_orient = True
        self.Snet_constR = True

    @on_trait_change('button_principal_mesh')
    def set_principal_mesh(self): 
        self.orthogonal = True  
        self.planarity = True
        
    @on_trait_change('button_funicularity')
    def set_funicularity(self): 
        self.orthogonal = True  
        self.equilibrium = True
        
    @on_trait_change('button_principal_stress')
    def set_principal_stress(self): 
        self.orthogonal = True  
        self.opt_planar_polyline1 = True
        #self.opt_planar_polyline2 = False ##or
        self.equilibrium = True

    @on_trait_change('button_ortho_planarPolyline_funicularity')
    def set_planar_long_range_supporting_beams(self): 
        self.orthogonal = True  
        self.planarity = True
        self.equilibrium = True    
        
    @on_trait_change('circular_mesh,conical_mesh')
    def set_principal_meshes(self): 
        if self.circular_mesh or self.conical_mesh:
            self.planarity = True
        else:
            self.planarity = False
    
    @on_trait_change('button_Multinets_Orthogonal')
    def set_multinets_orthogonal(self):
        self.multinets_orthogonal = True
        # self.if_set_weight =True
        self.weigth_multinets_orthogonal = 1.0

    #---------------------------------------------------------        
    #                       Ploting
    #---------------------------------------------------------
    @on_trait_change('show_isogonal_face_based_vector')
    def plot_isogonal_facebased_cross_vector(self):
        name = 'isg_f_v'
        if self.show_isogonal_face_based_vector:
            _,_,t1,t2,an = self.mesh.get_quad_midpoint_cross_vectors()
            ang = np.arccos(np.einsum('ij,ij->i',t1,t2))/np.pi*180
            print('\n mean_angle =',np.mean(ang))
            print(' max_angle   =',np.max(ang))
            self.meshmanager.plot_vectors(anchor=an,vectors=t1,
                                          position = 'center',
                                          glyph_type = 'line',
                                          color = 'b',
                                          name = name+'11')
            self.meshmanager.plot_vectors(anchor=an,vectors=t2,
                                          position = 'center',
                                          glyph_type = 'line',
                                          color = 'r',
                                          name = name+'22')
        else:
            self.meshmanager.remove([name+'11',name+'22'])              

            
    @on_trait_change('show_midpoint_edge1,show_midpoint_edge2')
    def plot_midpoint_edges(self):
        "same was as plot_isogonal_facebased_cross_vector; but different visulization"
        name = 'mid_edge'
        if self.show_midpoint_edge1 or self.show_midpoint_edge2:
            pl1,pl2 = self.mesh.get_quad_midpoint_cross_vectors(plot=True)
            if self.show_midpoint_edge1:
                self.meshmanager.plot_polyline(pl1,
                                               tube_radius=1*self.meshmanager.r,
                                               color=(162,20,47),glossy=0.8,
                                               name = name+'1')
            else:
                self.meshmanager.remove([name+'1'])  
            if self.show_midpoint_edge2 and pl2 is not None:
                self.meshmanager.plot_polyline(pl2,
                                               tube_radius=1*self.meshmanager.r,
                                               color=(20,162,47),glossy=0.8,
                                               name = name+'2')
            else:
                self.meshmanager.remove([name+'2'])                                         
        else:
            self.meshmanager.remove([name+'1',name+'2'])      

    @on_trait_change('show_midpoint_polyline1,show_midpoint_polyline2')
    def plot_midpoint_polylines(self):
        name = 'mid_pl'
        if self.show_midpoint_polyline1 or self.show_midpoint_polyline2:
            pl1,pl2 = self.mesh.get_quad_midline()
            if self.show_midpoint_polyline1:
                self.meshmanager.plot_polyline(pl1,
                                               tube_radius=1*self.meshmanager.r,
                                               color=(162,20,47),glossy=0.8,
                                               name = name+'1')
            else:
                self.meshmanager.remove([name+'1'])  
            if self.show_midpoint_polyline2 and pl2 is not None:
                self.meshmanager.plot_polyline(pl2,
                                               tube_radius=1*self.meshmanager.r,
                                               color=(20,162,47),glossy=0.8,
                                               name = name+'2')
            else:
                self.meshmanager.remove([name+'2'])                                         
        else:
            self.meshmanager.remove([name+'1',name+'2']) 
    
    @on_trait_change('show_quad_diagonal1,show_quad_diagonal2') # Maryam
    def plot_diagonals_polylines(self):
        name = 'quad_diagonals'
        if self.show_quad_diagonal1 or self.show_quad_diagonal2:
            print("show diagonal ")
            pl1,pl2,pl3,pl4 = self.mesh.get_quad_diagonals()
            if self.show_quad_diagonal1:
                self.meshmanager.plot_polyline(pl1,
                                               tube_radius=0.8*self.meshmanager.r,
                                               color=(200,200,128),glossy=0.8,
                                               name = name+'1')
                self.meshmanager.plot_polyline(pl2,
                                               tube_radius=0.8*self.meshmanager.r,
                                               color=(128,128,128),glossy=0.8,
                                               name = name+'2')
            else:
                self.meshmanager.remove([name+'1', name+'2'])
            if self.show_quad_diagonal2 and pl2 is not None:
                self.meshmanager.plot_polyline(pl3,
                                               tube_radius=0.8*self.meshmanager.r,
                                               color=(20,162,47),glossy=0.8,
                                               name = name+'3')
                self.meshmanager.plot_polyline(pl4,
                                               tube_radius=0.8*self.meshmanager.r,
                                               color=(20,162,47),glossy=0.8,
                                               name = name+'4')
            else:
                self.meshmanager.remove([name+'3', name+'4'])                                         
        else:
            self.meshmanager.remove([name+'1',name+'2', name+'3', name+'4'])    
    @on_trait_change('show_vertex_normals') # Maryam
    def plot_normals(self):
        name = 'normals'
        if self.show_vertex_normals:
            pl = self.mesh.get_normals()
            self.meshmanager.plot_polyline(pl,tube_radius=1*self.meshmanager.r,
                                               color=(162,20,47),glossy=0.8, name=name)  
        else:
            self.meshmanager.remove([name]) 
    @on_trait_change('show_midline_mesh')
    def plot_midline_mesh_checkboard(self):
        name = 'mid_mesh'
        if self.show_midline_mesh:
            dm = self.mesh.get_midline_mesh()
            self.save_new_mesh = dm
            self.label = name
            showe = Edges(dm,color=(0,59,117),
                          tube_radius=0.6*self.meshmanager.r,
                          glossy=0.8,
                          name=name+'e')
            self.meshmanager.add([showe])
        else:
            self.meshmanager.remove([name+'e',name+'f'])   

    @on_trait_change('show_multinets_diagonals') 
    def plot_multinets_diagonals(self):
        name = 'multi_diagonal'
        if self.show_multinets_diagonals:
            pl1,pl2 = self.mesh.Get_Diagonals_of_Multinets()
            self.meshmanager.plot_polyline(pl1,
                                           tube_radius=0.5*self.meshmanager.r,
                                           color=(138,43,226),glossy=1,
                                           name = name+'1')
            
            self.meshmanager.plot_polyline(pl2,
                                            tube_radius=0.5*self.meshmanager.r,
                                            color=(255,215,0),glossy=1,
                                            name = name+'2')
            
        else:
            self.meshmanager.remove([name+'1',name+'2'])  


    #---------------------------------------------------------        
    #               Planar Polyline (PP) - Ploting
    #---------------------------------------------------------
    @on_trait_change('show_planar_poly1_normal')
    def plot_polyline1_planar_normal(self):
        name = 'ply1_n'
        if self.show_planar_poly1_normal:  
            an,vn = self.optimizer.get_mesh_planar_normal_or_plane(pl1=True)
            self.meshmanager.plot_vectors(anchor=an,vectors=vn,position='tail',
                                          color = 'r',name = name) 
        else:
            self.meshmanager.remove([name])  
            
    @on_trait_change('show_planar_poly2_normal')
    def plot_polyline2_planar_normal(self):
        name = 'ply2_n'
        if self.show_planar_poly2_normal:  
            an,vn = self.optimizer.get_mesh_planar_normal_or_plane(pl2=True)
            self.meshmanager.plot_vectors(anchor=an,vectors=vn,position='tail',
                                          color = 'r',name = name) 
        else:
            self.meshmanager.remove([name])  
            
    @on_trait_change('show_planar_poly1_plane')
    def plot_polyline1_planar_plane(self):
        name = 'ply1_pln'
        if self.show_planar_poly1_plane:  
            sm= self.optimizer.get_mesh_planar_normal_or_plane(pl1=True,pln=True)
            showf = Faces(sm,glossy=0.5,opacity=0.7,
                          color='black',name=name+'f')           
            self.meshmanager.add([showf])
            self.save_new_mesh = sm
        else:
            self.meshmanager.remove([name+'e',name+'f'])   
            
    @on_trait_change('show_planar_poly2_plane')
    def plot_polyline2_planar_plane(self):
        name = 'ply2_pln'
        if self.show_planar_poly2_plane:  
            sm= self.optimizer.get_mesh_planar_normal_or_plane(pl2=True,pln=True)
            showf = Faces(sm,glossy=0.5,opacity=0.9,
                          color='yellow',name=name+'f')           
            self.meshmanager.add([showf])
            self.save_new_mesh = sm
        else:
            self.meshmanager.remove([name+'e',name+'f'])   
        

    #---------------------------------------------------------        
    #               Anet / Snet - Ploting
    #---------------------------------------------------------
    @on_trait_change('show_snet_center')
    def plot_snet_center(self):
        name = 'snetc'
        if self.show_snet_center:
            if self.Snet:
                is_diag = False
            elif self.Snet_diagnet:
                is_diag = True
            C,data = self.optimizer.get_snet_data(is_diag,center=True)
            r = self.meshmanager.r
            self.meshmanager.plot_glyph(points=C,vertex_data=data,
                                        color='blue-red',lut_range='0:+',
                                        radius=2*r,name=name)
        else:
            self.meshmanager.remove(name)

    @on_trait_change('show_snet_normal')
    def plot_snet_normal(self):
        name = 'snetn'
        if self.show_snet_normal:
            if self.Snet:
                is_diag = False
            elif self.Snet_diagnet:
                is_diag = True
            an,N = self.optimizer.get_snet_data(is_diag,normal=True)
            self.meshmanager.plot_vectors(anchor=an,vectors=N,#normal neg or pos
                                          position='tail',color='black',
                                          name=name)
            
            N0 = self.mesh.vertex_normals()
            N0[self.mesh.ver_rrv4f4] = N
            self.snet_normal = N0
        else:
            self.meshmanager.remove(name)
            
    @on_trait_change('show_snet_tangent')
    def plot_snet_tangent(self):
        name = 'snett'
        if self.show_snet_tangent:
            if self.Snet:
                is_diag = False
            elif self.Snet_diagnet:
                is_diag = True
            an,t1,t2 = self.optimizer.get_snet_data(is_diag,tangent=True)

            self.meshmanager.plot_vectors(anchor=an,vectors=t1,position='center',
                                          glyph_type = 'line',color='black',
                                          name=name+'1')
            self.meshmanager.plot_vectors(anchor=an,vectors=t2,position='center',
                                          glyph_type = 'line',color='black',
                                          name=name+'2')
        else:
            self.meshmanager.remove([name+'1',name+'2'])  

    @on_trait_change('show_snet_ss')
    def plot_snet_support_structure(self):
        name = 'snetss'
        if self.show_snet_ss:
            ss = self.mesh.get_both_support_structures_from_edges(
                N=self.snet_normal,
                dist=self.scale_dist_offset,
                diagnet=self.Snet_diagnet,
                is_central=self.is_middle_ss,
                ss1=False,ss2=False)
            data = ss.face_planarity()
            print('planarity of strips max=','%.2g' % np.max(data))
            val = 0.02#1e-4#
            showf = Faces(ss, face_data = data,
                          glossy=1,opacity=1,
                          color='bwr',lut_range=[-val,val],#'-:0:+',
                          name=name+'f')
            showe = Edges(ss,color = 'black',name=name+'e')
            self.meshmanager.add([showf,showe])
        else:
            self.meshmanager.remove([name+'e',name+'f'])
    
    @on_trait_change('show_vs_sphere')
    def plot_vertex_star_sphere(self):
        "S-net: vs-common-sphere"
        name='vs_sphere'
        v = self.meshmanager.selected_vertices
        if len(v) !=0:
            if self.show_vs_sphere:
                vv = np.array(v)
                C,r,Vneib = get_vs_interpolated_sphere(self.mesh.vertices,vv,self.mesh.ringlist)
                self.meshmanager.plot_glyph(points=Vneib,color=(123,123,0),
                                            name=name+'vi') 
                self.meshmanager.plot_glyph(points=C,color='black',
                                            name=name+'c')
                s = get_sphere_packing(C,r,Fa=50,Fv=50)
                shows = Faces(s,color = 'gray_40',opacity=0.3,name=name)
                self.meshmanager.add(shows)
            else:
                self.meshmanager.remove([name,name+'vi',name+'c'])
        else:
            print('Select a vertex first.')     
            
    @on_trait_change('show_circumcircle')
    def plot_face_circum_circle(self):
        name = 'circum'
        if self.show_circumcircle:
            cir = self.optimizer.face_circum_circles()
            r = self.meshmanager.r
            self.meshmanager.plot_polyline(polyline=cir,tube_radius=0.7*r,
                                           color =(251,144,37),name=name)
        else:
            self.meshmanager.remove(name)
    #---------------------------------------------------------        
    #               Multinets diagonals - Ploting (Xinye)
    #---------------------------------------------------------
    # @on_trait_change('show_multinets_diagonal')
    # def plot_multinets_diagonal(self):
    # name = 'mul_nets_diag'
    #  if self.show_multinets_diagonal:
                
 

    #--------------------------------------------------------------------------
    #                         Printing / Check
    #--------------------------------------------------------------------------         
    @on_trait_change('print_orthogonal')
    def print_orthogonal_data(self):
        pass
        #angle_min, angle_mean, angle_max = 
        #print('angle:[min,mean,max]=','%.3f'%angle_min,'%.3f'%angle_mean,'%.3f'%angle_max)

    @on_trait_change('print_computation')
    def print_computation_info(self):
        print('No. of all vertices: ', self.mesh.V)
        print('No. of all faces: ', self.mesh.F)
        print('No. of all edges: ', self.mesh.E)
        
        print('No. of rr_vertices: ', self.mesh.num_rrv4f4)
        print('No. of rr_quad_faces: ', self.mesh.num_rrf) #circular mesh
        
        print('#variables: ', len(self.optimizer.X))
        print('#constraints: ', len(self.optimizer._r))
        #print('time[s] per iteration: ',  )
    #--------------------------------------------------------------------------
    #                                    Save txt / obj
    #--------------------------------------------------------------------------
    @on_trait_change('save_button')
    def save_file(self):
        name = ('{}').format(self.label)
        if self.save_new_mesh is None:
            self.save_new_mesh = self.mesh
            
 
        save_path =  'archgeolab/objs'     
        completeName = os.path.join(save_path, name)   
        self.save_new_mesh.make_obj_file(completeName)

        print('\n\n NOTE: <'+self.label+'> mesh has been saved in <'+completeName+'>\n')
    # -------------------------------------------------------------------------
    #                              Settings
    # -------------------------------------------------------------------------
    def set_settings(self):
        # ---------------------------------------------------------------------
        #                     GP-ALGORITHM common used:
        # ---------------------------------------------------------------------
        self.optimizer.threshold = 1e-20
        #self.optimizer.itera_run = self.itera_run
        #self.optimizer.iterations = self.itera_run
        self.optimizer.epsilon = self.epsilon
        self.optimizer.step = self.step
        self.optimizer.fairness_reduction = self.fairness_reduction
        
        self.optimizer.add_weight('mesh_fairness', self.mesh_fairness)
        self.optimizer.add_weight('tangential_fairness', self.tangential_fairness)
        self.optimizer.add_weight('boundary_fairness', self.boundary_fairness)
        self.optimizer.add_weight('corner_fairness', self.corner_fairness)
        self.optimizer.add_weight('spring_fairness', self.spring_fairness)
        self.optimizer.add_weight('fairness_4diff', self.fairness_4diff)
        self.optimizer.add_weight('fairness_diag_4diff', self.fairness_diag_4diff)
        self.optimizer.add_weight('fairness_diagmesh', self.fairness_diagmesh)
        self.optimizer.add_weight('GGG_weight', self.GGG_weight) # Maryam 
        self.optimizer.add_weight('vertex_control_weight', self.vertex_control_weight) # Maryam 


 
        self.optimizer.add_weight('reference_closeness', self.reference_closeness)
        self.optimizer.add_weight('self_closeness', self.self_closeness)
        
        self.optimizer.add_weight('boundary_glide', self.boundary_glide)
        self.optimizer.add_weight('i_boundary_glide', self.i_boundary_glide)
        self.optimizer.add_weight('avoid_shrinkage', self.avoid_shrinkage)
        self.optimizer.add_weight('fixed_vertices', self.weight_fix)
        self.optimizer.add_weight('fix_point',  self.fix_p_weight)
        self.optimizer.add_weight('fix_corners',  self.fix_corner)
        self.optimizer.set_weight('sharp_corner',  self.sharp_corner) 
        self.optimizer.set_weight('z0', self.z0)
        self.optimizer.set_weight('boundary_z0', self.boundary_z0)
        
        # ---------------------------------------------------------------------
        self.optimizer.set_weight('planarity', self.planarity)
        self.optimizer.set_weight('orthogonal',  self.orthogonal*1)
        self.optimizer.set_weight('circularity', self.circular_mesh)
        self.optimizer.set_weight('equilibrium',  self.equilibrium)
        
        self.optimizer.set_weight('Anet',  self.Anet)
        self.optimizer.set_weight('Anet_diagnet',  self.Anet_diagnet)

        self.optimizer.set_weight('AAG',  self.AAG) # Maryam
        self.optimizer.set_weight('AGG',  self.AGG) # Maryam
        self.optimizer.set_weight('GGG',  self.GGG) # Maryam

        self.optimizer.set_weight('Snet', self.Snet)
        self.optimizer.set_weight('Snet_diagnet', self.Snet_diagnet)
        self.optimizer.set_weight('Snet_orient', self.Snet_orient)
        self.optimizer.set_weight('Snet_constR', self.Snet_constR)
        self.optimizer.if_uniqradius = self.if_uniqR
        self.optimizer.assigned_snet_radius = self.Snet_constR_assigned
        
        self.optimizer.set_weight('multinets_orthogonal', self.weigth_multinets_orthogonal)
        
        self.optimizer.set_another_polyline = self.set_another_poly
        self.optimizer.set_weight('planar_ply1', self.opt_planar_polyline1)
        self.optimizer.set_weight('planar_ply2', self.opt_planar_polyline2)

    @on_trait_change('print_error')
    def print_errors(self):
        self.optimizer.make_errors()
    # -------------------------------------------------------------------------
    #                         Reset + Optimization
    # -------------------------------------------------------------------------
    def reinitialize_constraints(self):# Hui: to set
        self.optimizer.is_initial = True 
        #self.planarity = False
        #self.orthogonal = False

    @on_trait_change('reinitialize')
    def reinitialize_optimizer(self):
        self.reinitialize_constraints()
        self.set_settings()
        self.optimizer.reinitialize = True
        self.optimizer.initialization() # Hui add
        self.mesh.vertices = self.mesh.vertices_0 # Huinote:add
        self.meshmanager.update_plot()
        print('\n---------------------\n')

    def updating_plot(self):
        pass

    def optimization_step(self):
        if not self.interactive:
            self.handler.set_state(None)
        self.set_settings()
        self.optimizer.optimize()
        print("Opt Step")
        #self.print_error()
        self.updating_plot()
        print("before mesh manager in opt")
        self.meshmanager.update_plot()
        if self.show_quad_diagonal1:
            name = 'quad_diagonals'
            self.meshmanager.remove([name+'1', name+'2'])
            self.plot_diagonals_polylines()
        print("after mesh manager in opt")
        if self.fairness_reduction !=0:
            self.mesh_fairness = self.mesh_fairness/(10**(self.fairness_reduction))
            self.tangential_fairness = self.tangential_fairness/(10**
                                        (self.fairness_reduction))
            self.boundary_fairness = self.boundary_fairness/(10**
                                    (self.fairness_reduction))
            self.spring_fairness = self.spring_fairness/(10**
                                    (self.fairness_reduction)) 
        print("faces in opt")

    @on_trait_change('optimize')
    def optimize_mesh(self):
        import time
        start_time = time.time()
        itera = self.itera_run
        self.meshmanager.iterate(self.optimization_step, itera) # note:iterations from gpbase.py
        self.meshmanager.update_plot()
        
        print('time[s] per iteration:','%.3g s' %((time.time() - start_time)/itera))
        

    # @on_trait_change('mesh_propogate')  # Maryam
    def mesh_propogation(self):
        print("mesh_propogation")
        print(self.mesh._vertices)
        print(type(self.mesh._vertices))
        # Store the initial strip.
        if (self.mesh._copy is None):
            self.mesh._copy = self.mesh.copy_mesh()

        # Number of vertices belonging to the first polyline   
        numVperPoly  = (self.mesh._copy._V/2).astype(int)

        # Number of all the vertices
        V = self.mesh._V # last index = V - 1 here V = 23
        
        # Initialize the new strip
        new_v = np.zeros((numVperPoly,3))
        # print("the 2 polylines: ", self.mesh._copy._vertices[numVperPoly:] )
        # print(self.mesh._copy._vertices[:numVperPoly-1])
        # print('the directions: ', self.mesh._copy._vertices[numVperPoly:] - self.mesh._copy._vertices[:numVperPoly])
        offset_dirs = self.mesh._copy._vertices[numVperPoly:] - self.mesh._copy._vertices[:numVperPoly]
        # print('the directions: ', offset_dirs)
        for i in range(numVperPoly):
            new_v[i] = self.mesh._vertices[V-numVperPoly+i]
            # new_v[i][2] += 1.2 # offset in the negative y-direction 
            new_v[i] += offset_dirs[i]
        
        # Add the new faces
        new_f = np.zeros((numVperPoly-1,4))
        last_poly = np.arange(V-numVperPoly,V)
        new_poly = np.arange(V,V+numVperPoly)
        for i in range(numVperPoly-1):
            new_f[i] =  np.array([last_poly[i+1],new_poly[i+1],new_poly[i],last_poly[i]])

        #print(self.mesh._vertices)
        #print(self.mesh._faces)
        #print(self.mesh._V,self.mesh._F,self.mesh._E)

        # Update the new mesh info
        self.mesh._V += len(new_v)
        self.mesh._F += len(new_f)
        self.mesh._E += 2*len(new_v)-1
        # print(self.mesh._V,self.mesh._F,self.mesh._E)
        self.mesh._vertices = np.vstack((self.mesh._vertices, new_v))
        self.mesh._faces = np.vstack((self.mesh._faces, new_f))
        self.mesh.make_mesh(self.mesh._vertices, self.mesh._faces)
        
        # Create an obj file of the mesh
        with open('3strip', 'w') as obj_file:
            # Write vertices
            for vert in self.mesh._vertices:
                obj_file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
            # Write faces
            for face in self.mesh._faces:
                # Note: In OBJ format, vertex indices start from 1 (not 0), so we add 1 to each index.
                obj_file.write(f"f {int(face[0] + 1)} {int(face[1] + 1)} {int(face[2] + 1)} {int(face[3] + 1)}\n")

        # Update the plot
        self.meshmanager.update_plot()
        if self.show_quad_diagonal1:
            name = 'quad_diagonals'
            self.meshmanager.remove([name+'1', name+'2'])
            self.plot_diagonals_polylines()

    @on_trait_change('mesh_propogate')  # Maryam
    def propogate(self):
        print("propogate")
        self.mesh._mesh_propogate = True
        # Store the initial strip.
        if (self.mesh._copy is None):
            self.mesh._copy = self.mesh.copy_mesh()

        # Number of vertices belonging to the first polyline   
        numVperPoly  = (self.mesh._copy._V/2).astype(int)

        # Number of all the vertices
        V = self.mesh._V # last index = V - 1 here V = 23
        vertices = self.mesh._vertices

        # Define a list of indices for v*, v1, v2, v3, v4, va, vb, vc, vd

        #  va  v1  vd  #
        #  v2  v*  v4  #
        #  vb  v3  vc  #

        v_star = np.arange(V-numVperPoly+1,V-1)
        v1 = v_star - 1
        v2 = v_star - numVperPoly
        v3 = v_star + 1
        v4 = v_star + numVperPoly  # This one is not defined yet
        va = v2 - 1
        vb = v2 + 1
        vc = v4 + 1                # This one is not defined yet
        vd = v4 - 1                # This one is not defined yet

        # initiate bisectors for each v_star and call them normals // check if vectors same dire, norm of cross prod is small
        delta = (vertices[v1]-vertices[v_star])+(vertices[v3]-vertices[v_star]) # (a-b)
        # print(v1,v_star)
        # print(vertices[v1],vertices[v_star])
        # print((vertices[v1]-vertices[v_star]))
        # print(delta)
        norms = np.linalg.norm(delta, axis = 1) 
        # print(norms)
        normals = delta / norms[:, np.newaxis]
        # print(normals)

        # # find the lengths of new sides of the inner triangles
            
        # 1) define the angles A, B, C
        A = []
        B = []
        C = []
        c = []
        for i in range(len(v_star)):
            print("i: ", i)
            # A = arccos( (v1-v_star)*(va-v_star) / l1 * la)
            a_vec = (vertices[v1[i]]-vertices[v_star[i]])
            b_vec = (vertices[va[i]]-vertices[v_star[i]])
            la = np.linalg.norm(a_vec)
            lb = np.linalg.norm(b_vec)
            A_temp = np.arccos(np.dot(a_vec,b_vec)/(la*lb))
            A = np.r_[A,A_temp]  
            
            if(i != len(v_star) - 1):  # makes sure we are still in the inner triangles
                # B = arccos( (v_star-v3)*(v_star-v2))
                a_vec = (vertices[v3[i+1]]-vertices[v_star[i+1]])
                print("v3 should be 7: ", v3[i+1], v_star[i+1])
                b_vec = (vertices[v2[i+1]]-vertices[v_star[i+1]])
                la = np.linalg.norm(a_vec)
                lb = np.linalg.norm(b_vec)
                B_temp = np.arccos(np.dot(a_vec,b_vec)/(la*lb))
                B = np.r_[B,B_temp] 

                # C = 180 - A - B
                C = np.r_[C,np.pi - A_temp - B_temp] 

                # c = || v_star[i]-v_star[i+1] ||
                c = np.r_[c,np.linalg.norm(vertices[v_star[i]] - vertices[v_star[i+1]])]

        print("A: ", np.degrees(A))
        print("B: ", np.degrees(B))
        print("C: ", np.degrees(C))
            
        print(A)
        a = np.abs(c*np.sin(A[1:])/np.sin(C))
        b = np.abs(c*np.sin(B)/np.sin(C))
        print("a: ", a)
        print("b: ", b)
        print("c: ", c)

        # find the direction of the new sides of the inner triangles
        va_vec = (vertices[va] - vertices[v_star])
        va_norms = np.linalg.norm(va_vec, axis = 1)  
        va_normalized = va_vec / va_norms[:, np.newaxis]
        e1_dir =  2 * np.einsum('ij,ij->i', normals, va_normalized)[:, np.newaxis]* normals - va_normalized
        # print("calculate manually, ", normals, va_normalized, e1_dir)
        # print(e1_dir)
        # print(va_normalized)
        # print(normals)

        print("va dot n: ", np.einsum('ij,ij->i', normals, va_normalized))
        print("n dot vc: ", np.einsum('ij,ij->i', normals, e1_dir))

        v2_vec = (vertices[v2] - vertices[v_star])
        v2_norms = np.linalg.norm(v2_vec, axis = 1) 
        v2_normalized = v2_vec / v2_norms[:, np.newaxis]
        e2_dir =  2 * np.einsum('ij,ij->i', normals, v2_normalized)[:, np.newaxis]* normals - v2_normalized
        print(e2_dir)

        print("v2 dot n: ", np.einsum('ij,ij->i', normals, v2_normalized))
        print("n dot v4: ", np.einsum('ij,ij->i', normals, e2_dir))

        print("brod: ",  e1_dir[:len(v_star)-1], b[:, np.newaxis])
        mid_points1  = vertices[v_star][:len(v_star)-1] + e1_dir[:len(v_star)-1]*b[:, np.newaxis]
        print("mid points: ", mid_points1)

        mid_points2  = vertices[v_star][1:] + e2_dir[1:]*a[:, np.newaxis]
        print("mid points: ", mid_points2)

        mid_points = (mid_points1+ mid_points2)/2
        print(mid_points)


        # find the outer three vertices.    

        # case 1
        c = np.r_[c, np.abs(np.linalg.norm(vertices[V-1]-vertices[V-1-numVperPoly]))]
        vec = (vertices[vb[-1]] - vertices[v3[-1]])
        vec_normalized = vec / np.linalg.norm(vec)
        dir1 = 2 * np.dot(normals[-1],vec_normalized)* normals[-1] - vec_normalized
        last_point = vertices[V-1] + dir1 * c[-1]


        # case 2
        length = np.abs(np.linalg.norm(vertices[v_star[0]]-vertices[v2[0]]))
        vec2 = (-vertices[v_star[0]] + vertices[v2[0]])

        vec2_normalized = vec2 / np.linalg.norm(vec2)
        dir2 = 2 * np.dot(normals[0],vec2_normalized)* normals[0] - vec2_normalized
        second_point = vertices[V-numVperPoly+1] + dir2 * length
        print("case2: v2 * normal vs e* normal", np.dot(normals[0], vec2_normalized), np.dot(normals[0], dir2))

        # case 3
        first_point = vertices[V-numVperPoly] + (vertices[V-numVperPoly]-vertices[V-2*numVperPoly])

        new_v = np.vstack((first_point, second_point, mid_points, last_point))
        # new_points = np.vstack((first_point, second_point, mid_points, last_point))

        print(new_v)

        # Add the new faces
        new_f = np.zeros((numVperPoly-1,4))
        last_poly = np.arange(V-numVperPoly,V)
        new_poly = np.arange(V,V+numVperPoly)
        for i in range(numVperPoly-1):
            new_f[i] =  np.array([last_poly[i+1],new_poly[i+1],new_poly[i],last_poly[i]])

        #print(self.mesh._vertices)
        #print(self.mesh._faces)
        #print(self.mesh._V,self.mesh._F,self.mesh._E)

        # Update the new mesh info
        self.mesh._V += len(new_v)
        self.mesh._F += len(new_f)
        self.mesh._E += 2*len(new_v)-1
        # print(self.mesh._V,self.mesh._F,self.mesh._E)
        self.mesh._vertices = np.vstack((self.mesh._vertices, new_v))
        self.mesh._faces = np.vstack((self.mesh._faces, new_f))
        self.mesh.make_mesh(self.mesh._vertices, self.mesh._faces)

        # Update the plot
        self.meshmanager.update_plot()
        if self.show_quad_diagonal1:
            name = 'quad_diagonals'
            self.meshmanager.remove([name+'1', name+'2'])
            self.plot_diagonals_polylines()
      
        ############################################### update the variables:
        ########### X = [Vx, Vy, Vz, 
                        #  Nx, Ny, Nz,
                        #  B1x, B1y, B1z, 
                        #  B2x, B2y, B2z,
                        #  B3x, B3y, B3z]
        ########### for each new added v_star (numVperPoly - 2), we add 4 vectors to the auxiliary variables
        print("ADD Auxilary: ")
        # N = kwargs.get("N")
        # self.build_added_weight()
        # Nanet = N   




    @on_trait_change('reset_mesh_propogation')  # Maryam
    def reset_mesh(self):

        # self.mesh._rrv4f4 = None
        # self.mesh._rr_star = None
        # self._ind_rr_star_v4f4 = None
        # self._num_rrv4f4 = 0
        # self._ver_rrv4f4  = None
        # self._rr_star_corner = None


        print("reset_mesh")
        copy = self.mesh._copy
        print(copy._vertices)
        print(copy._faces)
        self.mesh.make_mesh(copy._vertices, copy._faces)
        self.meshmanager.update_plot()
        print("Is GGG applied?",self.GGG)


    @on_trait_change('interactive')
    def interactive_optimize_mesh(self):
        print("interactive_optimize_mesh") # Maryam
        self.handler.set_state('kr_interactive')
        if self.interactive:
            print("self.interactive is true") # Maryam 
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
