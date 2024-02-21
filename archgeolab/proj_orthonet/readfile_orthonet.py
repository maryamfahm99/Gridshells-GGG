# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 22:16:21 2022

@author: WANGH0M
"""
#------------------------------------------------------------------------------
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(path)
# import igl  # Maryam
#print(path)
#------------------------------------------------------------------------------
# -------------------------------------------------------------------------
#                              Run
# -------------------------------------------------------------------------
if __name__ == '__main__':

    a = path + '\archgeolab\objs'
    local = '/Users/memo2/Desktop/2022-2023/Summer2023/WebImplementation/geometry-lab-main/archgeolab/objs'

    # pq = a + r'\obj_pq'
    # anet = a + r'\obj_anet'
    # snet = a + r'\obj_snet'
    # equ = a +'\obj_equilibrium'
    # multinets=a+r'\obj_multinets'

    # #file = pq + r'\baku_quad2000.obj'
    # #file = anet + r'\knet1.obj'
    # file = snet + r'\cmc1.obj'
    # #file = multinets + r'\2by2.obj'
    # file = equ + '\quad_dome.obj'
    
    
    # file = local + r'\obj_PQ' + r'\evolute_ex8_new2_circular.obj'

    # arch = local + r'\obj_architecture'
    # #file = arch+r'\TrainStation_quad.obj'
    # file= local+r'\obj_all' + r'\twist_circle_srf.obj'
    # # file = arch + r'\force_faired_5iters_tf100_ff1_3257ms_cut.obj'
    # #
    #file = local + '/obj_anet' + '/agg-bolun.obj' ##M1_eq TC2_eq.
    # file = local + '/obj_anet' + '/agg1-bolun.obj' ##M1_eq TC2_eq.
    #file = local + '/obj_anet' + '/aag-bolun.obj' ##M1_eq TC2_eq.
    #file = local + '/obj_anet' + '/stripPy.obj' ##M1_eq TC2_eq.
    #file = local + '/obj_anet' + '/mesh_initialization/3strip.obj' ##M1_eq TC2_eq.
    # file = local + '/obj_anet' + '/mesh_initialization/2faces3.obj' ##M1_eq TC2_eq.
    file = local + '/obj_anet' + '/mesh_initialization/stripp111.obj' ##M1_eq TC2_eq.
    file = local + '/obj_anet' + '/mesh_initialization/stripp17.obj' ##M1_eq TC2_eq.
    # file = local + '/obj_anet' + '/singltonQuad.obj' ##M1_eq TC2_eq.
    # file = local + '/obj_anet' + '/mesh_initialization/stripp11.obj' ##M1_eq TC2_eq.
    # file = local + '/obj_anet' + '/mesh_initialization/3quads.obj' ##M1_eq TC2_eq.
    #file = local + '/obj_anet' + '/mesh_initialization/7.obj' ##M1_eq TC2_eq.
    #file = local + '/mesh4.obj' ##M1_eq TC2_eq.
    #file = local + '/obj_anet' + '/polyline.obj' ##M1_eq TC2_eq.
    #file = local + '/obj_anet' + '/PolyLine10V.obj' ##M1_eq TC2_eq.
    # file = local + '/obj_anet' + '/aag2-bolun.obj' ##M1_eq TC2_eq.
    #file = local + r'\obj_caigui' + r'\caigui_6_patch_anchor.obj'
    # wein = local + r'\obj_weingarten'
    # file = wein + r'\wein_6noid_90_1.obj'
    # file = arch+r'\planarizednormalized_six_quad.obj'
    #file = local + r'\obj_pq\multinet_ex3_2.obj'    
    
    #file = local + r'\obj_agnet' + r'\schwarzh_02_diag_unitscale_AAG_AAG.obj'
    #----------------------------------------

    '''Instantiate the sample component'''
    print("Before OrthoNet")
    from opt_gui_orthonet import OrthoNet
    print("After OrthoNet")
    component = OrthoNet()
    #Bolun change this
    # component.optimization_step()  # this only
    component.optimizer.optimize()
    

    '''Instantiate the main geolab application'''
    # import geometrylab as geo ##Hui: replaced by below
    # GUI = geo.gui.GeolabGUI()
    ###########the following is commented out Bolun
    # from archgeolab.archgeometry.gui_basic import GeolabGUI
    # GUI = GeolabGUI()
    # print("After GeolabGUI")
    # '''Add the component to geolab'''
    # GUI.add_component(component)
    # print("After add_component")
    # '''Open an obj file'''
    # GUI.open_obj_file(file)
    
    # '''Open another obj file'''
    # #GUI.open_obj_file(reffile)
    
    # '''Start geolab main loop''' 
    # print("Geo start")
    # GUI.start()

