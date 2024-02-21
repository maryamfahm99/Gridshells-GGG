#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import os

import time

import shutil

import numpy as np

from scipy import sparse

from scipy.sparse import linalg

from traits.api import HasTraits, Instance, Property, Enum, Button,String,\
                       on_trait_change, Float, Bool, Int, Constant, ReadOnly,\
                       List, Array, Range

from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor, HGroup,\
                         Group, ListEditor, Tabbed, VGroup, CheckListEditor,\
                         ArrayEditor, RangeEditor

#------------------------------------------------------------------------------

'''-'''

__author__ = 'Davide Pellis'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                            MeshMovieManager
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


class MeshMovieManager(HasTraits):

    #--------------------------------------------------------------------------
    #                                 Traits
    #--------------------------------------------------------------------------

    record = Bool(False)

    shoot_view = Button()

    shoot_frame = Button()

    make_movie = Button()

    clear_all = Button(label='Clear')

    height = Int(800)

    width = Int(1200)

    fps = Int(20)

    speed = Float(1)

    intro = Float(1)

    final = Float(2)

    title = String('movie')

    key_frame = Int(0)

    K = Int(0)

    time = String('0.00', label='Time [s]')

    duration = Float(8)

    pre_time = Float(1)

    post_time = Float(0)

    camera_rotation = Button(label='Rotate')

    angle = Int(360)

    #--------------------------------------------------------------------------
    #                                  View
    #--------------------------------------------------------------------------

    view = View(
                VGroup(
                       HGroup(
                              VGroup(
                                     Item('fps',resizable=True,),
                                     Item('height',resizable=True,),
                                     Item('intro',resizable=True,),

                                     ),
                              VGroup(
                                     Item('speed',resizable=True,),
                                     Item('width',resizable=True,),
                                     Item('final',resizable=True,),
                                     ),
                              show_border=True),
                       VGroup(
                              Item('key_frame',
                                   editor = RangeEditor(low=0,
                                                        high_name='K',
                                                        mode='slider'),
                                   ),
                              Item('time',
                                   style='readonly'),
                              show_border=True),
                       HGroup(Item('shoot_frame'),
                              Item('shoot_view'),
                              Item('camera_rotation'),
                              show_border=True,
                              show_labels=False),
                       HGroup(
                              Item('clear_all'),
                              Item('make_movie'),
                              Item('record',show_label=True),
                              show_border=True,
                              show_labels=False),
                       ),
                resizable=False,
                width = 0.1,
                title = 'Movie studio',
                )

    #--------------------------------------------------------------------------
    #                             Initialization
    #--------------------------------------------------------------------------

    def __init__(self):
        HasTraits.__init__(self)

        self.plot_callback = None

        self.key_frames = None

        self.key_vertices = None

        self.key_times =  np.array([])

        self.key_views = {}

        self.camera_rotations = {}

        self.key_vertex_data = None

        self.key_face_data = None

        self.key_edge_data = None

        self.key_selected_vertices = []

        self.vertices = None

        self.edge_data = None

        self.face_data = None

        self.face_data_range = None

        self.edge_data_range = None

        self.selected_vertices = {}

        self.t0 = 0

        self.state = 'start'

        self._R = 0

        self._N = 0

        self._meshmanager = None

    #--------------------------------------------------------------------------
    #                            Set attributes
    #--------------------------------------------------------------------------

    @property
    def mesh(self):
        return self._meshmanager.mesh

    @property
    def meshmanager(self):
        return self._meshmanager

    @meshmanager.setter
    def meshmanager(self, plot_manager):
        self._meshmanager = plot_manager

    @property
    def total_frames(self):
        intro = int(self.intro * self.fps)
        final = int(self.final * self.fps)
        T = self._R + self._N + intro + final
        return T

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------

    def initialize_keys(self):
        V = self.mesh.V
        F = self.mesh.F
        E = self.mesh.E
        self.key_vertices = np.zeros((0,V,3))
        self.key_edge_data = np.zeros((0,E))
        self.key_face_data = np.zeros((0,F))
        self.key_times =  np.array([])
        self.key_selected_vertices = []
        self.key_views = {}

    def clear(self):
        self.key_frames = None
        self.key_vertices = None
        self.key_times =  np.array([])
        self.key_views = {}
        self.camera_rotations = {}
        self.key_vertex_data = None
        self.key_face_data = None
        self.key_edge_data = None
        self.key_selected_vertices = []
        self.vertices = None
        self.edge_data = None
        self.face_data = None
        self.face_data_range = None
        self.edge_data_range = None
        self.selected_vertices = {}
        self.t0 = 0
        self.state = 'start'
        self._R = 0
        self._N = 0
        self.K = 0
        self.key_frame = 0
        self.time = '0.00'

    def shoot(self):
        if self.record:
            t = time.time()
            if self.state == 'start':
                self.t0 = t
                self.state = 'record'
                self.initialize_keys()
            if self.state == 'pause':
                self.t0 = t - self.t0
                self.state = 'record'
            if self.key_vertices is None:
                self.initialize_keys()
            self.key_times = np.hstack((self.key_times, np.array([t - self.t0])))
            self.K = len(self.key_times) - 1
            #self.key_frame = int(self.K)
            frame = np.copy(np.array([self.meshmanager.mesh.vertices]))
            if self.meshmanager.edge_data is not None:
                edge_data = np.copy(np.array([self.meshmanager.edge_data]))
            else:
                edge_data = np.zeros((1, self.mesh.E))
            if self.meshmanager.face_data is not None:
                face_data = np.copy(np.array([self.meshmanager.face_data]))
            else:
                face_data = np.zeros((1, self.mesh.F))
            self.key_vertices = np.vstack((self.key_vertices, frame))
            self.key_edge_data = np.vstack((self.key_edge_data, edge_data))
            self.key_face_data = np.vstack((self.key_face_data, face_data))
            Emin = np.min(self.key_edge_data); Emax = np.max(self.key_edge_data)
            Fmin = np.min(self.key_face_data); Fmax = np.max(self.key_face_data)
            self.edge_data_range = ([Emin, Emax])
            self.face_data_range = ([Fmin, Fmax])
            self.key_selected_vertices.append(self.meshmanager.selected_vertices)

    def pause(self, time_step=0.1):
        if self.state == 'pause':
            return
        self.state = 'pause'
        t = time.time()
        self.t0 = t - self.t0 + time_step

    def unpause(self):
        if self.state == 'record':
            return
        self.state = 'record'
        t = time.time()
        self.t0 = t - self.t0

    #--------------------------------------------------------------------------
    #                                 Camera
    #--------------------------------------------------------------------------

    @on_trait_change('shoot_view')
    def shoot_view(self):
        view = self.meshmanager.get_position()
        self.key_views[str(self.key_frame)] = view

    @on_trait_change('camera_rotation')
    def camera_rotation(self):
        rotation = (self.angle, self.duration, self.pre_time, self.post_time)
        self.camera_rotations[str(self.key_frame)] = rotation
        self._R += int(self.duration * self.fps / self.speed)

    def correct_views(self):
        for key in self.camera_rotations:
            if key in self.key_views:
                self.key_views[key][3:6] = np.array([0,0,1])


    #--------------------------------------------------------------------------
    #                             Plot functions
    #--------------------------------------------------------------------------

    def plot_frame(self, frame):
        self.meshmanager.record = True
        self.update_meshmanager(frame)
        self.meshmanager.update_plot()
        self.meshmanager.render()
        self.save_image()
        self.meshmanager.record = False

    def update_meshmanager(self, frame):
        self.meshmanager.mesh.vertices[:,:] = self.vertices[frame,:,:]
        self.meshmanager.edge_data = self.edge_data[frame,:]
        self.meshmanager.face_data = self.face_data[frame,:]
        self.meshmanager.set_view(self.views[frame,:])
        if str(frame) in self.selected_vertices:
            sv = self.selected_vertices[str(frame)]
            self.meshmanager.selected_vertices = sv


    @on_trait_change('key_frame')
    def plot_key_frame(self):
        self.time = "%.2f"%(self.key_times[self.key_frame])
        self.mesh.vertices = np.copy(self.key_vertices[self.key_frame,:,:])
        self.meshmanager.edge_data = np.copy(self.key_edge_data[self.key_frame,:])
        self.meshmanager.face_data = np.copy(self.key_face_data[self.key_frame,:])
        #self.meshmanager.selected_vertices = np.copy(
                                      #self.key_selected_vertices[self.key_frame])
        self.meshmanager.update_plot()

    #--------------------------------------------------------------------------
    #                            Interpolation
    #--------------------------------------------------------------------------

    def interpolate_view(self):
        if len(self.key_times) < 2:
            return
        if len(self.key_views) == 0:
            self.shoot_view()
        self.correct_views()
        K = len(self.key_times) - 1
        Q = np.zeros((K+1,12))
        key = []
        ind = []
        kmin = K + 1
        kmax = 0
        for k in self.key_views:
            if int(k) > kmax:
                kmax = int(k)
            if int(k) < kmin:
                kmin = int(k)
            Q[int(k)] = self.key_views[k][:-1]
            key.append(self.key_times[int(k)])
            ind.append(int(k))
        if kmin > 0:
            key.append(0)
            ind.append(0)
            Q[0,:] = Q[kmin,:]
        if kmax < K:
            ind.append(self.K)
            key.append(self.key_times[-1])
            Q[K,:] = Q[kmax,:]
        M, key1, N = self.interpolation_matrix(key)
        views = np.zeros((N,12))
        views[key1,:] = Q[ind,:]
        for i in range(12):
            X = linalg.spsolve(M,views[:,i], use_umfpack=False)
            views[:,i] = X
        self.views = views

    def interpolate_motion(self):
        if len(self.key_times) < 2:
            return
        M, key, N = self.interpolation_matrix(self.key_times)
        self.key_frames = key
        self._N = N
        np.savetxt('test.txt', M.toarray(),fmt='%.1f')
        V = self.mesh.V
        vertices = np.zeros((N,V,3))
        for v in range(V):
            KV = self.key_vertices[:,v,:]
            P = np.zeros((N,3))
            P[key] = KV
            X = linalg.spsolve(M,P[:,0], use_umfpack=False)
            Y = linalg.spsolve(M,P[:,1], use_umfpack=False)
            Z = linalg.spsolve(M,P[:,2], use_umfpack=False)
            P = np.vstack((X,Y,Z)).T
            vertices[:,v,:] = P
        self.vertices = vertices

        E = self.mesh.E
        edge_data = np.zeros((N,E))
        for e in range(E):
            KD = self.key_edge_data[:,e]
            P = np.zeros(N)
            P[key] = KD
            X = linalg.spsolve(M,P, use_umfpack=False)
            edge_data[:,e] = X
        self.edge_data = edge_data
        e_max = np.max(np.abs(edge_data))
        self.meshmanager.edge_data_range = [-e_max,e_max]

        F = self.mesh.F
        face_data = np.zeros((N,F))
        for f in range(F):
            KD = self.key_face_data[:,f]
            P = np.zeros(N)
            P[key] = KD
            X = linalg.spsolve(M,P, use_umfpack=False)
            face_data[:,f] = X
        self.face_data = face_data
        f_max = np.max(np.abs(face_data))
        self.meshmanager.face_data_range = [-f_max,f_max]

        for i in range(len(key)):
            self.selected_vertices[str(key[i])] = self.key_selected_vertices[i]

    def interpolation_matrix(self, key):
        duration = self.key_times[-1]
        N = int(duration * (self.fps/self.speed)) + 1
        key = np.array(key) * (self.fps/self.speed)
        key = np.array(key, dtype=np.int)
        n = np.arange(N)
        i = np.hstack((n,n,n,n,n))
        j = np.hstack((n, np.roll(n,1), np.roll(n,-1)))
        j = np.hstack((j, np.roll(n,2), np.roll(n,-2)))
        d00 = np.repeat(3., N)
        d1L = np.repeat(-2., N)
        d1R = np.repeat(-2., N)
        d2L = np.repeat(0.5, N)
        d2R = np.repeat(0.5, N)
        d1L[1] = -1
        d1R[N-2] = -1
        d2L[1] = 0
        d00[1] = 2.5
        d00[N-2] = 2.5
        d2R[N-2] = 0
        d00[key] = 1
        d1L[key] = 0
        d1R[key] = 0
        d2L[key] = 0
        d2R[key] = 0
        data = np.hstack((d00,d1L,d1R,d2L,d2R))
        M = sparse.coo_matrix((data,(i,j)), shape=(N,N))
        M = sparse.csc_matrix(M)
        return M, key, N

    def smooth_rotation(self, degrees, n):
        phi = np.linspace(0, 1, n//2)
        phi = 1 / (1 + np.e**(-15*(phi-0.4)))
        phi = np.hstack((phi,phi[::-1]))
        phi = degrees * phi / np.sum(phi)
        if len(phi) < n:
            phi = np.hstack((phi,[phi[-1]]))
        return phi

    #--------------------------------------------------------------------------
    #                                Movie maker
    #--------------------------------------------------------------------------

    def make_paths(self):
        c_path = os.getcwd()
        n = 0
        forbidden = True
        while forbidden:
            video_name = '{}_{}.mp4'.format(self.title, n)
            folder = '{}_{}_img'.format(self.title, n)
            out_path = os.path.join(c_path,folder)
            video_path = os.path.join(c_path,video_name)
            prefix = '{}_{}'.format(self.title, n)
            n += 1
            if not os.path.exists(out_path) and not os.path.exists(video_path):
                os.makedirs(out_path)
                forbidden = False
        self._ext = '.png'
        self._out_path = out_path
        self._prefix = prefix
        self._padding = len(str(self.total_frames))
        self._i = 0

    def save_image(self):
        self._i += 1
        zeros = '0'*(self._padding - len(str(self._i)))
        suffix = '{}_{}{}{}'.format(self._prefix, zeros, self._i, self._ext)
        name = os.path.join(self._out_path, suffix)
        self._last_image = name
        self.meshmanager.save(name , size=(self.width,self.height))

    def yield_image(self, time):
        N = int(time * self.fps)
        for i in range(N):
            self._i += 1
            zeros = '0'*(self._padding - len(str(self._i)))
            suffix = '{}_{}{}{}'.format(self._prefix, zeros,self._i,self._ext)
            name = os.path.join(self._out_path, suffix)
            shutil.copyfile(self._last_image, name)

    def rotate_view(self, time, degrees):
        N = int(time * self.fps)
        phi = self.smooth_rotation(degrees, N)
        for j in range(N):
            center = self.mesh.mesh_center()
            self.meshmanager.camera_zRotation(center, phi[j])
            self.meshmanager.render()
            self.save_image()

    def update_view(self, frame):
        k = np.where(self.key_frames == frame)[0]
        try:
            k = str(k[0])
            if k in self.camera_rotations:

                degrees = self.camera_rotations[k][0]
                time = self.camera_rotations[k][1]
                pre_time = self.camera_rotations[k][2]
                post_time = self.camera_rotations[k][3]
                self.yield_image(pre_time)
                self.rotate_view(time, degrees)
                self.yield_image(post_time)
        except:
                pass

    def make_mp4(self):
        form = '{}_%0{}d{}'.format(self._prefix, self._padding, self._ext)
        ffmpeg_fname = os.path.join(self._out_path, form)
        cm = 'ffmpeg -f image2 -r {} -i {} -vcodec mpeg4 -qscale:v 1 -y {}.mp4'
        cm = cm.format(self.fps, ffmpeg_fname, self._prefix)
        try:
            os.system(cm)
            shutil.rmtree(self._out_path)
        except:
            print('Video not created: install FFmpeg on your system!')

    @on_trait_change('make_movie')
    def make_movie(self):
        self.interpolate_motion()
        self.interpolate_view()
        self.make_paths()
        self.meshmanager.clear()
        self.plot_frame(0)
        self.yield_image(self.intro)
        for i in range(self._N):
            self.plot_frame(i)
            self.update_view(i)
        self.yield_image(self.final)
        self.make_mp4()
