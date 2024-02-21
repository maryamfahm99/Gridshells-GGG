#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import os

import shutil

import numpy as np

from scipy import sparse

from scipy.sparse import linalg

from traits.api import HasTraits, Instance, Property, Enum, Button,String,\
                       on_trait_change, Float, Bool, Int, Constant, ReadOnly,\
                       List, Array, Range, Str

from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor, HGroup,\
                         Group, ListEditor, Tabbed, VGroup, CheckListEditor,\
                         ArrayEditor, RangeEditor

from tvtk.pyface.scene_editor import SceneEditor

from mayavi.tools.mlab_scene_model import MlabSceneModel

from mayavi.core.ui.mayavi_scene import MayaviScene

from pyface.image_resource import ImageResource

from geometrylab.gui.geolabscene import GeolabScene

from geometrylab.vtkplot.plotmanager import PlotManager

#------------------------------------------------------------------------------

'''-'''

__author__ = 'Davide Pellis'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                            MeshMovieManager
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

'''viewer.py: The viewer for vtkplot souce classes'''

__author__ = 'Davide Pellis'

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                                    Viewer
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class Screen(HasTraits):

    scene = Instance(MlabSceneModel, ())

    editor = SceneEditor(scene_class=GeolabScene)

    height = Int(1080)

    width = Int(1920)

    @property
    def position(self):
        return self.editor.scene_class._position

    @position.setter
    def position(self, position):
        self.editor.scene_class._position = position

    @property
    def background(self):
        return self._plotmanager.background

    @background.setter
    def background(self, background):
        self._plotmanager.background = background

    def start(self):
        view = View(Item('scene',
                     editor=self.editor,
                     show_label=False,
                     resizable=True,
                     height=self.height,
                     width=self.width,
                     ),
                resizable=True,
                title = 'Geometrylab Animator',
                icon = ImageResource(path + '/gui/img/new2/logo3.png')
                )
        self.configure_traits(view=view)


class Animator(HasTraits):

    # -------------------------------------------------------------------------
    #                                  Traits
    # -------------------------------------------------------------------------

    scene = Instance(MlabSceneModel, ())

    editor = SceneEditor(scene_class=GeolabScene)

    height = Int(1080)

    width = Int(1920)

    frames = Int(10)

    frame = Int(0)

    fps = Int(0)

    title = Str('movie')


    # -------------------------------------------------------------------------
    #                                   View
    # -------------------------------------------------------------------------
    view = View(VGroup(Item('title'),
                       Item('height'),
                       Item('width'),
                       Item('fps'),
                       Item('frames'),
                       Item('frame'),
                       show_border=True),
                resizable=True,
                title = 'Geometrylab Animator',
                icon = ImageResource(path + '/gui/img/new2/logo3.png')
                )


    # -------------------------------------------------------------------------
    #                                 Initialize
    # -------------------------------------------------------------------------

    def __init__(self):
        HasTraits.__init__(self)

        self._plotmanager = PlotManager(scene_model=self.scene)

        self._position = None

        self.title = 'movie'

        self.fps = 30

        self._make_paths()



    # -------------------------------------------------------------------------
    #                                 Methods
    # -------------------------------------------------------------------------

    def start(self):

        self.configure_traits()


    #--------------------------------------------------------------------------
    #                                 Camera
    #--------------------------------------------------------------------------

    def correct_views(self):
        for key in self.camera_rotations:
            if key in self.key_views:
                self.key_views[key][3:6] = np.array([0,0,1])

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

    def clear(self):
        self._make_paths()

    def _make_paths(self):
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
        self._padding = len(str(self.frames))
        self._i = 0

    def save_image(self):
        self._i += 1
        zeros = '0'*(self._padding - len(str(self._i)))
        suffix = '{}_{}{}{}'.format(self._prefix, zeros, self._i, self._ext)
        name = os.path.join(self._out_path, suffix)
        self._last_image = name
        self.meshmanager.save(name)

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


A = Animator()
A.start()