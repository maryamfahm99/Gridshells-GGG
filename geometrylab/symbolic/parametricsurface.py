#!/usr/bin/env python

# -*- coding: utf-8 -*-



from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

import sympy as sy

#------------------------------------------------------------------------------

from geometrylab.geometry import mesh_plane

#------------------------------------------------------------------------------

'''_'''

__author__ = 'Davide Pellis'

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                           Parametric Surface Class
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class ParametricSurface:

    def __init__(self, x=None, y=None, z=None):

        self._x = x

        self._y = y

        self._z = z

        self._f = None

        self._fu = None

        self._fv = None

        self._I = None

        self._Iinv = None

        self._II = None

        self._S = None

        self._k1 = None

        self._k2 = None

        self._update = True

    #--------------------------------------------------------------------------

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, equation):
        self._x = equation
        self._update = True

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, equation):
        self._y = equation
        self._update = True

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, equation):
        self._z = equation
        self._update = True

    @property
    def evaluation_points(self):
        return self._evaluation_points

    @evaluation_points.setter
    def evaluation_points(self, points):
        p = np.array(points)
        if len(p.shape) == 1:
            p = np.array([points])
        self._evaluation_points = p

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    def __str__(self):
        s = 'Parametric surface: x = {}, y = {}, z = {}'
        out = s.format(self.x, self.y, self.z)
        return out

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    def _dummify_undefined_functions(self, expr):
        mapping = {}
        for der in expr.atoms(sy.Derivative):
            f_name = der.expr.func.__name__
            var_names = [var.name for var in der.variables]
            name = "d%s_d%s" % (f_name, 'd'.join(var_names))
            mapping[der] = sy.Symbol(name)
        from sympy.core.function import AppliedUndef
        for f in expr.atoms(AppliedUndef):
            f_name = f.func.__name__
            mapping[f] = sy.Symbol(f_name)
        return expr.subs(mapping)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    def _update_values(self):
        if not self._update:
            return
        u, v = sy.symbols('u v')
        self._f = sy.Matrix((sy.sympify(self._x),
                             sy.sympify(self._y),
                             sy.sympify(self._z)))
        self._fu = sy.diff(self._f,u)
        self._fv = sy.diff(self._f,v)
        self._I = sy.Matrix((((self._fu.transpose()*self._fu)[0,0],
                              (self._fu.transpose()*self._fv)[0,0]),
                             ((self._fu.transpose()*self._fv)[0,0],
                              (self._fv.transpose()*self._fv)[0,0])))
        self._g = self._I[0,0]*self._I[1,1] - self._I[0,1]*self._I[1,0]

        self._Iinv = self._g**-1 * sy.Matrix(((self._I[1,1], -self._I[0,1]),
                                              (-self._I[1,0], self._I[0,0])))
        fuu = sy.diff(self._fu, u)
        fuv = sy.diff(self._fu, v)
        fvv = sy.diff(self._fv, v)
        self._n = self._fu.cross(self._fv) / (self._fu.cross(self._fv)).norm()
        self._II = sy.Matrix((((fuu.transpose()*self._n)[0,0],
                               (fuv.transpose()*self._n)[0,0]),
                              ((fuv.transpose()*self._n)[0,0],
                               (fvv.transpose()*self._n)[0,0])))

        self._S = self._Iinv * self._II

        self._k1 = (self._S[0,0]/2
                    + self._S[1,1]/2
                    - sy.sqrt(self._S[0,0]**2
                              - 2*self._S[0,0]*self._S[1,1]
                              + 4*self._S[0,1]*self._S[1,0]
                              + self._S[1,1]**2)/2)

        self._k2 = (self._S[0,0]/2
                    + self._S[1,1]/2
                    + sy.sqrt(self._S[0,0]**2
                              - 2*self._S[0,0]*self._S[1,1]
                              + 4*self._S[0,1]*self._S[1,0]
                              + self._S[1,1]**2)/2)

        self._update = False

    def _format_vector(self, vector):
        v = np.array(vector)
        if v.shape != self.evaluation_points.shape:
            v = np.tile(v, (len(self.evaluation_points),1))
        return v

    def _evaluate_function(self, function):
        #function = sy.re(self._dummify_undefined_functions(function))
        u, v = sy.symbols('u v')
        f = sy.lambdify((u,v), function, 'numpy')
        value = f(self.evaluation_points[:,0], self.evaluation_points[:,1])
        if type(value) == int or type(value) == float:
            return np.repeat(value, len(self._evaluation_points))
        else:
            return value

    def _evaluate_vector(self, vector):
        x = self._evaluate_function(vector[0,0])
        y = self._evaluate_function(vector[1,0])
        z = self._evaluate_function(vector[2,0])
        return np.column_stack((x, y, z))

    def _evaluate_matrix(self, matrix):
        M11 = self._evaluate_function(matrix[0,0])
        M12 = self._evaluate_function(matrix[0,1])
        M21 = self._evaluate_function(matrix[1,0])
        M22 = self._evaluate_function(matrix[1,1])
        return np.moveaxis(np.array([[M11,M12],[M21,M22]]), -1, 0)

    def evaluate(self):
        self._update_values()
        return self._evaluate_vector(self._f)

    def f(self):
        self._update_values()
        return self._f

    def fu(self):
        self._update_values()
        return self._evaluate_vector(self._fu)

    def fv(self):
        self._update_values()
        return self._evaluate_vector(self._fv)

    def n(self):
        self._update_values()
        return self._evaluate_vector(self._n)

    def I(self, evaluate=True):
        self._update_values()
        if not evaluate:
            return self._I
        else:
            return self._evaluate_matrix(self._I)

    def II(self, evaluate=True):
        self._update_values()
        if not evaluate:
            return self._II
        else:
            return self._evaluate_matrix(self._II)

    def S(self, evaluate=True):
        self._update_values()
        if not evaluate:
            return self._S
        else:
            return self._evaluate_matrix(self._S)

    def k1(self, evaluate=True):
        self._update_values()
        if not evaluate:
            return self._k1
        else:
            return self._evaluate_function(self._k1)

    def k2(self, evaluate=True):
        self._update_values()
        if not evaluate:
            return self._k2
        else:
            return self._evaluate_function(self._k2)

    def K(self, evaluate=True):
        self._update_values()
        if not evaluate:
            return self._k1 * self._k2
        else:
            return (self._evaluate_function(self._k1) *
                    self._evaluate_function(self._k2))

    def H(self, evaluate=True):
        self._update_values()
        if not evaluate:
            return 0.5*(self._k1 + self._k2)
        else:
            return 0.5*(self._evaluate_function(self._k1) +
                        self._evaluate_function(self._k2))

    def principal_directions(self):
        self._update_values()
        S = self.S()
        k1 = self.k1()
        k2 = self.k2()
        D1 = np.zeros((len(k1),2))
        D2 = np.zeros((len(k1),2))
        i = S[:,1,0] != 0
        j = np.logical_and(S[:,1,0] == 0, S[:,0,1] != 0)
        k = np.logical_and(S[:,1,0] == 0, S[:,0,1] == 0)
        m = np.logical_and(k, S[:,0,0] <= S[:,1,1])
        n = np.logical_and(k, S[:,0,0] > S[:,1,1])
        D1[i,0] = k2[i] - S[i,1,1]
        D1[i,1] = S[i,1,0]
        D2[i,0] = k1[i] - S[i,1,1]
        D2[i,1] = S[i,1,0]
        D1[j,0] = S[j,0,1]
        D1[j,1] = k2[j] - S[j,0,0]
        D2[j,0] = S[j,0,1]
        D2[j,1] = k1[j] - S[j,0,0]
        D1[m,0] = D2[m,1] = 1
        D1[m,1] = D2[m,0] = 0
        D1[n,0] = D2[n,1] = 0
        D1[n,1] = D2[n,0] = 1
        D1 = D1/np.linalg.norm(D1, axis=1, keepdims=True)
        D2 = D2/np.linalg.norm(D2, axis=1, keepdims=True)
        return ((k1, D1), (k2, D2))

    def principal_curvatures(self):
        return (self.k1, self.k2)

    def dot(self, vector_1, vector_2):
        v_1 = self._format_vector(vector_1)
        v_2 = self._format_vector(vector_2)
        return np.einsum('ki,kij,kj->k', v_1, self.I(), v_2)

    def normal_curvature(self, direction):
        d = self._format_vector(direction)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        return np.einsum('ki,kij,kj->k', d, self.S(), d)

    def norm(self, vector):
        return self.dot(vector, vector)**0.5

    def vector(self, vector):
        v = self._format_vector(vector)
        u = np.column_stack((v[:,0],v[:,0],v[:,0]))
        v = np.column_stack((v[:,1],v[:,1],v[:,1]))
        return u*self.fu() + v*self.fv()

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------

    def mesh(self, u_domain=(-1,1), v_domain=(-1,1), n_u=20, n_v=20):
        M = mesh_plane(n_u, n_v, u_domain[0], u_domain[1],
                        v_domain[0], v_domain[1])
        self.evaluation_points = np.copy(M.vertices[:,[0,1]])
        M.vertices = self.evaluate()
        return M





#------------------------------------------------------------------------------

if __name__ == '__main__':

    f = ParametricSurface('u','v','u**2-v**2')

    f.evaluation_points = np.array([[-1,0]])

    v1 = [[-1,1]]

    A,B = f.principal_directions()

    print(A[1],B[1])
    print(f.dot(A[1],B[1]))
    print(f.normal_curvature(v1))
    print(f.n())










