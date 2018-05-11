## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2011 mlpy Developers.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np


class DLDA:
    """Diagonal Linear Discriminant Analysis classifier.
    The algorithm uses the procedure called Nearest Shrunken
    Centroids (NSC).
    """

    def __init__(self, delta):
        """Initialization.

        :Parameters:
           delta : float
              regularization parameter
        """

        self._delta = float(delta)
        self._xstd = None # s_j
        self._dprime = None # d'_kj
        self._xmprime = None # xbar'_kj
        self._p = None # class prior probability
        self._labels = None

    def learn(self, x, y):
        """Learning method.

        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer
              target values (N)
        """

        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)

        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")

        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        self._labels = np.unique(yarr)
        k = self._labels.shape[0]

        if k < 2:
            raise ValueError("number of classes must be >= 2")

        xm = np.mean(xarr, axis=0)
        self._xstd = np.std(xarr, axis=0, ddof=1)
        s0 = np.median(self._xstd)
        self._dprime = np.empty((k, xarr.shape[1]), dtype=np.float)
        self._xmprime = np.empty((k, xarr.shape[1]), dtype=np.float)
        n = yarr.shape[0]
        self._p = np.empty(k, dtype=np.float)

        for i in range(k):
            yi = (yarr == self._labels[i])
            xim = np.mean(xarr[yi], axis=0)
            nk = np.sum(yi)
            mk = np.sqrt(np.float_power(nk,1) - np.float_power(n,-1))
            d = (xim - xm) / (mk * (self._xstd + s0))

            # soft thresholding
            tmp = np.abs(d) - self._delta
            tmp[tmp<0] = 0.0
            self._dprime[i] = np.sign(d) * tmp

            self._xmprime[i] = xm + (mk * (self._xstd + s0) * self._dprime[i])
            self._p[i] = float(nk) / float(n)

    def labels(self):
        """Outputs the name of labels.
        """

        return self._labels

    def sel(self):
        """Returns the most important features (the features that
        have a nonzero dprime for at least one of the classes).
        """

        return np.where(np.sum(self._dprime, axis=0) != 0)[0]

    def dprime(self):
        """Return the dprime d'_kj (C, P), where C is the
        number of classes.
        """

        return self._dprime

    def _score(self, x):
        """Return the discriminant score"""

        return - np.sum((x-self._xmprime)**2/self._xstd**2,
                        axis=1) + (2 * np.log(self._p))

    def _prob(self, x):
        """Return the probability estimates"""

        score = self._score(x)
        tmp = np.exp(score * 0.5)
        return tmp / np.sum(tmp)

    def pred(self, t):
        """Does classification on test vector(s) t.

        :Parameters:
           t : 1d (one sample) or 2d array_like object
              test data ([M,] P)

        :Returns:
           p : int or 1d numpy array
              the predicted class(es) for t is returned.
        """

        if self._xmprime is None:
            raise ValueError("no model computed.")

        tarr = np.asarray(t, dtype=np.float)

        if tarr.ndim == 1:
            return self._labels[np.argmax(self._score(tarr))]
        else:
            ret = np.empty(tarr.shape[0], dtype=np.int)
            for i in range(tarr.shape[0]):
                ret[i] = self._labels[np.argmax(self._score(tarr[i]))]
            return ret

    def prob(self, t):
        """For each sample returns C (number of classes)
        probability estimates.
        """

        if self._xmprime is None:
            raise ValueError("no model computed.")

        tarr = np.asarray(t, dtype=np.float)

        if tarr.ndim == 1:
            return self._prob(tarr)
        else:
            ret = np.empty((tarr.shape[0], self._labels.shape[0]),
                dtype=np.float)
            for i in range(tarr.shape[0]):
                ret[i] = self._prob(tarr[i])
            return ret

class Parzen:
    """Parzen based classifier (binary).
    """

    def __init__(self, kernel=None):
        """Initialization.

        :Parameters:
           kernel : None or mlpy.Kernel object.
              if kernel is None, K and Kt in .learn()
              and in .pred() methods must be precomputed kernel
              matricies, else K and Kt must be training (resp.
              test) data in input space.
        """

        self._alpha = None
        self._b = None
        self._labels = None
        self._kernel = kernel
        self._x = None

    def learn(self, K, y):
        """Compute alpha and b.

        Parameters:
           K: 2d array_like object
              precomputed training kernel matrix (if kernel=None);
              training data in input space (if kernel is a Kernel object)
           y : 1d array_like object (N)
              target values
        """

        K_arr = np.asarray(K, dtype=np.float)
        y_arr = np.asarray(y, dtype=np.int)

        if K_arr.ndim != 2:
            raise ValueError("K must be a 2d array_like object")

        if y_arr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        if K_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("K, y shape mismatch")

        if self._kernel is None:
            if K_arr.shape[0] != K_arr.shape[1]:
                raise ValueError("K must be a square matrix")
        else:
            self._x = K_arr.copy()
            K_arr = self._kernel.kernel(K_arr, K_arr)

        self._labels = np.unique(y_arr)
        if self._labels.shape[0] != 2:
            raise ValueError("number of classes != 2")

        ynew = np.where(y_arr==self._labels[0], -1., 1.)
        n = K_arr.shape[0]

        # from Kernel Methods for Pattern Analysis
        # Algorithm 5.6

        nplus = np.sum(ynew==1)
        nminus = n - nplus
        alphaplus = np.where(ynew==1, np.float_power(nplus,-1), 0)
        alphaminus = np.where(ynew==-1, np.float_power(nminus,-1), 0)
        self._b = -0.5 * (np.dot(np.dot(alphaplus, K_arr), alphaplus) - \
                         np.dot(np.dot(alphaminus, K_arr), alphaminus))
        self._alpha = alphaplus - alphaminus

    def pred(self, Kt):
        """Compute the predicted class.

        :Parameters:
           Kt : 1d or 2d array_like object
              precomputed test kernel matrix. (if kernel=None);
              test data in input space (if kernel is a Kernel object).

        :Returns:
           p : integer or 1d numpy array
              predicted class
        """

        if self._alpha is None:
            raise ValueError("no model computed; run learn()")

        Kt_arr = np.asarray(Kt, dtype=np.float)
        if self._kernel is not None:
            Kt_arr = self._kernel.kernel(Kt_arr, self._x)

        try:
            s = np.sign(np.dot(self._alpha, Kt_arr.T) + self._b)
        except ValueError:
            raise ValueError("Kt, alpha: shape mismatch")

        return np.where(s==-1, self._labels[0], self._labels[1]) \
            .astype(np.int)

    def alpha(self):
        """Return alpha.
        """
        return self._alpha

    def b(self):
        """Return b.
        """
        return self._b

    def labels(self):
        """Outputs the name of labels.
        """

        return self._labels