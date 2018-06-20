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

########
# DLDA #
########

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

##########
# Parzen #
##########

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

############################
# ElasticNet / ElasticNetC #
############################

def softt(beta0, lmb):
    """Soft thresholding.
    """

    lmb_half = lmb / 2.0
    idx = np.abs(beta0) < lmb_half
    beta = beta0 - np.sign(beta0) * lmb_half
    beta[idx] = 0.0

    return beta


def elasticnet_base(x, y, lmb, eps, supp=True, tol=0.01):
    """Elastic Net Regularization via Iterative Soft Thresholding.

    `x` should be centered and normalized by columns, and `y`
    should be centered.

    Computes the coefficient vector which solves the elastic-net
    regularization problem min {\|\| X beta - Y \|\|^2 + lambda(\|beta\|^2_2
    + eps \|beta\|_1}. The solution beta is computed via iterative
    soft-thresholding, with damping factor 1/(1+eps*lambda), thresholding
    factor eps*lambda, null initialization vector and step
    1 / (eig_max(XX^T)*1.1).

    :Parameters:
       x : 2d array_like object (N x P)
          matrix of regressors
       y : 1d array_like object (N)
          response
       lmb : float
          regularization parameter controlling overfitting.
          `lmb` can be tuned via cross validation.
       eps : float
          correlation parameter preserving correlation
          among variables against sparsity. The solutions
          obtained for different values of the correlation
          parameter have the same prediction properties but
          different feature representation.
       supp : bool
          if True, the algorithm stops when the support of beta
          reached convergence. If False, the algorithm stops when
          the coefficients reached convergence, that is when
          the beta_{l}(i) - beta_{l+1}(i) > tol * beta_{l}(i)
          for all i.
       tol : double
          tolerance for convergence

    :Returns:
       beta, iters : 1d numpy array, int
          beta, number of iterations performed
    """

    xarr = np.asarray(x)
    yarr = np.asarray(y)

    xt = xarr.T
    xy = np.dot(xt, yarr)

    n, p = xarr.shape
    if p >= n:
        xx = np.dot(xarr, xt)
    else:
        xx = np.dot(xt, xarr)

    step = 1. / (np.linalg.eigvalsh(xx).max() * 1.1)
    lmb = lmb * n * step
    damp = 1. / (1 + lmb * eps)
    beta0 = np.zeros(p, dtype=np.float)

    k, i = 0, 0
    kmax = 100000
    imax = 10

    tmp = beta0 + (step * (np.dot(-xt, np.dot(xarr, beta0)) + xy))
    beta = softt(tmp, lmb) * damp

    while ((k < kmax) and (i < imax)):
        if supp:
            if np.any(((beta0 != 0) ^ (beta != 0)) != 0):
                i = 0
        else:
            if np.any(np.abs(beta0 - beta) > tol * np.abs(beta0)):
                i = 0

        beta0 = beta
        tmp = beta0 + (step * (np.dot(-xt, np.dot(xarr, beta0)) + xy))
        beta = softt(tmp, lmb) * damp

        i, k = i+1, k+1
        imax = np.max((1000, 10**np.floor(np.log10(k))))

    return beta, k


class ElasticNet(object):
    """Elastic Net Regularization via Iterative Soft Thresholding.

    Computes the coefficient vector which solves the elastic-net
    regularization problem min {\|\| X beta - Y \|\|^2 + lambda(\|beta\|^2_2
    + eps \|beta\|_1}. The solution beta is computed via iterative
    soft-thresholding, with damping factor 1/(1+eps*lambda), thresholding
    factor eps*lambda, null initialization vector and step
    1 / (eig_max(XX^T)*1.1).
    """

    def __init__(self, lmb, eps, supp=True, tol=0.01):
        """Initialization.

        :Parameters:
            lmb : float
               regularization parameter controlling overfitting.
               `lmb` can be tuned via cross validation.
            eps : float
               correlation parameter preserving correlation
               among variables against sparsity. The solutions
               obtained for different values of the correlation
               parameter have the same prediction properties but
               different feature representation.
            supp : bool
               if True, the algorithm stops when the support of beta
               reached convergence. If False, the algorithm stops when
               the coefficients reached convergence, that is when
               the beta_{l}(i) - beta_{l+1}(i) > tol * beta_{l}(i)
               for all i.
            tol : double
               tolerance for convergence
        """

        self._lmb = float(lmb)
        self._eps = float(eps)
        self._supp = supp
        self._tol = float(tol)

        self._beta = None
        self._beta0 = None
        self._iters = None

    def learn(self, x, y):
        """Compute the regression coefficients.

        :Parameters:
          x : 2d array_like object (N x P)
            matrix of regressors
          y : 1d array_like object (N)
            response
        """

        xarr = np.array(x, dtype=np.float, copy=True)
        yarr = np.array(y, dtype=np.float, copy=True)

        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")

        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y shape mismatch")

        # center x
        xmean = np.mean(xarr, axis=0)
        xarr -= xmean

        # normalize x
        xnorm = np.sqrt(np.sum((xarr)**2, axis=0))
        xarr /= xnorm

        # center y
        ymean = np.mean(yarr)

        self._beta, self._iters = elasticnet_base(xarr, yarr,
            lmb=self._lmb, eps=self._eps, supp=self._supp, tol=self._tol)
        self._beta /= xnorm
        self._beta0 = ymean - np.dot(xmean, self._beta)

    def pred(self, t):
        """Compute the predicted response.

        :Parameters:
           t : 1d or 2d array_like object ([M,] P)
              test data

        :Returns:
           p : float or 1d numpy array
              predicted response
        """

        if self._beta0 is None:
            raise ValueError('no mode computed; run learn() first')

        tarr = np.asarray(t, dtype=np.float)

        if tarr.ndim > 2 or tarr.ndim < 1:
            raise ValueError("t must be an 1d or a 2d array_like object")

        try:
            p = np.dot(tarr, self._beta) + self._beta0
        except ValueError:
            raise ValueError("t, beta: shape mismatch")

        return p

    def iters(self):
        """Return the number of iterations performed.
        """

        return self._iters

    def beta(self):
        """Return b_1, ..., b_p.
        """

        return self._beta

    def beta0(self):
        """Return b_0.
        """

        return self._beta0


class ElasticNetC(ElasticNet):
    """Elastic Net Regularization via Iterative Soft Thresholding
    for classification.

    See the ElasticNet class documentation.
    """

    def __init__(self, lmb, eps, supp=True, tol=0.01):
        """Initialization.

        :Parameters:
            lmb : float
               regularization parameter controlling overfitting.
               `lmb` can be tuned via cross validation.
            eps : float
               correlation parameter preserving correlation
               among variables against sparsity. The solutions
               obtained for different values of the correlation
               parameter have the same prediction properties but
               different feature representation.
            supp : bool
               if True, the algorithm stops when the support of beta
               reached convergence. If False, the algorithm stops when
               the coefficients reached convergence, that is when
               the beta_{l}(i) - beta_{l+1}(i) > tol * beta_{l}(i)
               for all i.
            tol : double
               tolerance for convergence
        """

        ElasticNet.__init__(self, lmb, eps, supp=True, tol=0.01)
        self._labels = None

    def learn(self, x, y):
        """Compute the classification coefficients.

        :Parameters:
          x : 2d array_like object (N x P)
            matrix
          y : 1d array_like object integer (N)
            class labels
        """

        yarr = np.asarray(y, dtype=np.int)
        self._labels = np.unique(yarr)

        k = self._labels.shape[0]
        if k != 2:
            raise ValueError("number of classes must be = 2")

        ynew = np.where(yarr == self._labels[0], -1, 1)

        ElasticNet.learn(self, x, ynew)

    def pred(self, t):
        """Compute the predicted labels.

        :Parameters:
           t : 1d or 2d array_like object ([M,] P)
              test data

        :Returns:
           p : integer or 1d numpy array
              predicted labels
        """

        p = ElasticNet.pred(self, t)
        ret = np.where(p > 0, self._labels[1], self._labels[0])

        return ret

    def w(self):
        """Returns the coefficients.
        """
        if ElasticNet.beta(self) is None:
            raise ValueError("No model computed")

        return super(ElasticNetC, self).beta()

    def bias(self):
        """Returns the bias.
        """
        if ElasticNet.beta0(self) is None:
            raise ValueError("No model computed")

        return super(ElasticNetC, self).beta0()

    def labels(self):
        """Outputs the name of labels.
        """

        return self._labels
