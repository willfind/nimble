"""
Method implementations and helpers acting specifically on each element
Sparse object.
"""

from __future__ import absolute_import

import numpy

import nimble
from .elements import Elements
from .elements_view import ElementsView
from .dataHelpers import denseCountUnique

scipy = nimble.importModule('scipy')
if scipy is not None:
    from scipy.sparse import coo_matrix

class SparseElements(Elements):
    """
    Sparse method implementations performed on each element.

    Parameters
    ----------
    base : Sparse
        The Sparse instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _transform_implementation(self, toTransform, points, features):
        if toTransform.preserveZeros:
            self._transformEachElement_zeroPreserve_implementation(
                toTransform, points, features)
        else:
            self._transformEachElement_noPreserve_implementation(
                toTransform, points, features)

    ################################
    # Higher Order implementations #
    ################################

    def _calculate_implementation(self, function, points, features,
                                  preserveZeros, outputType):
        if not isinstance(self._base, nimble.data.BaseView):
            data = self._base.data.data
            row = self._base.data.row
            col = self._base.data.col
        else:
            # initiate generic implementation for view types
            preserveZeros = False
        # all data
        if preserveZeros and points is None and features is None:
            try:
                data = function(data)
            except Exception:
                function.otypes = [numpy.object_]
                data = function(data)
            shape = self._base.data.shape
            values = coo_matrix((data, (row, col)), shape=shape)
            # note: even if function transforms nonzero values into zeros
            # our init methods will filter them out from the data attribute
            return values
        # subset of data
        if preserveZeros:
            dataSubset = []
            rowSubset = []
            colSubset = []
            for idx, val in enumerate(data):
                if row[idx] in points and col[idx] in features:
                    rowSubset.append(row[idx])
                    colSubset.append(col[idx])
                    dataSubset.append(val)
            dataSubset = function(dataSubset)
            values = coo_matrix((dataSubset, (rowSubset, colSubset)))
            # note: even if function transforms nonzero values into zeros
            # our init methods will filter them out from the data attribute
            return values
        # zeros not preserved
        return self._calculate_genericVectorized(
            function, points, features, outputType)

    #########################
    # Query implementations #
    #########################

    def _countUnique_implementation(self, points, features):
        uniqueCount = {}
        isView = isinstance(self._base, nimble.data.BaseView)
        if points is None and features is None and not isView:
            source = self._base
        else:
            pWanted = points if points is not None else slice(None)
            fWanted = features if features is not None else slice(None)
            source = self._base[pWanted, fWanted]
        uniqueCount = denseCountUnique(source.data.data)
        totalValues = (len(source.points) * len(source.features))
        numZeros = totalValues - len(source.data.data)
        if numZeros > 0:
            uniqueCount[0] = numZeros
        return uniqueCount

    #############################
    # Numerical implementations #
    #############################

    def _multiply_implementation(self, other):
        """
        Perform element wise multiplication of this nimble Base object
        against the provided other nimble Base object. Both objects must
        contain only numeric data. The pointCount and featureCount of
        both objects must be equal. The types of the two objects may be
        different, but the returned object will be the inplace
        modification of the calling object.
        """
        # CHOICE OF OUTPUT WILL BE DETERMINED BY SCIPY!!!!!!!!!!!!
        # for other.data as any dense or sparse matrix
        toMul = None
        directMul = isinstance(other, (nimble.data.Sparse, nimble.data.Matrix))
        notView = not isinstance(other, nimble.data.BaseView)
        if directMul and notView:
            toMul = other.data
        else:
            toMul = other.copy(to='numpyarray')
        raw = self._base.data.multiply(coo_matrix(toMul))
        if scipy.sparse.isspmatrix(raw):
            self._base.data = raw.tocoo()
        else:
            self._base.data = coo_matrix(raw, shape=self._base.data.shape)
        self._base._sorted = None

    ######################
    # Structural helpers #
    ######################

    def _transformEachElement_noPreserve_implementation(self, toTransform,
                                                        points, features):
        # returns None if outside of the specified points and feature so that
        # when calculateForEach is called we are given a full data object
        # with only certain values modified.
        def wrapper(value, pID, fID):
            if points is not None and pID not in points:
                return None
            if features is not None and fID not in features:
                return None

            if toTransform.oneArg:
                return toTransform(value)
            else:
                return toTransform(value, pID, fID)

        # perserveZeros is always False in this helper, skipNoneReturnValues
        # is being hijacked by the wrapper: even if it was False, Sparse can't
        # contain None values.
        ret = self.calculate(wrapper, None, None, preserveZeros=False,
                             skipNoneReturnValues=True, useLog=False)

        pnames = self._base.points._getNamesNoGeneration()
        fnames = self._base.features._getNamesNoGeneration()
        self._base.referenceDataFrom(ret, useLog=False)
        self._base.points.setNames(pnames, useLog=False)
        self._base.features.setNames(fnames, useLog=False)
        self._base._sorted = None


    def _transformEachElement_zeroPreserve_implementation(
            self, toTransform, points, features):
        for index, val in enumerate(self._base.data.data):
            pID = self._base.data.row[index]
            fID = self._base.data.col[index]
            if points is not None and pID not in points:
                continue
            if features is not None and fID not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(val)
            else:
                currRet = toTransform(val, pID, fID)

            self._base.data.data[index] = currRet

class SparseElementsView(ElementsView, SparseElements):
    """
    Limit functionality of SparseElements to read-only.

    Parameters
    ----------
    base : SparseView
        The SparseView instance that will be queried.
    """
    pass
