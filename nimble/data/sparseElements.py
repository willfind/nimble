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
    source : nimble data object
        The object containing point and feature data.
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
        if not isinstance(self._source, nimble.data.BaseView):
            data = self._source.data.data
            row = self._source.data.row
            col = self._source.data.col
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
            shape = self._source.data.shape
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
        isView = self._source.data.data is None
        if points is None and features is None and not isView:
            source = self._source
        else:
            pWanted = points if points is not None else slice(None)
            fWanted = features if features is not None else slice(None)
            source = self._source[pWanted, fWanted]
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
        raw = self._source.data.multiply(coo_matrix(toMul))
        if scipy.sparse.isspmatrix(raw):
            self._source.data = raw.tocoo()
        else:
            self._source.data = coo_matrix(raw, shape=self._source.data.shape)

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

        pnames = self._source.points._getNamesNoGeneration()
        fnames = self._source.features._getNamesNoGeneration()
        self._source.referenceDataFrom(ret, useLog=False)
        self._source.points.setNames(pnames, useLog=False)
        self._source.features.setNames(fnames, useLog=False)


    def _transformEachElement_zeroPreserve_implementation(
            self, toTransform, points, features):
        for index, val in enumerate(self._source.data.data):
            pID = self._source.data.row[index]
            fID = self._source.data.col[index]
            if points is not None and pID not in points:
                continue
            if features is not None and fID not in features:
                continue

            if toTransform.oneArg:
                currRet = toTransform(val)
            else:
                currRet = toTransform(val, pID, fID)

            self._source.data.data[index] = currRet

        self._source.data.eliminate_zeros()

class SparseElementsView(ElementsView, SparseElements):
    """
    Limit functionality of SparseElements to read-only
    """
    pass
