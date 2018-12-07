"""

"""
from __future__ import absolute_import

import numpy

import UML
from .elements import Elements

scipy = UML.importModule('scipy')
if scipy is not None:
    from scipy.sparse import coo_matrix

class SparseElements(Elements):
    """

    """
    def __init__(self, source):
        self.source = source
        super(SparseElements, self).__init__()

    ################################
    # Higher Order implementations #
    ################################

    def _calculate_implementation(self, function, points, features,
                                  preserveZeros, outputType):
        if not isinstance(self.source, UML.data.BaseView):
            data = self.source.data.data
            row = self.source.data.row
            col = self.source.data.col
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
            shape = self.source.data.shape
            values = coo_matrix((data, (row, col)), shape=shape)
            # note: even if function transforms nonzero values into zeros
            # our init methods will filter them out from the data attribute
            return UML.createData(outputType, values)
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
            return UML.createData(outputType, values)
        # zeros not preserved
        return self._calculateForEachElementGenericVectorized(
            function, points, features, outputType)
