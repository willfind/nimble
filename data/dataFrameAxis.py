"""

"""
from __future__ import absolute_import

import numpy

import UML
from .axis import Axis

class DataFrameAxis(Axis):
    """

    """
    def __init__(self):
        super(DataFrameAxis, self).__init__()

    def _structuralBackend_implementation(self, structure, targetList):
        """
        Backend for points/features.extract points/features.delete,
        points/features.retain, and points/features.copy. Returns a new
        object containing only the members in targetList and performs
        some modifications to the original object if necessary. This
        function does not perform all of the modification or process how
        each function handles the returned value, these are managed
        separately by each frontend function.
        """
        df = self.source.data

        if self.axis == 'point':
            ret = df.iloc[targetList, :]
            axis = 0
            name = 'pointNames'
            nameList = [self.source.getPointName(i) for i in targetList]
            otherName = 'featureNames'
            otherNameList = self.source.getFeatureNames()
        elif self.axis == 'feature':
            ret = df.iloc[:, targetList]
            axis = 1
            name = 'featureNames'
            nameList = [self.source.getFeatureName(i) for i in targetList]
            otherName = 'pointNames'
            otherNameList = self.source.getPointNames()

        if structure.lower() != "copy":
            df.drop(targetList, axis=axis, inplace=True)

        if axis == 0:
            df.index = numpy.arange(len(df.index), dtype=df.index.dtype)
        else:
            df.columns = numpy.arange(len(df.columns), dtype=df.columns.dtype)

        return UML.data.DataFrame(ret, **{name: nameList,
                                          otherName: otherNameList})
