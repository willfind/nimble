"""

"""
from __future__ import absolute_import

import numpy

import UML
from .axis import Axis

class MatrixAxis(Axis):
    """

    """
    def __init__(self):
        super(MatrixAxis, self).__init__()

    def _structuralBackend_implementation(self, structure, targetList):
        """
        Backend for points/features.extract points/features.delete,
        points/features.retain, and points/features.copy. Returns a new
        object containing only the points or features in targetList and
        performs some modifications to the original object if necessary.
        This function does not perform all of the modification or
        process how each function handles the returned value, these are
        managed separately by each frontend function.
        """
        nameList = []
        if self.axis == 'point':
            axisVal = 0
            getName = self.source.getPointName
            ret = self.source.data[targetList]
            pointNames = nameList
            featureNames = self.source.getFeatureNames()
        else:
            axisVal = 1
            getName = self.source.getFeatureName
            ret = self.source.data[:, targetList]
            featureNames = nameList
            pointNames = self.source.getPointNames()

        if structure != 'copy':
            self.source.data = numpy.delete(self.source.data,
                                            targetList, axisVal)

        # construct nameList
        for index in targetList:
            nameList.append(getName(index))

        return UML.data.Matrix(ret, pointNames=pointNames,
                               featureNames=featureNames)
