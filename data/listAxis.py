"""

"""
from __future__ import absolute_import

import numpy

import UML
from .axis import Axis

class ListAxis(Axis):
    """

    """
    def __init__(self):
        super(ListAxis, self).__init__()

    def _structuralBackend_implementation(self, structure, targetList):
        """
        Backend for .points/.features.extract .points/.features.delete,
        .points/.features.retain, and .points/.features.copy. Returns a
        new object containing only the members in targetList and
        performs some modifications to the original object if necessary.
        This function does not perform all of the modification or
        process how each function handles the returned value, these are
        managed separately by each frontend function.
        """
        pnames = []
        fnames = []
        data = numpy.matrix(self.source.data, dtype=object)

        if self.axis == 'point':
            keepList = []
            for idx in range(self.source.pts):
                if idx not in targetList:
                    keepList.append(idx)
            satisfying = data[targetList, :]
            if structure != 'copy':
                keep = data[keepList, :]
                self.source.data = keep.tolist()

            for index in targetList:
                pnames.append(self.source.getPointName(index))
            fnames = self.source.getFeatureNames()

        else:
            if self.source.data == []:
                # create empty matrix with correct shape
                empty = numpy.empty((self.source.pts, self.source.fts))
                data = numpy.matrix(empty, dtype=numpy.object_)

            keepList = []
            for idx in range(self.source.fts):
                if idx not in targetList:
                    keepList.append(idx)
            satisfying = data[:, targetList]
            if structure != 'copy':
                keep = data[:, keepList]
                self.source.data = keep.tolist()

            for index in targetList:
                fnames.append(self.source.getFeatureName(index))
            pnames = self.source.getPointNames()

            if structure != 'copy':
                remainingFts = self.source._numFeatures - len(targetList)
                self.source._numFeatures = remainingFts

        return UML.data.List(satisfying, pointNames=pnames,
                             featureNames=fnames, reuseData=True)
