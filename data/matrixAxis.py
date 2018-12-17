"""

"""
from __future__ import absolute_import

import numpy

import UML
from .axis import Axis
from .base import cmp_to_key

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

    def _sort_implementation(self, sortBy, sortHelper):
        if self.axis == 'point':
            test = self.source.pointView(0)
            viewIter = self.source.pointIterator()
            indexGetter = self.source.getPointIndex
            nameGetter = self.source.getPointName
            nameGetterStr = 'getPointName'
            names = self.source.getPointNames()
        else:
            test = self.source.featureView(0)
            viewIter = self.source.featureIterator()
            indexGetter = self.source.getFeatureIndex
            nameGetter = self.source.getFeatureName
            nameGetterStr = 'getFeatureName'
            names = self.source.getFeatureNames()

        if isinstance(sortHelper, list):
            if self.axis == 'point':
                self.source.data = self.source.data[sortHelper, :]
            else:
                self.source.data = self.source.data[:, sortHelper]
            newNameOrder = [names[idx] for idx in sortHelper]
            return newNameOrder

        scorer = None
        comparator = None
        try:
            sortHelper(test)
            scorer = sortHelper
        except TypeError:
            pass
        try:
            sortHelper(test, test)
            comparator = sortHelper
        except TypeError:
            pass

        if sortHelper is not None and scorer is None and comparator is None:
            raise ArgumentException("sortHelper is neither a scorer or a comparator")

        if comparator is not None:
            # make array of views
            viewArray = []
            for v in viewIter:
                viewArray.append(v)

            viewArray.sort(key=cmp_to_key(comparator))
            indexPosition = []
            for i in range(len(viewArray)):
                index = indexGetter(getattr(viewArray[i], nameGetterStr)(0))
                indexPosition.append(index)
            indexPosition = numpy.array(indexPosition)
        elif hasattr(scorer, 'permuter'):
            scoreArray = scorer.indices
            indexPosition = numpy.argsort(scoreArray)
        else:
            # make array of views
            viewArray = []
            for v in viewIter:
                viewArray.append(v)

            scoreArray = viewArray
            if scorer is not None:
                # use scoring function to turn views into values
                for i in range(len(viewArray)):
                    scoreArray[i] = scorer(viewArray[i])
            else:
                for i in range(len(viewArray)):
                    scoreArray[i] = viewArray[i][sortBy]

            # use numpy.argsort to make desired index array
            # this results in an array whose ith entry contains the the
            # index into the data of the value that should be in the ith
            # position.
            indexPosition = numpy.argsort(scoreArray)

        # use numpy indexing to change the ordering
        if self.axis == 'point':
            self.source.data = self.source.data[indexPosition, :]
        else:
            self.source.data = self.source.data[:, indexPosition]

        # we convert the indices of the their previous location into their feature names
        newNameOrder = []
        for i in range(len(indexPosition)):
            oldIndex = indexPosition[i]
            newName = nameGetter(oldIndex)
            newNameOrder.append(newName)
        return newNameOrder
