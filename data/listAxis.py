"""

"""
from __future__ import absolute_import
import copy

import numpy

import UML
from .axis import Axis
from .base import cmp_to_key

class ListAxis(Axis):
    """

    """
    def __init__(self):
        super(ListAxis, self).__init__()

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
        pnames = []
        fnames = []
        data = numpy.matrix(self.source.data, dtype=object)

        if self.axis == 'point':
            keepList = []
            for idx in range(len(self.source.points)):
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
                empty = numpy.empty((len(self.source.points), len(self.source.features)))
                data = numpy.matrix(empty, dtype=numpy.object_)

            keepList = []
            for idx in range(len(self.source.features)):
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
            sortData = numpy.array(self.source.data, dtype=numpy.object_)
            if self.axis == 'point':
                sortData = sortData[sortHelper, :]
            else:
                sortData = sortData[:, sortHelper]
            self.source.data = sortData.tolist()
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

        # make array of views
        viewArray = []
        for v in viewIter:
            viewArray.append(v)

        if comparator is not None:
            # try:
            #     viewArray.sort(cmp=comparator)#python2
            # except:
            viewArray.sort(key=cmp_to_key(comparator))#python2 and 3
            indexPosition = []
            for i in range(len(viewArray)):
                index = indexGetter(getattr(viewArray[i], nameGetterStr)(0))
                indexPosition.append(index)
        else:
            #scoreArray = viewArray
            scoreArray = []
            if scorer is not None:
                # use scoring function to turn views into values
                for i in range(len(viewArray)):
                    scoreArray.append(scorer(viewArray[i]))
            else:
                for i in range(len(viewArray)):
                    scoreArray.append(viewArray[i][sortBy])

            # use numpy.argsort to make desired index array
            # this results in an array whole ith index contains the the
            # index into the data of the value that should be in the ith
            # position
            indexPosition = numpy.argsort(scoreArray)

        # run through target axis and change indices
        if self.axis == 'point':
            source = copy.copy(self.source.data)
            for i in range(len(self.source.data)):
                self.source.data[i] = source[indexPosition[i]]
        else:
            for i in range(len(self.source.data)):
                currPoint = self.source.data[i]
                temp = copy.copy(currPoint)
                for j in range(len(indexPosition)):
                    currPoint[j] = temp[indexPosition[j]]

        # we convert the indices of the their previous location into their feature names
        newNameOrder = []
        for i in range(len(indexPosition)):
            oldIndex = indexPosition[i]
            newName = nameGetter(oldIndex)
            newNameOrder.append(newName)
        return newNameOrder
