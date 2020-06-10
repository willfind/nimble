"""
Method implementations and helpers acting specifically on features in a
List object.
"""

import numpy

from .axis_view import AxisView
from .listAxis import ListAxis
from .features import Features
from .features_view import FeaturesView

class ListFeatures(ListAxis, Features):
    """
    List method implementations performed on the feature axis.

    Parameters
    ----------
    base : List
        The List instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _insert_implementation(self, insertBefore, toInsert):
        """
        Insert the features from the toInsert object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        insert = toInsert.copy('pythonlist')
        if insertBefore != 0 and insertBefore != len(self):
            breakIdx = insertBefore - 1
            restartIdx = insertBefore
            start = self._base.view(featureEnd=breakIdx).copy('pythonlist')
            end = self._base.view(featureStart=restartIdx).copy('pythonlist')
            zipChunks = zip(start, insert, end)
            allData = list(map(lambda pt: pt[0] + pt[1] + pt[2], zipChunks))
        elif insertBefore == 0:
            end = self._base.copy('pythonlist')
            allData = list(map(lambda pt: pt[0] + pt[1], zip(insert, end)))
        else:
            start = self._base.copy('pythonlist')
            allData = list(map(lambda pt: pt[0] + pt[1], zip(start, insert)))

        self._base.data = allData
        self._base._numFeatures += len(toInsert.features)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)

            for i in range(len(self._base.points)):
                self._base.data[i][j] = currRet[i]

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        tmpData = numpy.empty(shape=(len(self._base.points), numRetFeatures),
                              dtype=numpy.object_)

        tmpData[:, :featureIndex] = [ft[:featureIndex] for ft
                                     in self._base.data]
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])
            tmpData[:, featureIndex + i] = newFeat
        existingData = [ft[featureIndex + 1:] for ft in self._base.data]
        tmpData[:, featureIndex + numResultingFts:] = existingData

        self._base.data = tmpData.tolist()
        self._base._numFeatures = numRetFeatures


class ListFeaturesView(FeaturesView, AxisView, ListFeatures):
    """
    Limit functionality of ListFeatures to read-only.

    Parameters
    ----------
    base : ListView
        The ListView instance that will be queried.
    """
    pass
