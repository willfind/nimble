"""
Method implementations and helpers acting specifically on points in a
List object.
"""

import numpy

from nimble.exceptions import InvalidArgumentValue

from .axis_view import AxisView
from .listAxis import ListAxis
from .points import Points
from .points_view import PointsView
from .dataHelpers import fillArrayWithCollapsedFeatures
from .dataHelpers import fillArrayWithExpandedFeatures

class ListPoints(ListAxis, Points):
    """
    List method implementations performed on the points axis.

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
        Insert the points from the toInsert object below the provided
        index in this object, the remaining points from this object will
        continue below the inserted points.
        """
        insert = toInsert.copy('pythonlist')
        if insertBefore != 0 and insertBefore != len(self):
            breakIdx = insertBefore - 1
            restartIdx = insertBefore
            start = self._base.view(pointEnd=breakIdx).copy('pythonlist')
            end = self._base.view(pointStart=restartIdx).copy('pythonlist')
            allData = start + insert + end
        elif insertBefore == 0:
            allData = insert + self._base.copy('pythonlist')
        else:
            allData = self._base.copy('pythonlist') + insert

        self._base.data = allData

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)
            if len(currRet) != len(self._base.features):
                msg = "function must return an iterable with as many elements "
                msg += "as features in this object"
                raise InvalidArgumentValue(msg)

            self._base.data[i] = currRet

    # def _flattenToOne_implementation(self):
    #     onto = self._base.data[0]
    #     for _ in range(1, len(self._base.points)):
    #         onto += self._base.data[1]
    #         del self._base.data[1]
    #
    #     self._base._numFeatures = len(onto)
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     result = []
    #     numPoints = divideInto
    #     numFeatures = len(self._base.features) // numPoints
    #     for i in range(numPoints):
    #         temp = self._base.data[0][(i*numFeatures):((i+1)*numFeatures)]
    #         result.append(temp)
    #
    #     self._base.data = result
    #     self._base._numFeatures = numFeatures

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = []
        retainData = []
        for pt in self._base.data:
            collapseFeatures = []
            retainFeatures = []
            for idx in collapseIndices:
                collapseFeatures.append(pt[idx])
            for idx in retainIndices:
                retainFeatures.append(pt[idx])
            collapseData.append(collapseFeatures)
            retainData.append(retainFeatures)

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, numpy.array(collapseData),
            currNumPoints, currFtNames, numRetPoints, numRetFeatures)

        self._base.data = tmpData.tolist()
        self._base._numFeatures = numRetFeatures

    def _combineByExpandingFeatures_implementation(
            self, uniqueDict, namesIdx, uniqueNames, numRetFeatures):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures)

        self._base.data = tmpData.tolist()
        self._base._numFeatures = numRetFeatures


class ListPointsView(PointsView, AxisView, ListPoints):
    """
    Limit functionality of ListPoints to read-only.

    Parameters
    ----------
    base : ListView
        The ListView instance that will be queried.
    """
    pass
