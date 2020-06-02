"""
Method implementations and helpers acting specifically on points in a
List object.
"""

import numpy

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
        insert = toInsert.copy('List').data
        if insertBefore != 0 and insertBefore != len(self):
            start = self._base.data[:insertBefore]
            end = self._base.data[insertBefore:]
            allData = start + insert + end
        elif insertBefore == 0:
            allData = insert + self._base.data
        else:
            allData = self._base.data + insert

        self._base.data = allData

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)

            self._base.data[i] = list(currRet)

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

    def _combineByExpandingFeatures_implementation(self, uniqueDict, namesIdx,
                                                   uniqueNames, numRetFeatures,
                                                   numExpanded):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures,
                                                numExpanded)

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
