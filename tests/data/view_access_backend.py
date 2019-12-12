"""
Tests functionality that is limited only to View objects.
"""

from __future__ import absolute_import

import nimble
from nimble.data import Base, BaseView
from nimble.data.points import Points
from nimble.data.points_view import PointsView
from nimble.data.features import Features
from nimble.data.features_view import FeaturesView
from .baseObject import DataTestObject


class ViewAccess(DataTestObject):

    def test_exceptionDocstring_decorator_Base(self):
        """ test wrapper prepends view object information to Base docstring
        when for a method that raises an exception"""
        viewDoc = getattr(BaseView, 'transpose').__doc__
        baseDoc = getattr(Base, 'transpose').__doc__

        viewLines = viewDoc.split('\n')
        baseLines = baseDoc.split('\n')

        assert len(viewLines) == len(baseLines) + 3
        assert viewLines[3:] == baseLines

    def test_exceptionDocstring_decorator_Points(self):
        """ test wrapper prepends view object information to Base docstring
        when for a method that raises an exception"""
        viewDoc = getattr(PointsView, 'splitByCollapsingFeatures').__doc__
        baseDoc = getattr(Points, 'splitByCollapsingFeatures').__doc__

        viewLines = viewDoc.split('\n')
        baseLines = baseDoc.split('\n')

        assert len(viewLines) == len(baseLines) + 3
        assert viewLines[3:] == baseLines

    def test_exceptionDocstring_decorator_Features(self):
        """ test wrapper prepends view object information to Base docstring
        when for a method that raises an exception"""
        viewDoc = getattr(FeaturesView, 'splitByParsing').__doc__
        baseDoc = getattr(Features, 'splitByParsing').__doc__

        viewLines = viewDoc.split('\n')
        baseLines = baseDoc.split('\n')

        assert len(viewLines) == len(baseLines) + 3
        assert viewLines[3:] == baseLines

    def test_BaseView_exceptionsForModifying(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        testObject = self.constructor(data)

        try:
            testObject.fillUsingAllData(match='a', fill=0)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.replaceFeatureWithBinaryFeatures(0)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.transformFeatureToIntegers(0)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.transformElements(lambda elem: elem + 1)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.transpose()
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.referenceDataFrom(self, testObject)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.fillWith(self, [99, 99, 99], 0, 0, 0, 2)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.fillUsingAllData(0, nimble.fill.kNeighborsClassifier)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.flattenToOnePoint()
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.flattenToOneFeature()
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.unflattenFromOnePoint(3)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.unflattenFromOneFeature(3)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            mergeData = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
            toMerge = self.constructor(mergeData)
            testObject.merge(toMerge, point='strict', feature='union')
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject @= 2
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject *= 2
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject += 1
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject -= 1
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject /= 1
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject //= 1
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject %= 1
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject **= 2
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)


    def test_PointsView_exceptionsForModifying(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        testObject = self.constructor(data)

        try:
            testObject.points.setName(0, 'set')
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.setNames(None)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.extract(lambda pt: True)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.delete(lambda pt: False)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.retain(lambda pt: True)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            new = self.constructor([[0, 0, 0]])
            testObject.points.append(new)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.shuffle()
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.sort(sortHelper=[2, 0, 1])
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.transform(lambda pt: [v + 1 for v in pt])
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.fill(0, 99)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.normalize(subtract=0, divide=1)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.splitByCollapsingFeatures([1, 2], 'names', 'values')
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.points.combineByExpandingFeatures(1, 2)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)


    def test_FeaturesView_exceptionsForModifying(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        testObject = self.constructor(data)

        try:
            testObject.features.setName(0, 'set')
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.features.setNames(None)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.features.extract(lambda pt: True)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.features.delete(lambda pt: False)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.features.retain(lambda pt: True)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            new = self.constructor([[0], [0], [0]])
            testObject.features.append(new)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.features.shuffle()
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.features.sort(sortHelper=[2, 0, 1])
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.features.transform(lambda ft: [v + 1 for v in ft])
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.features.fill(0, 99)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.features.normalize(subtract=0, divide=1)
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

        try:
            testObject.features.splitByParsing(0, lambda v: [1, 1], ['new1', 'new2'])
            assert False # expected TypeError
        except TypeError as e:
            assert "disallowed for View objects" in str(e)

# get retType TODO -- still not sure what is correct functionality.
