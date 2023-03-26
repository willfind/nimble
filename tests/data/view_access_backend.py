"""
Tests functionality that is limited only to View objects.
"""

from nimble.core.data import Base, BaseView
from nimble.core.data.points import Points
from nimble.core.data.views import PointsView
from nimble.core.data.features import Features
from nimble.core.data.views import FeaturesView

from tests.helpers import raises
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

        with raises(TypeError, match="disallowed for View objects"):
            testObject.replaceFeatureWithBinaryFeatures(0)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.transformFeatureToIntegers(0)

        with raises(TypeError, match="disallowed for View objects"):
            testObject._referenceFrom(testObject)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.transformElements(lambda elem: elem + 1)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.transpose()

        with raises(TypeError, match="disallowed for View objects"):
            testObject.replaceRectangle([99, 99, 99], 0, 0, 0, 2)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.flatten()

        with raises(TypeError, match="disallowed for View objects"):
            testObject.unflatten((3, 3))

        with raises(TypeError, match="disallowed for View objects"):
            mergeData = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
            toMerge = self.constructor(mergeData)
            testObject.merge(toMerge, point='strict', feature='union')

        with raises(TypeError, match="disallowed for View objects"):
            testObject @= 2

        with raises(TypeError, match="disallowed for View objects"):
            testObject *= 2

        with raises(TypeError, match="disallowed for View objects"):
            testObject += 1

        with raises(TypeError, match="disallowed for View objects"):
            testObject -= 1

        with raises(TypeError, match="disallowed for View objects"):
            testObject /= 1

        with raises(TypeError, match="disallowed for View objects"):
            testObject //= 1

        with raises(TypeError, match="disallowed for View objects"):
            testObject %= 1

        with raises(TypeError, match="disallowed for View objects"):
            testObject **= 2


    def test_PointsView_exceptionsForModifying(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        testObject = self.constructor(data)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.setNames('set', oldIdentifiers=0)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.setNames(None, useLog=False)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.extract(lambda pt: True)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.delete(lambda pt: False)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.retain(lambda pt: True)

        with raises(TypeError, match="disallowed for View objects"):
            new = self.constructor([[0, 0, 0]])
            testObject.points.append(new)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.permute()

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.sort(by=0)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.transform(lambda pt: [v + 1 for v in pt])

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.fillMatching(99, 0)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.splitByCollapsingFeatures([1, 2], 'names', 'values')

        with raises(TypeError, match="disallowed for View objects"):
            testObject.points.combineByExpandingFeatures(1, 2)


    def test_FeaturesView_exceptionsForModifying(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        testObject = self.constructor(data)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.setNames('set', oldIdentifiers=0)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.setNames(None, useLog=False)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.extract(lambda pt: True)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.delete(lambda pt: False)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.retain(lambda pt: True)

        with raises(TypeError, match="disallowed for View objects"):
            new = self.constructor([[0], [0], [0]])
            testObject.features.append(new)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.permute()

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.sort(by=0)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.transform(lambda ft: [v + 1 for v in ft])

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.fillMatching(99, 0)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.normalize(lambda x: x)

        with raises(TypeError, match="disallowed for View objects"):
            testObject.features.splitByParsing(0, lambda v: [1, 1], ['new1', 'new2'])

# get retType TODO -- still not sure what is correct functionality.
