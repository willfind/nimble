"""
Backend for unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, these
functions are to be called by unit tests of the derived class, with the appropiate
objects provided.

Methods tested in this file:

In object HighLevelDataSafe:
points.calculate, features.calculate, calculateOnElements, points.count,
features.count, countElements, countUniqueElements, points.unique,
features.unique, points.mapReduce, isApproximatelyEqual, trainAndTestSets,
points.repeat, features.repeat, points.matching, features.matching,
matchingElements

In object HighLevelModifying:
replaceFeatureWithBinaryFeatures, points.permute, features.permute,
features.normalize, points.fill, features.fill, fillMatching,
points.splitByCollapsingFeatures, points.combineByExpandingFeatures,
features.splitByParsing, points.replace, features.replace
"""

from copy import deepcopy
import os.path
import tempfile
import datetime
import functools
import copy

import numpy as np
import pytest
import scipy.sparse
import pandas as pd

import nimble
from nimble import match
from nimble import fill
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction
from nimble.random import numpyRandom
from tests.helpers import raises
from tests.helpers import logCountAssertionFactory
from tests.helpers import noLogEntryExpected, oneLogEntryExpected
from tests.helpers import assertNoNamesGenerated
from tests.helpers import assertCalled
from tests.helpers import getDataConstructors
from .baseObject import DataTestObject


preserveName = "PreserveTestName"
preserveAPath = os.path.join(os.getcwd(), "correct", "looking", "path")
preserveRPath = os.path.relpath(preserveAPath)
preservePair = (preserveAPath, preserveRPath)

### Helpers used by tests in the test class ###

def simpleMapper(point):
    idInt = point[0]
    intList = []
    for i in range(1, len(point)):
        intList.append(point[i])
    ret = []
    for value in intList:
        ret.append((idInt, value))
    return ret


def simpleReducer(identifier, valuesList):
    total = 0
    for value in valuesList:
        total += value
    return (identifier, total)


def oddOnlyReducer(identifier, valuesList):
    if identifier % 2 == 0:
        return None
    return simpleReducer(identifier, valuesList)


def passThrough(value):
    return value


def plusOne(value):
    return (value + 1)


def plusOneOnlyEven(value):
    if value % 2 == 0:
        return (value + 1)
    else:
        return None

def noChange(value):
    return value

def noNormalize(values1, values2=None):
    if values2 is None:
        return values1
    return values1, values2


class HighLevelDataSafe(DataTestObject):
    #######################
    # .points.calculate() #
    #######################

    @raises(InvalidArgumentType)
    def test_points_calculate_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
        origObj.points.calculate(None)

    @raises(ImproperObjectAction)
    def test_points_calculate_exceptionPEmpty(self):
        data = [[], []]
        data = np.array(data).T
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        origObj.points.calculate(emitLower)

    @raises(ImproperObjectAction)
    def test_points_calculate_exceptionFEmpty(self):
        data = [[], []]
        data = np.array(data)
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        origObj.points.calculate(emitLower)

    def test_points_calculate_functionReturns2D(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def return2D(point):
            return [list(point)]

        calc = toTest.points.calculate(return2D)
        assert calc._dims == [4, 1, 3]

    @raises(InvalidArgumentValue)
    def test_points_calculate_functionReturnsInvalidObj(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnInvalidObj(point):
            return int

        calc = toTest.points.calculate(returnInvalidObj)

    @raises(InvalidArgumentValue)
    def test_points_calculate_dictReturn(self):

        def dictReturn(pt):
            return {str(i): pt for i in range(len(pt))}

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        ret = orig.points.calculate(dictReturn)

    @assertCalled(nimble.core.data.axis, 'constructIndicesList')
    def test_points_calculate_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        ret = toTest.points.calculate(noChange, points=['a', 'b'])

    @oneLogEntryExpected
    def test_points_calculate_Handmade(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        lowerCounts = origObj.points.calculate(emitLower)

        expectedOut = [[0.1], [0.1], [0.1], [0.2]]
        exp = self.constructor(expectedOut, pointNames=pointNames)

        assert lowerCounts.isIdentical(exp)

    @oneLogEntryExpected
    def test_points_calculate_Handmade_lazyNameGeneration(self):
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData))

        def emitLower(point):
            return point[1]

        lowerCounts = origObj.points.calculate(emitLower)

        assertNoNamesGenerated(origObj)
        assertNoNamesGenerated(lowerCounts)

    @oneLogEntryExpected
    def test_points_calculate_functionReturnsNimbleObject(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnNimbleObj(point):
            ret = point * 2
            assert isinstance(ret, nimble.core.data.Base)
            return ret

        calc = toTest.points.calculate(returnNimbleObj)

        expData = [[2, 0.2, 0.02], [2, 0.2, 0.04], [2, 0.2, 0.06], [2, 0.4, 0.04]]
        exp = self.constructor(expData, pointNames=pointNames,
                               featureNames=featureNames)

        assert calc.isIdentical(exp)

    def test_points_calculate_NamePathPreservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames, name=preserveName, paths=preservePair)

        def emitLower(point):
            return point[toTest.features.getIndex('deci')]

        ret = toTest.points.calculate(emitLower)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.name is None
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

        assert ret.points.getNames() == toTest.points.getNames()

    def test_points_calculate_HandmadeLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        lowerCounts = origObj.points.calculate(emitLower, points=['three', 2])

        expectedOut = [[0.2], [0.1]]
        expPnames = ['three', 'two']
        exp = self.constructor(expectedOut, pointNames=expPnames)

        assert lowerCounts.isIdentical(exp)

    def test_points_calculate_functionReturnsNimbleObject_limited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnNimbleObj(point):
            ret = point * 2
            assert isinstance(ret, nimble.core.data.Base)
            return ret

        calc = toTest.points.calculate(returnNimbleObj, points=['two', 'zero'])

        expData = [[2, 0.2, 0.06], [2, 0.2, 0.02]]
        exp = self.constructor(expData, pointNames=['two', 'zero'],
                               featureNames=featureNames)

        assert calc.isIdentical(exp)

    def test_points_calculate_functionReturnsNimbleObject_featureLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitLowers(point):
            return point[['centi', 'deci']]

        lowerCounts = origObj.points.calculate(emitLowers)

        expectedOut = [[0.01, 0.1], [0.02, 0.1], [0.03, 0.1], [0.02, 0.2]]
        exp = self.constructor(expectedOut, pointNames=pointNames,
                               featureNames=['centi', 'deci'])

        assert lowerCounts.isIdentical(exp)

    def test_points_calculate_functionReturnsNimbleObject_newFtNames(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def changeFtNames(point):
            pt = point.copy()
            pt.features.setNames(['new1', 'new2', 'new3'])
            return pt

        lowerCounts = origObj.points.calculate(changeFtNames)

        exp = self.constructor(origData, pointNames=pointNames,
                               featureNames=['new1', 'new2', 'new3'])

        assert lowerCounts.isIdentical(exp)

    def test_points_calculate_nonZeroItAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(deepcopy(origData))

        def emitNumNZ(point):
            ret = 0
            assert len(point) == 3
            for value in point.iterateElements(only=match.nonZero):
                ret += 1
            return ret

        counts = origObj.points.calculate(emitNumNZ)

        expectedOut = [[3], [2], [2], [1]]
        exp = self.constructor(expectedOut)

        assert counts.isIdentical(exp)

    def test_points_calculate_zerosReturned(self):

        def returnAllZero(pt):
            return [0 for val in pt]

        orig1 = self.constructor([[1, 2, 3], [1, 2, 3], [0, 0, 0]])
        exp1 = self.constructor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        ret1 = orig1.points.calculate(returnAllZero)
        assert ret1 == exp1

        def invert(pt):
            return [0 if v == 1 else 1 for v in pt]

        orig2 = self.constructor([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        exp2 = self.constructor([[0, 0, 0], [1, 0, 1], [1, 1, 1]])

        ret2 = orig2.points.calculate(invert)
        assert ret2 == exp2

    def test_points_calculate_conversionWhenIntType(self):

        def addTenth(pt):
            return [v + 0.1 for v in pt]

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        exp = self.constructor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [0.1, 0.1, 0.1]])

        ret = orig.points.calculate(addTenth)
        assert ret == exp

    def test_points_calculate_stringReturnsPreserved(self):

        def toString(pt):
            return [str(v) for v in pt]

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        exp = self.constructor([['1', '2', '3'], ['4', '5', '6'], ['0', '0', '0']])

        ret = orig.points.calculate(toString)
        assert ret == exp

    def test_points_calculate_featureVector(self):

        def asFeature(pt):
            return pt.T

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        exp = self.constructor([[[1], [2], [3]], [[4], [5], [6]], [[0], [0], [0]]])

        ret = orig.points.calculate(asFeature)
        assert ret == exp

    def test_points_calculate_reshape(self):

        def reshape3D(pt):
            pt = pt.copy()
            pt.unflatten((2, 2))
            return pt

        orig = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]])
        exp3D = self.constructor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[0, 0], [0, 0]]])

        ret = orig.points.calculate(reshape3D)
        assert ret == exp3D

        def reshape5D(pt):
            pt = pt.copy()
            pt.unflatten((1, 2, 2, 1))
            return pt

        exp5D = self.constructor([[[[[1], [2]], [[3], [4]]]],
                                  [[[[5], [6]], [[7], [8]]]],
                                  [[[[0], [0]], [[0], [0]]]]])

        ret = orig.points.calculate(reshape5D)
        assert ret == exp5D

    def test_points_calculate_toDatetime(self):
        data = [[1, 1, 2020], [1, 2, 2020], [1, 3, 2020]]
        fnames = ['month', 'day', 'year']
        toTest = self.constructor(data, featureNames=fnames)

        def toDatetime(pt):
            return datetime.datetime(pt['year'], pt['month'], pt['day'])

        expData = [[datetime.datetime(2020, 1, 1)],
                   [datetime.datetime(2020, 1, 2)],
                   [datetime.datetime(2020, 1, 3)]]

        exp = self.constructor(expData)

        datetimes = toTest.points.calculate(toDatetime)

        assert datetimes == exp

    def test_points_calculate_fromDatetime(self):
        data = [[datetime.datetime(2020, 1, 1)],
                [datetime.datetime(2020, 1, 2)],
                [datetime.datetime(2020, 1, 3)]]
        toTest = self.constructor(data)

        def fromDatetime(pt):
            return [pt[0].month, pt[0].day, pt[0].year]

        expData = [[1, 1, 2020], [1, 2, 2020], [1, 3, 2020]]

        exp = self.constructor(expData)

        datetimes = toTest.points.calculate(fromDatetime)

        assert datetimes == exp

    ##########################
    # .features.calculate() #
    #########################

    @raises(ImproperObjectAction)
    def test_features_calculate_exceptionPEmpty(self):
        data = [[], []]
        data = np.array(data).T
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.features.calculate(emitAllEqual)

    @raises(ImproperObjectAction)
    def test_features_calculate_exceptionFEmpty(self):
        data = [[], []]
        data = np.array(data)
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.features.calculate(emitAllEqual)

    @raises(InvalidArgumentType)
    def test_features_calculate_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
        origObj.features.calculate(None)

    @raises(ImproperObjectAction)
    def test_features_calculate_functionReturns2D(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def return2D(feature):
            return [[val for val in feature]]

        calc = toTest.features.calculate(return2D)

    @raises(InvalidArgumentValue)
    def test_features_calculate_functionReturnsInvalidObj(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnInvalidObj(feature):
            return int

        calc = toTest.features.calculate(returnInvalidObj)

    @raises(InvalidArgumentValue)
    def test_features_transform_dictReturn(self):

        def dictReturn(ft):
            return {str(i): ft for i in range(len(ft))}

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        ret = orig.features.calculate(dictReturn)

    @assertCalled(nimble.core.data.axis, 'constructIndicesList')
    def test_features_calculate_calls_constructIndicesList(self):
        toTest = self.constructor([[1,2],[3,4]], featureNames=['a', 'b'])

        ret = toTest.features.calculate(noChange, features=['a', 'b'])

    @oneLogEntryExpected
    def test_features_calculate_Handmade(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllEqual(feature):
            first = feature['zero']
            for value in feature:
                if value != first:
                    return 0
            return 1

        lowerCounts = origObj.features.calculate(emitAllEqual)
        expectedOut = [[1, 0, 0]]
        exp = self.constructor(expectedOut, featureNames=featureNames)
        assert lowerCounts.isIdentical(exp)

    @oneLogEntryExpected
    def test_features_calculate_Handmade_lazyNameGeneration(self):
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData))

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        lowerCounts = origObj.features.calculate(emitAllEqual)

        assertNoNamesGenerated(origObj)
        assertNoNamesGenerated(lowerCounts)

    @oneLogEntryExpected
    def test_features_calculate_functionReturnsNimbleObject(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnNimbleObj(feature):
            ret = feature * 2
            assert isinstance(ret, nimble.core.data.Base)
            return ret

        calc = toTest.features.calculate(returnNimbleObj)

        expData = [[2, 0.2, 0.02], [2, 0.2, 0.04], [2, 0.2, 0.06], [2, 0.4, 0.04]]
        exp = self.constructor(expData, pointNames=pointNames,
                               featureNames=featureNames)

        assert calc.isIdentical(exp)

    def test_features_calculate_NamePath_preservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(origData, pointNames=pointNames,
                                  featureNames=featureNames, name=preserveName, paths=preservePair)

        def emitAllEqual(feature):
            first = feature['zero']
            for value in feature:
                if value != first:
                    return 0
            return 1

        ret = toTest.features.calculate(emitAllEqual)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.name is None
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

        assert toTest.features.getNames() == ret.features.getNames()

    def test_features_calculate_HandmadeLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        lowerCounts = origObj.features.calculate(emitAllEqual, features=[0, 'centi'])
        expectedOut = [[1, 0]]
        expFNames = ['number', 'centi']
        exp = self.constructor(expectedOut, featureNames=expFNames)
        assert lowerCounts.isIdentical(exp)

    def test_features_calculate_functionReturnsNimbleObject_limited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnNimbleObj(feature):
            ret = feature * 2
            assert isinstance(ret, nimble.core.data.Base)
            return ret

        calc = toTest.features.calculate(returnNimbleObj, features=['deci', 'number'])

        expData = [[0.2, 2], [0.2, 2], [0.2, 2], [0.4, 2]]
        exp = self.constructor(expData, pointNames=pointNames,
                               featureNames=['deci', 'number'])

        assert calc.isIdentical(exp)

    def test_features_calculate_functionReturnsNimbleObject_pointLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def firstAndLastPoint(feature):
            return feature[['zero', 'three']]

        calc = toTest.features.calculate(firstAndLastPoint)

        expData = [[1, 0.1, 0.01], [1, 0.2, 0.02]]
        exp = self.constructor(expData, pointNames=['zero', 'three'],
                               featureNames=featureNames)

        assert calc.isIdentical(exp)

    def test_features_calculate_functionReturnsNimbleObject_newPointNames(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def changePtNames(feature):
            ft = feature.copy()
            ft.points.setNames(['new1', 'new2', 'new3', 'new4'])
            return ft

        calc = toTest.features.calculate(changePtNames)

        exp = self.constructor(origData, featureNames=featureNames,
                               pointNames=['new1', 'new2', 'new3', 'new4'])

        assert calc.isIdentical(exp)

    def test_features_calculate_nonZeroIterAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(deepcopy(origData))

        def emitNumNZ(feature):
            ret = 0
            assert len(feature) == 4
            for value in feature.iterateElements(order='feature', only=match.nonZero):
                ret += 1
            return ret

        counts = origObj.features.calculate(emitNumNZ)

        expectedOut = [[3, 3, 2]]
        exp = self.constructor(expectedOut)

        assert counts.isIdentical(exp)

    def test_features_calculate_zerosReturned(self):

        def returnAllZero(ft):
            return [0 for val in ft]

        orig1 = self.constructor([[1, 2, 3], [1, 2, 3], [0, 0, 0]])
        exp1 = self.constructor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        ret1 = orig1.features.calculate(returnAllZero)
        assert ret1 == exp1

        def invert(ft):
            return [0 if v == 1 else 1 for v in ft]

        orig2 = self.constructor([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        exp2 = self.constructor([[0, 0, 0], [1, 0, 1], [1, 1, 1]])

        ret2 = orig2.features.calculate(invert)
        assert ret2 == exp2

    def test_features_calculate_conversionWhenIntType(self):

        def addTenth(ft):
            return [v + 0.1 for v in ft]

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        exp = self.constructor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [0.1, 0.1, 0.1]])

        ret = orig.features.calculate(addTenth)
        assert ret == exp

    def test_features_calculate_stringReturnsPreserved(self):

        def toString(ft):
            return [str(v) for v in ft]

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        exp = self.constructor([['1', '2', '3'], ['4', '5', '6'], ['0', '0', '0']])

        ret = orig.features.calculate(toString)
        assert ret == exp

    @raises(ImproperObjectAction)
    def test_features_calculate_pointVector(self):

        def asPoint(ft):
            return ft.T

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        ret = orig.features.calculate(asPoint)

    @raises(ImproperObjectAction)
    def test_features_calculate_reshape(self):

        def reshape3D(ft):
            ft = ft.copy()
            ft.unflatten((2, 2))
            return ft

        orig = self.constructor([[1, 5, 0], [2, 6, 0], [3, 7, 0], [4, 8, 0]])
        ret = orig.features.calculate(reshape3D)

    def test_features_calculate_toDatetime(self):
        data = [['2020-01-01'], ['2020-01-02'], ['2020-01-03']]
        toTest = self.constructor(data)

        def toDatetime(ft):
            return [datetime.datetime.strptime(dt, '%Y-%m-%d') for dt in ft]

        expData = [[datetime.datetime(2020, 1, 1)],
                   [datetime.datetime(2020, 1, 2)],
                   [datetime.datetime(2020, 1, 3)]]

        exp = self.constructor(expData)

        datetimes = toTest.features.calculate(toDatetime)

        assert datetimes == exp

    def test_features_calculate_fromDatetime(self):
        data = [[datetime.datetime(2020, 1, 1)],
                [datetime.datetime(2020, 1, 2)],
                [datetime.datetime(2020, 1, 3)]]
        toTest = self.constructor(data)

        def fromDatetime(ft):
            return ['-'.join(map(str, [d.year, d.month, d.day])) for d in ft]

        expData = [['2020-1-1'], ['2020-1-2'], ['2020-1-3']]

        exp = self.constructor(expData)

        datetimes = toTest.features.calculate(fromDatetime)

        assert datetimes == exp

    #######################
    # calculateOnElements #
    #######################

    @assertCalled(nimble.core.data.base, 'constructIndicesList')
    def test_calculateOnElements_calls_constructIndicesList1(self):
        toTest = self.constructor([[1,2],[3,4]], pointNames=['a', 'b'])

        def noChange(point):
            return point

        ret = toTest.calculateOnElements(noChange, points=['a', 'b'])

    @assertCalled(nimble.core.data.base, 'constructIndicesList')
    def test_calculateOnElements_calls_constructIndicesList2(self):
        toTest = self.constructor([[1,2],[3,4]], featureNames=['a', 'b'])

        def noChange(point):
            return point

        ret = toTest.calculateOnElements(noChange, features=['a', 'b'])

    @raises(InvalidArgumentValue)
    def test_calculateOnElements_invalidElementReturned(self):
        data = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]
        toTest = self.constructor(data)
        toTest.calculateOnElements(lambda e: [e])

    def test_calculateOnElements_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, name=preserveName, paths=preservePair)

        ret = toTest.calculateOnElements(passThrough)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.name is None
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

    @oneLogEntryExpected
    def test_calculateOnElements_passthrough(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.calculateOnElements(passThrough)
        retRaw = ret.copy(to="python list")

        assert [1, 2, 3] in retRaw
        assert [4, 5, 6] in retRaw
        assert [7, 8, 9] in retRaw
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(ret)

    def test_calculateOnElements_builtin(self):
        # builtins are implemented in C, so may behave differently
        data = [['1', '0', '3'], ['0', '5', '6'], ['7', '0', '9']]
        toTest = self.constructor(data)
        assert all(isinstance(x, str) for x in toTest.iterateElements())

        ret = toTest.calculateOnElements(int)
        assert all(isinstance(x, (int, np.integer)) for x in ret.iterateElements())

        ret = toTest.calculateOnElements(float)
        assert all(isinstance(x, (float, np.floating)) for x in ret.iterateElements())

        ret = toTest.calculateOnElements(str)
        assert all(isinstance(x, str) for x in ret.iterateElements())

    def test_calculateOnElements_plusOnePreserve(self):
        data = [[1, 0, 3], [0, 5, 6], [7, 0, 9]]
        toTest = self.constructor(data)
        ret = toTest.calculateOnElements(plusOne, preserveZeros=True)
        retRaw = ret.copy(to="python list")

        assert [2, 0, 4] in retRaw
        assert [0, 6, 7] in retRaw
        assert [8, 0, 10] in retRaw
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(ret)

    def test_calculateOnElements_plusOneExclude(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.calculateOnElements(plusOneOnlyEven, skipNoneReturnValues=True)
        retRaw = ret.copy(to="python list")

        assert [1, 3, 3] in retRaw
        assert [5, 5, 7] in retRaw
        assert [7, 9, 9] in retRaw

    def test_calculateOnElements_plusOneLimited(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)

        ret = toTest.calculateOnElements(plusOneOnlyEven, points='4', features=[1, 'three'],
                                             skipNoneReturnValues=True)
        retRaw = ret.copy(to="python list")

        assert [5, 7] in retRaw

    def test_calculateOnElements_All_zero(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret1 = toTest.calculateOnElements(lambda x: 0)
        ret2 = toTest.calculateOnElements(lambda x: 0, preserveZeros=True)

        expData = [[0,0,0],[0,0,0],[0,0,0]]
        expObj = self.constructor(expData)
        assert ret1 == expObj
        assert ret2 == expObj

    def test_calculateOnElements_String_conversion_manipulations(self):
        def allString(val):
            return str(val)

        toSMap = {2:'two', 4:'four', 6:'six', 8:'eight'}
        def f1(val):
            return toSMap[val] if val in toSMap else val

        toIMap = {'two':2, 'four':4, 'six':6, 'eight':8}
        def f2(val):
            return toIMap[val] if val in toIMap else val

        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret0A = toTest.calculateOnElements(allString)
        ret0B = toTest.calculateOnElements(allString, preserveZeros=True)

        exp0Data = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
        exp0Obj = self.constructor(exp0Data)
        assert ret0A == exp0Obj
        assert ret0B == exp0Obj

        ret1 = toTest.calculateOnElements(f1)

        exp1Data = [[1, 'two', 3], ['four', 5, 'six'], [7, 'eight', 9]]
        exp1Obj = self.constructor(exp1Data)

        assert ret1 == exp1Obj

        ret2 = ret1.calculateOnElements(f2)

        exp2Obj = self.constructor(data)

        assert ret2 == exp2Obj

    def test_calculateOnElements_dictionaryMapping(self):
        data = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        reverseMap = {k: v for k, v in zip(range(9), range(8, -1, -1))}

        toTest = self.constructor(data)
        ret = toTest.calculateOnElements(reverseMap)
        exp = self.constructor([[8, 7, 6], [5, 4, 3], [2, 1, 0]])
        assert ret == exp

        toTest = self.constructor(data)
        ret = toTest.calculateOnElements(reverseMap, points=[0, 2], features=[1, 2])
        exp = self.constructor([[7, 6], [1, 0]])
        assert ret == exp

        toTest = self.constructor(data)
        ret = toTest.calculateOnElements(reverseMap, preserveZeros=True)
        exp = self.constructor([[0, 7, 6], [5, 4, 3], [2, 1, 0]])
        assert ret == exp

        reverseMap[2] = None
        reverseMap[7] = None

        toTest = self.constructor(data)
        ret = toTest.calculateOnElements(reverseMap, skipNoneReturnValues=True)
        exp = self.constructor([[8, 7, 2], [5, 4, 3], [2, 7, 0]])
        assert ret == exp

        toTest = self.constructor(data)
        ret = toTest.calculateOnElements(reverseMap, skipNoneReturnValues=False)
        exp = self.constructor([[8, 7, None], [5, 4, 3], [2, None, 0]])
        assert ret == exp

    def test_calculateOnElements_zerosReturned(self):

        def returnAllZero(elem):
            return 0

        orig1 = self.constructor([[1, 2, 3], [1, 2, 3], [0, 0, 0]])
        exp1 = self.constructor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        ret1 = orig1.calculateOnElements(returnAllZero)
        assert ret1 == exp1

        def invert(elem):
            return 0 if elem == 1 else 1

        orig2 = self.constructor([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        exp2 = self.constructor([[0, 0, 0], [1, 0, 1], [1, 1, 1]])

        ret2 = orig2.calculateOnElements(invert)
        assert ret2 == exp2

        orig3 = self.constructor([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
        exp3 = self.constructor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        ret3 = orig3.calculateOnElements(invert, preserveZeros=True)
        assert ret3 == exp3

    def test_calculateOnElements_conversionWhenIntType(self):

        def addTenth(elem):
            return elem + 0.1

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        exp = self.constructor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [0.1, 0.1, 0.1]])

        ret = orig.calculateOnElements(addTenth)
        assert ret == exp

    def test_calculateOnElements_stringReturnsPreserved(self):

        def toString(e):
            return str(e)

        orig = self.constructor([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        exp = self.constructor([['1', '2', '3'], ['4', '5', '6'], ['0', '0', '0']])

        ret = orig.calculateOnElements(toString)
        assert ret == exp

    def test_calculateOnElements_toDatetime(self):
        data = [['2019-01-01', '2019-12-31'],
                ['2020-01-01', '2020-12-31'],
                ['2021-01-01', '2021-12-31']]
        toTest = self.constructor(data)

        def toDatetime(elem):
            return datetime.datetime.strptime(elem, '%Y-%m-%d')

        expData = [[datetime.datetime(2019, 1, 1), datetime.datetime(2019, 12, 31)],
                   [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 12, 31)],
                   [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 12, 31)]]

        exp = self.constructor(expData)

        datetimes = toTest.calculateOnElements(toDatetime)

        assert datetimes == exp

    def test_calculateOnElements_fromDatetime(self):
        data = [[datetime.datetime(2019, 1, 1), datetime.datetime(2019, 12, 31)],
                [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 12, 31)],
                [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 12, 31)]]
        toTest = self.constructor(data)

        def fromDatetime(elem):
            return '-'.join(map(str, [elem.year, elem.month, elem.day]))

        expData = [['2019-1-1', '2019-12-31'],
                   ['2020-1-1', '2020-12-31'],
                   ['2021-1-1', '2021-12-31']]

        exp = self.constructor(expData)

        datetimes = toTest.calculateOnElements(fromDatetime)

        assert datetimes == exp

    ######################
    # points.mapReduce() #
    ######################

    @raises(ImproperObjectAction)
    def test_points_mapReduce_ExceptionNoFeatures(self):
        """ Test points.mapReduce() for ImproperObjectAction when there are no features  """
        data = [[], []]
        data = np.array(data)
        toTest = self.constructor(data)
        toTest.points.mapReduce(simpleMapper, simpleReducer)

    def test_points_mapReduce_emptyResultNoPoints(self):
        """ Test points.mapReduce() when given point empty data """
        data = [[], []]
        data = np.array(data).T
        toTest = self.constructor(data)
        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        data = np.empty(shape=(0, 0))
        exp = self.constructor(data)
        assert ret.isIdentical(exp)

    @raises(InvalidArgumentType)
    def test_points_mapReduce_ExceptionNoneMap(self):
        """ Test points.mapReduce() for InvalidArgumentType when mapper is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce(None, simpleReducer)

    @raises(InvalidArgumentType)
    def test_points_mapReduce_ExceptionNoneReduce(self):
        """ Test points.mapReduce() for InvalidArgumentType when reducer is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce(simpleMapper, None)

    @raises(InvalidArgumentType)
    def test_points_mapReduce_ExceptionUncallableMap(self):
        """ Test points.mapReduce() for InvalidArgumentType when mapper is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce("hello", simpleReducer)

    @raises(InvalidArgumentType)
    def test_points_mapReduce_ExceptionUncallableReduce(self):
        """ Test points.mapReduce() for InvalidArgumentType when reducer is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce(simpleMapper, 5)

    @oneLogEntryExpected
    def test_points_mapReduce_handmade(self):
        """ Test points.mapReduce() against handmade output """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        exp = self.constructor([[1, 5], [4, 11], [7, 17]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))

    def test_points_mapReduce_handmade_lazyNameGeneration(self):
        """ Test points.mapReduce() against handmade output """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(ret)

    def test_points_mapReduce_NamePath_preservation(self):
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames,
                                  name=preserveName, paths=preservePair)

        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.name is None
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

    def test_points_mapReduce_handmadeNoneReturningReducer(self):
        """ Test points.mapReduce() against handmade output with a None returning Reducer """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.points.mapReduce(simpleMapper, oddOnlyReducer)

        exp = self.constructor([[1, 5], [7, 17]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))

    ###################
    # countElements() #
    ###################
    @noLogEntryExpected
    def test_countElements(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.countElements('>= 5')
        assert ret == 5

        ret = toTest.countElements('==5')
        assert ret == 1

        ret = toTest.countElements(lambda x: x % 2 == 1)
        assert ret == 5

    ##################
    # points.count() #
    ##################
    @noLogEntryExpected
    def test_points_count(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=['one', 'two', 'three'],
                                  featureNames=['a', 'b', 'c'])
        ret = toTest.points.count('b >= 5')
        assert ret == 2

        ret = toTest.points.count(lambda x: x['b'] >= 5)
        assert ret == 2

    @noLogEntryExpected    
    @raises(InvalidArgumentValue)
    def test_points_count_pointNameQuery(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=['one', 'two', 'three'],
                                  featureNames=['a', 'b', 'c'])
        ret = toTest.points.count('one')


    ####################
    # features.count() #
    ####################
    @noLogEntryExpected
    def test_features_count(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=['one', 'two', 'three'],
                                  featureNames=['a', 'b', 'c'])
        ret = toTest.features.count('two >= 5')
        assert ret == 2

        ret = toTest.features.count(lambda x: x['two'] >= 5)
        assert ret == 2
        
    @noLogEntryExpected
    @raises(InvalidArgumentValue)
    def test_features_count_featureNameQuery(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=['one', 'two', 'three'],
                                  featureNames=['a', 'b', 'c'])
        ret = toTest.features.count('c')


    ##########################
    # isApproximatelyEqual() #
    ##########################

    @pytest.mark.slow
    @noLogEntryExpected
    def test_isApproximatelyEqual_randomTest(self):
        """ Test isApproximatelyEqual() using randomly generated data """

        for x in range(2):
            points = 100
            features = 40
            data = np.zeros((points, features))

            for i in range(points):
                for j in range(features):
                    data[i, j] = numpyRandom.rand() * numpyRandom.randint(0, 5)

            toTest = self.constructor(data)

            for constructor in getDataConstructors():
                currObj = constructor(data, useLog=False)
                assert toTest.isApproximatelyEqual(currObj)
                assert toTest.hashCode() == currObj.hashCode()
                assertNoNamesGenerated(toTest)
                assertNoNamesGenerated(currObj)


    ######################
    # trainAndTestSets() #
    ######################

    @raises(InvalidArgumentValue)
    def test_trainAndTestSets_invalidFraction(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, featureNames=featureNames)

        trX, teX = toTest.trainAndTestSets(20)

    # simple sucess - no labels
    @logCountAssertionFactory(2)
    def test_trainAndTestSets_simple_nolabels(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, featureNames=featureNames)

        trX, teX = toTest.trainAndTestSets(.5)

        assert len(trX.points) == 2
        assert len(trX.features) == 5
        assert len(teX.points) == 2
        assert len(teX.features) == 5

        # try the same test with a default named object
        toTest = self.constructor(data)

        trX, teX = toTest.trainAndTestSets(.5)
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(trX)
        assertNoNamesGenerated(teX)

    # simple sucess - single label
    @logCountAssertionFactory(2)
    def test_trainAndTestSets_simple_singlelabel(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, featureNames=featureNames)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=0)

        assert len(trX.points) == 2
        assert len(trX.features) == 4
        assert len(trY.points) == 2
        assert len(trY.features) == 1
        assert len(teX.points) == 2
        assert len(teX.features) == 4
        assert len(teY.points) == 2
        assert len(teY.features) == 1

        # try the same test with a default named object
        toTest = self.constructor(data)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=0)
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(trX)
        assertNoNamesGenerated(trY)
        assertNoNamesGenerated(teX)
        assertNoNamesGenerated(teY)

    # simple sucess - single label
    @logCountAssertionFactory(3)
    def test_trainAndTestSets_simple_nimbleLabels(self):
        data = [[5, -1, 3, 33], [5, -2, 6, 66], [5, -2, 9, 99], [5, -4, 12, 111]]
        labels = [[1], [3], [2], [4]]
        featureNames = ['fives', 'labs2', 'bozo', 'long']
        pointNames = ['pt0', 'pt1', 'pt2', 'pt3']
        labelsFtName = 'labs1'
        toTest = self.constructor(data, pointNames=pointNames,
                                  featureNames=featureNames)
        toTestLabels = self.constructor(labels, pointNames=pointNames,
                                        featureNames=[labelsFtName])

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=toTestLabels)
        assert len(trX.points) == 2
        assert len(trX.features) == 4
        assert len(trY.points) == 2
        assert len(trY.features) == 1
        assert trX.points.getNames() == trY.points.getNames()
        assert len(teX.points) == 2
        assert len(teX.features) == 4
        assert len(teY.points) == 2
        assert len(teY.features) == 1
        assert teX.points.getNames() == teY.points.getNames()

        # try the same test with labels also being present in toTest
        # Since we test on View objects, we have to remake from the start
        # instead of modifying the current toTest
        data = [[5, -1, 3, 33, 1], [5, -2, 6, 66, 3], [5, -2, 9, 99, 2], [5, -4, 12, 111, 4]]
        featureNames = ['fives', 'labs2', 'bozo', 'long', 'labs1']
        pointNames = ['pt0', 'pt1', 'pt2', 'pt3']
        labelsFtName = 'labs1'
        toTest = self.constructor(data, pointNames=pointNames,
                                  featureNames=featureNames)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=toTestLabels)
        assert not trX.features.hasName(labelsFtName)
        assert len(trX.features) == 4
        assert not teX.features.hasName(labelsFtName)
        assert len(teX.features) == 4

        # try the same test with a default named object
        toTest = self.constructor(data)
        toTestLabels = self.constructor(labels)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=toTestLabels)
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(trX)
        assertNoNamesGenerated(trY)
        assertNoNamesGenerated(teX)
        assertNoNamesGenerated(teY)

    @raises(InvalidArgumentValue)
    def test_trainAndTestSets_simple_nimbleLabels_invalidLength(self):
        data = [[5, -1, 3, 33], [5, -2, 6, 66], [5, -2, 9, 99], [5, -4, 12, 111]]
        featureNames = ['fives', 'labs2', 'bozo', 'long']
        pointNames = ['pt0', 'pt1', 'pt2', 'pt3']
        toTest = self.constructor(data, pointNames=pointNames,
                                  featureNames=featureNames)
        toTestLabels = self.constructor([[1], [2]])

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, toTestLabels)

    @raises(InvalidArgumentValue)
    def test_trainAndTestSets_simple_nimbleLabels_invalidPointNames(self):
        data = [[5, -1, 3, 33], [5, -2, 6, 66], [5, -2, 9, 99], [5, -4, 12, 111]]
        featureNames = ['fives', 'labs2', 'bozo', 'long']
        pointNamesX = ['pt0', 'pt2', 'pt1', 'pt3']
        toTest = self.constructor(data, pointNames=pointNamesX,
                                  featureNames=featureNames)
        pointNamesY = ['pt0', 'pt1', 'pt2', 'pt3']
        toTestLabels = self.constructor([[1], [3], [2], [4]],
                                        pointNames=pointNamesY)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, toTestLabels)

    def test_trainAndTestSets_simple_nimbleLabels_defaultPointNames(self):
        # Only testing that the default names do not generate an error
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        pointNamesX = ['pt0', 'pt1', None, 'pt3']
        pointNamesY = ['pt0', 'pt1', None, 'pt3']
        toTest = self.constructor(data, pointNames=pointNamesX,
                                  featureNames=featureNames)
        toTestLabels = self.constructor([[4], [3], [2], [1]],
                                        pointNames=pointNamesY)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, toTestLabels)

        pointNamesX = [None, 'pt1', 'pt2', 'pt3']
        pointNamesY = ['pt0', 'pt1', 'pt2', None]
        toTest = self.constructor(data, pointNames=pointNamesX,
                                  featureNames=featureNames)
        toTestLabels = self.constructor([[4], [3], [2], [1]],
                                        pointNames=pointNamesY)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, toTestLabels)

        pointNamesX = [None, 'pt0', 'pt2' , 'pt3']
        pointNamesY = [None, 'pt0', None, 'pt3']
        toTest = self.constructor(data, pointNames=pointNamesX,
                                  featureNames=featureNames)
        toTestLabels = self.constructor([[4], [3], [2], [1]],
                                        pointNames=pointNamesY)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, toTestLabels)

    # simple sucess - multi label
    @logCountAssertionFactory(3)
    def test_trainAndTestSets_simple_multilabel(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, featureNames=featureNames)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=[0, 'labs2'])

        assert len(trX.points) == 2
        assert len(trX.features) == 3
        assert len(trY.points) == 2
        assert len(trY.features) == 2
        assert len(teX.points) == 2
        assert len(teX.features) == 3
        assert len(teY.points) == 2
        assert len(teY.features) == 2

        # try the same test with a nimble labels object and labels in toTest
        toTest = self.constructor(data, featureNames=featureNames)
        toTestLabels = toTest.features.copy([0, 'labs2'], useLog=False)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=toTestLabels)
        assert not trX.features.hasName("labs1")
        assert not trX.features.hasName("labs2")
        assert len(trX.features) == 3
        assert not teX.features.hasName("labs1")
        assert not teX.features.hasName("labs2")
        assert len(teX.features) == 3

        # try the same test with a default named object
        toTest = self.constructor(data)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=[0, 2])
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(trX)
        assertNoNamesGenerated(trY)
        assertNoNamesGenerated(teX)
        assertNoNamesGenerated(teY)

    # edge cases 0/1 test portions
    def test_trainAndTestSets_0or1_testFraction(self):
        data = [[1, 2, 3, 33], [2, 5, 6, 66], [3, 8, 9, 99], [4, 11, 12, 111]]
        toTest = self.constructor(data)

        trX, trY, teX, teY = toTest.trainAndTestSets(0, 0)

        assert len(trX.points) == 4
        assert len(trY.points) == 4
        assert len(teX.points) == 0
        assert len(teY.points) == 0

        trX, trY, teX, teY = toTest.trainAndTestSets(1, 0)

        assert len(trX.points) == 0
        assert len(trY.points) == 0
        assert len(teX.points) == 4
        assert len(teY.points) == 4

    # each returned set independant of calling set
    def test_trainAndTestSets_unconnectedReturn(self):
        data = [[1, 1], [2, 2], [3, 3], [4, 4]]
        toTest = self.constructor(data)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0, randomOrder=False)

        assert trX == trY
        assert teX == teY

        def changeFirst(point):
            ret = []
            first = True
            for val in point:
                if first:
                    ret.append(-val)
                    first = False
                else:
                    ret.append(val)
            return ret

        # change the returned data
        trX.points.transform(changeFirst)
        trY.points.transform(changeFirst)
        assert trX[0, 0] == -1
        assert trY[0, 0] == -1

        # check all other data is unchanged
        assert toTest[0, 0] == 1
        assert teX[0, 0] == 3
        assert teY[0, 0] == 3

        # check that our multiple references are still the same

        assert trX == trY
        assert teX == teY

    def test_trainAndTestSets_objectNameCreation(self):
        data = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]

        toTest = self.constructor(data)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0)

        assert trX.name == 'trainX'
        assert trY.name == 'trainY'
        assert teX.name == 'testX'
        assert teY.name == 'testY'

    def test_trainAndTestSets_nameAppend_PathPreserve(self):
        data = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
        toTest = self.constructor(data, )
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmpFile:
            toTest.save(tmpFile.name, fileFormat='csv')

            toTest = self.constructor(tmpFile.name, name='toTest')

            trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0)

            assert trX.name == 'toTest_trainX'
            assert trX.path == tmpFile.name
            assert trX.absolutePath == tmpFile.name
            assert trX.relativePath == os.path.relpath(tmpFile.name)

            assert trY.name == 'toTest_trainY'
            assert trY.path == tmpFile.name
            assert trY.path == tmpFile.name
            assert trY.absolutePath == tmpFile.name
            assert trY.relativePath == os.path.relpath(tmpFile.name)

            assert teX.name == 'toTest_testX'
            assert teX.path == tmpFile.name
            assert teX.path == tmpFile.name
            assert teX.absolutePath == tmpFile.name
            assert teX.relativePath == os.path.relpath(tmpFile.name)

            assert teY.name == 'toTest_testY'
            assert teY.path == tmpFile.name
            assert teY.path == tmpFile.name
            assert teY.absolutePath == tmpFile.name
            assert teY.relativePath == os.path.relpath(tmpFile.name)


    def test_trainAndTestSets_PandFnamesPerserved_identifier(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        pnames = ['one', 'two', 'three', 'four']
        fnames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=0, randomOrder=False)

        assert trX.points.getNames() == ['one', 'two']
        assert trX.features.getNames() == ['fives', 'labs2', 'bozo', 'long']

        assert trY.points.getNames() == ['one', 'two']
        assert trY.features.getNames() == ['labs1']

        assert teX.points.getNames() == ['three', 'four']
        assert teX.features.getNames() == ['fives', 'labs2', 'bozo', 'long']

        assert teY.points.getNames() == ['three', 'four']
        assert teY.features.getNames() == ['labs1']

    def test_trainAndTestSets_PandFnamesPerserved_nimbleObject(self):
        data = [[5, -1, 3, 33], [5, -2, 6, 66], [5, -2, 9, 99], [5, -4, 12, 111]]
        labels = [[1], [2], [3], [4]]
        pnames = ['one', 'two', 'three', 'four']
        fnames = ['fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        toTestLabels = self.constructor(labels, pointNames=pnames,
                                        featureNames=['labs1'])
        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=toTestLabels,
                                                     randomOrder=False)

        assert trX.points.getNames() == ['one', 'two']
        assert trX.features.getNames() == ['fives', 'labs2', 'bozo', 'long']

        assert trY.points.getNames() == ['one', 'two']
        assert trY.features.getNames() == ['labs1']

        assert teX.points.getNames() == ['three', 'four']
        assert teX.features.getNames() == ['fives', 'labs2', 'bozo', 'long']

        assert teY.points.getNames() == ['three', 'four']
        assert teY.features.getNames() == ['labs1']


    def test_trainAndTestSets_randomOrder(self):
        data = [[1, 1], [2, 2], [3, 3], [4, 4]]
        toTest = self.constructor(data)

        for i in range(100):
            trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0, randomOrder=False)

            assert trX == trY
            assert trX[0] == 1
            assert trX[1] == 2

            assert teX == teY
            assert teX[0] == 3
            assert teX[1] == 4

        for i in range(100):
            trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0)

            # just to make sure everything looks right
            assert trX == trY
            assert teX == teY
            # this means point ordering was randomized, so we return successfully
            if trX[0] != 1:
                return

        assert False  # implausible number of checks for random order were unsucessful

    #################
    # points.unique #
    #################

    def test_points_unique_allNames_string(self):
        data = [['George', 'Washington'], ['George', 'Washington'],
                ['John', 'Adams'], ['John', 'Adams'], ['John', 'Adams'],
                ['Thomas', 'Jefferson'],  ['Thomas', 'Jefferson'],
                ['James', 'Madison']]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['George', 'Washington'], ['John', 'Adams'],
                   ['Thomas', 'Jefferson'], ['James', 'Madison']]
        exp = self.constructor(expData, pointNames=["p0", "p2", "p5", "p7"], featureNames=ftNames)

        ret = test.points.unique()

        assert ret == exp

    def test_points_unique_allNames_numeric(self):
        data = [[0, 0], [0, 0],
                [99, 99], [99, 99],
                [5.5, 11], [5.5, 11],  [5.5, 11],
                [-1, -2]]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [[0, 0], [99, 99], [5.5, 11], [-1, -2]]
        exp = self.constructor(expData, pointNames=["p0", "p2", "p4", "p7"], featureNames=ftNames)

        ret = test.points.unique()

        assert ret == exp

    @noLogEntryExpected
    def test_points_unique_allNames_mixed(self):
        data = [['George', 0], ['George', 0],
                ['John', 1], ['John', 1], ['John', 1],
                ['Thomas', 2],  ['Thomas', 2],
                ['James', 3]]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['George', 0], ['John', 1],
                   ['Thomas', 2], ['James', 3]]
        exp = self.constructor(expData, pointNames=["p0", "p2", "p5", "p7"], featureNames=ftNames)

        ret = test.points.unique()

        assert ret == exp

    def test_points_unique_allNames_allUnique(self):
        data = [['George', 'Washington'], ['John', 'Adams'],
                ['Thomas', 'Jefferson'], ['James', 'Madison'],
                ['James', 'Monroe'], ['John Quincy', 'Adams'],
                ['Andrew', 'Jackson'], ['Martin', 'Van Buren']]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        exp = test.copy()

        ret = test.points.unique()

        assert ret == exp

    def test_points_unique_allDefaultNames(self):
        data = [['George', 'Washington'], ['George', 'Washington'],
                ['John', 'Adams'], ['John', 'Adams'], ['John', 'Adams'],
                ['Thomas', 'Jefferson'],  ['Thomas', 'Jefferson'],
                ['James', 'Madison']]
        test = self.constructor(data)
        expData = [['George', 'Washington'], ['John', 'Adams'],
                   ['Thomas', 'Jefferson'], ['James', 'Madison']]
        exp = self.constructor(expData)

        ret = test.points.unique()

        assert ret == exp
        assertNoNamesGenerated(test)
        assertNoNamesGenerated(ret)

    def test_points_unique_subsetFeature0(self):
        data = [['George', 'Washington'], ['John', 'Adams'],
                ['Thomas', 'Jefferson'], ['James', 'Madison'],
                ['James', 'Monroe'], ['John Quincy', 'Adams'],
                ['Andrew', 'Jackson'], ['Martin', 'Van Buren']]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['George'], ['John'], ['Thomas'], ['James'],
                   ['John Quincy'], ['Andrew'], ['Martin']]
        expPtNames = ["p0", "p1", "p2", "p3", "p5", "p6", "p7"]
        exp = self.constructor(expData, pointNames=expPtNames, featureNames=["firstName"])

        ret = test[:, 0].points.unique()

        assert ret == exp

    def test_points_unique_subsetFeature1(self):
        data = [['George', 'Washington'], ['John', 'Adams'],
                ['Thomas', 'Jefferson'], ['James', 'Madison'],
                ['James', 'Monroe'], ['John Quincy', 'Adams'],
                ['Andrew', 'Jackson'], ['Martin', 'Van Buren']]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['Washington'], ['Adams'], ['Jefferson'], ['Madison'],
                   ['Monroe'], ['Jackson'], ['Van Buren']]
        expPtNames = ["p0", "p1", "p2", "p3", "p4", "p6", "p7"]
        exp = self.constructor(expData, pointNames=expPtNames, featureNames=["lastName"])

        ret = test[:, 1].points.unique()

        assert ret == exp

    ###################
    # features.unique #
    ###################

    def test_features_unique_allNames_string(self):
        data = [['a','b','c','a','b','c'],
                ['1','2','3','1','2','4'],
                ['0','0','0','0','0','0']]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['a','b','c','c'], ['1','2','3','4'], ['0','0','0','0']]
        exp = self.constructor(expData, pointNames=ptNames, featureNames=["f0", "f1", "f2", "f5"])

        ret = test.features.unique()

        assert ret == exp

    def test_features_unique_allNames_numeric(self):
        data = [[0, 1, 2, 0, 1, 2],
                [0, 0, 0, 0, 0, 0],
                [-1, -2, -3, -1, -1, -3]]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [[0, 1, 2, 1], [0, 0, 0, 0], [-1, -2, -3, -1]]
        exp = self.constructor(expData, pointNames=ptNames, featureNames=["f0", "f1", "f2", "f4"])

        ret = test.features.unique()

        assert ret == exp

    @noLogEntryExpected
    def test_features_unique_allNames_mixed(self):
        data = [['George', 0, 'George', 0, 'George', 0],
                ['John', 1, 'James', 3, 'John', 3],
                ['Thomas', 2, 'Thomas', 2, 'Thomas', 2]]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['George', 0, 'George', 0],
                   ['John', 1, 'James', 3],
                   ['Thomas', 2, 'Thomas', 2]]
        exp = self.constructor(expData, pointNames=ptNames, featureNames=["f0", "f1", "f2", "f3"])

        ret = test.features.unique()

        assert ret == exp

    def test_features_unique_allNames_allUnique(self):
        data = [['George', 0, 'George', 1, 'James', 2],
                ['John', 1, 'James', 0, 'John', 1],
                ['Thomas', 2, 'Thomas', 2, 'Thomas', 0]]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        exp = test.copy()

        ret = test.features.unique()

        assert ret == exp

    def test_features_unique_allDefaultNames(self):
        data = [['George', 0, 'George', 0, 'George', 0],
                ['John', 1, 'James', 3, 'John', 3],
                ['Thomas', 2, 'Thomas', 2, 'Thomas', 2]]
        test = self.constructor(data)

        expData = [['George', 0, 'George', 0],
                   ['John', 1, 'James', 3],
                   ['Thomas', 2, 'Thomas', 2]]
        exp = self.constructor(expData)

        ret = test.features.unique()

        assert ret == exp
        assertNoNamesGenerated(test)
        assertNoNamesGenerated(ret)

    def test_features_unique_subsetPoint0(self):
        data = [['George', 0, 'George', 0, 'George', 0],
                ['John', 1, 'James', 3, 'John', 3],
                ['Thomas', 2, 'Thomas', 2, 'Thomas', 2]]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['George', 0]]
        exp = self.constructor(expData, pointNames=["p0"], featureNames=["f0", "f1"])

        ret = test[0, :].features.unique()

        assert ret == exp

    def test_features_unique_subsetPoint1(self):
        data = [['George', 0, 'George', 0, 'George', 0],
                ['John', 1, 'James', 3, 'John', 3],
                ['Thomas', 2, 'Thomas', 2, 'Thomas', 2]]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['John', 1, 'James', 3]]
        exp = self.constructor(expData, pointNames=["p1"], featureNames=["f0", "f1", "f2", "f3"])

        ret = test[1, :].features.unique()

        assert ret == exp

    #######################
    # countUniqueElements #
    #######################
    @noLogEntryExpected
    def test_countUniqueElements_allPtsAndFtrs(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        toTest = self.constructor(data)
        unique = toTest.countUniqueElements()

        assert len(unique) == 6
        assert unique[1] == 2
        assert unique[2] == 2
        assert unique[3] == 2
        assert unique['a'] == 1
        assert unique['b'] == 1
        assert unique['c'] == 1
        # for Sparse, 0 is added to returned dictionary manually
        # want to test 0 is not added if the data doesn't contain zeros
        assert 0 not in unique
        assertNoNamesGenerated(toTest)

    def test_countUniqueElements_limitPoints(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        pNames = ['p1', 'p2', 'p3']
        toTest = self.constructor(data, pointNames=pNames)
        unique = toTest.countUniqueElements(points=0)

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[2] == 1
        assert unique[3] == 1

        unique = toTest.countUniqueElements(points='p1')

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[2] == 1
        assert unique[3] == 1

        unique = toTest.countUniqueElements(points=[0,'p3'])

        assert len(unique) == 3
        assert unique[1] == 2
        assert unique[2] == 2
        assert unique[3] == 2

    def test_countUniqueElements_limitFeatures(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        fNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(data, featureNames=fNames)
        unique = toTest.countUniqueElements(features=0)

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[3] == 1
        assert unique['a'] == 1

        unique = toTest.countUniqueElements(features='f1')

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[3] == 1
        assert unique['a'] == 1

        unique = toTest.countUniqueElements(features=[0,'f3'])

        assert len(unique) == 4
        assert unique[1] == 2
        assert unique[3] == 2
        assert unique['a'] == 1
        assert unique['c'] == 1

    @noLogEntryExpected
    def test_countUniqueElements_limitPointsAndFeatures_cornercase(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        fNames = ['f1', 'f2', 'f3']
        pNames = ['p1', 'p2', 'p3']
        toTest = self.constructor(data, featureNames=fNames, pointNames=pNames)

        unique = toTest.countUniqueElements(features=[0,'f3'], points=[0,'p3'])

        assert len(unique) == 2
        assert unique[1] == 2
        assert unique[3] == 2

    def test_countUniqueElements_zeroCount(self):
        data = [[0, 0, 0, 0, 1], [2, 0, 0, 0, 0], [0, 0, 3, 0, 0]]
        toTest = self.constructor(data)
        unique = toTest.countUniqueElements()

        assert 0 in unique
        assert unique[0] == 12

    ####################
    # points.repeat #
    ####################

    @assertCalled(nimble.core.data.Base, 'copy')
    def test_points_repeat_OneCopyCallsCopy(self):
        data = [0, 1, 2, 3]
        ptNames = ['pt']
        ftNames = ['a', 'b', 'c', 'd']
        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)
        repeated = toTest.points.repeat(totalCopies=1, copyPointByPoint=True)

    @logCountAssertionFactory(2)
    def test_points_repeat_1D(self):
        data = [0, 1, 2, 3]
        ptNames = ['pt']
        ftNames = ['a', 'b', 'c', 'd']
        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)
        repeated1 = toTest.points.repeat(3, copyPointByPoint=False)
        repeated2 = toTest.points.repeat(3, copyPointByPoint=True)

        expData = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        expPtNames = ['pt_1', 'pt_2', 'pt_3']
        exp = self.constructor(expData, pointNames=expPtNames, featureNames=ftNames)

        assert repeated1 == exp
        # return is same for either copyPointByPoint when 1D
        assert repeated1 == repeated2

    @oneLogEntryExpected
    def test_points_repeat_2D_copyPointByPointFalse(self):
        data = [[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0]]
        ptNames = ['1', '4', '0']
        ftNames = ['a', 'b', 'c', 'd']
        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)
        repeated = toTest.points.repeat(3, copyPointByPoint=False)

        expData = [[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0],
                   [1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0],
                   [1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0]]
        expPtNames = ['1_1', '4_1', '0_1', '1_2', '4_2', '0_2','1_3', '4_3', '0_3']
        exp = self.constructor(expData, pointNames=expPtNames, featureNames=ftNames)

        assert repeated == exp

    @noLogEntryExpected
    def test_points_repeat_2D_copyPointByPointTrue(self):
        data = [[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0]]
        ptNames = ['1', '4', '0']
        ftNames = ['a', 'b', 'c', 'd']
        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)
        repeated = toTest.points.repeat(3, copyPointByPoint=True, useLog=False)

        expData = [[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 3, 0],
                   [4, 5, 6, 0], [4, 5, 6, 0], [4, 5, 6, 0],
                   [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        expPtNames = ['1_1', '1_2', '1_3', '4_1', '4_2', '4_3', '0_1', '0_2', '0_3']
        exp = self.constructor(expData, pointNames=expPtNames, featureNames=ftNames)

        assert repeated == exp

    @raises(InvalidArgumentType)
    def test_points_repeat_invalidCopyCount_float(self):
        data = [[0, 1, 2], [3, 4, 5]]
        toTest = self.constructor(data)
        repeated = toTest.points.repeat(1.5, copyPointByPoint=False)

    @raises(InvalidArgumentType)
    def test_points_repeat_invalidCopyCount_negative(self):
        data = [[0, 1, 2], [3, 4, 5]]
        toTest = self.constructor(data)
        repeated = toTest.points.repeat(-1, copyPointByPoint=False)

    ######################
    # features.repeat #
    ######################

    @assertCalled(nimble.core.data.Base, 'copy')
    def test_features_repeat_OneCopyCallsCopy(self):
        data = [0, 1, 2, 3]
        ptNames = ['pt']
        ftNames = ['a', 'b', 'c', 'd']
        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)
        repeated = toTest.features.repeat(totalCopies=1, copyFeatureByFeature=False)

    @logCountAssertionFactory(2)
    def test_features_repeat_1D(self):
        data = [[0], [1], [2], [3]]
        ptNames = ['pt0', 'pt1', 'pt2', 'pt3']
        ftNames = ['a']
        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)
        repeated1 = toTest.features.repeat(3, copyFeatureByFeature=False)
        repeated2 = toTest.features.repeat(3, copyFeatureByFeature=True)

        expData = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
        expFtNames = ['a_1', 'a_2', 'a_3']
        exp = self.constructor(expData, pointNames=ptNames, featureNames=expFtNames)

        assert repeated1 == exp
        # return is same for either copyFeatureByFeature when 1D
        assert repeated1 == repeated2

    @oneLogEntryExpected
    def test_features_repeat_2D_copyFeatureByFeatureFalse(self):
        data = [[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0]]
        ptNames = ['1', '4', '0']
        ftNames = ['a', 'b', 'c', 'd']
        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)
        repeated = toTest.features.repeat(3, copyFeatureByFeature=False)

        expData = [[1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0],
                   [4, 5, 6, 0, 4, 5, 6, 0, 4, 5, 6, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        expFtNames = ['a_1', 'b_1', 'c_1', 'd_1', 'a_2', 'b_2', 'c_2', 'd_2', 'a_3', 'b_3', 'c_3', 'd_3']
        exp = self.constructor(expData, pointNames=ptNames, featureNames=expFtNames)

        assert repeated == exp

    @noLogEntryExpected
    def test_features_repeat_2D_copyFeatureByFeatureTrue(self):
        data = [[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 0]]
        ptNames = ['1', '4', '0']
        ftNames = ['a', 'b', 'c', 'd']
        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)
        repeated = toTest.features.repeat(3, copyFeatureByFeature=True, useLog=False)

        expData = [[1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0],
                   [4, 4, 4, 5, 5, 5, 6, 6, 6, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        expFtNames = ['a_1', 'a_2', 'a_3', 'b_1', 'b_2', 'b_3', 'c_1', 'c_2', 'c_3', 'd_1', 'd_2', 'd_3']
        exp = self.constructor(expData, pointNames=ptNames, featureNames=expFtNames)

        assert repeated == exp

    @raises(InvalidArgumentType)
    def test_features_repeat_invalidCopyCount_float(self):
        data = [[0, 1, 2], [3, 4, 5]]
        toTest = self.constructor(data)
        repeated = toTest.features.repeat(1.5, copyFeatureByFeature=False)

    @raises(InvalidArgumentType)
    def test_features_repeat_invalidCopyCount_negative(self):
        data = [[0, 1, 2], [3, 4, 5]]
        toTest = self.constructor(data)
        repeated = toTest.features.repeat(-1, copyFeatureByFeature=False)

    ####################
    # matchingElements #
    ####################

    @raises(InvalidArgumentValue)
    def test_matchingElements_funcDoesNotReturnBoolean(self):
        def returnVal(val):
            return val

        raw = [[1, 2, 3], [-1, -2, -3]]
        obj = self.constructor(raw)
        obj.matchingElements(returnVal)

    @oneLogEntryExpected
    def test_matchingElements_allElementsAreBool(self):
        raw = [[1, 2, 3], [-1, -2, -3]]
        obj = self.constructor(raw)
        matches = obj.matchingElements(lambda x: x > 0)

        assert matches[0, 0] is True or matches[0, 0] is np.bool_(True)
        assert matches[0, 1] is True or matches[0, 1] is np.bool_(True)
        assert matches[0, 2] is True or matches[0, 2] is np.bool_(True)
        assert matches[1, 0] is False or matches[1, 0] is np.bool_(False)
        assert matches[1, 1] is False or matches[1, 1] is np.bool_(False)
        assert matches[1, 2] is False or matches[1, 2] is np.bool_(False)

    def test_matchingElements_valueInput(self):
        raw = [[1, 2, 3], [-1, -2, -3], [0, 'a', 0]]
        obj = self.constructor(raw)
        match1 = obj.matchingElements(lambda x: x == 0)
        match2 = obj.matchingElements(0)
        assert match1 == match2

        match1 = obj.matchingElements(lambda x: x == 'a')
        match2 = obj.matchingElements('a')
        assert match1 == match2

    def test_matchingElements_comparisonStringInput(self):
        raw = [[1, 2, 3], [-1, -2, -3], [0, 0, 0]]
        obj = self.constructor(raw)

        match1 = obj.matchingElements(lambda x: x == 0)
        match2 = obj.matchingElements('==0')
        match3 = obj.matchingElements('== 0')
        assert match1 == match2 == match3

        match1 = obj.matchingElements(lambda x: x != 0)
        match2 = obj.matchingElements('!=0')
        match3 = obj.matchingElements('!= 0')
        assert match1 == match2 == match3

        match1 = obj.matchingElements(lambda x: x > 0)
        match2 = obj.matchingElements('>0')
        match3 = obj.matchingElements('> 0')
        assert match1 == match2 == match3

        match1 = obj.matchingElements(lambda x: x < 0)
        match2 = obj.matchingElements('<0')
        match3 = obj.matchingElements('< 0')
        assert match1 == match2 == match3

        match1 = obj.matchingElements(lambda x: x >= 0)
        match2 = obj.matchingElements('>=0')
        match3 = obj.matchingElements('>= 0')
        assert match1 == match2 == match3

        match1 = obj.matchingElements(lambda x: x <= 0)
        match2 = obj.matchingElements('<=0')
        match3 = obj.matchingElements('<= 0')
        assert match1 == match2 == match3

    @oneLogEntryExpected
    def test_matchingElements_pfLimited(self):
        raw = [[1, 2, -3], [-1, -2, 3]]
        pnames = ['1', '-1']
        fnames = ['a', 'b', 'c']
        obj = self.constructor(raw, pnames, fnames)
        matches = obj.matchingElements(lambda x: x > 0, points='1', features=[1, 2])
        expRaw = [[True, False]]
        expected = self.constructor(expRaw, ['1'], ['b', 'c'])
        assert matches == expected

    @logCountAssertionFactory(4)
    def test_matchingElements_varietyOfFuncs(self):
        raw = [[1, 2, 3], [-1, -2, -3], [0, 0, 0]]
        obj = self.constructor(raw)

        exp = [[True, True, True], [False, False, False], [False, False, False]]
        expObj = self.constructor(exp)
        matchPositive = obj.matchingElements(match.positive)

        assert matchPositive == expObj

        exp = [[True, True, True], [True, False, False], [True, True, True]]
        expObj = self.constructor(exp)
        greaterEqualToNeg1 = obj.matchingElements(lambda x: x >= -1)

        assert greaterEqualToNeg1 == expObj

        raw = [['a', None, 'c'], [np.nan, None, -3], [0, 'zero', None]]
        obj = self.constructor(raw)

        exp = [[False, True, False], [True, True, False], [False, False, True]]
        expObj = self.constructor(exp)
        isMissing = obj.matchingElements(match.missing)

        assert isMissing == expObj

        # None is converted to nan by nimble.data, here we explicitly pass the
        # value the underlying representation uses, so we avoid making it
        # look like None is considered a numeric
        raw = [['a', np.nan, 'c'], [np.nan, np.nan, -3], [0, 'zero', np.nan]]
        obj = self.constructor(raw)
        exp = [[True, False, True], [False, False, False], [False, True, False]]

        expObj = self.constructor(exp)
        isNonNumeric = obj.matchingElements(match.nonNumeric)

        assert isNonNumeric == expObj

    def test_matchingElements_pfname_preservation(self):
        raw = [[1, 2, 3], [-1, -2, -3], [0, 0, 0]]
        pnames = ['pos', 'neg', 'zero']
        fnames = ['1', '2', '3']

        obj = self.constructor(raw, pointNames=pnames, featureNames=fnames)
        matchPositive = obj.matchingElements(match.positive)

        assert matchPositive.points.getNames() == pnames
        assert matchPositive.features.getNames() == fnames

    def test_matchingElements_namePath_preservation(self):
        raw = [[1, 2, 3], [-1, -2, -3], [0, 0, 0]]

        preserveName = "PreserveTestName"
        preserveAPath = os.path.join(os.getcwd(), "correct", "looking", "path")
        preserveRPath = os.path.relpath(preserveAPath)
        preservePair = (preserveAPath, preserveRPath)

        obj = self.constructor(raw, name=preserveName, paths=preservePair)
        matchPositive = obj.matchingElements(match.positive)

        assert matchPositive.absolutePath == preserveAPath
        assert matchPositive.relativePath == preserveRPath
        assert matchPositive.name != preserveName
        assert matchPositive.name is None

    #####################################
    # points/features matching backends #
    #####################################

    @raises(InvalidArgumentValue)
    def back_pointsfeatures_matching_funcDoesNotReturnBoolean(self, axis):
        def returnInput(input):
            return input

        raw = [[1, 2, 3], [-1, -2, -3]]
        obj = self.constructor(raw)
        if axis == 'point':
            obj.points.matching(returnInput)
        else:
            obj.features.matching(returnInput)

    @logCountAssertionFactory(5)
    def back_pointsfeatures_matching_varietyOfFuncs(self, axis):
        raw = [[1, 2, 3], [-1, -2, -3], [0, 0, 0]]
        obj = self.constructor(raw)
        exp = [[True], [False], [False]]
        expObj = self.constructor(exp, featureNames=['allPositive'])
        if axis == 'point':
            matchPositive = obj.points.matching(match.allPositive)
        else:
            obj = obj.T
            expObj = expObj.T
            matchPositive = obj.features.matching(match.allPositive)

        assert matchPositive == expObj

        exp = [[True], [False], [True]]
        expObj = self.constructor(exp)
        if axis == 'point':
            anyGreaterNeg1 = obj.points.matching(lambda view: any(x > -1 for x in view))
        else:
            expObj = expObj.T
            anyGreaterNeg1 = obj.features.matching(lambda view: any(x > -1 for x in view))

        assert anyGreaterNeg1 == expObj

        raw = [[None, None, np.nan], [np.nan, None, -3], [0, 'zero', None]]
        obj = self.constructor(raw)

        exp = [[True], [False], [False]]
        expObj = self.constructor(exp, featureNames=['allMissing'])
        if axis == 'point':
            allMissing = obj.points.matching(match.allMissing)
        else:
            obj = obj.T
            expObj = expObj.T
            allMissing = obj.features.matching(match.allMissing)

        assert allMissing == expObj

        # None is converted to nan by nimble.data, here we explicitly pass the
        # value the underlying representation uses, so we avoid making it
        # look like None is considered a numeric
        raw = [['a', np.nan, 'c'], [np.nan, np.nan, -3], [0, 'zero', np.nan]]
        obj = self.constructor(raw)

        exp = [[True], [False], [True]]
        expObj = self.constructor(exp, featureNames=['anyNonNumeric'])
        if axis == 'point':
            anyNonNumeric = obj.points.matching(match.anyNonNumeric)
        else:
            obj = obj.T
            expObj = expObj.T
            anyNonNumeric = obj.features.matching(match.anyNonNumeric)

        assert anyNonNumeric == expObj

        raw = [['a', np.nan, 'c'], [np.nan, np.nan, -3], [0, 'zero', np.nan]]
        obj = self.constructor(raw, featureNames=['a 1', 'a 2', 'a 3'])

        exp = [[True], [True], [False]]
        expObj = self.constructor(exp)
        if axis == 'point':
            a2Missing = obj.points.matching("a 2 is missing")
        else:
            obj = obj.T
            expObj = expObj.T
            a2Missing = obj.features.matching("a 2 is missing")

        assert a2Missing == expObj

    ###################
    # points.matching #
    ###################

    def test_points_matching_funcDoesNotReturnBoolean(self):
        self.back_pointsfeatures_matching_funcDoesNotReturnBoolean('point')

    @oneLogEntryExpected
    def test_points_matching_allElementsAreBool(self):
        raw = [[1, 2, 3], [-1, -2, -3], [1, 2, -3]]
        obj = self.constructor(raw)
        matches = obj.points.matching(match.allPositive)

        assert len(matches.features) == 1
        assert len(matches.points) == 3
        assert matches[0, 0] is True or matches[0, 0] is np.bool_(True)
        assert matches[1, 0] is False or matches[1, 0] is np.bool_(False)
        assert matches[2, 0] is False or matches[2, 0] is np.bool_(False)

    def test_points_matching_varietyOfFuncs(self):
        self.back_pointsfeatures_matching_varietyOfFuncs('point')

    def test_points_matching_pfname_preservation(self):
        raw = [[1, 2, 3], [-1, -2, -3], [0, 0, 0]]
        pnames = ['pos', 'neg', 'zero']
        fnames = ['1', '2', '3']

        obj = self.constructor(raw, pointNames=pnames, featureNames=fnames)
        allNeg = obj.points.matching(match.allNegative)

        assert allNeg.points.getNames() == pnames
        assert allNeg.features.getNames() == ['allNegative']

    def test_points_matching_namePath_preservation(self):
        raw = [[1, 2, 3], [-1, -2, -3], [0, 0, 0]]

        preserveName = "PreserveTestName"
        preserveAPath = os.path.join(os.getcwd(), "correct", "looking", "path")
        preserveRPath = os.path.relpath(preserveAPath)
        preservePair = (preserveAPath, preserveRPath)

        obj = self.constructor(raw, name=preserveName, paths=preservePair)
        allNeg = obj.points.matching(match.allNegative)

        assert allNeg.absolutePath == preserveAPath
        assert allNeg.relativePath == preserveRPath
        assert allNeg.name != preserveName
        assert allNeg.name is None

    #####################
    # features.matching #
    #####################

    def test_features_matching_funcDoesNotReturnBoolean(self):
        self.back_pointsfeatures_matching_funcDoesNotReturnBoolean('feature')

    @oneLogEntryExpected
    def test_features_matching_allElementsAreBool(self):
        raw = [[1, 2, 3], [1, -2, -3], [1, 2, -3]]
        obj = self.constructor(raw)
        matches = obj.features.matching(match.allPositive)

        assert len(matches.points) == 1
        assert len(matches.features) == 3
        assert matches[0, 0] is True or matches[0, 0] is np.bool_(True)
        assert matches[0, 1] is False or matches[0, 1] is np.bool_(False)
        assert matches[0, 2] is False or matches[0, 2] is np.bool_(False)

    def test_features_matching_varietyOfFuncs(self):
        self.back_pointsfeatures_matching_varietyOfFuncs('feature')

    def test_features_matching_pfname_preservation(self):
        raw = [[1, -1, 0], [2, -2, 0], [3, -3, 0]]
        fnames = ['pos', 'neg', 'zero']
        pnames = ['1', '2', '3']

        obj = self.constructor(raw, pointNames=pnames, featureNames=fnames)
        allZeros = obj.features.matching(match.allZero)

        assert allZeros.features.getNames() == fnames
        assert allZeros.points.getNames() == ['allZero']

    def test_features_matching_namePath_preservation(self):
        raw = [[1, -1, 0], [2, -2, 0], [3, -3, 0]]

        preserveName = "PreserveTestName"
        preserveAPath = os.path.join(os.getcwd(), "correct", "looking", "path")
        preserveRPath = os.path.relpath(preserveAPath)
        preservePair = (preserveAPath, preserveRPath)

        obj = self.constructor(raw, name=preserveName, paths=preservePair)
        allZeros = obj.points.matching(match.allZero)

        assert allZeros.absolutePath == preserveAPath
        assert allZeros.relativePath == preserveRPath
        assert allZeros.name != preserveName
        assert allZeros.name is None


class HighLevelModifying(DataTestObject):

    ####################################
    # replaceFeatureWithBinaryFeatures #
    ####################################

    @raises(ImproperObjectAction)
    def test_replaceFeatureWithBinaryFeatures_PemptyException(self):
        """ Test replaceFeatureWithBinaryFeatures() with a point empty object """
        data = [[], []]
        data = np.array(data).T
        toTest = self.constructor(data)
        toTest.replaceFeatureWithBinaryFeatures(0)

    @raises(IndexError)
    def test_replaceFeatureWithBinaryFeatures_FemptyException(self):
        """ Test replaceFeatureWithBinaryFeatures() with a feature empty object """
        data = [[], []]
        data = np.array(data)
        toTest = self.constructor(data)
        toTest.replaceFeatureWithBinaryFeatures(0)

    @oneLogEntryExpected
    def test_replaceFeatureWithBinaryFeatures_handmade(self):
        """ Test replaceFeatureWithBinaryFeatures() against handmade output """
        data = [[1], [2], [3]]
        featureNames = ['col']
        toTest = self.constructor(data, featureNames=featureNames)
        getNames = self.constructor(data, featureNames=featureNames)
        ret = toTest.replaceFeatureWithBinaryFeatures(0)

        expData = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        expFeatureNames = []
        for point in getNames.points:
            expFeatureNames.append('col=' + str(point[0]))
        exp = self.constructor(expData, featureNames=expFeatureNames)

        assert toTest.isIdentical(exp)
        assert ret == expFeatureNames

    @oneLogEntryExpected
    def test_replaceFeatureWithBinaryFeatures_insertLocation(self):
        """ Test replaceFeatureWithBinaryFeatures() replaces at same index """
        data = [['a', 1, 'a'], ['b', 2, 'b'], ['c', 3, 'c']]
        featureNames = ['stay1', 'replace', 'stay2']
        toTest = self.constructor(data, featureNames=featureNames)
        getNames = self.constructor(data, featureNames=featureNames)
        ret = toTest.replaceFeatureWithBinaryFeatures(1)

        expData = [['a', 1, 0, 0, 'a'], ['b', 0, 1, 0, 'b'], ['c', 0, 0, 1, 'c']]
        expFeatureNames = []
        for point in getNames.points:
            expFeatureNames.append('replace=' + str(point[1]))
        expFeatureNames.insert(0, 'stay1')
        expFeatureNames.append('stay2')
        exp = self.constructor(expData, featureNames=expFeatureNames)

        assert toTest.isIdentical(exp)
        assert ret == expFeatureNames[1: -1]

    def test_replaceFeatureWithBinaryFeatures_NamePath_preservation(self):
        data = [[1], [2], [3]]
        featureNames = ['col']
        toTest = self.constructor(data, featureNames=featureNames)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.replaceFeatureWithBinaryFeatures(0)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    ##############################
    # transformFeatureToIntegers #
    ##############################

    @raises(ImproperObjectAction)
    def test_transformFeatureToIntegers_PemptyException(self):
        """ Test transformFeatureToIntegers() with an point empty object """
        data = [[], []]
        data = np.array(data).T
        toTest = self.constructor(data)
        toTest.transformFeatureToIntegers(0)

    @raises(IndexError)
    def test_transformFeatureToIntegers_FemptyException(self):
        """ Test transformFeatureToIntegers() with an feature empty object """
        data = [[], []]
        data = np.array(data)
        toTest = self.constructor(data)
        toTest.transformFeatureToIntegers(0)

    @oneLogEntryExpected
    def test_transformFeatureToIntegers_handmade(self):
        """ Test transformFeatureToIntegers() against handmade output """
        data = [['a'], ['b'], ['c'], ['b'], ['a']]
        featureNames = ['col']
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.transformFeatureToIntegers(0)

        assert toTest[0, 0] == toTest[4, 0]
        assert toTest[1, 0] == toTest[3, 0]
        assert toTest[0, 0] != toTest[1, 0]
        assert toTest[0, 0] != toTest[2, 0]

        # ensure data was transformed to a numeric type.
        for i in range(len(toTest.points)):
            # Matrix and Sparse might store values as floats or numpy types
            assert isinstance(toTest[i, 0], (int, float, np.number))

        # check ret
        assert len(ret) == 3
        assert all(isinstance(key, int) for key in ret.keys())
        for value in ret.values():
            assert value in ['a', 'b', 'c']

    def test_transformFeatureToIntegers_handmade_lazyNameGeneration(self):
        """ Test transformFeatureToIntegers() against handmade output """
        data = [['a'], ['b'], ['c'], ['b'], ['a']]
        toTest = self.constructor(data)
        ret = toTest.transformFeatureToIntegers(0)

        assertNoNamesGenerated(toTest)

    def test_transformFeatureToIntegers_pointNames(self):
        """ Test transformFeatureToIntegers preserves pointNames """
        data = [['a'], ['b'], ['c'], ['b'], ['a']]
        pnames = ['1a', '2a', '3', '2b', '1b']
        fnames = ['col']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = toTest.transformFeatureToIntegers(0)

        assert toTest.points.getName(0) == '1a'
        assert toTest.points.getName(1) == '2a'
        assert toTest.points.getName(2) == '3'
        assert toTest.points.getName(3) == '2b'
        assert toTest.points.getName(4) == '1b'

        # ensure data was transformed to a numeric type.
        for i in range(len(toTest.points)):
            # Matrix and Sparse might store values as floats or numpy types
            assert isinstance(toTest[i, 0], (int, float, np.number))

        # check ret
        assert len(ret) == 3
        assert all(isinstance(key, int) for key in ret.keys())
        for value in ret.values():
            assert value in ['a', 'b', 'c']

    def test_transformFeatureToIntegers_positioning(self):
        """ Test transformFeatureToIntegers preserves featurename mapping """
        data = [['a', 0], ['b', 1], ['c', 2], ['b', 3], ['a', 4]]
        pnames = ['1a', '2a', '3', '2b', '1b']
        fnames = ['col', 'pos']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = toTest.transformFeatureToIntegers(0)

        assert toTest[0, 0] == toTest[4, 0]
        assert toTest[1, 0] == toTest[3, 0]
        assert toTest[0, 0] != toTest[1, 0]
        assert toTest[0, 0] != toTest[2, 0]

        assert toTest[0, 1] == 0
        assert toTest[1, 1] == 1
        assert toTest[2, 1] == 2
        assert toTest[3, 1] == 3
        assert toTest[4, 1] == 4

        # ensure data was transformed to a numeric type.
        for i in range(len(toTest.points)):
            # Matrix and Sparse might store values as floats or numpy types
            assert isinstance(toTest[i, 0], (int, float, np.number))

        # check ret
        assert len(ret) == 3
        assert all(isinstance(key, int) for key in ret.keys())
        for value in ret.values():
            assert value in ['a', 'b', 'c']

    def test_transformFeatureToIntegers_ZerosInFeatureValuesPreserved(self):
        data = [['a'], [52], [0], [0], [0], [52], ['a']]

        toTest = self.constructor(data, featureNames=False)
        ret = toTest.transformFeatureToIntegers(0)

        assert ret[0] == 0
        assert toTest[0, 0] == toTest[6, 0]
        assert toTest[1, 0] == toTest[5, 0]
        assert toTest[2, 0] == 0
        assert toTest[3, 0] == 0
        assert toTest[4, 0] == 0

    def test_transformFeatureToIntegers_NamePath_preservation(self):
        data = [[10], [20], [30.5], [20], [10]]
        featureNames = ['col']
        toTest = self.constructor(data, featureNames=featureNames)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.transformFeatureToIntegers(0)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    ####################
    # points.permute() #
    ####################

    def testpoints_permute_random_noLongerEqual(self):
        """ Tests points.permute() results in a changed object """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        # it is possible that it permutes it into the same configuration.
        # the odds are vanishingly low that it will do so over consecutive calls
        # however. We will pass as long as it changes once
        returns = []
        for _ in range(5):
            ret = toTest.points.permute() # RET CHECK
            returns.append(ret)
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert not toTest.isApproximatelyEqual(toCompare)

        assert all(ret is None for ret in returns)
        assertNoNamesGenerated(toTest)


    def testpoints_permute_random_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        # odds are vanishingly low that it generates same configuration over
        # consecutive calls. We will pass as long as soon as it changes once
        for _ in range(5):
            toTest.points.permute()
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert not toTest.isApproximatelyEqual(toCompare)
        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    def test_points_permute_dataTypeRetainedFromList(self):
        """ Test points.permute() data not converted when permuting by list"""
        data = [['a', 2, 3.0], ['b', 5, 6.0], ['c', 8, 9.0]]
        toTest = self.constructor(data)

        toTest.points.permute([2, 1, 0])

        expData = [['c', 8, 9.0], ['b', 5, 6.0], ['a', 2, 3.0]]
        exp = self.constructor(expData)

        assert toTest == exp
        assertNoNamesGenerated(toTest)

    def test_points_permute_indicesList(self):
        """ Test points.permute() when we specify a list of indices """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        toTest = self.constructor(data)

        toTest.points.permute([2, 1, 0])

        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        exp = self.constructor(expData)

        assert toTest == exp

    def test_points_permute_namesList(self):
        """ Test points.permute() when we specify a list of point names """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        pnames = ['3', '6', '9']
        toTest = self.constructor(data, pointNames=pnames)

        toTest.points.permute(['9', '6', '3'])

        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        expNames = ['9', '6', '3']
        exp = self.constructor(expData, pointNames=expNames)

        assert toTest == exp

    @oneLogEntryExpected
    def test_points_permute_mixedList(self):
        """ Test points.permute() when we specify a mixed list (names/indices) """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        pnames = ['3', '6', '9']
        toTest = self.constructor(data, pointNames=pnames)

        toTest.points.permute(['9', '6', 0])

        expData = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        expNames = ['9', '6', '3']
        exp = self.constructor(expData, pointNames=expNames)

        assert toTest == exp

    @raises(IndexError)
    def test_points_permute_exceptionIndicesPEmpty(self):
        """ tests points.permute() throws an IndexError when given invalid indices """
        data = [[], []]
        data = np.array(data).T
        toTest = self.constructor(data)
        toTest.points.permute([1, 3])

    @raises(InvalidArgumentValue)
    def test_points_permute_exceptionIndicesSmall(self):
        """ tests points.permute() throws an InvalidArgumentValue when given an incorrectly sized indices list """
        data = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
        toTest = self.constructor(data)

        toTest.points.permute([1, 0])

    @raises(InvalidArgumentValue)
    def test_points_permute_exceptionNotUniqueIds(self):
        """ tests points.permute() throws an InvalidArgumentValue when given duplicate indices """
        data = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
        toTest = self.constructor(data)
        toTest.points.permute([1, 1, 0])


    ######################
    # features.permute() #
    ######################

    def test_features_permute_random_noLongerEqual(self):
        """ Tests features.permute() results in a changed object """
        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 1111111]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        # it is possible that it permutes it into the same configuration.
        # the odds are vanishly low that it will do so over consecutive calls
        # however. We will pass as long as it changes once
        returns = []
        for _ in range(5):
            ret = toTest.features.permute() # RET CHECK
            returns.append(ret)
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert not toTest.isApproximatelyEqual(toCompare)

        assert all(ret is None for ret in returns)
        assertNoNamesGenerated(toTest)


    def test_features_permute_random_NamePath_preservation(self):
        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 1111111]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        # odds are vanishingly low that it generates same configuration over
        # consecutive calls. We will pass as long as soon as it changes once
        for _ in range(5):
            toTest.features.permute()
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert not toTest.isApproximatelyEqual(toCompare)
        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    def test_features_permute_dataTypeRetainedFromList(self):
        """ Test features.permute() data not converted when permuting by list"""
        data = [['a', 2, 3.0], ['b', 5, 6.0], ['c', 8, 9.0]]
        toTest = self.constructor(data)

        toTest.features.permute([2, 1, 0])

        expData = [[3.0, 2, 'a'], [6.0, 5, 'b'], [9.0, 8, 'c']]
        exp = self.constructor(expData)

        assert toTest == exp
        assertNoNamesGenerated(toTest)

    def test_features_permute_indicesList(self):
        """ Test features.permute() when we specify a list of indices """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        toTest = self.constructor(data)

        toTest.features.permute([2, 1, 0])

        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        exp = self.constructor(expData)

        assert toTest == exp

    def test_features_permute_namesList(self):
        """ Test features.permute() when we specify a list of feature names """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        fnames = ['third', 'second', 'first']
        toTest = self.constructor(data, featureNames=fnames)

        toTest.features.permute(['first', 'second', 'third'])

        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expNames = ['first', 'second', 'third']
        exp = self.constructor(expData, featureNames=expNames)

        assert toTest == exp

    @oneLogEntryExpected
    def test_features_permute_mixedList(self):
        """ Test features.permute() when we specify a mixed list (names/indices) """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        fnames = ['third', 'second', 'first']
        toTest = self.constructor(data, featureNames=fnames)

        toTest.features.permute(['first', 'second', 0])

        expData = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expNames = ['first', 'second', 'third']
        exp = self.constructor(expData, featureNames=expNames)

        assert toTest == exp

    @raises(IndexError)
    def test_features_permute_exceptionIndicesFEmpty(self):
        """ tests features.permute() throws an IndexError when given invalid indices """
        data = [[], []]
        data = np.array(data)
        toTest = self.constructor(data)
        toTest.features.permute([1, 3])


    @raises(InvalidArgumentValue)
    def test_features_permute_exceptionIndicesSmall(self):
        """ tests features.permute() throws an InvalidArgumentValue when given an incorrectly sized indices list """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        toTest = self.constructor(data)

        toTest.features.permute([1, 0])


    @raises(InvalidArgumentValue)
    def test_features_permute_exceptionNotUniqueIds(self):
        """ tests features.permute() throws an InvalidArgumentValue when given duplicate indices """
        data = [[3, 2, 1], [6, 5, 4],[9, 8, 7]]
        data = np.array(data)
        toTest = self.constructor(data)
        toTest.features.permute([1, 1, 0])

    ########################
    # features.normalize() #
    ########################

    def test_features_normalize_parameterCount(self):
        obj = self.constructor([[1, 2], [3, 4]])
        a, _, _, d = nimble._utility.inspectArguments(obj.features.normalize)
        assert len(a) == 4 # self, function, applyResultTo, features
        assert d == (None, None)

    def test_features_normalize_exception_unexpectedInputType(self):
        obj = self.constructor([[1, 2], [3, 4]])
        with raises(InvalidArgumentType):
            obj.features.normalize({})

        with raises(InvalidArgumentType):
            obj.features.normalize(set([1]))

        with raises(InvalidArgumentType):
            obj.features.normalize(noNormalize, applyResultTo={})

    def test_features_normalize_exception_applyResultToWrongShape(self):
        obj = self.constructor([[1, 2], [3, 4]])
        alsoLong = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8,9]])
        alsoShort = self.constructor([[11], [12], [11]])

        with raises(InvalidArgumentValue):
            obj.features.normalize(noNormalize, applyResultTo=alsoLong)

        with raises(InvalidArgumentValue):
            obj.features.normalize(noNormalize, applyResultTo=alsoShort)

    def test_features_normalize_nameCompatibility(self):
        obj = self.constructor([[1, 2], [3, 4]])
        also = obj.copy()

        obj.features.normalize(noNormalize, applyResultTo=also)
        assert not obj.features._namesCreated()
        assert not also.features._namesCreated()

        obj.features.setNames(['a', 'b'])

        obj.features.normalize(noNormalize, applyResultTo=also)
        assert obj.features.getNames() == ['a', 'b']
        assert not also.features._namesCreated()

        also.features.setNames(['a', 'b'])

        obj.features.normalize(noNormalize, applyResultTo=also)
        assert obj.features.getNames() == ['a', 'b']
        assert also.features.getNames() == ['a', 'b']

        obj.features.setNames(None)

        obj.features.normalize(noNormalize, applyResultTo=also)
        assert not obj.features._namesCreated()
        assert also.features.getNames() == ['a', 'b']

        obj.features.setNames(['1', '2'])

        with raises(InvalidArgumentValue):
            obj.features.normalize(noNormalize, applyResultTo=also)

    @oneLogEntryExpected
    def test_features_normalize_success_singleInputFunction(self):
        obj = self.constructor([[1, 1, 1], [3, 3, 3], [7, 7, 7]])
        expObj = self.constructor([[0, 0, 0], [2, 2, 2], [6, 6, 6]])

        def subMin(ft):
            return ft - nimble.calculate.minimum(ft)

        ret = obj.features.normalize(subMin)

        assert ret is None
        assert expObj == obj
        assertNoNamesGenerated(obj)

    @oneLogEntryExpected
    def test_features_normalize_success_dualInputFunction(self):
        obj1 = self.constructor([[1, 2, 3], [3, 3, 3], [7, 7, 7]])
        obj2 = self.constructor([[2, 2, 2], [3, 3, 3]])
        expObj1 = self.constructor([[0, 0, 0], [2, 1, 0], [6, 5, 4]])
        expObj2 = self.constructor([[1, 0 , -1], [2, 1, 0]])

        def subMin(ft1, ft2):
            min = nimble.calculate.minimum(ft1)
            return ft1 - min, ft2 - min

        ret = obj1.features.normalize(subMin, applyResultTo=obj2)

        assert ret is None
        assert expObj1 == obj1
        assert expObj2 == obj2
        assertNoNamesGenerated(obj1)
        assertNoNamesGenerated(obj2)


    @oneLogEntryExpected
    def test_features_normalize_success_featuresLimited(self):
        obj1 = self.constructor([[1, 2, 3], [3, 3, 3], [7, 7, 7]])
        obj2 = self.constructor([[2, 2, 2], [3, 3, 3]])
        expObj1 = self.constructor([[1, 0, 0], [3, 1, 0], [7, 5, 4]])
        expObj2 = self.constructor([[2, 0 , -1], [3, 1, 0]])

        def subMin(ft1, ft2):
            min = nimble.calculate.minimum(ft1)
            return ft1 - min, ft2 - min

        ret = obj1.features.normalize(subMin, applyResultTo=obj2,
                                      features=[1, 2])

        assert ret is None
        assert expObj1 == obj1
        assert expObj2 == obj2
        assertNoNamesGenerated(obj1)
        assertNoNamesGenerated(obj2)

    #########################
    # features.fillMatching #
    #########################
    @logCountAssertionFactory(3)
    def test_features_fillMatching_mean_missing(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        ret = obj1.features.fillMatching(fill.mean, match.missing) #RET CHECK
        exp1 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        exp1.features.setNames(['a', 'b', 'c'], useLog=False)
        assert obj1 == exp1
        assert ret is None

        obj2 = obj0.copy()
        obj2.features.fillMatching(None, [3, 7])
        obj2.features.fillMatching(fill.mean, match.missing)
        exp2 = self.constructor([[1, 2, 9], [1, 11, 9], [1, 11, 9], [1, 8, 9]])
        exp2.features.setNames(['a', 'b', 'c'], useLog=False)
        assert obj2 == exp2

    def test_features_fillMatching_mean_nonNumeric(self):
        obj0 = self.constructor([[1, 2, 3], ['na', 11, 'na'], [7, 11, 'na'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.features.fillMatching(fill.mean, match.nonNumeric)
        exp1 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        exp1.features.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

        obj2 = obj0.copy()
        obj2.features.fillMatching('na', [3, 7])
        obj2.features.fillMatching(fill.mean, match.nonNumeric)
        exp2 = self.constructor([[1, 2, 9], [1, 11, 9], [1, 11, 9], [1, 8, 9]])
        exp2.features.setNames(['a', 'b', 'c'])
        assert obj2 == exp2

    def test_features_fillMatching_queryString(self):
        obj0 = self.constructor([[1, 2, 3], ['na', 11, 'na'], [7, 11, 'na'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj0.features.fillMatching(fill.mean, "== na")
        exp0 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        exp0.features.setNames(['a', 'b', 'c'])
        assert obj0 == exp0

        obj1 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj1.features.fillMatching(fill.mean, "is missing")
        exp1 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        exp1.features.setNames(['a', 'b', 'c'], useLog=False)
        assert obj1 == exp1

    @raises(InvalidArgumentValue)
    def test_features_fillMatching_mean_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.features.fillMatching(fill.mean, match.missing)

    def test_features_fillMatching_median_missing(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fillMatching(None, 11)
        obj.features.fillMatching(fill.median, match.missing)
        exp = self.constructor([[1, 2, 3], [7, 5, 6], [7, 5, 6], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fillMatching_median_nonNumeric(self):
        obj = self.constructor([[1, 2, 3], ['na', 11, 'na'], [7, 11, 'na'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fillMatching('na', 11)
        obj.features.fillMatching(fill.median, match.nonNumeric)
        exp = self.constructor([[1, 2, 3], [7, 5, 6], [7, 5, 6], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_features_fillMatching_median_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.features.fillMatching(fill.median, match.missing)

    def test_features_fillMatching_mode(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj0.features.fillMatching(None, 9)
        obj0.features.fillMatching(fill.mode, match.missing)
        exp0 = self.constructor([[1, 2, 3], [7, 11, 3], [7, 11, 3], [7, 8, 3]])
        exp0.features.setNames(['a', 'b', 'c'])
        assert obj0 == exp0

        obj1 = self.constructor([['a','b','c'], [None, 'd', None], ['e','d','c'], ['e','f','g']], featureNames=['a', 'b', 'c'])
        obj1.features.fillMatching(None, 'c')
        obj1.features.fillMatching(fill.mode, match.missing)
        exp1 = self.constructor([['a','b','g'], ['e','d', 'g'], ['e','d', 'g'], ['e','f', 'g']])
        exp1.features.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

    @raises(InvalidArgumentValue)
    def test_features_fillMatching_mode_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.features.fillMatching(fill.mode, match.missing)

    def test_features_fillMatching_zero(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fillMatching(None, 11)
        obj.features.fillMatching(0, match.missing, features=['b', 'c'])
        exp = self.constructor([[1, 2, 3], [None, 0, 0], [7, 0, 0], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fillMatching_constant(self):
        obj = self.constructor([[1, 2, 3], [0, 0, 0], [7, 0, 0], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fillMatching(100, 0)
        exp = self.constructor([[1, 2, 3], [100, 100, 100], [7, 100, 100], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fillMatching_forwardFill(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fillMatching(fill.forwardFill, match.missing)
        exp = self.constructor([[1, 2, 3], [1, 11, 3], [7, 11, 3], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_features_fillMatching_forwardFill_firstFeatureValueMissing(self):
        obj = self.constructor([[1, None, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fillMatching(fill.forwardFill, match.missing)

    def test_features_fillMatching_backwardFill(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fillMatching(None, 11)
        obj.features.fillMatching(fill.backwardFill, match.missing)
        exp = self.constructor([[1, 2, 3], [7, 8, 9], [7, 8, 9], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_features_fillMatching_backwardFill_lastFeatureValueMissing(self):
        obj = self.constructor([[1, None, 3], [None, 11, None], [7, 11, None], [7, None, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fillMatching(fill.backwardFill, match.missing)

    def test_features_fillMatching_interpolate(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fillMatching(fill.interpolate, match.missing)
        exp = self.constructor([[1, 2, 3], [4, 11, 5], [7, 11, 7], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fillMatching_custom_match(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        toTest.features.fillMatching(0, negative)
        assert toTest == exp

    def test_features_fillMatching_custom_fill(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, -3, 4], [5, 2, -3, 8], [9, 10, 11, 4]]
        exp = self.constructor(expData)

        def firstValue(feat, match):
            first = feat[0]
            ret = []
            for i, val in enumerate(feat):
                if match(val):
                    ret.append(first)
                else:
                    ret.append(val)
            return ret

        toTest.features.fillMatching(firstValue, match.negative)
        assert toTest == exp

    def test_features_fillMatching_custom_fillAndMatch(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, -3, 4], [5, 2, -3, 8], [9, 10, 11, 4]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        def firstValue(feat, match):
            first = feat[0]
            ret = []
            for i, val in enumerate(feat):
                if match(val):
                    ret.append(first)
                else:
                    ret.append(val)
            return ret

        toTest.features.fillMatching(firstValue, negative)
        assert toTest == exp
        assertNoNamesGenerated(toTest)

    def test_features_fillMatching_limited_outOfOrder(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, -12]]
        exp = self.constructor(expData)

        toTest.features.fillMatching(0, match.negative, features=[2, 1])
        assert toTest == exp
        assertNoNamesGenerated(toTest)

    def test_features_fillMatching_fillValuesWithNaN_constant(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj1 = self.constructor(data)
        obj2 = self.constructor(data)
        obj3 = self.constructor(data)
        obj1.features.fillMatching(float('nan'), 999)
        obj2.features.fillMatching(None, 999)
        obj3.features.fillMatching(np.nan, 999)
        obj1.features.fillMatching(0, np.nan)
        obj2.features.fillMatching(0, np.nan)
        obj3.features.fillMatching(0, float('nan'))

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj1 == exp
        assert obj2 == obj1
        assert obj3 == obj1

    def test_features_fillMatching_fillValuesWithNaN_list(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.features.fillMatching(None, 999)
        obj.features.fillMatching(0, [1, np.nan])

        exp = self.constructor([[0, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_features_fillMatching_fillValuesWithNaN_function(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.features.fillMatching(None, 999)
        obj.features.fillMatching(0, match.missing)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_features_fillMatching_fillNumericWithNonNumeric(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.features.fillMatching('na', 999)

        exp = self.constructor([[1, 2, 'na', 4], [5, 'na', 'na', 8], [9, 10, 11, 'na']])
        assert obj == exp

    def test_features_fillMatching_NamePath_preservation(self):
        data = [['a'], ['b'], [1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = os.path.abspath("TestAbsPath")
        toTest._relPath = "testRelPath"

        toTest.features.fillMatching(0, match.nonNumeric)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == os.path.abspath("TestAbsPath")
        assert toTest.relativePath == 'testRelPath'

    #######################
    # points.fillMatching #
    #######################
    @logCountAssertionFactory(3)
    def test_points_fillMatching_mean_missing(self):
        obj0 = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        ret = obj1.points.fillMatching(fill.mean, match.missing) # RET CHECK
        exp1 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        exp1.points.setNames(['a', 'b', 'c'], useLog=False)
        assert obj1 == exp1
        assert ret is None

        obj2 = obj0.copy()
        obj2.points.fillMatching(None, [4, 8])
        obj2.points.fillMatching(fill.mean, match.missing)
        exp2 = self.constructor([[1, 2, 3, 2], [6, 6, 6, 6], [9, 1, 11, 7]])
        exp2.points.setNames(['a', 'b', 'c'], useLog=False)
        assert obj2 == exp2

    def test_points_fillMatching_mean_nonNumeric(self):
        obj0 = self.constructor([[1, 2, 3, 4], ['na', 6, 'na', 8], [9, 1, 11, 'na']], pointNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.points.fillMatching(fill.mean, match.nonNumeric)
        exp1 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        exp1.points.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

        obj2 = obj0.copy()
        obj2.points.fillMatching('na', [4, 8])
        obj2.points.fillMatching(fill.mean, match.nonNumeric)
        exp2 = self.constructor([[1, 2, 3, 2], [6, 6, 6, 6], [9, 1, 11, 7]])
        exp2.points.setNames(['a', 'b', 'c'])
        assert obj2 == exp2

    def test_points_fillMatching_queryString(self):
        obj0 = self.constructor([[1, 2, 3, 4], ['na', 6, 'na', 8], [9, 1, 11, 'na']], pointNames=['a', 'b', 'c'])
        obj0.points.fillMatching(fill.mean, "is not numeric")
        exp0 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        exp0.points.setNames(['a', 'b', 'c'])
        assert obj0 == exp0

        obj1 = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj1.points.fillMatching(fill.mean, "is missing")
        exp1 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        exp1.points.setNames(['a', 'b', 'c'], useLog=False)
        assert obj1 == exp1

    @raises(InvalidArgumentValue)
    def test_points_fillMatching_mean_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.points.fillMatching(fill.mean, match.missing)

    def test_points_fillMatching_median_missing(self):
        obj = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fillMatching(fill.median, match.missing)
        exp = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 9]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_points_fillMatching_median_nonNumeric(self):
        obj = self.constructor([[1, 2, 3, 4], ['na', 6, 'na', 8], [9, 1, 11, 'na']], pointNames=['a', 'b', 'c'])
        obj.points.fillMatching('na', 11)
        obj.points.fillMatching(fill.median, match.nonNumeric)
        exp = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 5, 5]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_points_fillMatching_median_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.points.fillMatching(fill.median, match.missing)

    def test_points_fillMatching_mode(self):
        obj0 = self.constructor([[1, 2, 3, 3], [None, 6, 8, 8], [9, 9, 11, None]], pointNames=['a', 'b', 'c'])
        obj0.points.fillMatching(None, 9)
        obj0.points.fillMatching(fill.mode, match.missing)
        exp0 = self.constructor([[1, 2, 3, 3], [8, 6, 8, 8], [11, 11, 11, 11]], pointNames=['a', 'b', 'c'])
        exp0.points.setNames(['a', 'b', 'c'])
        assert obj0 == exp0

        obj1 = self.constructor([['a', 'b', 'c', 'c'], [None, 'f', 'h', 'h'], ['i', 'i', 'k', None]], pointNames=['a', 'b', 'c'])
        obj1.points.fillMatching(None, 'b')
        obj1.points.fillMatching(fill.mode, match.missing)
        exp1 = self.constructor([['a', 'c', 'c', 'c'], ['h', 'f', 'h', 'h'], ['i', 'i', 'k', 'i']], pointNames=['a', 'b', 'c'])
        exp1.points.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

    @raises(InvalidArgumentValue)
    def test_points_fillMatching_mode_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.points.fillMatching(fill.mode, match.missing)

    def test_points_fillMatching_zero(self):
        obj = self.constructor([[1, 2, None], [None, 11, 6], [7, 11, None], [7, 8, 9]], pointNames=['a', 'b', 'c', 'd'])
        obj.points.fillMatching(None, 11)
        obj.points.fillMatching(0, match.missing, points=['b', 'c'])
        exp = self.constructor([[1, 2, None], [0, 0, 6], [7, 0, 0], [7, 8, 9]])
        exp.points.setNames(['a', 'b', 'c', 'd'])
        assert obj == exp

    def test_points_fillMatching_constant(self):
        obj = self.constructor([[1, 2, 3], [0, 0, 0], [7, 0, 0], [7, 8, 9]], pointNames=['a', 'b', 'c', 'd'])
        obj.points.fillMatching(100, 0)
        exp = self.constructor([[1, 2, 3], [100, 100, 100], [7, 100, 100], [7, 8, 9]])
        exp.points.setNames(['a', 'b', 'c', 'd'])
        assert obj == exp

    def test_points_fillMatching_forwardFill(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fillMatching(fill.forwardFill, match.missing)
        exp = self.constructor([[1, 2, 3, 4], [5, 5, 5, 8], [9, 1, 11, 11]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_points_fillMatching_forwardFill_firstFeatureValueMissing(self):
        obj = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fillMatching(fill.forwardFill, match.missing)

    def test_points_fillMatching_backwardFill(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [None, 1, 11, 2]], pointNames=['a', 'b', 'c'])
        obj.points.fillMatching(None, 11)
        obj.points.fillMatching(fill.backwardFill, match.missing)
        exp = self.constructor([[1, 2, 3, 4], [5, 8, 8, 8], [1, 1, 2, 2]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_points_fillMatching_backwardFill_lastFeatureValueMissing(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fillMatching(fill.backwardFill, match.missing)

    def test_points_fillMatching_interpolate(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [None, 1, None, 5]], pointNames=['a', 'b', 'c'])
        obj.points.fillMatching(fill.interpolate, match.missing)
        exp = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 3, 5]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_points_fillMatching_custom_match(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        toTest.points.fillMatching(0, negative)
        assert toTest == exp
        assertNoNamesGenerated(toTest)

    def test_points_fillMatching_custom_fill(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 1, 4], [5, 5, 5, 8], [9, 10, 11, 9]]
        exp = self.constructor(expData)

        def firstValue(feat, match):
            first = feat[0]
            ret = []
            for i, val in enumerate(feat):
                if match(val):
                    ret.append(first)
                else:
                    ret.append(val)
            return ret

        toTest.points.fillMatching(firstValue, match.negative)
        assert toTest == exp

    def test_points_fillMatching_custom_fillAndMatch(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 1, 4], [5, 5, 5, 8], [9, 10, 11, 9]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        def firstValue(feat, match):
            first = feat[0]
            ret = []
            for i, val in enumerate(feat):
                if match(val):
                    ret.append(first)
                else:
                    ret.append(val)
            return ret

        toTest.points.fillMatching(firstValue, negative)
        assert toTest == exp

    def test_points_fillMatching_limited_outOfOrder(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, -3, 4], [5, 0, 0, 8], [9, 10, 11, 0]]
        exp = self.constructor(expData)

        toTest.points.fillMatching(0, match.negative, points=[2, 1])
        assert toTest == exp
        assertNoNamesGenerated(toTest)

    def test_points_fillMatching_fillValuesWithNaN_constant(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj1 = self.constructor(data)
        obj2 = self.constructor(data)
        obj3 = self.constructor(data)
        obj1.points.fillMatching(float('nan'), 999)
        obj2.points.fillMatching(None, 999)
        obj3.points.fillMatching(np.nan, 999)
        obj1.points.fillMatching(0, np.nan)
        obj2.points.fillMatching(0, np.nan)
        obj3.points.fillMatching(0, float('nan'))

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj1 == exp
        assert obj2 == obj1
        assert obj3 == obj1

    def test_points_fillMatching_fillValuesWithNaN_list(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.points.fillMatching(None, 999)
        obj.points.fillMatching(0, [1, np.nan])

        exp = self.constructor([[0, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_points_fillMatching_fillValuesWithNaN_function(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.points.fillMatching(None, 999)
        obj.points.fillMatching(0, match.missing)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_points_fillMatching_fillNumericWithNonNumeric(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.points.fillMatching('na', 999)

        exp = self.constructor([[1, 2, 'na', 4], [5, 'na', 'na', 8], [9, 10, 11, 'na']])
        assert obj == exp

    def test_points_fillMatching_NamePath_preservation(self):
        data = [['a', 'b', 1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = os.path.abspath("TestAbsPath")
        toTest._relPath = "testRelPath"

        toTest.points.fillMatching(0, match.nonNumeric)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == os.path.abspath("TestAbsPath")
        assert toTest.relativePath == 'testRelPath'

    ####################################
    # points.splitByCollapsingFeatures #
    ####################################
    @logCountAssertionFactory(3)
    def test_points_splitByCollapsingFeatures_sequentialFeatures(self):
        data = [[0,0,1,2,3,4], [1,1,5,6,7,8], [2,2,-1,-2,-3,-4]]
        ptNames = ["0", "1", "2"]
        ftNames = ["ret0", "ret1", "coll0", "coll1", "coll2", "coll3"]

        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [[0,0,"coll0",1], [0,0,"coll1",2], [0,0,"coll2",3], [0,0,"coll3",4],
                   [1,1,"coll0",5], [1,1,"coll1",6], [1,1,"coll2",7], [1,1,"coll3",8],
                   [2,2,"coll0",-1], [2,2,"coll1",-2], [2,2,"coll2",-3], [2,2,"coll3",-4]]
        expPnames = ["0_0", "0_1", "0_2", "0_3",
                     "1_0", "1_1", "1_2", "1_3",
                     "2_0", "2_1", "2_2", "2_3"]
        expFNames = ["ret0", "ret1", "ftNames", "ftValues"]

        exp = self.constructor(expData, pointNames=expPnames, featureNames=expFNames)

        nameFeatureNames = "ftNames"
        nameFeatureValues = "ftValues"
        test0 = toTest.copy()
        toCollapse = ["coll0", "coll1", "coll2", "coll3"]
        test0.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test0 == exp

        test1 = toTest.copy()
        toCollapse = [2, 3, 4, 5]
        test1.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test1 == exp

        test2 = toTest.copy()
        toCollapse = [2, "coll1", 4, "coll3"]
        test2.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test2 == exp

    @logCountAssertionFactory(3)
    def test_points_splitByCollapsingFeatures_nonSequentialFeatures(self):
        data = [[1,0,2,0,3,4], [5,1,6,1,7,8], [-1,2,-2,2,-3,-4]]
        ptNames = ["0", "1", "2"]
        ftNames = ["coll0", "ret0", "coll1", "ret1", "coll2", "coll3"]

        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [[0,0,"coll3",4], [0,0,"coll1",2], [0,0,"coll2",3], [0,0,"coll0",1],
                   [1,1,"coll3",8], [1,1,"coll1",6], [1,1,"coll2",7], [1,1,"coll0",5],
                   [2,2,"coll3",-4], [2,2,"coll1",-2], [2,2,"coll2",-3], [2,2,"coll0",-1]]
        expPnames = ["0_0", "0_1", "0_2", "0_3",
                     "1_0", "1_1", "1_2", "1_3",
                     "2_0", "2_1", "2_2", "2_3"]
        expFNames = ["ret0", "ret1", "ftNames", "ftValues"]

        exp = self.constructor(expData, pointNames=expPnames, featureNames=expFNames)

        nameFeatureNames = "ftNames"
        nameFeatureValues = "ftValues"
        test0 = toTest.copy()
        toCollapse = ["coll3", "coll1", "coll2", "coll0"]
        test0.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test0 == exp

        test1 = toTest.copy()
        toCollapse = [5, 2, 4, 0]
        test1.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test1 == exp

        test2 = toTest.copy()
        toCollapse = [5, "coll1", "coll2", 0]
        test2.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test2 == exp

    def test_points_splitByCollapsingFeatures_noPointNames(self):
        data = [[0,0,1,2,3,4], [1,1,5,6,7,8], [2,2,-1,-2,-3,-4]]
        ftNames = ["ret0", "ret1", "coll0", "coll1", "coll2", "coll3"]

        toTest = self.constructor(data, featureNames=ftNames)

        expData = [[0,0,"coll0",1], [0,0,"coll1",2], [0,0,"coll2",3], [0,0,"coll3",4],
                   [1,1,"coll0",5], [1,1,"coll1",6], [1,1,"coll2",7], [1,1,"coll3",8],
                   [2,2,"coll0",-1], [2,2,"coll1",-2], [2,2,"coll2",-3], [2,2,"coll3",-4]]
        expFNames = ["ret0", "ret1", "ftNames", "ftValues"]

        exp = self.constructor(expData, featureNames=expFNames)

        toCollapse = ["coll0", "coll1", "coll2", "coll3"]
        toTest.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert toTest == exp

    def test_points_splitByCollapsingFeatures_noNames(self):
        data = [[0,0,1,2,3,4], [1,1,5,6,7,8], [2,2,-1,-2,-3,-4]]

        toTest = self.constructor(data)

        expData = [[0,0,2,1], [0,0,3,2], [0,0,4,3], [0,0,5,4],
                   [1,1,2,5], [1,1,3,6], [1,1,4,7], [1,1,5,8],
                   [2,2,2,-1], [2,2,3,-2], [2,2,4,-3], [2,2,5,-4]]

        exp = self.constructor(expData)
        exp.features.setNames("ftIndex", oldIdentifiers=2)
        exp.features.setNames("ftValues", oldIdentifiers=3)

        toCollapse = [2, 3, 4, 5]
        toTest.points.splitByCollapsingFeatures(toCollapse, "ftIndex", "ftValues")

        assert toTest == exp

    #####################################
    # points.combineByExpandingFeatures #
    #####################################
    @logCountAssertionFactory(3)
    def test_points_combineByExpandingFeatures_singleValuesFeature(self):
        data = [["p1", 100, 'r1', 9.5], ["p1", 100, 'r2', 9.9], ["p1", 100, 'r3', 9.8],
                ["p2", 100, 'r1', 6.5], ["p2", 100, 'r2', 6.0], ["p2", 100, 'r3', 5.9],
                ["p3", 100, 'r1', 11], ["p3", 100, 'r2', 11.2], ["p3", 100, 'r3', 11.0],
                ["p1", 200, 'r1', 18.1], ["p1", 200, 'r2', 20.1], ["p1", 200, 'r3', 19.8]]
        pNames = [str(i) for i in range(12)]
        fNames = ['type', 'dist', 'run', 'time']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["p1", 100, 9.5, 9.9, 9.8],
                   ["p2", 100, 6.5, 6.0, 5.9],
                   ["p3", 100, 11, 11.2, 11.0],
                   ["p1", 200, 18.1, 20.1, 19.8]]
        expFNames = ['type', 'dist', 'r1', 'r2', 'r3']
        expPNames = ["0", "3", "6", "9"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=expFNames)

        test0 = toTest.copy()
        test0.points.combineByExpandingFeatures('run', 'time')
        assert test0 == exp

        test1 = toTest.copy()
        test1.points.combineByExpandingFeatures(2, 3)
        assert test1 == exp

        test2 = toTest.copy()
        test2.points.combineByExpandingFeatures('run', 3)
        assert test2 == exp

    @logCountAssertionFactory(3)
    def test_points_combineByExpandingFeatures_multipleValuesFeatures(self):
        data = [["p1", 100, 'r1', 9.5, 9.4], ["p1", 100, 'r2', 9.9, 9.7], ["p1", 100, 'r3', 9.8, 9.8],
                ["p2", 100, 'r1', 6.5, 6.5], ["p2", 100, 'r2', 6.0, 6.2], ["p2", 100, 'r3', 5.9, 6.1],
                ["p3", 100, 'r1', 11.0, 10.9], ["p3", 100, 'r2', 11.2, 11.1], ["p3", 100, 'r3', 11.0, 11.0],
                ["p1", 200, 'r1', 18.1, 18.0], ["p1", 200, 'r2', 20.1, 20.2], ["p1", 200, 'r3', 19.8, 19.9]]
        pNames = [str(i) for i in range(12)]
        fNames = ['type', 'dist', 'run', 'timer1', 'timer2']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["p1", 100, 9.5, 9.4, 9.9, 9.7, 9.8, 9.8],
                   ["p2", 100, 6.5, 6.5, 6.0, 6.2, 5.9, 6.1],
                   ["p3", 100, 11, 10.9, 11.2, 11.1, 11.0, 11.0],
                   ["p1", 200, 18.1, 18.0, 20.1, 20.2, 19.8, 19.9]]
        expFNames = ['type', 'dist', 'r1_timer1', 'r1_timer2', 'r2_timer1',
                     'r2_timer2', 'r3_timer1', 'r3_timer2']
        expPNames = ["0", "3", "6", "9"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=expFNames)

        test0 = toTest.copy()
        test0.points.combineByExpandingFeatures('run', ['timer1', 'timer2'])
        assert test0 == exp

        test1 = toTest.copy()
        test1.points.combineByExpandingFeatures(2, [3, 4])
        assert test1 == exp

        test2 = toTest.copy()
        test2.points.combineByExpandingFeatures('run', [3, 4])
        assert test2 == exp

    def test_points_combineByExpandingFeatures_withMissing(self):
        data = [["p1", 100, 'r2', 9.9, 9.7], ["p1", 100, 'r3', 9.8, 9.8],
                ["p2", 100, 'r1', 6.5, 6.5], ["p2", 100, 'r2', 6.0, 6.2], ["p2", 100, 'r3', 5.9, 6.1],
                ["p3", 100, 'r1', 11.0, 10.9], ["p3", 100, 'r2', 11.2, 11.1],
                ["p1", 200, 'r1', 18.1, 18.0], ["p1", 200, 'r2', 20.1, 20.2], ["p1", 200, 'r3', 19.8, 19.9]]
        pNames = [str(i) for i in range(10)]
        fNames = ['type', 'dist', 'run', 'timer1', 'timer2']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["p1", 100, 9.9, 9.7, 9.8, 9.8, None, None],
                   ["p2", 100, 6.0, 6.2, 5.9, 6.1, 6.5, 6.5],
                   ["p3", 100, 11.2, 11.1, None, None, 11.0, 10.9],
                   ["p1", 200, 20.1, 20.2, 19.8, 19.9, 18.1, 18.0]]
        expFNames = ['type', 'dist', 'r2_timer1', 'r2_timer2', 'r3_timer1',
                     'r3_timer2', 'r1_timer1', 'r1_timer2']
        expPNames = ["0", "2", "5", "7"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=expFNames)

        toTest.points.combineByExpandingFeatures('run', ['timer1', 'timer2'])

        assert toTest == exp

    def test_points_combineByExpandingFeatures_nonConcurrentNamesAndValues(self):
        data = [[100, 'r1', "p1", 9.5, 9.4], [100, 'r2', "p1", 9.9, 9.7], [100, 'r3', "p1", 9.8, 9.8],
                [100, 'r1', "p2", 6.5, 6.5], [100, 'r2', "p2", 6.0, 6.2], [100, 'r3', "p2", 5.9, 6.1],
                [100, 'r1', "p3", 11.0, 10.9], [100, 'r2', "p3", 11.2, 11.1], [100, 'r3', "p3", 11.0, 11.0],
                [200, 'r1', "p1", 18.1, 18.0], [200, 'r2', "p1", 20.1, 20.2], [200, 'r3', "p1", 19.8, 19.9]]
        pNames = [str(i) for i in range(12)]
        fNames = ['type', 'run', 'dist', 'timer1', 'timer2']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [[100, 9.5, 9.4, 9.9, 9.7, 9.8, 9.8, "p1"],
                   [100, 6.5, 6.5, 6.0, 6.2, 5.9, 6.1, "p2"],
                   [100, 11, 10.9, 11.2, 11.1, 11.0, 11.0, "p3"],
                   [200, 18.1, 18.0, 20.1, 20.2, 19.8, 19.9, "p1"]]
        expFNames = ['type', 'r1_timer1', 'r1_timer2', 'r2_timer1',
                     'r2_timer2', 'r3_timer1', 'r3_timer2', 'dist']
        expPNames = ["0", "3", "6", "9"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=expFNames)

        toTest.points.combineByExpandingFeatures('run', ['timer1', 'timer2'])
        assert toTest == exp

    def test_points_combineByExpandingFeatures_noPointNames(self):
        data = [["p1", 100, 'r1', 9.5, 9.4], ["p1", 100, 'r2', 9.9, 9.7], ["p1", 100, 'r3', 9.8, 9.8],
                ["p2", 100, 'r1', 6.5, 6.5], ["p2", 100, 'r2', 6.0, 6.2], ["p2", 100, 'r3', 5.9, 6.1],
                ["p3", 100, 'r1', 11.0, 10.9], ["p3", 100, 'r2', 11.2, 11.1], ["p3", 100, 'r3', 11.0, 11.0],
                ["p1", 200, 'r1', 18.1, 18.0], ["p1", 200, 'r2', 20.1, 20.2], ["p1", 200, 'r3', 19.8, 19.9]]
        fNames = ['type', 'dist', 'run', 'timer1', 'timer2']
        toTest = self.constructor(data, featureNames=fNames)

        expData = [["p1", 100, 9.5, 9.4, 9.9, 9.7, 9.8, 9.8],
                   ["p2", 100, 6.5, 6.5, 6.0, 6.2, 5.9, 6.1],
                   ["p3", 100, 11, 10.9, 11.2, 11.1, 11.0, 11.0],
                   ["p1", 200, 18.1, 18.0, 20.1, 20.2, 19.8, 19.9]]
        expFNames = ['type', 'dist', 'r1_timer1', 'r1_timer2', 'r2_timer1',
                     'r2_timer2', 'r3_timer1', 'r3_timer2']
        exp = self.constructor(expData, featureNames=expFNames)

        toTest.points.combineByExpandingFeatures('run', ['timer1', 'timer2'])
        assert toTest == exp

    def test_points_combineByExpandingFeatures_noNames(self):
        data = [["p1", 100, 'r1', 9.5, 9.4], ["p1", 100, 'r2', 9.9, 9.7], ["p1", 100, 'r3', 9.8, 9.8],
                ["p2", 100, 'r1', 6.5, 6.5], ["p2", 100, 'r2', 6.0, 6.2], ["p2", 100, 'r3', 5.9, 6.1],
                ["p3", 100, 'r1', 11.0, 10.9], ["p3", 100, 'r2', 11.2, 11.1], ["p3", 100, 'r3', 11.0, 11.0],
                ["p1", 200, 'r1', 18.1, 18.0], ["p1", 200, 'r2', 20.1, 20.2], ["p1", 200, 'r3', 19.8, 19.9]]
        toTest = self.constructor(data)

        expData = [["p1", 100, 9.5, 9.4, 9.9, 9.7, 9.8, 9.8],
                   ["p2", 100, 6.5, 6.5, 6.0, 6.2, 5.9, 6.1],
                   ["p3", 100, 11, 10.9, 11.2, 11.1, 11.0, 11.0],
                   ["p1", 200, 18.1, 18.0, 20.1, 20.2, 19.8, 19.9]]

        exp = self.constructor(expData,)
        exp.features.setNames('r1_3', oldIdentifiers=2)
        exp.features.setNames('r1_4', oldIdentifiers=3)
        exp.features.setNames('r2_3', oldIdentifiers=4)
        exp.features.setNames('r2_4', oldIdentifiers=5)
        exp.features.setNames('r3_3', oldIdentifiers=6)
        exp.features.setNames('r3_4', oldIdentifiers=7)

        toTest.points.combineByExpandingFeatures(2, [3, 4])
        assert toTest == exp

    @raises(ImproperObjectAction)
    def test_points_combineByExpandingFeatures_2valuesSameFeature(self):
        data = [["p1", 100, 'r1', 9.5, 9.4], ["p1", 100, 'r2', 9.9, 9.7], ["p1", 100, 'r3', 9.8, 9.8],
                ["p2", 100, 'r1', 6.5, 6.5], ["p2", 100, 'r2', 6.0, 6.2], ["p2", 100, 'r3', 5.9, 6.1],
                ["p3", 100, 'r1', 11.0, 10.9], ["p3", 100, 'r2', 11.2, 11.1], ["p3", 100, 'r1', 11.0, 11.0], # r1 in p3 twice
                ["p1", 200, 'r1', 18.1, 18.0], ["p1", 200, 'r2', 20.1, 20.2], ["p1", 200, 'r3', 19.8, 19.9]]
        pNames = [str(i) for i in range(12)]
        fNames = ['type', 'dist', 'run', 'timer1', 'timer2']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        toTest.points.combineByExpandingFeatures('run', ['timer1', 'timer2'])

    def test_points_combineByExpandingFeatures_single_valuesMapToSameString(self):

        data = [["p1", 100, 1, 9.5], ["p1", 100, 2, 9.9], ["p1", 100, '2', 9.8],
                ["p2", 100, 1, 6.5], ["p2", 100, 2, 6.0], ["p2", 100, '2', 5.9],
                ["p3", 100, 1, 11], ["p3", 100, 2, 11.2], ["p3", 100, '2', 11.0],
                ["p1", 200, 1, 18.1], ["p1", 200, 2, 20.1], ["p1", 200, '2', 19.8]]
        pNames = [str(i) for i in range(12)]
        fNames = ['type', 'dist', 'run', 'time']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["p1", 100, 9.5, 9.9, 9.8],
                   ["p2", 100, 6.5, 6.0, 5.9],
                   ["p3", 100, 11, 11.2, 11.0],
                   ["p1", 200, 18.1, 20.1, 19.8]]
        expFNames = ['type', 'dist', '1', '2(int)', '2(str)']
        expPNames = ["0", "3", "6", "9"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=expFNames)

        msg = 'Identical strings returned by classes that are unequal in '
        msg += 'featureWithFeatureNames.'
        with raises(ImproperObjectAction, match=msg):
            toTest.points.combineByExpandingFeatures('run', 'time')

        toTest.points.combineByExpandingFeatures('run', 'time',
                                                 modifyDuplicateFeatureNames=True)
        assert toTest == exp

    def test_points_combineByExpandingFeatures_multiple_valuesMapToSameString(self):
        data = [["p1", 100, '1', 9.5, 9.4], ["p1", 100, '2', 9.9, 9.7], ["p1", 100, 2, 9.8, 9.8],
                ["p2", 100, '1', 6.5, 6.5], ["p2", 100, '2', 6.0, 6.2], ["p2", 100, 2, 5.9, 6.1],
                ["p3", 100, '1', 11.0, 10.9], ["p3", 100, '2', 11.2, 11.1], ["p3", 100, 2, 11.0, 11.0],
                ["p1", 200, '1', 18.1, 18.0], ["p1", 200, '2', 20.1, 20.2], ["p1", 200, 2, 19.8, 19.9]]
        pNames = [str(i) for i in range(12)]
        fNames = ['type', 'dist', 'run', 'timer1', 'timer2']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["p1", 100, 9.5, 9.4, 9.9, 9.7, 9.8, 9.8],
                   ["p2", 100, 6.5, 6.5, 6.0, 6.2, 5.9, 6.1],
                   ["p3", 100, 11, 10.9, 11.2, 11.1, 11.0, 11.0],
                   ["p1", 200, 18.1, 18.0, 20.1, 20.2, 19.8, 19.9]]
        expFNames = ['type', 'dist', '1_timer1', '1_timer2', '2(str)_timer1',
                     '2(str)_timer2', '2(int)_timer1', '2(int)_timer2']
        expPNames = ["0", "3", "6", "9"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=expFNames)

        msg = 'Identical strings returned by classes that are unequal in '
        msg += 'featureWithFeatureNames.'
        with raises(ImproperObjectAction, match=msg):
            toTest.points.combineByExpandingFeatures('run', ['timer1', 'timer2'])

        toTest.points.combineByExpandingFeatures('run', ['timer1', 'timer2'],
                                                 modifyDuplicateFeatureNames=True)
        assert toTest == exp

    ###########################
    # features.splitByParsing #
    ###########################
    @oneLogEntryExpected
    def test_features_splitByParsing_integer(self):
        data = [[0, "a1", 0], [1, "b2", 1], [2, "c3", 2]]
        pNames = ["0", "1", "2"]
        fNames = ["f0", "merged", "f1"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [[0, "a", "1", 0], [1, "b", "2", 1], [2, "c", "3", 2]]
        expFNames = ["f0", "split0", "split1", "f1"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        toTest.features.splitByParsing(1, 1, ["split0", "split1"])
        assert toTest == exp

    @oneLogEntryExpected
    def test_features_splitByParsing_string(self):
        data = [["a-1", 0], ["b-2", 1], ["c-3", 2]]
        pNames = ["a", "b", "c"]
        fNames = ["merged", "f0"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["a", "1", 0], ["b", "2", 1], ["c", "3", 2]]
        expFNames = ["split0", "split1", "f0"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        toTest.features.splitByParsing(0, '-', ["split0", "split1"])
        assert toTest == exp

    def test_features_splitByParsing_listIntegers(self):
        data = [[0, "a1z9000AAA"], [1, "b2y8000BBB"], [2, "c3x7000CCC"]]
        pNames = ["0", "1", "2"]
        fNames = ["f0", "merged"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [[0, "a1", "z9", "000", "AAA"], [1, "b2", "y8", "000", "BBB"],
                   [2, "c3", "x7", "000", "CCC"]]
        expFNames = ["f0", "split0", "split1", "split2", "split3"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        toTest.features.splitByParsing("merged", [2,4,7], ["split0", "split1", "split2", "split3"])
        assert toTest == exp

    def test_features_splitByParsing_listStrings(self):
        data = [[0, "a1/9000-AAA"], [1, "b2/8000-BBB"], [2, "c3/7000-CCC"]]
        pNames = ["0", "1", "2"]
        fNames = ["f0", "merged"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [[0, "a1", "9000", "AAA"], [1, "b2", "8000", "BBB"],
                   [2, "c3", "7000", "CCC"]]
        expFNames = ["f0", "split0", "split1", "split2"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        toTest.features.splitByParsing(1, ['/','-'], ["split0", "split1", "split2"])
        assert toTest == exp

    @oneLogEntryExpected
    def test_features_splitByParsing_listMixed(self):
        data = [[0, "a1/9z000-AAA"], [1, "b2/8y000-BBB"], [2, "c3/7x000-CCC"]]
        pNames = ["0", "1", "2"]
        fNames = ["f0", "merged"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [[0, "a1", "9z", "000", "AAA"], [1, "b2", "8y", "000", "BBB"],
                   [2, "c3", "7x", "000", "CCC"]]
        expFNames = ["f0", "split0", "split1", "split2", "split3"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        toTest.features.splitByParsing(1, ['/', 5, '-'], ["split0", "split1", "split2", "split3"])
        assert toTest == exp

    def test_features_splitByParsing_function(self):
        data = [["a1z9000AAA"], ["b2y8000BBB"], ["c3x7000CCC"]]
        pNames = ["a", "b", "c"]
        fNames = ["merged"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["a1z9", "AAA"], ["b2y8", "BBB"], ["c3x7", "CCC"]]
        expFNames = ["split0", "split1"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        def splitter(value):
            return value.split('000')

        toTest.features.splitByParsing("merged", splitter, ["split0", "split1"])
        assert toTest == exp

    def test_features_splitByParsing_regex(self):
        data = [["a1z9000AAA", '001'], ["b2y8000BBB", '001'], ["c3x7000CCC", '002']]
        pNames = ["a", "b", "c"]
        fNames = ["merged", "f0"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["a1", "9000AAA", '001'], ["b2", "8000BBB", '001'], ["c3", "7000CCC", '002']]
        expFNames = ["split0", "split1", "f0"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        def splitter(value):
            import re
            return re.split('[xyz]', value)

        toTest.features.splitByParsing("merged", splitter, ["split0", "split1"])
        assert toTest == exp

    def test_features_splitByParsing_noNames(self):
        data = [[0, "a1", 0], [1, "b2", 1], [2, "c3", 2]]
        toTest = self.constructor(data)

        expData = [[0, "a", "1", 0], [1, "b", "2", 1], [2, "c", "3", 2]]
        exp = self.constructor(expData)
        exp.features.setNames("split0", oldIdentifiers=1)
        exp.features.setNames("split1", oldIdentifiers=2)

        toTest.features.splitByParsing(1, 1, ["split0", "split1"])
        assert toTest == exp

    @raises(InvalidArgumentValueCombination)
    def test_features_splitByParsing_shortSplitList(self):
        data = [["a-1", 0], ["b-2", 1], ["c3", 2]]
        pNames = ["a", "b", "c"]
        fNames = ["merged", "f0"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        toTest.features.splitByParsing("merged", '-', ["split0", "split1"])

    @raises(InvalidArgumentValueCombination)
    def test_features_splitByParsing_longSplitList(self):
        data = [["a-1", 0], ["b-2-2", 1], ["c-3", 2]]
        pNames = ["a", "b", "c"]
        fNames = ["merged", "f0"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        toTest.features.splitByParsing("merged", '-', ["split0", "split1"])

    ###################
    # replace helpers #
    ###################

    def replacements(self, axis, single):
        replace = [list, tuple, np.array, np.matrix, scipy.sparse.coo_matrix,
                   pd.DataFrame]

        if single:
            replace.append(pd.Series)

        nimLst = functools.partial(nimble.data, returnType="List", useLog=False)
        nimMtx = functools.partial(nimble.data, returnType="Matrix", useLog=False)
        nimSp = functools.partial(nimble.data, returnType="Sparse", useLog=False)
        nimDF = functools.partial(nimble.data, returnType="DataFrame", useLog=False)

        if axis == 'point':
            replace.extend([nimLst, nimMtx, nimSp, nimDF])
        elif single:
            replace.extend([lambda *args, **kwargs: nimLst(*args, **kwargs).T,
                            lambda *args, **kwargs: nimMtx(*args, **kwargs).T,
                            lambda *args, **kwargs: nimSp(*args, **kwargs).T,
                            lambda *args, **kwargs: nimDF(*args, **kwargs).T])

        return replace

    @noLogEntryExpected
    def back_replace(self, axis, data, expData, replace, replaceLocs):
        toTest = self.constructor(data)

        exp = self.constructor(expData)

        replaceMap = {0: 'a', 1: 'b', 2: 'c'}
        if axis == 'point':
            kwarg = 'points'
            opposite = 'features'
        else:
            kwarg = 'features'
            opposite = 'points'

        single = len(replaceLocs) == 1

        test = toTest.copy()
        with tempfile.NamedTemporaryFile('w+') as tmp:
            if single:
                tmp.write(','.join(map(str, replace)))
                tmp.write('\n')
            else:
                for lst in replace:
                    tmp.write(','.join(map(str, lst)))
                    tmp.write('\n')
            tmp.seek(0)
            test._getAxis(axis).replace(tmp.name, useLog=False,
                                        **{kwarg: replaceLocs})
            assert test == exp

        for rep in self.replacements(axis, single):
            replacement = rep(replace)
            required = "{}s argument is required".format(axis)
            with raises(InvalidArgumentValue, match=required):
                toTest._getAxis(axis).replace(replacement)

            names = ['a', 'b', 'c', 'd'][:len(toTest._getAxis(axis))]
            namedTest = toTest.copy()
            namedTest._getAxis(axis).setNames(names, useLog=False)
            exp._getAxis(axis).setNames(names, useLog=False)

            needsNames = "data must have {}Names".format(axis)
            with raises(InvalidArgumentValue, match=needsNames):
                namedTest._getAxis(axis).replace(replacement)

            replaceNames = [replaceMap[i] for i in replaceLocs]
            # providing points/features argument
            test = namedTest.copy()
            test._getAxis(axis).replace(replacement, useLog=False,
                                        **{kwarg: replaceNames})
            assert test == exp

            test = namedTest.copy()
            if isinstance(replacement, nimble.core.data.Base):
                # Base with axis name set
                replacement._getAxis(axis).setNames(replaceNames, useLog=False)
                test._getAxis(axis).replace(replacement, useLog=False)
            else:
                # providing point/featureNames keyword argument
                axisNames = {axis + 'Names': replaceNames}
                test._getAxis(axis).replace(replacement, useLog=False,
                                            **axisNames)
            assert test == exp

            with raises(InvalidArgumentValue, match="The number of locations"):
                tooManyLocs = replaceLocs.copy()
                tooManyLocs.append(0)
                test._getAxis(axis).replace(replacement, tooManyLocs)

            wrongElemCount = 'as many elements as {}'.format(opposite)
            with raises(InvalidArgumentValue, match=wrongElemCount):
                badReplace = copy.deepcopy(replace)
                if single:
                    badReplace.append(99)
                elif axis == 'point':
                    for lst in badReplace:
                        lst.append(99)
                else:
                    badReplace.append([99] * len(badReplace[0]))
                replacement = rep(badReplace)
                test._getAxis(axis).replace(replacement,
                                            **{kwarg: replaceNames})

    def back_replace_all(self, axis):
        data = [['x', 'x', 'x'], ['y', 'y', 'y'], ['z', 'z', 'z']]
        exp = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        toTest = self.constructor(data)
        toTest._getAxis(axis).replace(exp)

        assert toTest == self.constructor(exp)

    ##################
    # points.replace #
    ##################

    @oneLogEntryExpected
    def test_points_replace_single(self):
        data = [[0, 1, 2], ['x', 'x', 'x'], [6, 7, 8]]
        exp = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        replace = [3, 4, 5]
        replaceLocs = [1]
        self.back_replace('point', data, exp, replace, replaceLocs)
        # logging test
        toTest = self.constructor(data)
        toTest.points.replace([3, 4, 5], 1)

    def test_points_replace_multiple(self):
        data = [[0, 1, 2], ['x', 'x', 'x'], ['x', 'x', 'x'], [0, -1, -2]]
        exp = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, -1, -2]]
        replace = [[3, 4, 5], [6, 7, 8]]
        replaceLocs = [1, 2]
        self.back_replace('point', data, exp, replace, replaceLocs)

    def test_points_replace_all(self):
        self.back_replace_all('point')

    def test_points_replace_order(self):
        data = [['x', 'x', 'x'], ['y', 'y', 'y'], ['z', 'z', 'z']]
        repl = [[0, 1, 2], [6, 7, 8]]
        exp = [[6, 7, 8], ['y', 'y', 'y'], [0, 1, 2]]

        toTest = self.constructor(data)
        toTest.points.replace(repl, points=[2, 0])

        assert toTest == self.constructor(exp)

    ####################
    # features.replace #
    ####################

    @oneLogEntryExpected
    def test_features_replace_single(self):
        data = [[0, 'x', 2], [3, 'x', 5], [6, 'x', 8]]
        exp = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        replace = [1, 4, 7]
        replaceLocs = [1]
        self.back_replace('feature', data, exp, replace, replaceLocs)
        # logging test
        toTest = self.constructor(data)
        toTest.features.replace([1, 4, 7], 1)

    def test_features_replace_multiple(self):
        data = [[0, 'x', 'x', 3], [4, 'x', 'x', 7], [8, 'x', 'x', -1]]
        exp = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, -1]]
        replace = [[1, 2], [5, 6], [9, 0]]
        replaceLocs = [1, 2]
        self.back_replace('feature', data, exp, replace, replaceLocs)

    def test_features_replace_all(self):
        self.back_replace_all('feature')

    def test_features_replace_order(self):
        data = [['x', 'y', 'z'], ['x', 'y', 'z'], ['x', 'y', 'z']]
        repl = [[0, 2], [3, 5], [6, 8]]
        exp = [[2, 'y', 0], [5, 'y', 3], [8, 'y', 6]]

        toTest = self.constructor(data)
        toTest.features.replace(repl, features=[2, 0])

        assert toTest == self.constructor(exp)


class HighLevelAll(HighLevelDataSafe, HighLevelModifying):
    pass
