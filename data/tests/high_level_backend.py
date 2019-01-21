"""
Backend for unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, these
functions are to be called by unit tests of the derived class, with the appropiate
objects provided.

Methods tested in this file:

In object HighLevelDataSafe:
points.calculate, features.calculate, elements.calculate, points.count,
features.count, elements.countUnique, points.mapReduce,
isApproximatelyEqual, trainAndTestSets

In object HighLevelModifying:
replaceFeatureWithBinaryFeatures, points.shuffle, features.shuffle,
points.normalize, features.normalize, points.fill, features.fill,
fillUsingAllData
"""

from __future__ import absolute_import
from copy import deepcopy
import os.path
import tempfile
import inspect

import numpy
import six
from six.moves import range
from nose.tools import *
from nose.plugins.attrib import attr
try:
    from unittest import mock #python >=3.3
except:
    import mock

import UML
from UML import match
from UML import fill
from UML.exceptions import ArgumentException, ImproperActionException
from UML.data.tests.baseObject import DataTestObject
from UML.randomness import numpyRandom

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

class CalledFunctionException(Exception):
    def __init__(self):
        pass

def calledException(*args, **kwargs):
    raise CalledFunctionException()

def noChange(value):
    return value


class HighLevelDataSafe(DataTestObject):
    #######################
    # .points.calculate() #
    #######################

    @raises(ArgumentException)
    def test_points_calculate_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
        origObj.points.calculate(None)

    @raises(ImproperActionException)
    def test_points_calculate_exceptionPEmpty(self):
        data = [[], []]
        data = numpy.array(data).T
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        origObj.points.calculate(emitLower)

    @raises(ImproperActionException)
    def test_points_calculate_exceptionFEmpty(self):
        data = [[], []]
        data = numpy.array(data)
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        origObj.points.calculate(emitLower)

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_points_calculate_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        ret = toTest.points.calculate(noChange, points=['a', 'b'])

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

    def test_points_calculate_NamePathPreservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames, name=preserveName, path=preservePair)

        def emitLower(point):
            return point[toTest.features.getIndex('deci')]

        ret = toTest.points.calculate(emitLower)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
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

        expectedOut = [[0.1], [0.2]]
        expPnames = ['two', 'three']
        exp = self.constructor(expectedOut, pointNames=expPnames)

        assert lowerCounts.isIdentical(exp)

    def test_points_calculate_nonZeroItAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(deepcopy(origData))

        def emitNumNZ(point):
            ret = 0
            assert len(point) == 3
            for value in point.points.nonZeroIterator():
                ret += 1
            return ret

        counts = origObj.points.calculate(emitNumNZ)

        expectedOut = [[3], [2], [2], [1]]
        exp = self.constructor(expectedOut)

        assert counts.isIdentical(exp)

    ##########################
    # .features.calculate() #
    #########################

    @raises(ImproperActionException)
    def test_features_calculate_exceptionPEmpty(self):
        data = [[], []]
        data = numpy.array(data).T
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.features.calculate(emitAllEqual)

    @raises(ImproperActionException)
    def test_features_calculate_exceptionFEmpty(self):
        data = [[], []]
        data = numpy.array(data)
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.features.calculate(emitAllEqual)

    @raises(ArgumentException)
    def test_features_calculate_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
        origObj.features.calculate(None)

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_features_calculate_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], featureNames=['a', 'b'])

        ret = toTest.features.calculate(noChange, features=['a', 'b'])

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

    def test_features_calculate_NamePath_preservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(origData, pointNames=pointNames,
                                  featureNames=featureNames, name=preserveName, path=preservePair)

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

        assert ret.nameIsDefault()
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

    def test_features_calculate_nonZeroIterAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(deepcopy(origData))

        def emitNumNZ(feature):
            ret = 0
            assert len(feature) == 4
            for value in feature.features.nonZeroIterator():
                ret += 1
            return ret

        counts = origObj.features.calculate(emitNumNZ)

        expectedOut = [[3, 3, 2]]
        exp = self.constructor(expectedOut)

        assert counts.isIdentical(exp)

    #######################
    # .elements.calculate #
    #######################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_elements_calculate_calls_constructIndicesList1(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], pointNames=['a', 'b'])

        def noChange(point):
            return point

        ret = toTest.elements.calculate(noChange, points=['a', 'b'])

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_elements_calculate_calls_constructIndicesList2(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], featureNames=['a', 'b'])

        def noChange(point):
            return point

        ret = toTest.elements.calculate(noChange, features=['a', 'b'])

    def test_elements_calculate_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, name=preserveName, path=preservePair)

        ret = toTest.elements.calculate(passThrough)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

    def test_elements_calculate_passthrough(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.elements.calculate(passThrough)
        retRaw = ret.copyAs(format="python list")

        assert [1, 2, 3] in retRaw
        assert [4, 5, 6] in retRaw
        assert [7, 8, 9] in retRaw

    def test_elements_calculate_plusOnePreserve(self):
        data = [[1, 0, 3], [0, 5, 6], [7, 0, 9]]
        toTest = self.constructor(data)
        ret = toTest.elements.calculate(plusOne, preserveZeros=True)
        retRaw = ret.copyAs(format="python list")

        assert [2, 0, 4] in retRaw
        assert [0, 6, 7] in retRaw
        assert [8, 0, 10] in retRaw

    def test_elements_calculate_plusOneExclude(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.elements.calculate(plusOneOnlyEven, skipNoneReturnValues=True)
        retRaw = ret.copyAs(format="python list")

        assert [1, 3, 3] in retRaw
        assert [5, 5, 7] in retRaw
        assert [7, 9, 9] in retRaw

    def test_elements_calculate_plusOneLimited(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)

        ret = toTest.elements.calculate(plusOneOnlyEven, points='4', features=[1, 'three'],
                                             skipNoneReturnValues=True)
        retRaw = ret.copyAs(format="python list")

        assert [5, 7] in retRaw

    def test_elements_calculate_All_zero(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret1 = toTest.elements.calculate(lambda x: 0)
        ret2 = toTest.elements.calculate(lambda x: 0, preserveZeros=True)

        expData = [[0,0,0],[0,0,0],[0,0,0]]
        expObj = self.constructor(expData)
        assert ret1 == expObj
        assert ret2 == expObj

    def test_elements_calculate_String_conversion_manipulations(self):
        def allString(val):
            return str(val)

        toSMap = {2:'two', 4:'four', 6:'six', 8:'eight'}
        def f1(val):
            return toSMap[val] if val in toSMap else val

        toIMap = {'two':2, 'four':4, 'six':6, 'eight':8}
        def f2(val):
            return toIMap[val] if val in toIMap else val

        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, elementType=object)
        ret0A = toTest.elements.calculate(allString)
        ret0B = toTest.elements.calculate(allString, preserveZeros=True)

        exp0Data = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
        exp0Obj = self.constructor(exp0Data, elementType=object)
        assert ret0A == exp0Obj
        assert ret0B == exp0Obj

        ret1 = toTest.elements.calculate(f1)

        exp1Data = [[1, 'two', 3], ['four', 5, 'six'], [7, 'eight', 9]]
        exp1Obj = self.constructor(exp1Data)

        assert ret1 == exp1Obj

        ret2 = ret1.elements.calculate(f2)

        exp2Obj = self.constructor(data)

        assert ret2 == exp2Obj

    ######################
    # points.mapReduce() #
    ######################

    @raises(ImproperActionException)
    def test_points_mapReduce_argumentExceptionNoFeatures(self):
        """ Test points.mapReduce() for ImproperActionException when there are no features  """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.points.mapReduce(simpleMapper, simpleReducer)

    def test_points_mapReduce_emptyResultNoPoints(self):
        """ Test points.mapReduce() when given point empty data """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        data = numpy.empty(shape=(0, 0))
        exp = self.constructor(data)
        assert ret.isIdentical(exp)

    @raises(ArgumentException)
    def test_points_mapReduce_argumentExceptionNoneMap(self):
        """ Test points.mapReduce() for ArgumentException when mapper is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce(None, simpleReducer)

    @raises(ArgumentException)
    def test_points_mapReduce_argumentExceptionNoneReduce(self):
        """ Test points.mapReduce() for ArgumentException when reducer is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce(simpleMapper, None)

    @raises(ArgumentException)
    def test_points_mapReduce_argumentExceptionUncallableMap(self):
        """ Test points.mapReduce() for ArgumentException when mapper is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce("hello", simpleReducer)

    @raises(ArgumentException)
    def test_points_mapReduce_argumentExceptionUncallableReduce(self):
        """ Test points.mapReduce() for ArgumentException when reducer is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce(simpleMapper, 5)


    def test_points_mapReduce_handmade(self):
        """ Test points.mapReduce() against handmade output """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        exp = self.constructor([[1, 5], [4, 11], [7, 17]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))

    def test_points_mapReduce_NamePath_preservation(self):
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames,
                                  name=preserveName, path=preservePair)

        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
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


    ########################
    # features.mapReduce() #
    ########################

    @raises(ImproperActionException)
    def test_features_mapReduce_argumentExceptionNoPoints(self):
        """ Test features.mapReduce() for ImproperActionException when there are no points  """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        toTest.features.mapReduce(simpleMapper, simpleReducer)

    def test_features_mapReduce_emptyResultNoFeatures(self):
        """ Test features.mapReduce() when given feature empty data """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        ret = toTest.features.mapReduce(simpleMapper, simpleReducer)

        data = numpy.empty(shape=(0, 0))
        exp = self.constructor(data)
        assert ret.isIdentical(exp)

    @raises(ArgumentException)
    def test_features_mapReduce_argumentExceptionNoneMap(self):
        """ Test features.mapReduce() for ArgumentException when mapper is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.mapReduce(None, simpleReducer)

    @raises(ArgumentException)
    def test_features_mapReduce_argumentExceptionNoneReduce(self):
        """ Test features.mapReduce() for ArgumentException when reducer is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.mapReduce(simpleMapper, None)

    @raises(ArgumentException)
    def test_features_mapReduce_argumentExceptionUncallableMap(self):
        """ Test features.mapReduce() for ArgumentException when mapper is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.mapReduce("hello", simpleReducer)

    @raises(ArgumentException)
    def test_features_mapReduce_argumentExceptionUncallableReduce(self):
        """ Test features.mapReduce() for ArgumentException when reducer is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.mapReduce(simpleMapper, 5)

    def test_features_mapReduce_handmade(self):
        """ Test features.mapReduce() against handmade output """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.features.mapReduce(simpleMapper, simpleReducer)

        exp = self.constructor([[1, 11], [2, 13], [3, 15]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))

    def test_features_mapReduce_NamePath_preservation(self):
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames,
                                  name=preserveName, path=preservePair)

        ret = toTest.features.mapReduce(simpleMapper, simpleReducer)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

    def test_features_mapReduce_handmadeNoneReturningReducer(self):
        """ Test features.mapReduce() against handmade output with a None returning Reducer """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.features.mapReduce(simpleMapper, oddOnlyReducer)

        exp = self.constructor([[1, 11], [3, 15]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))

    ####################
    # elements.count() #
    ####################

    def test_elements_count(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.elements.count('>=5')
        assert ret == 5

        ret = toTest.elements.count(lambda x: x % 2 == 1)
        assert ret == 5

    ##################
    # points.count() #
    ##################

    def test_points_count(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=['one', 'two', 'three'], featureNames=['a', 'b', 'c'])
        ret = toTest.points.count('b>=5')
        assert ret == 2

        ret = toTest.points.count(lambda x: x['b'] >= 5)
        assert ret == 2

    ####################
    # features.count() #
    ####################

    def test_features_count(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=['one', 'two', 'three'], featureNames=['a', 'b', 'c'])
        ret = toTest.features.count('two>=5')
        assert ret == 2

        ret = toTest.features.count(lambda x: x['two'] >= 5)
        assert ret == 2

    ##########################
    # isApproximatelyEqual() #
    ##########################

    @attr('slow')
    def test_isApproximatelyEqual_randomTest(self):
        """ Test isApproximatelyEqual() using randomly generated data """

        for x in range(2):
            points = 100
            features = 40
            data = numpy.zeros((points, features))

            for i in range(points):
                for j in range(features):
                    data[i, j] = numpyRandom.rand() * numpyRandom.randint(0, 5)

            toTest = self.constructor(data)

            for retType in UML.data.available:
                currObj = UML.createData(retType, data)
                assert toTest.isApproximatelyEqual(currObj)
                assert toTest.hashCode() == currObj.hashCode()


    ######################
    # trainAndTestSets() #
    ######################

    # simple sucess - no labels
    def test_trainAndTestSets_simple_nolabels(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, featureNames=featureNames)

        trX, teX = toTest.trainAndTestSets(.5)

        assert len(trX.points) == 2
        assert len(trX.features) == 5
        assert len(teX.points) == 2
        assert len(teX.features) == 5

    # simple sucess - single label
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

    # simple sucess - multi label
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

    def test_trainAndTestSets_nameAppend_PathPreserve(self):
        data = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
        toTest = self.constructor(data, )
        tmpFile = tempfile.NamedTemporaryFile(suffix='.csv')
        toTest.writeFile(tmpFile.name, format='csv')

        toTest = self.constructor(tmpFile.name, name='toTest')

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0)

        assert trX.name == 'toTest trainX'
        assert trX.path == tmpFile.name
        assert trX.absolutePath == tmpFile.name
        assert trX.relativePath == os.path.relpath(tmpFile.name)

        assert trY.name == 'toTest trainY'
        assert trY.path == tmpFile.name
        assert trY.path == tmpFile.name
        assert trY.absolutePath == tmpFile.name
        assert trY.relativePath == os.path.relpath(tmpFile.name)

        assert teX.name == 'toTest testX'
        assert teX.path == tmpFile.name
        assert teX.path == tmpFile.name
        assert teX.absolutePath == tmpFile.name
        assert teX.relativePath == os.path.relpath(tmpFile.name)

        assert teY.name == 'toTest testY'
        assert teY.path == tmpFile.name
        assert teY.path == tmpFile.name
        assert teY.absolutePath == tmpFile.name
        assert teY.relativePath == os.path.relpath(tmpFile.name)


    def test_trainAndTestSets_PandFnamesPerserved(self):
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

    ########################
    # elements.countUnique #
    ########################

    def test_elements_countUnique_allPtsAndFtrs(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        toTest = self.constructor(data)
        unique = toTest.elements.countUnique()

        assert len(unique) == 6
        assert unique[1] == 2
        assert unique[2] == 2
        assert unique[3] == 2
        assert unique['a'] == 1
        assert unique['b'] == 1
        assert unique['c'] == 1

    def test_elements_countUnique_limitPoints(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        pNames = ['p1', 'p2', 'p3']
        toTest = self.constructor(data, pointNames=pNames)
        unique = toTest.elements.countUnique(points=0)

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[2] == 1
        assert unique[3] == 1

        unique = toTest.elements.countUnique(points='p1')

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[2] == 1
        assert unique[3] == 1

        unique = toTest.elements.countUnique(points=[0,'p3'])

        assert len(unique) == 3
        assert unique[1] == 2
        assert unique[2] == 2
        assert unique[3] == 2

    def test_elements_countUnique_limitFeatures(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        fNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(data, featureNames=fNames)
        unique = toTest.elements.countUnique(features=0)

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[3] == 1
        assert unique['a'] == 1

        unique = toTest.elements.countUnique(features='f1')

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[3] == 1
        assert unique['a'] == 1

        unique = toTest.elements.countUnique(features=[0,'f3'])

        assert len(unique) == 4
        assert unique[1] == 2
        assert unique[3] == 2
        assert unique['a'] == 1
        assert unique['c'] == 1

    def test_elements_countUnique_limitPointsAndFeatures_cornercase(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        fNames = ['f1', 'f2', 'f3']
        pNames = ['p1', 'p2', 'p3']
        toTest = self.constructor(data, featureNames=fNames, pointNames=pNames)

        unique = toTest.elements.countUnique(features=[0,'f3'], points=[0,'p3'])

        assert len(unique) == 2
        assert unique[1] == 2
        assert unique[3] == 2

class HighLevelModifying(DataTestObject):

    ####################################
    # replaceFeatureWithBinaryFeatures #
    ####################################

    @raises(ImproperActionException)
    def test_replaceFeatureWithBinaryFeatures_PemptyException(self):
        """ Test replaceFeatureWithBinaryFeatures() with a point empty object """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        toTest.replaceFeatureWithBinaryFeatures(0)

    @raises(ArgumentException)
    def test_replaceFeatureWithBinaryFeatures_FemptyException(self):
        """ Test replaceFeatureWithBinaryFeatures() with a feature empty object """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.replaceFeatureWithBinaryFeatures(0)


    def test_replaceFeatureWithBinaryFeatures_handmade(self):
        """ Test replaceFeatureWithBinaryFeatures() against handmade output """
        data = [[1], [2], [3]]
        featureNames = ['col']
        toTest = self.constructor(data, featureNames=featureNames)
        getNames = self.constructor(data, featureNames=featureNames)
        ret = toTest.replaceFeatureWithBinaryFeatures(0) # RET CHECK

        expData = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        expFeatureNames = []
        for point in getNames.points:
            expFeatureNames.append('col=' + str(point[0]))
        exp = self.constructor(expData, featureNames=expFeatureNames)

        assert toTest.isIdentical(exp)
        assert ret == expFeatureNames

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

    @raises(ImproperActionException)
    def test_transformFeatureToIntegers_PemptyException(self):
        """ Test transformFeatureToIntegers() with an point empty object """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        toTest.transformFeatureToIntegers(0)

    @raises(ArgumentException)
    def test_transformFeatureToIntegers_FemptyException(self):
        """ Test transformFeatureToIntegers() with an feature empty object """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.transformFeatureToIntegers(0)

    def test_transformFeatureToIntegers_handmade(self):
        """ Test transformFeatureToIntegers() against handmade output """
        data = [[10], [20], [30.5], [20], [10]]
        featureNames = ['col']
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.transformFeatureToIntegers(0)  # RET CHECK

        assert toTest[0, 0] == toTest[4, 0]
        assert toTest[1, 0] == toTest[3, 0]
        assert toTest[0, 0] != toTest[1, 0]
        assert toTest[0, 0] != toTest[2, 0]
        assert ret is None

    def test_transformFeatureToIntegers_pointNames(self):
        """ Test transformFeatureToIntegers preserves pointNames """
        data = [[10], [20], [30.5], [20], [10]]
        pnames = ['1a', '2a', '3', '2b', '1b']
        fnames = ['col']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = toTest.transformFeatureToIntegers(0)  # RET CHECK

        assert toTest.points.getName(0) == '1a'
        assert toTest.points.getName(1) == '2a'
        assert toTest.points.getName(2) == '3'
        assert toTest.points.getName(3) == '2b'
        assert toTest.points.getName(4) == '1b'
        assert ret is None

    def test_transformFeatureToIntegers_positioning(self):
        """ Test transformFeatureToIntegers preserves featurename mapping """
        data = [[10, 0], [20, 1], [30.5, 2], [20, 3], [10, 4]]
        pnames = ['1a', '2a', '3', '2b', '1b']
        fnames = ['col', 'pos']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = toTest.transformFeatureToIntegers(0)  # RET CHECK

        assert toTest[0, 1] == toTest[4, 1]
        assert toTest[1, 1] == toTest[3, 1]
        assert toTest[0, 1] != toTest[1, 1]
        assert toTest[0, 1] != toTest[2, 1]

        assert toTest[0, 0] == 0
        assert toTest[1, 0] == 1
        assert toTest[2, 0] == 2
        assert toTest[3, 0] == 3
        assert toTest[4, 0] == 4

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
    # points.shuffle() #
    ####################

    def testpoints_shuffle_noLongerEqual(self):
        """ Tests points.shuffle() results in a changed object """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        # it is possible that it shuffles it into the same configuration.
        # the odds are vanishingly low that it will do so over consecutive calls
        # however. We will pass as long as it changes once
        returns = []
        for i in range(5):
            ret = toTest.points.shuffle() # RET CHECK
            returns.append(ret)
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert not toTest.isApproximatelyEqual(toCompare)

        for ret in returns:
            assert ret is None


    def testpoints_shuffle_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        # it is possible that it shuffles it into the same configuration.
        # we only test after we're sure we've done something
        while True:
            toTest.points.shuffle()  # RET CHECK
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    ######################
    # features.shuffle() #
    ######################

    def test_features_shuffle_noLongerEqual(self):
        """ Tests features.shuffle() results in a changed object """
        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 1111111]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        # it is possible that it shuffles it into the same configuration.
        # the odds are vanishly low that it will do so over consecutive calls
        # however. We will pass as long as it changes once
        returns = []
        for i in range(5):
            ret = toTest.features.shuffle() # RET CHECK
            returns.append(ret)
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert not toTest.isApproximatelyEqual(toCompare)

        for ret in returns:
            assert ret is None


    def test_features_shuffle_NamePath_preservation(self):
        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 1111111]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        # it is possible that it shuffles it into the same configuration.
        # we only test after we're sure we've done something
        while True:
            toTest.features.shuffle()  # RET CHECK
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    ###########################################
    # points.normalize / features.normalize() #
    ###########################################

    def normalizeHelper(self, caller, axis, subtract=None, divide=None, also=None):
        if axis == 'point':
            func = caller.points.normalize
        else:
            func = caller.features.normalize
        if 'cython' in str(func.__func__.__class__):#if it is a cython function
            d = func.__func__.__defaults__
            assert (d is None) or (d == (None, None, None))
        else:#if it is a normal python function
            a, va, vk, d = UML.helpers.inspectArguments(func)
            assert d == (None, None, None, None)

        if axis == 'point':
            return caller.points.normalize(subtract=subtract, divide=divide, applyResultTo=also)
        else:
            caller.transpose()
            if also is not None:
                also.transpose()
            ret = caller.features.normalize(subtract=subtract, divide=divide, applyResultTo=also)
            caller.transpose()
            if also is not None:
                also.transpose()
            return ret

    #exception different type from expected inputs
    def test_points_normalize_exception_unexpected_input_type(self):
        self.back_normalize_exception_unexpected_input_type("point")

    def test_features_normalize_exception_unexpected_input_type(self):
        self.back_normalize_exception_unexpected_input_type("feature")

    def back_normalize_exception_unexpected_input_type(self, axis):
        obj = self.constructor([[1, 2], [3, 4]])

        try:
            self.normalizeHelper(obj, axis, subtract={})
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass

        try:
            self.normalizeHelper(obj, axis, divide=set([1]))
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass


    # exception non stats string
    def test_points_normalize_exception_unexpected_string_value(self):
        self.back_normalize_exception_unexpected_string_value('point')

    def test_features_normalize_exception_unexpected_string_value(self):
        self.back_normalize_exception_unexpected_string_value('feature')

    def back_normalize_exception_unexpected_string_value(self, axis):
        obj = self.constructor([[1, 2], [3, 4]])

        try:
            self.normalizeHelper(obj, axis, subtract="Hello")
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass

        try:
            self.normalizeHelper(obj, axis, divide="enumerate")
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass


    # exception wrong length vector shaped UML object
    def test_points_normalize_exception_wrong_vector_length(self):
        self.back_normalize_exception_wrong_vector_length('point')

    def test_features_normalize_exception_wrong_vector_length(self):
        self.back_normalize_exception_wrong_vector_length('feature')

    def back_normalize_exception_wrong_vector_length(self, axis):
        obj = self.constructor([[1, 2], [3, 4]])
        vectorLong = self.constructor([[1, 2, 3, 4]])
        vectorShort = self.constructor([[11]])

        try:
            self.normalizeHelper(obj, axis, subtract=vectorLong)
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass

        try:
            self.normalizeHelper(obj, axis, divide=vectorShort)
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass


    # exception wrong size of UML object
    def test_points_normalize_exception_wrong_size_object(self):
        self.back_normalize_exception_wrong_size_object('point')

    def test_features_normalize_exception_wrong_size_object(self):
        self.back_normalize_exception_wrong_size_object('feature')

    def back_normalize_exception_wrong_size_object(self, axis):
        obj = self.constructor([[1, 2, 2], [3, 4, 4], [5, 5, 5]])
        objBig = self.constructor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
        objSmall = self.constructor([[1, 1], [2, 2]])

        try:
            self.normalizeHelper(obj, axis, subtract=objBig)
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass

        try:
            self.normalizeHelper(obj, axis, divide=objSmall)
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass

    # applyResultTo is wrong shape in the normalized axis
    def test_points_normalize_exception_applyResultTo_wrong_shape(self):
        self.back_normalize_exception_applyResultTo_wrong_shape('point')

    def test_features_normalize_exception_applyResultTo_wrong_shape(self):
        self.back_normalize_exception_applyResultTo_wrong_shape('feature')

    def back_normalize_exception_applyResultTo_wrong_shape(self, axis):
        obj = self.constructor([[1, 2, 2], [3, 4, 4], [5, 5, 5]])
        alsoShort = self.constructor([[1, 2, 2], [3, 4, 4]])
        alsoLong = self.constructor([[1, 2, 2], [3, 4, 4], [5, 5, 5], [1, 23, 4]])

        try:
            self.normalizeHelper(obj, axis, subtract=1, also=alsoShort)
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass

        try:
            self.normalizeHelper(obj, axis, divide=2, also=alsoLong)
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass

    # applyResultTo is wrong shape when given obj subtract and divide
    def test_points_normalize_exception_applyResultTo_wrong_shape_obj_input(self):
        self.back_normalize_exception_applyResultTo_wrong_shape_obj_input('point')

    def test_features_normalize_exception_applyResultTo_wrong_shape_obj_input(self):
        self.back_normalize_exception_applyResultTo_wrong_shape_obj_input('feature')

    def back_normalize_exception_applyResultTo_wrong_shape_obj_input(self, axis):
        obj = self.constructor([[1, 2, 2], [3, 4, 4], [5, 5, 5]])
        alsoShort = self.constructor([[1, 2], [3, 4], [5, 5]])
        alsoLong = self.constructor([[1, 2, 2, 2], [3, 4, 4, 4], [5, 5, 5, 6]])

        sub_div = self.constructor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        try:
            self.normalizeHelper(obj, axis, subtract=sub_div, also=alsoShort)
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass

        try:
            self.normalizeHelper(obj, axis, divide=sub_div, also=alsoLong)
            assert False  # Expected ArgumentException
        except ArgumentException:
            pass

    # successful float valued inputs
    def test_points_normalize_success_float_int_inputs_NoAlso(self):
        self.back_normalize_success_float_int_inputs_NoAlso("point")

    def test_features_normalize_success_float_int_inputs_NoAlso(self):
        self.back_normalize_success_float_int_inputs_NoAlso("feature")

    def back_normalize_success_float_int_inputs_NoAlso(self, axis):
        obj = self.constructor([[1, 1, 1], [3, 3, 3], [7, 7, 7]])
        expObj = self.constructor([[0, 0, 0], [4, 4, 4], [12, 12, 12]])

        ret = self.normalizeHelper(obj, axis, subtract=1, divide=0.5)

        assert ret is None
        assert expObj == obj

        # vector versions
        obj = self.constructor([[1, 1, 1], [3, 3, 3], [7, 7, 7]])
        expObj = self.constructor([[0, 0, 0], [4, 4, 4], [12, 12, 12]])

        for retType in UML.data.available:
            currObj = obj.copy()
            sub = UML.createData(retType, [1] * 3)
            div = UML.createData(retType, [0.5] * 3)
            ret = self.normalizeHelper(currObj, axis, subtract=sub, divide=div)

            assert ret is None
            assert expObj == currObj


    # successful float valued inputs
    def test_points_normalize_success_float_int_inputs(self):
        self.back_normalize_success_float_int_inputs("point")

    def test_features_normalize_success_float_int_inputs(self):
        self.back_normalize_success_float_int_inputs("feature")

    def back_normalize_success_float_int_inputs(self, axis):
        obj = self.constructor([[1, 1, 1], [3, 3, 3], [7, 7, 7]])
        also = self.constructor([[-1, -1, -1], [.5, .5, .5], [2, 2, 2]])
        expObj = self.constructor([[0, 0, 0], [4, 4, 4], [12, 12, 12]])
        expAlso = self.constructor([[-4, -4, -4], [-1, -1, -1], [2, 2, 2]])

        ret = self.normalizeHelper(obj, axis, subtract=1, divide=0.5, also=also)

        assert ret is None
        assert expObj == obj
        assert expAlso == also


    # successful stats-string valued inputs
    def test_points_normalize_success_stat_string_inputs(self):
        self.back_normalize_success_stat_string_inputs("point")

    def test_features_normalize_success_stat_string_inputs(self):
        self.back_normalize_success_stat_string_inputs("feature")

    def back_normalize_success_stat_string_inputs(self, axis):
        obj = self.constructor([[1, 1, 1], [2, 2, 2], [-1, -1, -1]])
        also = self.constructor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        expObj = self.constructor([[0, 0, 0], [.5, .5, .5], [2, 2, 2]])
        expAlso = self.constructor([[0, 1, 2], [0, .5, 1], [0, -1, -2]])

        ret = self.normalizeHelper(obj, axis, subtract="unique count", divide="median", also=also)

        assert ret is None
        assert expObj == obj
        assert expAlso == also


    # successful vector object valued inputs
    def test_points_normalize_success_vector_object_inputs(self):
        self.back_normalize_success_vector_object_inputs("point")

    def test_features_normalize_success_vector_object_inputs(self):
        self.back_normalize_success_vector_object_inputs("feature")

    def back_normalize_success_vector_object_inputs(self, axis):
        obj = self.constructor([[1, 3, 7], [10, 30, 70], [100, 300, 700]])
        also = self.constructor([[2, 6, 14], [10, 30, 70], [100, 300, 700]])
        subVec = self.constructor([[1, 10, 100]])
        divVec = self.constructor([[.5], [5], [50]])
        expObj = self.constructor([[0, 4, 12], [0, 4, 12], [0, 4, 12]])
        expAlso = self.constructor([[2, 10, 26], [0, 4, 12], [0, 4, 12]])

        ret = self.normalizeHelper(obj, axis, subtract=subVec, divide=divVec, also=also)

        assert ret is None
        assert expObj == obj
        assert expAlso == also


    # successful matrix valued inputs
    def test_points_normalize_success_full_object_inputs(self):
        self.back_normalize_success_full_object_inputs("point")

    def test_features_normalize_success_full_object_inputs(self):
        self.back_normalize_success_full_object_inputs("feature")

    def back_normalize_success_full_object_inputs(self, axis):
        obj = self.constructor([[2, 10, 100], [3, 30, 300], [7, 70, 700]])
        also = self.constructor([[1, 5, 100], [3, 30, 300], [7, 70, 700]])

        subObj = self.constructor([[0, 5, 20], [3, 10, 60], [4, -30, 100]])
        divObj = self.constructor([[2, .5, 4], [2, 2, 4], [.25, 2, 6]])

        expObj = self.constructor([[1, 10, 20], [0, 10, 60], [12, 50, 100]])
        expAlso = self.constructor([[.5, 0, 20], [0, 10, 60], [12, 50, 100]])

        if axis == 'point':
            ret = self.normalizeHelper(obj, axis, subtract=subObj, divide=divObj, also=also)
        else:
            subObj.transpose()
            divObj.transpose()
            ret = self.normalizeHelper(obj, axis, subtract=subObj, divide=divObj, also=also)

        assert ret is None
        assert expObj == obj
        assert expAlso == also


    # string valued inputs and also values that are different in shape.
    def test_features_normalize_success_statString_diffSizeAlso(self):
        self.back_normalize_success_statString_diffSizeAlso("point")

    def test_points_normalize_success_statString_diffSizeAlso(self):
        self.back_normalize_success_statString_diffSizeAlso("feature")

    def back_normalize_success_statString_diffSizeAlso(self, axis):
        obj1 = self.constructor([[1, 1, 1], [2, 2, 2], [-1, -1, -1]])
        obj2 = self.constructor([[1, 1, 1], [2, 2, 2], [-1, -1, -1]])
        alsoLess = self.constructor([[1, 2], [1, 2], [1, 2]])
        alsoMore = self.constructor([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]])

        expObj = self.constructor([[0, 0, 0], [.5, .5, .5], [2, 2, 2]])
        expAlsoL = self.constructor([[0, 1], [0, .5], [0, -1]])
        expAlsoM = self.constructor([[0, 1, 0, 1], [0, .5, 0, .5], [0, -1, 0, -1]])

        ret1 = self.normalizeHelper(obj1, axis, subtract="unique count", divide="median", also=alsoLess)
        ret2 = self.normalizeHelper(obj2, axis, subtract="unique count", divide="median", also=alsoMore)

        assert ret1 is None
        assert ret2 is None
        assert expObj == obj1
        assert expObj == obj2
        assert expAlsoL == alsoLess
        assert expAlsoM == alsoMore

    #################
    # features.fill #
    #################

    def test_features_fill_mean_missing(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        ret = obj1.features.fill(match.missing, fill.mean) #RET CHECK
        exp1 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        exp1.features.setNames(['a', 'b', 'c'])
        assert obj1 == exp1
        assert ret is None

        obj2 = obj0.copy()
        obj2.features.fill([3, 7], None)
        obj2.features.fill(match.missing, fill.mean)
        exp2 = self.constructor([[1, 2, 9], [1, 11, 9], [1, 11, 9], [1, 8, 9]])
        exp2.features.setNames(['a', 'b', 'c'])
        assert obj2 == exp2

        obj3 = obj0.copy()
        ret = obj3.features.fill(match.missing, fill.mean, returnModified=True)
        exp3 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        exp3.features.setNames(['a', 'b', 'c'])
        expRet = self.constructor([[False, False, False], [True, False, True], [False, False, True], [False, False, False]])
        expRet.features.setNames(['a_modified', 'b_modified', 'c_modified'])
        assert obj3 == exp3
        assert ret == expRet

    def test_features_fill_mean_nonNumeric(self):
        obj0 = self.constructor([[1, 2, 3], ['na', 11, 'na'], [7, 11, 'na'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.features.fill(match.nonNumeric, fill.mean)
        exp1 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        exp1.features.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

        obj2 = obj0.copy()
        obj2.features.fill([3, 7], 'na')
        obj2.features.fill(match.nonNumeric, fill.mean)
        exp2 = self.constructor([[1, 2, 9], [1, 11, 9], [1, 11, 9], [1, 8, 9]])
        exp2.features.setNames(['a', 'b', 'c'])
        assert obj2 == exp2

        obj3 = obj0.copy()
        ret = obj3.features.fill(match.nonNumeric, fill.mean, features=['b',2], returnModified=True)
        exp3 = self.constructor([[1, 2, 3], ['na', 11, 6], [7, 11, 6], [7, 8, 9]])
        exp3.features.setNames(['a', 'b', 'c'])
        expRet = self.constructor([[False, False], [False, True], [False, True], [False, False]])
        expRet.features.setNames(['b_modified', 'c_modified'])
        assert obj3 == exp3
        assert ret == expRet

    @raises(ArgumentException)
    def test_features_fill_mean_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.features.fill(match.missing, fill.mean)

    def test_features_fill_median_missing(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(11, None)
        obj.features.fill(match.missing, fill.median)
        exp = self.constructor([[1, 2, 3], [7, 5, 6], [7, 5, 6], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fill_median_nonNumeric(self):
        obj = self.constructor([[1, 2, 3], ['na', 11, 'na'], [7, 11, 'na'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(11, 'na')
        obj.features.fill(match.nonNumeric, fill.median)
        exp = self.constructor([[1, 2, 3], [7, 5, 6], [7, 5, 6], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(ArgumentException)
    def test_features_fill_median_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.features.fill(match.missing, fill.median)

    def test_features_fill_mode(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj0.features.fill(9, None)
        obj0.features.fill(match.missing, fill.mode)
        exp0 = self.constructor([[1, 2, 3], [7, 11, 3], [7, 11, 3], [7, 8, 3]])
        exp0.features.setNames(['a', 'b', 'c'])
        assert obj0 == exp0

        obj1 = self.constructor([['a','b','c'], [None, 'd', None], ['e','d','c'], ['e','f','g']], featureNames=['a', 'b', 'c'])
        obj1.features.fill('c', None)
        obj1.features.fill(match.missing, fill.mode)
        exp1 = self.constructor([['a','b','g'], ['e','d', 'g'], ['e','d', 'g'], ['e','f', 'g']])
        exp1.features.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

    @raises(ArgumentException)
    def test_features_fill_mode_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.features.fill(match.missing, fill.mode)

    def test_features_fill_zero(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(11, None)
        obj.features.fill(match.missing, 0, features=['b', 'c'])
        exp = self.constructor([[1, 2, 3], [None, 0, 0], [7, 0, 0], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fill_constant(self):
        obj = self.constructor([[1, 2, 3], [0, 0, 0], [7, 0, 0], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(0, 100)
        exp = self.constructor([[1, 2, 3], [100, 100, 100], [7, 100, 100], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fill_forwardFill(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(match.missing, fill.forwardFill)
        exp = self.constructor([[1, 2, 3], [1, 11, 3], [7, 11, 3], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(ArgumentException)
    def test_features_fill_forwardFill_firstFeatureValueMissing(self):
        obj = self.constructor([[1, None, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(match.missing, fill.forwardFill)

    def test_features_fill_backwardFill(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(11, None)
        obj.features.fill(match.missing, fill.backwardFill)
        exp = self.constructor([[1, 2, 3], [7, 8, 9], [7, 8, 9], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(ArgumentException)
    def test_features_fill_backwardFill_lastFeatureValueMissing(self):
        obj = self.constructor([[1, None, 3], [None, 11, None], [7, 11, None], [7, None, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(match.missing, fill.backwardFill)

    def test_features_fill_interpolate(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(match.missing, fill.interpolate)
        exp = self.constructor([[1, 2, 3], [4, 11, 5], [7, 11, 7], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fill_custom_match(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        toTest.features.fill(negative, 0)
        assert toTest == exp

    def test_features_fill_custom_fill(self):
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

        toTest.features.fill(match.negative, firstValue)
        assert toTest == exp

    def test_features_fill_custom_fillAndMatch(self):
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

        toTest.features.fill(negative, firstValue)
        assert toTest == exp

    def test_features_fill_fillValuesWithNaN_constant(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj1 = self.constructor(data)
        obj2 = self.constructor(data)
        obj3 = self.constructor(data)
        obj1.features.fill(999, float('nan'))
        obj2.features.fill(999, None)
        obj3.features.fill(999, numpy.nan)
        obj1.features.fill(None, 0)
        obj2.features.fill(numpy.nan, 0)
        obj3.features.fill(float('nan'), 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj1 == exp
        assert obj2 == obj1
        assert obj3 == obj1

    def test_features_fill_fillValuesWithNaN_list(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.features.fill(999, None)
        obj.features.fill([1, numpy.nan], 0)

        exp = self.constructor([[0, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_features_fill_fillValuesWithNaN_function(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.features.fill(999, None)
        obj.features.fill(match.missing, 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_features_fill_fillNumericWithNonNumeric(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.features.fill(999, 'na')

        exp = self.constructor([[1, 2, 'na', 4], [5, 'na', 'na', 8], [9, 10, 11, 'na']])
        assert obj == exp

    def test_features_fill_NamePath_preservation(self):
        data = [['a'], ['b'], [1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.features.fill(match.nonNumeric, 0)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    ###################
    # points.fill #
    ###################

    def test_points_fill_mean_missing(self):
        obj0 = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        ret = obj1.points.fill(match.missing, fill.mean) # RET CHECK
        exp1 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        exp1.points.setNames(['a', 'b', 'c'])
        assert obj1 == exp1
        assert ret is None

        obj2 = obj0.copy()
        obj2.points.fill([4, 8], None)
        obj2.points.fill(match.missing, fill.mean)
        exp2 = self.constructor([[1, 2, 3, 2], [6, 6, 6, 6], [9, 1, 11, 7]])
        exp2.points.setNames(['a', 'b', 'c'])
        assert obj2 == exp2

        obj3 = obj0.copy()
        ret = obj3.points.fill(match.missing, fill.mean, returnModified=True)
        exp3 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        expRet = self.constructor([[False, False, False, False], [True, False, True, False], [False, False, False, True]])
        exp3.points.setNames(['a', 'b', 'c'])
        expRet.points.setNames(['a_modified', 'b_modified', 'c_modified'])
        assert obj3 == exp3
        assert ret == expRet

    def test_points_fill_mean_nonNumeric(self):
        obj0 = self.constructor([[1, 2, 3, 4], ['na', 6, 'na', 8], [9, 1, 11, 'na']], pointNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.points.fill(match.nonNumeric, fill.mean)
        exp1 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        exp1.points.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

        obj2 = obj0.copy()
        obj2.points.fill([4, 8], 'na')
        obj2.points.fill(match.nonNumeric, fill.mean)
        exp2 = self.constructor([[1, 2, 3, 2], [6, 6, 6, 6], [9, 1, 11, 7]])
        exp2.points.setNames(['a', 'b', 'c'])
        assert obj2 == exp2

        obj3 = obj0.copy()
        ret = obj3.points.fill(match.nonNumeric, fill.mean, points=['b', 2], returnModified=True)
        exp3 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        expRet = self.constructor([[True, False, True, False], [False, False, False, True]])
        exp3.points.setNames(['a', 'b', 'c'])
        expRet.points.setNames(['b_modified', 'c_modified'])
        assert obj3 == exp3
        assert ret == expRet

    @raises(ArgumentException)
    def test_points_fill_mean_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.points.fill(match.missing, fill.mean)

    def test_points_fill_median_missing(self):
        obj = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fill(match.missing, fill.median)
        exp = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 9]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_points_fill_median_nonNumeric(self):
        obj = self.constructor([[1, 2, 3, 4], ['na', 6, 'na', 8], [9, 1, 11, 'na']], pointNames=['a', 'b', 'c'])
        obj.points.fill(11, 'na')
        obj.points.fill(match.nonNumeric, fill.median)
        exp = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 5, 5]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(ArgumentException)
    def test_points_fill_median_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.points.fill(match.missing, fill.median)

    def test_points_fill_mode(self):
        obj0 = self.constructor([[1, 2, 3, 3], [None, 6, 8, 8], [9, 9, 11, None]], pointNames=['a', 'b', 'c'])
        obj0.points.fill(9, None)
        obj0.points.fill(match.missing, fill.mode)
        exp0 = self.constructor([[1, 2, 3, 3], [8, 6, 8, 8], [11, 11, 11, 11]], pointNames=['a', 'b', 'c'])
        exp0.points.setNames(['a', 'b', 'c'])
        assert obj0 == exp0

        obj1 = self.constructor([['a', 'b', 'c', 'c'], [None, 'f', 'h', 'h'], ['i', 'i', 'k', None]], pointNames=['a', 'b', 'c'])
        obj1.points.fill('b', None)
        obj1.points.fill(match.missing, fill.mode)
        exp1 = self.constructor([['a', 'c', 'c', 'c'], ['h', 'f', 'h', 'h'], ['i', 'i', 'k', 'i']], pointNames=['a', 'b', 'c'])
        exp1.points.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

    @raises(ArgumentException)
    def test_points_fill_mode_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.points.fill(match.missing, fill.mode)

    def test_points_fill_zero(self):
        obj = self.constructor([[1, 2, None], [None, 11, 6], [7, 11, None], [7, 8, 9]], pointNames=['a', 'b', 'c', 'd'])
        obj.points.fill(11, None)
        obj.points.fill(match.missing, 0, points=['b', 'c'])
        exp = self.constructor([[1, 2, None], [0, 0, 6], [7, 0, 0], [7, 8, 9]])
        exp.points.setNames(['a', 'b', 'c', 'd'])
        assert obj == exp

    def test_points_fill_constant(self):
        obj = self.constructor([[1, 2, 3], [0, 0, 0], [7, 0, 0], [7, 8, 9]], pointNames=['a', 'b', 'c', 'd'])
        obj.points.fill(0, 100)
        exp = self.constructor([[1, 2, 3], [100, 100, 100], [7, 100, 100], [7, 8, 9]])
        exp.points.setNames(['a', 'b', 'c', 'd'])
        assert obj == exp

    def test_points_fill_forwardFill(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fill(match.missing, fill.forwardFill)
        exp = self.constructor([[1, 2, 3, 4], [5, 5, 5, 8], [9, 1, 11, 11]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(ArgumentException)
    def test_points_fill_forwardFill_firstFeatureValueMissing(self):
        obj = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fill(match.missing, fill.forwardFill)

    def test_points_fill_backwardFill(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [None, 1, 11, 2]], pointNames=['a', 'b', 'c'])
        obj.points.fill(11, None)
        obj.points.fill(match.missing, fill.backwardFill)
        exp = self.constructor([[1, 2, 3, 4], [5, 8, 8, 8], [1, 1, 2, 2]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(ArgumentException)
    def test_points_fill_backwardFill_lastFeatureValueMissing(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fill(match.missing, fill.backwardFill)

    def test_points_fill_interpolate(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [None, 1, None, 5]], pointNames=['a', 'b', 'c'])
        obj.points.fill(match.missing, fill.interpolate)
        exp = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 3, 5]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_points_fill_custom_match(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        toTest.points.fill(negative, 0)
        assert toTest == exp

    def test_points_fill_custom_fill(self):
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

        toTest.points.fill(match.negative, firstValue)
        assert toTest == exp

    def test_points_fill_custom_fillAndMatch(self):
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

        toTest.points.fill(negative, firstValue)
        assert toTest == exp

    def test_points_fill_fillValuesWithNaN_constant(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj1 = self.constructor(data)
        obj2 = self.constructor(data)
        obj3 = self.constructor(data)
        obj1.points.fill(999, float('nan'))
        obj2.points.fill(999, None)
        obj3.points.fill(999, numpy.nan)
        obj1.points.fill(None, 0)
        obj2.points.fill(numpy.nan, 0)
        obj3.points.fill(float('nan'), 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj1 == exp
        assert obj2 == obj1
        assert obj3 == obj1

    def test_points_fill_fillValuesWithNaN_list(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.points.fill(999, None)
        obj.points.fill([1, numpy.nan], 0)

        exp = self.constructor([[0, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_points_fill_fillValuesWithNaN_function(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.points.fill(999, None)
        obj.points.fill(match.missing, 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_points_fill_fillNumericWithNonNumeric(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.points.fill(999, 'na')

        exp = self.constructor([[1, 2, 'na', 4], [5, 'na', 'na', 8], [9, 10, 11, 'na']])
        assert obj == exp

    def test_points_fill_NamePath_preservation(self):
        data = [['a', 'b', 1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.points.fill(match.nonNumeric, 0)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    #######################
    # fillUsingAllData #
    #######################

    def test_fillUsingAllData_kNeighborsRegressor_missing(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        ret = toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor, arguments) # RET CHECK
        assert toTest == expTest
        assert ret is None

    def test_fillUsingAllData_kNeighborsRegressor_returnModified_all(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        ret = toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor, arguments, returnModified=True)
        expRet = self.constructor([[False, True, True], [False, False, False], [False, False, False], [False, False, False], [True, False, True]])
        expRet.features.setNames([name + "_modified" for name in expRet.features.getNames()])
        assert toTest == expTest
        assert ret == expRet

    def test_fillUsingAllData_kNeighborsRegressor_nonNumeric(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, 'na', 'x'], [1, 3, 9], [2, 1, 6], [3, 2, 3], ['na', 3, 'x']]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.nonNumeric, fill.kNeighborsRegressor, arguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsRegressor_pointsLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor, arguments, points=[2, 3, 4])
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsRegressor_featuresLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 2, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor, arguments, features=[1,0])
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsRegressor_pointsFeaturesLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor, arguments, points=0, features=2)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_missing(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsClassifier, arguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_nonNumeric(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.nonNumeric, fill.kNeighborsClassifier, arguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_pointsLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsClassifier, arguments, points=[2, 3, 4])
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_featuresLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 3, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsClassifier, arguments, features=[1,0])
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_pointsFeaturesLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsClassifier, arguments, points=0, features=2)
        assert toTest == expTest

    def test_fillUsingAllData_NamePath_preservation(self):
        data = [[None, None, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

class HighLevelAll(HighLevelDataSafe, HighLevelModifying):
    pass
