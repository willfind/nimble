"""
Backend for unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, these
functions are to be called by unit tests of the derived class, with the appropiate
objects provided.

Methods tested in this file:

In object HighLevelDataSafe:
calculateForEachPoint, calculateForEachFeature, mapReducePoints, pointIterator,
featureIterator, calculateForEachElement, isApproximatelyEqual,
trainAndTestSets

In object HighLevelModifying:
replaceFeatureWithBinaryFeatures, transformFeatureToIntegers,
shufflePoints, shuffleFeatures,
normalizePoints, normalizeFeatures,
fillUsingPoints, fillUsingFeatures, fillUsingNeighbors

"""

from __future__ import absolute_import
from copy import deepcopy
from nose.tools import *
from nose.plugins.attrib import attr
try:
    from unittest import mock #python >=3.3
except:
    import mock

import os.path
import numpy
import tempfile
import inspect

import UML
from UML import match
from UML import fill
from UML.exceptions import ArgumentException, ImproperActionException

from UML.data.tests.baseObject import DataTestObject

from UML.randomness import numpyRandom
import six
from six.moves import range


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
    ###########################
    # calculateForEachPoint() #
    ###########################

    @raises(ArgumentException)
    def test_calculateForEachPoint_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
        origObj.calculateForEachPoint(None)

    @raises(ImproperActionException)
    def test_calculateForEachPoint_exceptionPEmpty(self):
        data = [[], []]
        data = numpy.array(data).T
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.getFeatureIndex('deci')]

        origObj.calculateForEachPoint(emitLower)

    @raises(ImproperActionException)
    def test_calculateForEachPoint_exceptionFEmpty(self):
        data = [[], []]
        data = numpy.array(data)
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.getFeatureIndex('deci')]

        origObj.calculateForEachPoint(emitLower)

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_calculateForEachPoint_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        ret = toTest.calculateForEachPoint(noChange, points=['a', 'b'])

    def test_calculateForEachPoint_Handmade(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitLower(point):
            return point[origObj.getFeatureIndex('deci')]

        lowerCounts = origObj.calculateForEachPoint(emitLower)

        expectedOut = [[0.1], [0.1], [0.1], [0.2]]
        exp = self.constructor(expectedOut, pointNames=pointNames)

        assert lowerCounts.isIdentical(exp)

    def test_calculateForEachPoint_NamePathPreservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames, name=preserveName, path=preservePair)

        def emitLower(point):
            return point[toTest.getFeatureIndex('deci')]

        ret = toTest.calculateForEachPoint(emitLower)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

        assert ret.getPointNames() == toTest.getPointNames()


    def test_calculateForEachPoint_HandmadeLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitLower(point):
            return point[origObj.getFeatureIndex('deci')]

        lowerCounts = origObj.calculateForEachPoint(emitLower, points=['three', 2])

        expectedOut = [[0.1], [0.2]]
        expPnames = ['two', 'three']
        exp = self.constructor(expectedOut, pointNames=expPnames)

        assert lowerCounts.isIdentical(exp)


    def test_calculateForEachPoint_nonZeroItAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(deepcopy(origData))

        def emitNumNZ(point):
            ret = 0
            assert len(point) == 3
            for value in point.nonZeroIterator():
                ret += 1
            return ret

        counts = origObj.calculateForEachPoint(emitNumNZ)

        expectedOut = [[3], [2], [2], [1]]
        exp = self.constructor(expectedOut)

        assert counts.isIdentical(exp)


    #############################
    # calculateForEachFeature() #
    #############################

    @raises(ImproperActionException)
    def test_calculateForEachFeature_exceptionPEmpty(self):
        data = [[], []]
        data = numpy.array(data).T
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.calculateForEachFeature(emitAllEqual)

    @raises(ImproperActionException)
    def test_calculateForEachFeature_exceptionFEmpty(self):
        data = [[], []]
        data = numpy.array(data)
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.calculateForEachFeature(emitAllEqual)

    @raises(ArgumentException)
    def test_calculateForEachFeature_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
        origObj.calculateForEachFeature(None)

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_calculateForEachFeature_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], featureNames=['a', 'b'])

        ret = toTest.calculateForEachFeature(noChange, features=['a', 'b'])

    def test_calculateForEachFeature_Handmade(self):
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

        lowerCounts = origObj.calculateForEachFeature(emitAllEqual)
        expectedOut = [[1, 0, 0]]
        exp = self.constructor(expectedOut, featureNames=featureNames)
        assert lowerCounts.isIdentical(exp)


    def test_calculateForEachFeature_NamePath_preservation(self):
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

        ret = toTest.calculateForEachFeature(emitAllEqual)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

        assert toTest.getFeatureNames() == ret.getFeatureNames()


    def test_calculateForEachFeature_HandmadeLimited(self):
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

        lowerCounts = origObj.calculateForEachFeature(emitAllEqual, features=[0, 'centi'])
        expectedOut = [[1, 0]]
        expFNames = ['number', 'centi']
        exp = self.constructor(expectedOut, featureNames=expFNames)
        assert lowerCounts.isIdentical(exp)


    def test_calculateForEachFeature_nonZeroIterAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(deepcopy(origData))

        def emitNumNZ(feature):
            ret = 0
            assert len(feature) == 4
            for value in feature.nonZeroIterator():
                ret += 1
            return ret

        counts = origObj.calculateForEachFeature(emitNumNZ)

        expectedOut = [[3, 3, 2]]
        exp = self.constructor(expectedOut)

        assert counts.isIdentical(exp)


    #####################
    # mapReducePoints() #
    #####################

    @raises(ImproperActionException)
    def test_mapReducePoints_argumentExceptionNoFeatures(self):
        """ Test mapReducePoints() for ImproperActionException when there are no features  """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.mapReducePoints(simpleMapper, simpleReducer)


    def test_mapReducePoints_emptyResultNoPoints(self):
        """ Test mapReducePoints() when given point empty data """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        ret = toTest.mapReducePoints(simpleMapper, simpleReducer)

        data = numpy.empty(shape=(0, 0))
        exp = self.constructor(data)
        assert ret.isIdentical(exp)


    @raises(ArgumentException)
    def test_mapReducePoints_argumentExceptionNoneMap(self):
        """ Test mapReducePoints() for ArgumentException when mapper is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.mapReducePoints(None, simpleReducer)

    @raises(ArgumentException)
    def test_mapReducePoints_argumentExceptionNoneReduce(self):
        """ Test mapReducePoints() for ArgumentException when reducer is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.mapReducePoints(simpleMapper, None)

    @raises(ArgumentException)
    def test_mapReducePoints_argumentExceptionUncallableMap(self):
        """ Test mapReducePoints() for ArgumentException when mapper is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.mapReducePoints("hello", simpleReducer)

    @raises(ArgumentException)
    def test_mapReducePoints_argumentExceptionUncallableReduce(self):
        """ Test mapReducePoints() for ArgumentException when reducer is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.mapReducePoints(simpleMapper, 5)


    def test_mapReducePoints_handmade(self):
        """ Test mapReducePoints() against handmade output """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.mapReducePoints(simpleMapper, simpleReducer)

        exp = self.constructor([[1, 5], [4, 11], [7, 17]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))

    def test_mapReducePoints_NamePath_preservation(self):
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames,
                                  name=preserveName, path=preservePair)

        ret = toTest.mapReducePoints(simpleMapper, simpleReducer)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath


    def test_mapReducePoints_handmadeNoneReturningReducer(self):
        """ Test mapReducePoints() against handmade output with a None returning Reducer """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.mapReducePoints(simpleMapper, oddOnlyReducer)

        exp = self.constructor([[1, 5], [7, 17]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))


    #######################
    # mapReduceFeatures() #
    #######################

    @raises(ImproperActionException)
    def test_mapReduceFeatures_argumentExceptionNoPoints(self):
        """ Test mapReduceFeatures() for ImproperActionException when there are no points  """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        toTest.mapReduceFeatures(simpleMapper, simpleReducer)


    def test_mapReduceFeatures_emptyResultNoFeatures(self):
        """ Test mapReduceFeatures() when given feature empty data """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        ret = toTest.mapReduceFeatures(simpleMapper, simpleReducer)

        data = numpy.empty(shape=(0, 0))
        exp = self.constructor(data)
        assert ret.isIdentical(exp)


    @raises(ArgumentException)
    def test_mapReduceFeatures_argumentExceptionNoneMap(self):
        """ Test mapReduceFeatures() for ArgumentException when mapper is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.mapReduceFeatures(None, simpleReducer)

    @raises(ArgumentException)
    def test_mapReduceFeatures_argumentExceptionNoneReduce(self):
        """ Test mapReduceFeatures() for ArgumentException when reducer is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.mapReduceFeatures(simpleMapper, None)

    @raises(ArgumentException)
    def test_mapReduceFeatures_argumentExceptionUncallableMap(self):
        """ Test mapReduceFeatures() for ArgumentException when mapper is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.mapReduceFeatures("hello", simpleReducer)

    @raises(ArgumentException)
    def test_mapReduceFeatures_argumentExceptionUncallableReduce(self):
        """ Test mapReduceFeatures() for ArgumentException when reducer is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.mapReduceFeatures(simpleMapper, 5)


    def test_mapReduceFeatures_handmade(self):
        """ Test mapReduceFeatures() against handmade output """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.mapReduceFeatures(simpleMapper, simpleReducer)

        exp = self.constructor([[1, 11], [2, 13], [3, 15]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))

    def test_mapReduceFeatures_NamePath_preservation(self):
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames,
                                  name=preserveName, path=preservePair)

        ret = toTest.mapReduceFeatures(simpleMapper, simpleReducer)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath


    def test_mapReduceFeatures_handmadeNoneReturningReducer(self):
        """ Test mapReduceFeatures() against handmade output with a None returning Reducer """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.mapReduceFeatures(simpleMapper, oddOnlyReducer)

        exp = self.constructor([[1, 11], [3, 15]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))


    #######################
    # pointIterator() #
    #######################

    def test_pointIterator_FemptyCorrectness(self):
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        pIter = toTest.pointIterator()

        pView = next(pIter)
        assert len(pView) == 0
        pView = next(pIter)
        assert len(pView) == 0

        try:
            next(pIter)
            assert False  # expected StopIteration from prev statement
        except StopIteration:
            pass

    def test_pointIterator_noNextPempty(self):
        """ test pointIterator() has no next value when object is point empty """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        viewIter = toTest.pointIterator()
        try:
            next(viewIter)
        except StopIteration:
            return
        assert False

    def test_pointIterator_exactValueViaFor(self):
        """ Test pointIterator() gives views that contain exactly the correct data """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        viewIter = toTest.pointIterator()

        toCheck = []
        for v in viewIter:
            toCheck.append(v)

        assert toCheck[0][0] == 1
        assert toCheck[0][1] == 2
        assert toCheck[0][2] == 3
        assert toCheck[1][0] == 4
        assert toCheck[1][1] == 5
        assert toCheck[1][2] == 6
        assert toCheck[2][0] == 7
        assert toCheck[2][1] == 8
        assert toCheck[2][2] == 9

    def test_pointIterator_allZeroVectors(self):
        """ Test pointIterator() works when there are all zero points """
        data = [[0, 0, 0], [4, 5, 6], [0, 0, 0], [7, 8, 9], [0, 0, 0], [0, 0, 0]]
        toTest = self.constructor(data)

        viewIter = toTest.pointIterator()
        toCheck = []
        for v in viewIter:
            toCheck.append(v)

        assert len(toCheck) == toTest.points

        assert toCheck[0][0] == 0
        assert toCheck[0][1] == 0
        assert toCheck[0][2] == 0

        assert toCheck[1][0] == 4
        assert toCheck[1][1] == 5
        assert toCheck[1][2] == 6

        assert toCheck[2][0] == 0
        assert toCheck[2][1] == 0
        assert toCheck[2][2] == 0

        assert toCheck[3][0] == 7
        assert toCheck[3][1] == 8
        assert toCheck[3][2] == 9

        assert toCheck[4][0] == 0
        assert toCheck[4][1] == 0
        assert toCheck[4][2] == 0

        assert toCheck[5][0] == 0
        assert toCheck[5][1] == 0
        assert toCheck[5][2] == 0


    #########################
    # featureIterator() #
    #########################

    def test_featureIterator_PemptyCorrectness(self):
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        fIter = toTest.featureIterator()

        fView = next(fIter)
        assert len(fView) == 0
        fView = next(fIter)
        assert len(fView) == 0

        try:
            next(fIter)
            assert False  # expected StopIteration from prev statement
        except StopIteration:
            pass

    def test_featureIterator_noNextFempty(self):
        """ test featureIterator() has no next value when object is feature empty """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        viewIter = toTest.featureIterator()
        try:
            next(viewIter)
        except StopIteration:
            return
        assert False


    def test_featureIterator_exactValueViaFor(self):
        """ Test featureIterator() gives views that contain exactly the correct data """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)

        viewIter = toTest.featureIterator()

        toCheck = []
        for v in viewIter:
            toCheck.append(v)

        assert toCheck[0][0] == 1
        assert toCheck[0][1] == 4
        assert toCheck[0][2] == 7
        assert toCheck[1][0] == 2
        assert toCheck[1][1] == 5
        assert toCheck[1][2] == 8
        assert toCheck[2][0] == 3
        assert toCheck[2][1] == 6
        assert toCheck[2][2] == 9


    def test_featureIterator_allZeroVectors(self):
        """ Test featureIterator() works when there are all zero points """
        data = [[0, 1, 0, 2, 0, 3, 0, 0], [0, 4, 0, 5, 0, 6, 0, 0], [0, 7, 0, 8, 0, 9, 0, 0]]
        toTest = self.constructor(data)

        viewIter = toTest.featureIterator()
        toCheck = []
        for v in viewIter:
            toCheck.append(v)

        assert len(toCheck) == toTest.features
        assert toCheck[0][0] == 0
        assert toCheck[0][1] == 0
        assert toCheck[0][2] == 0

        assert toCheck[1][0] == 1
        assert toCheck[1][1] == 4
        assert toCheck[1][2] == 7

        assert toCheck[2][0] == 0
        assert toCheck[2][1] == 0
        assert toCheck[2][2] == 0

        assert toCheck[3][0] == 2
        assert toCheck[3][1] == 5
        assert toCheck[3][2] == 8

        assert toCheck[4][0] == 0
        assert toCheck[4][1] == 0
        assert toCheck[4][2] == 0

        assert toCheck[5][0] == 3
        assert toCheck[5][1] == 6
        assert toCheck[5][2] == 9

        assert toCheck[6][0] == 0
        assert toCheck[6][1] == 0
        assert toCheck[6][2] == 0

        assert toCheck[7][0] == 0
        assert toCheck[7][1] == 0
        assert toCheck[7][2] == 0


    #############################
    # calculateForEachElement() #
    #############################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_calculateForEachElement_calls_constructIndicesList1(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], pointNames=['a', 'b'])

        def noChange(point):
            return point

        ret = toTest.calculateForEachElement(noChange, points=['a', 'b'])

    @raises(CalledFunctionException)
    @mock.patch('UML.data.base.Base._constructIndicesList', side_effect=calledException)
    def test_calculateForEachElement_calls_constructIndicesList2(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], featureNames=['a', 'b'])

        def noChange(point):
            return point

        ret = toTest.calculateForEachElement(noChange, features=['a', 'b'])

    def test_calculateForEachElement_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, name=preserveName, path=preservePair)

        ret = toTest.calculateForEachElement(passThrough)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

    def test_calculateForEachElement_passthrough(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.calculateForEachElement(passThrough)
        retRaw = ret.copyAs(format="python list")

        assert [1, 2, 3] in retRaw
        assert [4, 5, 6] in retRaw
        assert [7, 8, 9] in retRaw


    def test_calculateForEachElement_plusOnePreserve(self):
        data = [[1, 0, 3], [0, 5, 6], [7, 0, 9]]
        toTest = self.constructor(data)
        ret = toTest.calculateForEachElement(plusOne, preserveZeros=True)
        retRaw = ret.copyAs(format="python list")

        assert [2, 0, 4] in retRaw
        assert [0, 6, 7] in retRaw
        assert [8, 0, 10] in retRaw


    def test_calculateForEachElement_plusOneExclude(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.calculateForEachElement(plusOneOnlyEven, skipNoneReturnValues=True)
        retRaw = ret.copyAs(format="python list")

        assert [1, 3, 3] in retRaw
        assert [5, 5, 7] in retRaw
        assert [7, 9, 9] in retRaw


    def test_calculateForEachElement_plusOneLimited(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)

        ret = toTest.calculateForEachElement(plusOneOnlyEven, points='4', features=[1, 'three'],
                                             skipNoneReturnValues=True)
        retRaw = ret.copyAs(format="python list")

        assert [5, 7] in retRaw


    def test_calculateForEachElement_All_zero(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret1 = toTest.calculateForEachElement(lambda x: 0)
        ret2 = toTest.calculateForEachElement(lambda x: 0, preserveZeros=True)

        expData = [[0,0,0],[0,0,0],[0,0,0]]
        expObj = self.constructor(expData)
        assert ret1 == expObj
        assert ret2 == expObj

    def test_calculateForEachElement_String_conversion_manipulations(self):
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
        ret0A = toTest.calculateForEachElement(allString)
        ret0B = toTest.calculateForEachElement(allString, preserveZeros=True)

        exp0Data = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
        exp0Obj = self.constructor(exp0Data)
        assert ret0A == exp0Obj
        assert ret0B == exp0Obj

        ret1 = toTest.calculateForEachElement(f1)

        exp1Data =  [[1, 'two', 3], ['four', 5, 'six'], [7, 'eight', 9]]
        exp1Obj = self.constructor(exp1Data)

        assert ret1 == exp1Obj

        ret2 = ret.calculateForEachElement(f2)

        exp2Obj = self.constructor(data)

        assert ret2 == exp2Obj



    #############################
    # countElements() #
    #############################

    def test_countElements(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.countElements('>=5')
        assert ret == 5

        ret = toTest.countElements(lambda x: x % 2 == 1)
        assert ret == 5

    #############################
    # countPoints() #
    #############################

    def test_countPoints(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=['one', 'two', 'three'], featureNames=['a', 'b', 'c'])
        ret = toTest.countPoints('b>=5')
        assert ret == 2

        ret = toTest.countPoints(lambda x: x['b'] >= 5)
        assert ret == 2


    #############################
    # countFeatures() #
    #############################

    def test_countFeatures(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=['one', 'two', 'three'], featureNames=['a', 'b', 'c'])
        ret = toTest.countFeatures('two>=5')
        assert ret == 2

        ret = toTest.countFeatures(lambda x: x['two'] >= 5)
        assert ret == 2

    ########################
    # isApproximatelyEqual() #
    ########################

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

        assert trX.points == 2
        assert trX.features == 5
        assert teX.points == 2
        assert teX.features == 5

    # simple sucess - single label
    def test_trainAndTestSets_simple_singlelabel(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, featureNames=featureNames)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=0)

        assert trX.points == 2
        assert trX.features == 4
        assert trY.points == 2
        assert trY.features == 1
        assert teX.points == 2
        assert teX.features == 4
        assert teY.points == 2
        assert teY.features == 1

    # simple sucess - multi label
    def test_trainAndTestSets_simple_multilabel(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, featureNames=featureNames)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=[0, 'labs2'])

        assert trX.points == 2
        assert trX.features == 3
        assert trY.points == 2
        assert trY.features == 2
        assert teX.points == 2
        assert teX.features == 3
        assert teY.points == 2
        assert teY.features == 2

    # edge cases 0/1 test portions
    def test_trainAndTestSets_0or1_testFraction(self):
        data = [[1, 2, 3, 33], [2, 5, 6, 66], [3, 8, 9, 99], [4, 11, 12, 111]]
        toTest = self.constructor(data)

        trX, trY, teX, teY = toTest.trainAndTestSets(0, 0)

        assert trX.points == 4
        assert trY.points == 4
        assert teX.points == 0
        assert teY.points == 0

        trX, trY, teX, teY = toTest.trainAndTestSets(1, 0)

        assert trX.points == 0
        assert trY.points == 0
        assert teX.points == 4
        assert teY.points == 4

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
        trX.transformEachPoint(changeFirst)
        trY.transformEachPoint(changeFirst)
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

        assert trX.getPointNames() == ['one', 'two']
        assert trX.getFeatureNames() == ['fives', 'labs2', 'bozo', 'long']

        assert trY.getPointNames() == ['one', 'two']
        assert trY.getFeatureNames() == ['labs1']

        assert teX.getPointNames() == ['three', 'four']
        assert teX.getFeatureNames() == ['fives', 'labs2', 'bozo', 'long']

        assert teY.getPointNames() == ['three', 'four']
        assert teY.getFeatureNames() == ['labs1']


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


class HighLevelModifying(DataTestObject):

    #################################
    # replaceFeatureWithBinaryFeatures #
    #################################

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
        for point in getNames.pointIterator():
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


    #############################
    # transformFeatureToIntegers #
    #############################

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

        assert toTest.getPointName(0) == '1a'
        assert toTest.getPointName(1) == '2a'
        assert toTest.getPointName(2) == '3'
        assert toTest.getPointName(3) == '2b'
        assert toTest.getPointName(4) == '1b'
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


    ###################
    # shufflePoints() #
    ###################

    def test_shufflePoints_noLongerEqual(self):
        """ Tests shufflePoints() results in a changed object """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        # it is possible that it shuffles it into the same configuration.
        # the odds are vanishingly low that it will do so over consecutive calls
        # however. We will pass as long as it changes once
        returns = []
        for i in range(5):
            ret = toTest.shufflePoints() # RET CHECK
            returns.append(ret)
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert not toTest.isApproximatelyEqual(toCompare)

        for ret in returns:
            assert ret is None


    def test_shufflePoints_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        # it is possible that it shuffles it into the same configuration.
        # we only test after we're sure we've done something
        while True:
            toTest.shufflePoints()  # RET CHECK
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    #####################
    # shuffleFeatures() #
    #####################

    def test_shuffleFeatures_noLongerEqual(self):
        """ Tests shuffleFeatures() results in a changed object """
        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 1111111]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        # it is possible that it shuffles it into the same configuration.
        # the odds are vanishly low that it will do so over consecutive calls
        # however. We will pass as long as it changes once
        returns = []
        for i in range(5):
            ret = toTest.shuffleFeatures() # RET CHECK
            returns.append(ret)
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert not toTest.isApproximatelyEqual(toCompare)

        for ret in returns:
            assert ret is None


    def test_shuffleFeatures_NamePath_preservation(self):
        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 1111111]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        # it is possible that it shuffles it into the same configuration.
        # we only test after we're sure we've done something
        while True:
            toTest.shuffleFeatures()  # RET CHECK
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    #########################################
    # normalizePoints / normalizeFeatures() #
    #########################################

    def normalizeHelper(self, caller, axis, subtract=None, divide=None, also=None):
        if axis == 'point':
            func = caller.normalizePoints
        else:
            func = caller.normalizeFeatures
        if 'cython' in str(func.__func__.__class__):#if it is a cython function
            d = func.__func__.__defaults__
            assert (d is None) or (d == (None, None, None))
        else:#if it is a normal python function
            a, va, vk, d = inspect.getargspec(func)
            assert d == (None, None, None)

        if axis == 'point':
            return caller.normalizePoints(subtract=subtract, divide=divide, applyResultTo=also)
        else:
            caller.transpose()
            if also is not None:
                also.transpose()
            ret = caller.normalizeFeatures(subtract=subtract, divide=divide, applyResultTo=also)
            caller.transpose()
            if also is not None:
                also.transpose()
            return ret

    #exception different type from expected inputs
    def test_normalizePoints_exception_unexpected_input_type(self):
        self.back_normalize_exception_unexpected_input_type("point")

    def test_normalizeFeatures_exception_unexpected_input_type(self):
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
    def test_normalizePoints_exception_unexpected_string_value(self):
        self.back_normalize_exception_unexpected_string_value('point')

    def test_normalizeFeatures_exception_unexpected_string_value(self):
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
    def test_normalizePoints_exception_wrong_vector_length(self):
        self.back_normalize_exception_wrong_vector_length('point')

    def test_normalizeFeatures_exception_wrong_vector_length(self):
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
    def test_normalizePoints_exception_wrong_size_object(self):
        self.back_normalize_exception_wrong_size_object('point')

    def test_normalizeFeatures_exception_wrong_size_object(self):
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
    def test_normalizePoints_exception_applyResultTo_wrong_shape(self):
        self.back_normalize_exception_applyResultTo_wrong_shape('point')

    def test_normalizeFeatures_exception_applyResultTo_wrong_shape(self):
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
    def test_normalizePoints_exception_applyResultTo_wrong_shape_obj_input(self):
        self.back_normalize_exception_applyResultTo_wrong_shape_obj_input('point')

    def test_normalizeFeatures_exception_applyResultTo_wrong_shape_obj_input(self):
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
    def test_normalizePoints_success_float_int_inputs_NoAlso(self):
        self.back_normalize_success_float_int_inputs_NoAlso("point")

    def test_normalizeFeatures_success_float_int_inputs_NoAlso(self):
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
    def test_normalizePoints_success_float_int_inputs(self):
        self.back_normalize_success_float_int_inputs("point")

    def test_normalizeFeatures_success_float_int_inputs(self):
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
    def test_normalizePoints_success_stat_string_inputs(self):
        self.back_normalize_success_stat_string_inputs("point")

    def test_normalizeFeatures_success_stat_string_inputs(self):
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
    def test_normalizePoints_success_vector_object_inputs(self):
        self.back_normalize_success_vector_object_inputs("point")

    def test_normalizeFeatures_success_vector_object_inputs(self):
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
    def test_normalizePoints_success_full_object_inputs(self):
        self.back_normalize_success_full_object_inputs("point")

    def test_normalizeFeatures_success_full_object_inputs(self):
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
    def test_normalizeFeatures_success_statString_diffSizeAlso(self):
        self.back_normalize_success_statString_diffSizeAlso("point")

    def test_normalizePoints_success_statString_diffSizeAlso(self):
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


    #####################
    # fillUsingFeatures #
    #####################

    def test_fillUsingFeatures_mean_missing(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.fillUsingFeatures(match.missing, fill.mean)
        ret1 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        ret1.setFeatureNames(['a', 'b', 'c'])
        assert obj1 == ret1

        obj2 = obj0.copy()
        obj2.fillUsingFeatures([3, 7], None)
        obj2.fillUsingFeatures(match.missing, fill.mean)
        ret2 = self.constructor([[1, 2, 9], [1, 11, 9], [1, 11, 9], [1, 8, 9]])
        ret2.setFeatureNames(['a', 'b', 'c'])
        assert obj2 == ret2

        # obj3 = obj0.copy()
        # ret = obj3.fillUsingFeatures(match.missing, fill.mean, returnModified=True)
        # exp3 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        # exp3.setFeatureNames(['a', 'b', 'c'])
        # expRet = self.constructor([[False, False, False], [True, False, True], [False, False, True], [False, False, False]])
        # expRet.setFeatureNames(['a_modified', 'b_modified', 'c_modified'])
        # assert obj3 == exp3
        # assert ret == expRet

    def test_fillUsingFeatures_mean_nonNumeric(self):
        obj0 = self.constructor([[1, 2, 3], ['na', 11, 'na'], [7, 11, 'na'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.fillUsingFeatures(match.nonNumeric, fill.mean)
        ret1 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        ret1.setFeatureNames(['a', 'b', 'c'])
        assert obj1 == ret1

        obj2 = obj0.copy()
        obj2.fillUsingFeatures([3, 7], 'na')
        obj2.fillUsingFeatures(match.nonNumeric, fill.mean)
        ret2 = self.constructor([[1, 2, 9], [1, 11, 9], [1, 11, 9], [1, 8, 9]])
        ret2.setFeatureNames(['a', 'b', 'c'])
        assert obj2 == ret2

        # obj3 = obj0.copy()
        # ret = obj3.fillUsingFeatures(match.nonNumeric, fill.mean, features=['b',2], returnModified=True)
        # exp3 = self.constructor([[1, 2, 3], ['na', 11, 6], [7, 11, 6], [7, 8, 9]])
        # exp3.setFeatureNames(['a', 'b', 'c'])
        # expRet = self.constructor([[False, False], [False, True], [False, True], [False, False]])
        # expRet.setFeatureNames(['b_modified', 'c_modified'])
        # assert obj3 == exp3
        # assert ret == expRet

    @raises(ArgumentException)
    def test_fillUsingFeatures_mean_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.fillUsingFeatures(match.missing, fill.mean)

    def test_fillUsingFeatures_median_missing(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.fillUsingFeatures(11, None)
        obj.fillUsingFeatures(match.missing, fill.median)
        ret = self.constructor([[1, 2, 3], [7, 5, 6], [7, 5, 6], [7, 8, 9]])
        ret.setFeatureNames(['a', 'b', 'c'])
        assert obj == ret

    def test_fillUsingFeatures_median_nonNumeric(self):
        obj = self.constructor([[1, 2, 3], ['na', 11, 'na'], [7, 11, 'na'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.fillUsingFeatures(11, 'na')
        obj.fillUsingFeatures(match.nonNumeric, fill.median)
        ret = self.constructor([[1, 2, 3], [7, 5, 6], [7, 5, 6], [7, 8, 9]])
        ret.setFeatureNames(['a', 'b', 'c'])
        assert obj == ret

    @raises(ArgumentException)
    def test_fillUsingFeatures_median_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.fillUsingFeatures(match.missing, fill.median)

    def test_fillUsingFeatures_mode(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj0.fillUsingFeatures(9, None)
        obj0.fillUsingFeatures(match.missing, fill.mode)
        ret0 = self.constructor([[1, 2, 3], [7, 11, 3], [7, 11, 3], [7, 8, 3]])
        ret0.setFeatureNames(['a', 'b', 'c'])
        assert obj0 == ret0

        obj1 = self.constructor([['a','b','c'], [None, 'd', None], ['e','d','c'], ['e','f','g']], featureNames=['a', 'b', 'c'])
        obj1.fillUsingFeatures('c', None)
        obj1.fillUsingFeatures(match.missing, fill.mode)
        ret1 = self.constructor([['a','b','g'], ['e','d', 'g'], ['e','d', 'g'], ['e','f', 'g']])
        ret1.setFeatureNames(['a', 'b', 'c'])
        assert obj1 == ret1

    @raises(ArgumentException)
    def test_fillUsingFeatures_mode_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.fillUsingFeatures(match.missing, fill.mode)

    def test_fillUsingFeatures_zero(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.fillUsingFeatures(11, None)
        obj.fillUsingFeatures(match.missing, 0, features=['b', 'c'])
        ret = self.constructor([[1, 2, 3], [None, 0, 0], [7, 0, 0], [7, 8, 9]])
        ret.setFeatureNames(['a', 'b', 'c'])
        assert obj == ret

    def test_fillUsingFeatures_constant(self):
        obj = self.constructor([[1, 2, 3], [0, 0, 0], [7, 0, 0], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.fillUsingFeatures(0, 100)
        ret = self.constructor([[1, 2, 3], [100, 100, 100], [7, 100, 100], [7, 8, 9]])
        ret.setFeatureNames(['a', 'b', 'c'])
        assert obj == ret

    def test_fillUsingFeatures_forwardFill(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.fillUsingFeatures(match.missing, fill.forwardFill)
        ret = self.constructor([[1, 2, 3], [1, 11, 3], [7, 11, 3], [7, 8, 9]])
        ret.setFeatureNames(['a', 'b', 'c'])
        assert obj == ret

    @raises(ArgumentException)
    def test_fillUsingFeatures_forwardFill_firstFeatureValueMissing(self):
        obj = self.constructor([[1, None, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.fillUsingFeatures(match.missing, fill.forwardFill)

    def test_fillUsingFeatures_backwardFill(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.fillUsingFeatures(11, None)
        obj.fillUsingFeatures(match.missing, fill.backwardFill)
        ret = self.constructor([[1, 2, 3], [7, 8, 9], [7, 8, 9], [7, 8, 9]])
        ret.setFeatureNames(['a', 'b', 'c'])
        assert obj == ret

    @raises(ArgumentException)
    def test_fillUsingFeatures_backwardFill_lastFeatureValueMissing(self):
        obj = self.constructor([[1, None, 3], [None, 11, None], [7, 11, None], [7, None, 9]], featureNames=['a', 'b', 'c'])
        obj.fillUsingFeatures(match.missing, fill.backwardFill)

    def test_fillUsingFeatures_interpolate(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.fillUsingFeatures(match.missing, fill.interpolate)
        ret = self.constructor([[1, 2, 3], [4, 11, 5], [7, 11, 7], [7, 8, 9]])
        ret.setFeatureNames(['a', 'b', 'c'])
        assert obj == ret

    def test_fillUsingFeatures_custom_match(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        toTest.fillUsingFeatures(negative, 0)
        assert toTest == exp

    def test_fillUsingFeatures_custom_fill(self):
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

        toTest.fillUsingFeatures(match.negative, firstValue)
        assert toTest == exp

    def test_fillUsingFeatures_custom_fillAndMatch(self):
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

        toTest.fillUsingFeatures(negative, firstValue)
        assert toTest == exp

    def test_fillUsingPoints_fillValuesWithNaN_constant(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj1 = self.constructor(data)
        obj2 = self.constructor(data)
        obj3 = self.constructor(data)
        obj1.filllUsingFeatures(999, float('nan'))
        obj2.filllUsingFeatures(999, None)
        obj3.filllUsingFeatures(999, numpy.nan)
        obj1.filllUsingFeatures(None, 0)
        obj2.filllUsingFeatures(numpy.nan, 0)
        obj3.filllUsingFeatures(float('nan'), 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj1 == exp
        assert obj2 == obj1
        assert obj3 == obj1

    def test_fillUsingFeatures_fillValuesWithNaN_list(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.fillUsingFeatures(999, None)
        obj.fillUsingFeatures([1, numpy.nan], 0)

        exp = self.constructor([[0, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_fillUsingFeatures_fillValuesWithNaN_function(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.fillUsingFeatures(999, None)
        obj.fillUsingFeatures(match.missing, 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_fillUsingFeatures_fillNumericWithNonNumeric(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.fillUsingFeatures(999, 'na')

        exp = self.constructor([[1, 2, 'na', 4], [5, 'na', 'na', 8], [9, 10, 11, 'na']])
        assert obj == exp

    def test_fillUsingFeatures_NamePath_preservation(self):
        data = [['a'], ['b'], [1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.fillUsingFeatures(match.nonNumeric, 0)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    ###################
    # fillUsingPoints #
    ###################

    def test_fillUsingPoints_mean_missing(self):
        obj0 = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.fillUsingPoints(match.missing, fill.mean)
        ret1 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        ret1.setPointNames(['a', 'b', 'c'])
        assert obj1 == ret1

        obj2 = obj0.copy()
        obj2.fillUsingPoints([4, 8], None)
        obj2.fillUsingPoints(match.missing, fill.mean)
        ret2 = self.constructor([[1, 2, 3, 2], [6, 6, 6, 6], [9, 1, 11, 7]])
        ret2.setPointNames(['a', 'b', 'c'])
        assert obj2 == ret2

    def test_fillUsingPoints_mean_nonNumeric(self):
        obj0 = self.constructor([[1, 2, 3, 4], ['na', 6, 'na', 8], [9, 1, 11, 'na']], pointNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.fillUsingPoints(match.nonNumeric, fill.mean)
        ret1 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        ret1.setPointNames(['a', 'b', 'c'])
        assert obj1 == ret1

        obj2 = obj0.copy()
        obj2.fillUsingPoints([4, 8], 'na')
        obj2.fillUsingPoints(match.nonNumeric, fill.mean)
        ret2 = self.constructor([[1, 2, 3, 2], [6, 6, 6, 6], [9, 1, 11, 7]])
        ret2.setPointNames(['a', 'b', 'c'])
        assert obj2 == ret2

    @raises(ArgumentException)
    def test_fillUsingPoints_mean_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.fillUsingPoints(match.missing, fill.mean)

    def test_fillUsingPoints_median_missing(self):
        obj = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.fillUsingPoints(match.missing, fill.median)
        ret = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 9]])
        ret.setPointNames(['a', 'b', 'c'])
        assert obj == ret

    def test_fillUsingPoints_median_nonNumeric(self):
        obj = self.constructor([[1, 2, 3, 4], ['na', 6, 'na', 8], [9, 1, 11, 'na']], pointNames=['a', 'b', 'c'])
        obj.fillUsingPoints(11, 'na')
        obj.fillUsingPoints(match.nonNumeric, fill.median)
        ret = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 5, 5]])
        ret.setPointNames(['a', 'b', 'c'])
        assert obj == ret

    @raises(ArgumentException)
    def test_fillUsingPoints_median_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.fillUsingPoints(match.missing, fill.median)

    def test_fillUsingPoints_mode(self):
        obj0 = self.constructor([[1, 2, 3, 3], [None, 6, 8, 8], [9, 9, 11, None]], pointNames=['a', 'b', 'c'])
        obj0.fillUsingPoints(9, None)
        obj0.fillUsingPoints(match.missing, fill.mode)
        ret0 = self.constructor([[1, 2, 3, 3], [8, 6, 8, 8], [11, 11, 11, 11]], pointNames=['a', 'b', 'c'])
        ret0.setPointNames(['a', 'b', 'c'])
        assert obj0 == ret0

        obj1 = self.constructor([['a', 'b', 'c', 'c'], [None, 'f', 'h', 'h'], ['i', 'i', 'k', None]], pointNames=['a', 'b', 'c'])
        obj1.fillUsingPoints('b', None)
        obj1.fillUsingPoints(match.missing, fill.mode)
        ret1 = self.constructor([['a', 'c', 'c', 'c'], ['h', 'f', 'h', 'h'], ['i', 'i', 'k', 'i']], pointNames=['a', 'b', 'c'])
        ret1.setPointNames(['a', 'b', 'c'])
        assert obj1 == ret1

    @raises(ArgumentException)
    def test_fillUsingPoints_mode_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.fillUsingPoints(match.missing, fill.mode)

    def test_fillUsingPoints_zero(self):
        obj = self.constructor([[1, 2, None], [None, 11, 6], [7, 11, None], [7, 8, 9]], pointNames=['a', 'b', 'c', 'd'])
        obj.fillUsingPoints(11, None)
        obj.fillUsingPoints(match.missing, 0, points=['b', 'c'])
        ret = self.constructor([[1, 2, None], [0, 0, 6], [7, 0, 0], [7, 8, 9]])
        ret.setPointNames(['a', 'b', 'c', 'd'])
        assert obj == ret

    def test_fillUsingPoints_constant(self):
        obj = self.constructor([[1, 2, 3], [0, 0, 0], [7, 0, 0], [7, 8, 9]], pointNames=['a', 'b', 'c', 'd'])
        obj.fillUsingPoints(0, 100)
        ret = self.constructor([[1, 2, 3], [100, 100, 100], [7, 100, 100], [7, 8, 9]])
        ret.setPointNames(['a', 'b', 'c', 'd'])
        assert obj == ret

    def test_fillUsingPoints_forwardFill(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.fillUsingPoints(match.missing, fill.forwardFill)
        ret = self.constructor([[1, 2, 3, 4], [5, 5, 5, 8], [9, 1, 11, 11]])
        ret.setPointNames(['a', 'b', 'c'])
        assert obj == ret

    @raises(ArgumentException)
    def test_fillUsingPoints_forwardFill_firstFeatureValueMissing(self):
        obj = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.fillUsingPoints(match.missing, fill.forwardFill)

    def test_fillUsingPoints_backwardFill(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [None, 1, 11, 2]], pointNames=['a', 'b', 'c'])
        obj.fillUsingPoints(11, None)
        obj.fillUsingPoints(match.missing, fill.backwardFill)
        ret = self.constructor([[1, 2, 3, 4], [5, 8, 8, 8], [1, 1, 2, 2]])
        ret.setPointNames(['a', 'b', 'c'])
        assert obj == ret

    @raises(ArgumentException)
    def test_fillUsingPoints_backwardFill_lastFeatureValueMissing(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.fillUsingPoints(match.missing, fill.backwardFill)

    def test_fillUsingPoints_interpolate(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [None, 1, None, 5]], pointNames=['a', 'b', 'c'])
        obj.fillUsingPoints(match.missing, fill.interpolate)
        ret = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 3, 5]])
        ret.setPointNames(['a', 'b', 'c'])
        assert obj == ret

    def test_fillUsingPoints_custom_match(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        toTest.fillUsingPoints(negative, 0)
        assert toTest == exp

    def test_fillUsingPoints_custom_fill(self):
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

        toTest.fillUsingPoints(match.negative, firstValue)
        assert toTest == exp

    def test_fillUsingPoints_custom_fillAndMatch(self):
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

        toTest.fillUsingPoints(negative, firstValue)
        assert toTest == exp

    def test_fillUsingPoints_fillValuesWithNaN_constant(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj1 = self.constructor(data)
        obj2 = self.constructor(data)
        obj3 = self.constructor(data)
        obj1.fillUsingPoints(999, float('nan'))
        obj2.fillUsingPoints(999, None)
        obj3.fillUsingPoints(999, numpy.nan)
        obj1.fillUsingPoints(None, 0)
        obj2.fillUsingPoints(numpy.nan, 0)
        obj3.fillUsingPoints(float('nan'), 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj1 == exp
        assert obj2 == obj1
        assert obj3 == obj1

    def test_fillUsingPoints_fillValuesWithNaN_list(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.fillUsingPoints(999, None)
        obj.fillUsingPoints([1, numpy.nan], 0)

        exp = self.constructor([[0, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_fillUsingPoints_fillValuesWithNaN_function(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.fillUsingPoints(999, None)
        obj.fillUsingPoints(match.missing, 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_fillUsingPoints_fillNumericWithNonNumeric(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.fillUsingPoints(999, 'na')

        exp = self.constructor([[1, 2, 'na', 4], [5, 'na', 'na', 8], [9, 10, 11, 'na']])
        assert obj == exp

    def test_fillUsingPoints_NamePath_preservation(self):
        data = [['a', 'b', 1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.fillUsingPoints(match.nonNumeric, 0)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    #################
    # fillUsingNeighbors #
    #################

    def test_fillUsingNeighbors_kNeighborsRegressor_missing(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingNeighbors(match.missing, fill.kNeighborsRegressor, arguments)
        assert toTest == expTest

    # def test_fillUsingNeighbors_kNeighborsRegressor_nonNumeric(self):
    #     fNames = ['a', 'b', 'c']
    #     pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    #     data = [[1, 'na', 'x'], [1, 3, 9], [2, 1, 6], [3, 2, 3], ['na', 3, 'x']]
    #     arguments = {'n_neighbors': 3}
    #     toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
    #     expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
    #     expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
    #     toTest.fillUsingNeighbors(match.nonNumeric, fill.kNeighborsRegressor, arguments)
    #     assert toTest == expTest

    def test_fillUsingNeighbors_kNeighborsRegressor_pointsLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingNeighbors(match.missing, fill.kNeighborsRegressor, arguments, points=[2, 3, 4])
        assert toTest == expTest

    def test_fillUsingNeighbors_kNeighborsRegressor_featuresLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 2, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingNeighbors(match.missing, fill.kNeighborsRegressor, arguments, features=[1,0])
        assert toTest == expTest

    def test_fillUsingNeighbors_kNeighborsRegressor_pointsFeaturesLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingNeighbors(match.missing, fill.kNeighborsRegressor, arguments, points=0, features=2)
        assert toTest == expTest

    def test_fillUsingNeighbors_kNeighborsClassifier_missing(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingNeighbors(match.missing, fill.kNeighborsClassifier, arguments)
        assert toTest == expTest

    # def test_fillUsingNeighbors_kNeighborsClassifier_nonNumeric(self):
    #     fNames = ['a', 'b', 'c']
    #     pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
    #     data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
    #     arguments = {'n_neighbors': 3}
    #     toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
    #     expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
    #     expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
    #     toTest.fillUsingNeighbors(match.nonNumeric, fill.kNeighborsClassifier, arguments)
    #     assert toTest == expTest

    def test_fillUsingNeighbors_kNeighborsClassifier_pointsLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingNeighbors(match.missing, fill.kNeighborsClassifier, arguments, points=[2, 3, 4])
        print(toTest)
        assert toTest == expTest

    def test_fillUsingNeighbors_kNeighborsClassifier_featuresLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 3, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingNeighbors(match.missing, fill.kNeighborsClassifier, arguments, features=[1,0])
        assert toTest == expTest

    def test_fillUsingNeighbors_kNeighborsClassifier_pointsFeaturesLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        arguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingNeighbors(match.missing, fill.kNeighborsClassifier, arguments, points=0, features=2)
        assert toTest == expTest

    def test_fillUsingNeighbors_NamePath_preservation(self):
        data = [[None, None, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.fillUsingNeighbors(match.missing, fill.kNeighborsRegressor)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

class HighLevelAll(HighLevelDataSafe, HighLevelModifying):
    pass
