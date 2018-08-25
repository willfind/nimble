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
dropFeaturesContainingType, replaceFeatureWithBinaryFeatures,
transformFeatureToIntegers, extractPointsByCoinToss,
shufflePoints, shuffleFeatures, normalizePoints, normalizeFeatures


"""

from __future__ import absolute_import
from copy import deepcopy
from nose.tools import *
from nose.plugins.attrib import attr

import os.path
import numpy
import tempfile
import inspect

import UML
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

        ret2 = ret1.calculateForEachElement(f2)

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
    ###########################
    # dropFeaturesContainingType #
    ###########################

    def test_dropFeaturesContainingType_emptyTest(self):
        """ Test dropFeaturesContainingType() when the data is empty """
        data = []
        toTest = self.constructor(data)
        unchanged = self.constructor(data)
        ret = toTest.dropFeaturesContainingType(six.string_types) # RET CHECK
        assert toTest.isIdentical(unchanged)
        assert ret is None

    def test_dropFeaturesContainingType_intoFEmpty(self):
        """ Test dropFeaturesContainingType() when dropping all features """
        data = [[1.0], [2.0]]
        toTest = self.constructor(data)
        toTest.dropFeaturesContainingType(float)

        exp = numpy.array([[], []])
        exp = numpy.array(exp)
        exp = self.constructor(exp)

        assert toTest.isIdentical(exp)

    def test_dropFeaturesContainingType_ListOnlyTest(self):
        """ Test dropFeaturesContainingType() only on List data """
        data = [[1, 2], [3, 4]]
        toTest = self.constructor(data)
        stringData = [[5, 'six']]
        toAdd = UML.createData('List', stringData)
        if toTest.getTypeString() == 'List':
            toTest.appendPoints(toAdd)
            toTest.dropFeaturesContainingType(six.string_types)
            assert toTest.features == 1

    def test_dropFeaturesContainingType_NamePath_preservation(self):
        data = [[1.0], [2.0]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.dropFeaturesContainingType(float)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


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

    #########################
    # extractPointsByCoinToss #
    #########################

    #	@raises(ImproperActionException)
    #	def test_extractPointsByCoinToss_exceptionEmpty(self):
    #		""" Test extractPointsByCoinToss() for ImproperActionException when object is empty """
    #		data = []
    #		toTest = self.constructor(data)
    #		toTest.extractPointsByCoinToss(0.5)

    @raises(ArgumentException)
    def test_extractPointsByCoinToss_exceptionNoneProbability(self):
        """ Test extractPointsByCoinToss() for ArgumentException when extractionProbability is None """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        featureNames = ['1', '2', '3']
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.extractPointsByCoinToss(None)

    @raises(ArgumentException)
    def test_extractPointsByCoinToss_exceptionLEzero(self):
        """ Test extractPointsByCoinToss() for ArgumentException when extractionProbability is <= 0 """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        featureNames = ['1', '2', '3']
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.extractPointsByCoinToss(0)

    @raises(ArgumentException)
    def test_extractPointsByCoinToss_exceptionGEone(self):
        """ Test extractPointsByCoinToss() for ArgumentException when extractionProbability is >= 1 """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        featureNames = ['1', '2', '3']
        pointNames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        toTest.extractPointsByCoinToss(1)

    def test_extractPointsByCoinToss_intoPEmpty(self):
        """ Test extractPointsByCoinToss() when it removes all points """
        data = [[1]]
        toTest = self.constructor(data)
        retExp = self.constructor(data)
        while True:
            ret = toTest.extractPointsByCoinToss(.99)
            if ret.points == 1:
                break

        assert retExp.isIdentical(ret)

        data = [[]]
        data = numpy.array(data).T
        exp = self.constructor(data)

        assert toTest.isIdentical(exp)


    def test_extractPointsByCoinToss_handmade(self):
        """ Test extractPointsByCoinToss() produces sane results (ie a partition) """
        data = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]
        featureNames = ['a', 'b', 'c']
        pointNames = ['1', '2', '3', '4', '5', '6']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)
        orig = toTest.copy()
        ret = toTest.extractPointsByCoinToss(0.5)

        def checkEqual(v1, v2):
            assert len(v1) == len(v2)
            for i in range(len(v1)):
                assert v1[i] == v2[i]

        # everything in ret is in orig
        for pIndex in range(ret.points):
            currRetPoint = ret.pointView(pIndex)
            currName = ret.getPointName(pIndex)
            currOrigPoint = orig.pointView(currName)
            checkEqual(currRetPoint, currOrigPoint)

        # everything in toTest is in orig
        for pIndex in range(toTest.points):
            currToTestPoint = toTest.pointView(pIndex)
            currName = toTest.getPointName(pIndex)
            currOrigPoint = orig.pointView(currName)
            checkEqual(currToTestPoint, currOrigPoint)

        # everything in orig in either ret or toTest
        for pIndex in range(orig.points):
            currOrigPoint = orig.pointView(pIndex)
            currName = orig.getPointName(pIndex)
            if currName in ret.getPointNames():
                assert currName not in toTest.getPointNames()
                checkPoint = ret.pointView(currName)
            else:
                assert currName in toTest.getPointNames()
                assert currName not in ret.getPointNames()
                checkPoint = toTest.pointView(currName)

            checkEqual(checkPoint, currOrigPoint)


    def test_extractPointsByCoinToss_NamePath_preservation(self):
        data = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]
        featureNames = ['a', 'b', 'c']
        pointNames = ['1', '2', '3', '4', '5', '6']
        toTest = self.constructor(data, pointNames=pointNames, featureNames=featureNames)

        toTest._name = "testName"
        toTest._absPath = "testAbsPath"
        toTest._relPath = "testRelPath"

        ret = toTest.extractPointsByCoinToss(0.5)

        assert toTest.name == "testName"
        assert toTest.absolutePath == "testAbsPath"
        assert toTest.relativePath == 'testRelPath'

        assert ret.nameIsDefault()
        assert ret.absolutePath == 'testAbsPath'
        assert ret.relativePath == 'testRelPath'


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

    def test_handleMissingValues_remove_points(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj1 = obj0.copy()
        obj1.handleMissingValues(method='remove points')
        ret1 = self.constructor([[1, 2, 3], [7, 8, 9]])
        ret1.setFeatureNames(['a', 'b', 'c'])
        assert obj1 == ret1

        obj2 = obj0.copy()
        obj2.handleMissingValues(method='remove points', arguments='all', alsoTreatAsMissing=[11])
        ret2 = self.constructor([[1, 2, 3], [7, numpy.NaN, None], [7, 8, 9]])
        ret2.setFeatureNames(['a', 'b', 'c'])
        assert obj2 == ret2

        obj3 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj3.handleMissingValues(method='remove points', arguments='all', alsoTreatAsMissing=[1, 2, 3], markMissing=True)
        ret3 = self.constructor([[None, 11, None, True, False, True], [7, 11, None, False, False, True], [7, 8, 9, False, False, False]])
        ret3.setPointNames(['_DEFAULT_#1', '_DEFAULT_#2', '_DEFAULT_#3'])
        ret3.setFeatureNames(['a', 'b', 'c', 'a_missing', 'b_missing', 'c_missing'])
        try:
            obj3.data = obj3.data.astype(float)
        except Exception:
            pass
        assert obj3 == ret3

    def test_handleMissingValues_remove_features(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj1 = obj0.copy()
        obj1.handleMissingValues(method='remove features')
        ret1 = self.constructor([[2], [11], [11], [8]])
        ret1.setFeatureNames(['b'])
        assert obj1 == ret1

        obj2 = obj0.copy()
        obj2.handleMissingValues(method='remove features', arguments='all', alsoTreatAsMissing=[3, 9])
        ret2 = self.constructor([[1, 2], [None, 11], [7, 11], [7, 8]])
        ret2.setFeatureNames(['a', 'b'])
        assert obj2 == ret2

        obj3 = obj0.copy()
        obj3.handleMissingValues(method='remove features', arguments='all', alsoTreatAsMissing=[1, 7], markMissing=True)
        ret3 = self.constructor([[2, 3, False, False], [11, None, False, True], [11, None, False, True], [8, 9, False, False]])
        ret3.setFeatureNames(['b', 'c', 'b_missing', 'c_missing'])
        try:
            obj3.data = obj3.data.astype(float)
        except Exception:
            pass
        assert obj3 == ret3

    @raises(ArgumentException)
    def test_handleMissingValues_remove_points2(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj1 = obj0.copy()
        obj1.handleMissingValues(method='remove points', alsoTreatAsMissing=[2, 9])

    @raises(ArgumentException)
    def test_handleMissingValues_remove_features2(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj1 = obj0.copy()
        obj1.handleMissingValues(method='remove features', alsoTreatAsMissing=2)

    def test_handleMissingValues_feature_mean(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.handleMissingValues(method='feature mean')
        ret1 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        ret1.setFeatureNames(['a', 'b', 'c'])
        assert obj1 == ret1

        obj2 = obj0.copy()
        obj2.handleMissingValues(method='feature mean', alsoTreatAsMissing=[3, 7])
        ret2 = self.constructor([[1, 2, 9], [1, 11, 9], [1, 11, 9], [1, 8, 9]])
        ret2.setFeatureNames(['a', 'b', 'c'])
        assert obj2 == ret2

        obj3 = obj0.copy()
        obj3.handleMissingValues(method='feature mean', alsoTreatAsMissing=[3, 7], markMissing=True)
        ret3 = self.constructor([[1, 2, 9, False, False, True], [1, 11, 9, True, False, True], [1, 11, 9, True, False, True], [1, 8, 9, True, False, False]])
        ret3.setFeatureNames(['a', 'b', 'c', 'a_missing', 'b_missing', 'c_missing'])
        assert obj3 == ret3

    def test_handleMissingValues_feature_median(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj3 = obj0.copy()
        obj3.handleMissingValues(method='feature median', alsoTreatAsMissing=11, markMissing=True)
        ret3 = self.constructor([[1, 2, 3, False, False, False], [7, 5, 6, True, True, True], [7, 5, 6, False, True, True], [7, 8, 9, False, False, False]])
        ret3.setFeatureNames(['a', 'b', 'c', 'a_missing', 'b_missing', 'c_missing'])
        assert obj3 == ret3

    def test_handleMissingValues_feature_mode(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj3 = obj0.copy()
        obj3.handleMissingValues(method='feature mode', alsoTreatAsMissing=9, markMissing=True)
        ret3 = self.constructor([[1, 2, 3, False, False, False], [7, 11, 3, True, False, True], [7, 11, 3, False, False, True], [7, 8, 3, False, False, True]])
        ret3.setFeatureNames(['a', 'b', 'c', 'a_missing', 'b_missing', 'c_missing'])
        assert obj3 == ret3

        obj2 = self.constructor([['a','b','c'], [None, 'd', None], ['e','d',None], ['e','f','g']], featureNames=['a', 'b', 'c'])
        obj2.handleMissingValues(method='feature mode', markMissing=True, alsoTreatAsMissing=['c'])
        ret2 = self.constructor([['a','b','g', 0, 0, 1], ['e','d', 'g', 1, 0, 1], ['e','d', 'g', 0, 0, 1], ['e','f', 'g', 0, 0, 0]])
        ret2.setFeatureNames(['a', 'b', 'c', 'a_missing', 'b_missing', 'c_missing'])
        assert obj2 == ret2

    def test_handleMissingValues_zero(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj3 = obj0.copy()
        obj3.handleMissingValues(method='zero', alsoTreatAsMissing=11, markMissing=True, features=['b', 2])
        ret3 = self.constructor([[1, 2, 3, False, False], [None, 0, 0, True, True], [7, 0, 0, True, True], [7, 8, 9, False, False]])
        ret3.setFeatureNames(['a', 'b', 'c', 'b_missing', 'c_missing'])
        try:
            obj3.data = obj3.data.astype(float)
        except Exception:
            pass
        assert obj3 == ret3

    def test_handleMissingValues_constant(self):
        obj0 = self.constructor([[1, 2, 3], [None, 0, None], [7, 0, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj3 = obj0.copy()
        obj3.handleMissingValues(method='constant', alsoTreatAsMissing=[0], markMissing=True, arguments=100)
        ret3 = self.constructor([[1, 2, 3, False, False, False], [100, 100, 100, True, True, True], [7, 100, 100, False, True, True], [7, 8, 9, False, False, False]])
        ret3.setFeatureNames(['a', 'b', 'c', 'a_missing', 'b_missing', 'c_missing'])
        assert obj3 == ret3

    def test_handleMissingValues_forward_fill(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj3 = obj0.copy()
        obj3.handleMissingValues(method='forward fill', alsoTreatAsMissing=[11], markMissing=True)
        ret3 = self.constructor([[1, 2, 3, False, False, False], [1, 2, 3, True, True, True], [7, 2, 3, False, True, True], [7, 8, 9, False, False, False]])
        ret3.setFeatureNames(['a', 'b', 'c', 'a_missing', 'b_missing', 'c_missing'])
        assert obj3 == ret3

    def test_handleMissingValues_backward_fill(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj3 = obj0.copy()
        obj3.handleMissingValues(method='backward fill', alsoTreatAsMissing=[11], markMissing=True)
        ret3 = self.constructor([[1, 2, 3, False, False, False], [7, 8, 9, True, True, True], [7, 8, 9, False, True, True], [7, 8, 9, False, False, False]])
        ret3.setFeatureNames(['a', 'b', 'c', 'a_missing', 'b_missing', 'c_missing'])
        assert obj3 == ret3

    def test_handleMissingValues_interpolate(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])

        obj3 = obj0.copy()
        obj3.handleMissingValues(method='interpolate', markMissing=True)
        ret3 = self.constructor([[1, 2, 3, False, False, False], [4, 11, 5, True, False, True], [7, 11, 7, False, False, True], [7, 8, 9, False, False, False]])
        ret3.setFeatureNames(['a', 'b', 'c', 'a_missing', 'b_missing', 'c_missing'])
        assert obj3 == ret3

class HighLevelAll(HighLevelDataSafe, HighLevelModifying):
    pass
