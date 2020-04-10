"""
Unit tests of the low level functions defined by the base representation class.

Since these functions are stand alone in the base class, they can be tested directly.
FeatureName equality is defined using _features_implementation, which is to be defined
in the derived class, so all of these tests make a call to self.constructor instead
of directly instantiating a Base object. This function temporarily fills in
that missing implementation.

Methods tested in this file (none modify the data):
_pointNameDifference, _featureNameDifference, points._nameIntersection,
features._nameIntersection, _pointNameSymmetricDifference,
_featureNameSymmetricDifference, _pointNameUnion, _featureNameUnion,
points.setName, features.setName, points.setNames, features.setNames,
_removePointNameAndShift, _removeFeatureNameAndShift, _equalPointNames,
_equalFeatureNames, points.getNames, features.getNames, __len__,
features.getIndex, features.getName, points.getIndex, points.getName,
points.getIndices, features.getIndices, constructIndicesList, copy
features.hasName, points.hasName, __bool__
"""

import numpy
import pandas
try:
    from unittest import mock #python >=3.3
except ImportError:
    import mock

from nose.tools import *

from nimble import createData
from nimble.data import Base
from nimble.data import available
from nimble.utility import inheritDocstringsFactory, numpy2DArray
from nimble.data.dataHelpers import DEFAULT_PREFIX
from nimble.data.dataHelpers import DEFAULT_NAME_PREFIX
from nimble.data.dataHelpers import constructIndicesList
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination, ImproperObjectAction
from nimble.randomness import pythonRandom
from ..assertionHelpers import logCountAssertionFactory
from ..assertionHelpers import noLogEntryExpected, oneLogEntryExpected
from ..assertionHelpers import CalledFunctionException, calledException

###########
# helpers #
###########

class SimpleIterator(object):
    def __init__(self, *args):
        self.values = args

    def __iter__(self):
        return iter(self.values)

class GetItemOnly(object):
    def __init__(self, *args):
        self.values = args

    def __getitem__(self, i):
        return self.values[i]

class NotIterable(object):
    def __init__(self, *args):
        self.values = args

def confirmExpectedNames(toTest, axis, expected):
    if axis == 'point':
        names = toTest.pointNames
        namesInv = toTest.pointNamesInverse
    else:
        names = toTest.featureNames
        namesInv = toTest.featureNamesInverse
    if isinstance(expected, list):
        for i in range(len(expected)):
            expectedFeatureName = expected[i]
            if not expectedFeatureName.startswith(DEFAULT_PREFIX):
                actualIndex = names[expectedFeatureName]
                actualFeatureName = namesInv[i]
                assert (actualIndex == i)
                assert (actualFeatureName == expectedFeatureName)
    else:
        for k in expected.keys():
            kIndex = expected[k]
            assert k in names
            actualIndex = names[k]
            actualFeatureName = namesInv[kIndex]
            assert (actualIndex == kIndex)
            assert (actualFeatureName == k)


class LowLevelBackend(object):

    def test_objectValidationSetup(self):
        """ Test that object validation has been setup """
        assert hasattr(Base, 'objectValidation')

    ##########################
    # _pointNameDifference() #
    ##########################

    @raises(InvalidArgumentType)
    def test__pointNameDifference_exceptionOtherNone(self):
        """ Test _pointNameDifference() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameDifference(None)

    @raises(InvalidArgumentType)
    def test__pointNameDifference_exceptionWrongType(self):
        """ Test _pointNameDifference() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameDifference("wrong")

    def test__pointNameDifference_handmade(self):
        """ Test _pointNameDifference() against handmade output """
        toTest1 = self.constructor(pointNames=["one", "two", "three"])
        toTest2 = self.constructor(pointNames=["two", "four"])
        results = toTest1._pointNameDifference(toTest2)
        assert "one" in results
        assert "two" not in results
        assert "three" in results
        assert "four" not in results

    #####################
    # _featureNameDifference() #
    #####################

    @raises(InvalidArgumentType)
    def test_featureNameDifference_exceptionOtherNone(self):
        """ Test _featureNameDifference() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameDifference(None)

    @raises(InvalidArgumentType)
    def test_featureNameDifference_exceptionWrongType(self):
        """ Test _featureNameDifference() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameDifference("wrong")

    def test_featureNameDifference_handmade(self):
        """ Test _featureNameDifference() against handmade output """
        toTest1 = self.constructor(featureNames=["one", "two", "three"])
        toTest2 = self.constructor(featureNames=["two", "four"])
        results = toTest1._featureNameDifference(toTest2)
        assert "one" in results
        assert "two" not in results
        assert "three" in results
        assert "four" not in results


    ##############################
    # points._nameIntersection() #
    ##############################

    @raises(InvalidArgumentType)
    def test_points_nameIntersection_exceptionOtherNone(self):
        """ Test points._nameIntersection() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest.points._nameIntersection(None)

    @raises(InvalidArgumentType)
    def test_points_nameIntersection_exceptionWrongType(self):
        """ Test points._nameIntersection() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(pointNames=["hello"])
        toTest.points._nameIntersection("wrong")

    def test_points_nameIntersection_handmade(self):
        """ Test points._nameIntersection() against handmade output """
        toTest1 = self.constructor(pointNames=["one", "two", "three"])
        toTest2 = self.constructor(pointNames=["two", "four"])
        results = toTest1.points._nameIntersection(toTest2)
        assert "one" not in results
        assert "two" in results
        assert "three" not in results
        assert "four" not in results

    ################################
    # features._nameIntersection() #
    ################################

    @raises(InvalidArgumentType)
    def test_features_nameIntersection_exceptionOtherNone(self):
        """ Test features._nameIntersection() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest.features._nameIntersection(None)

    @raises(InvalidArgumentType)
    def test_features_nameIntersection_exceptionWrongType(self):
        """ Test features._nameIntersection() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(featureNames=["hello"])
        toTest.features._nameIntersection("wrong")

    def test_features_nameIntersection_handmade(self):
        """ Test features._nameIntersection() against handmade output """
        toTest1 = self.constructor(featureNames=["one", "two", "three"])
        toTest2 = self.constructor(featureNames=["two", "four"])
        results = toTest1.features._nameIntersection(toTest2)
        assert "one" not in results
        assert "two" in results
        assert "three" not in results
        assert "four" not in results

    ##############################
    # _pointNameSymmetricDifference() #
    ##############################

    @raises(InvalidArgumentType)
    def test__pointNameSymmetricDifference_exceptionOtherNone(self):
        """ Test _pointNameSymmetricDifference() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameSymmetricDifference(None)

    @raises(InvalidArgumentType)
    def test__pointNameSymmetricDifference_exceptionWrongType(self):
        """ Test _pointNameSymmetricDifference() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameSymmetricDifference("wrong")

    def test__pointNameSymmetricDifference_handmade(self):
        """ Test _pointNameSymmetricDifference() against handmade output """
        toTest1 = self.constructor(pointNames=["one", "two", "three"])
        toTest2 = self.constructor(pointNames=["two", "four"])
        results = toTest1._pointNameSymmetricDifference(toTest2)
        assert "one" in results
        assert "two" not in results
        assert "three" in results
        assert "four" in results

    ##############################
    # _featureNameSymmetricDifference() #
    ##############################

    @raises(InvalidArgumentType)
    def test_featureNameSymmetricDifference_exceptionOtherNone(self):
        """ Test _featureNameSymmetricDifference() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameSymmetricDifference(None)

    @raises(InvalidArgumentType)
    def test_featureNameSymmetricDifference_exceptionWrongType(self):
        """ Test _featureNameSymmetricDifference() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameSymmetricDifference("wrong")

    def test_featureNameSymmetricDifference_handmade(self):
        """ Test _featureNameSymmetricDifference() against handmade output """
        toTest1 = self.constructor(featureNames=["one", "two", "three"])
        toTest2 = self.constructor(featureNames=["two", "four"])
        results = toTest1._featureNameSymmetricDifference(toTest2)
        assert "one" in results
        assert "two" not in results
        assert "three" in results
        assert "four" in results

    ################
    # _pointNameUnion() #
    ################

    @raises(InvalidArgumentType)
    def test__pointNameUnion_exceptionOtherNone(self):
        """ Test _pointNameUnion() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameUnion(None)

    @raises(InvalidArgumentType)
    def test__pointNameUnion_exceptionWrongType(self):
        """ Test _pointNameUnion() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameUnion("wrong")

    def test__pointNameUnion_handmade(self):
        """ Test _pointNameUnion() against handmade output """
        toTest1 = self.constructor(pointNames=["one", "two", "three"])
        toTest2 = self.constructor(pointNames=["two", "four"])
        results = toTest1._pointNameUnion(toTest2)
        assert "one" in results
        assert "two" in results
        assert "three" in results
        assert "four" in results

    ################
    # _featureNameUnion() #
    ################

    @raises(InvalidArgumentType)
    def test_featureNameUnion_exceptionOtherNone(self):
        """ Test _featureNameUnion() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameUnion(None)

    @raises(InvalidArgumentType)
    def test_featureNameUnion_exceptionWrongType(self):
        """ Test _featureNameUnion() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameUnion("wrong")

    def test_featureNameUnion_handmade(self):
        """ Test _featureNameUnion() against handmade output """
        toTest1 = self.constructor(featureNames=["one", "two", "three"])
        toTest2 = self.constructor(featureNames=["two", "four"])
        results = toTest1._featureNameUnion(toTest2)
        assert "one" in results
        assert "two" in results
        assert "three" in results
        assert "four" in results

    ####################
    # points.setName() #
    ####################

    @raises(InvalidArgumentType)
    def test_points_setName_exceptionPrevWrongType(self):
        """ Test points.setName() for InvalidArgumentType when given the wrong type for prev"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setName(oldIdentifier=0.3, newName="New!")

    @raises(IndexError)
    def test_points_setName_exceptionPrevInvalidIndex(self):
        """ Test points.setName() for InvalidArgumentValue when given an invalid prev index"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setName(oldIdentifier=12, newName="New!")

    @raises(KeyError)
    def test_points_setName_exceptionPrevNotFound(self):
        """ Test points.setName() for InvalidArgumentValue when the prev pointName is not found"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setName(oldIdentifier="Previous!", newName="New!")

    @raises(InvalidArgumentType)
    def test_points_setName_exceptionNewInvalidType(self):
        """ Test points.setName() for InvalidArgumentValue when the new pointName is not a string"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setName(oldIdentifier="three", newName=4)

    @raises(InvalidArgumentValue)
    def test_points_setName_exceptionNonUnique(self):
        """ Test points.setName() for InvalidArgumentValue when a duplicate pointName is given"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setName(oldIdentifier="three", newName="two")

    @raises(ImproperObjectAction)
    def test_points_setName_exceptionNoPoints(self):
        toTest = self.constructor()
        toTest.points.setName("hello", "2")

    @logCountAssertionFactory(2)
    def test_points_setName_handmade_viaIndex(self):
        """ Test points.setName() against handmade input when specifying the pointName by index """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setName(0, "ZERO")
        toTest.points.setName(3, "3")
        expectedNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'point', expectedNames)

    def test_points_setName_handmade_viaPointName(self):
        """ Test points.setName() against handmade input when specifying the pointName by name """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setName("zero", "ZERO")
        toTest.points.setName("three", "3")
        expectedNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'point', expectedNames)

    def test_points_setName_NoneOutput(self):
        """ Test points.setName() return None as output """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        ret = toTest.points.setName("zero", "ZERO")
        assert ret is None


    ######################
    # features.setName() #
    ######################

    @raises(InvalidArgumentType)
    def test_features_setName_exceptionPrevWrongType(self):
        """ Test features.setName() for InvalidArgumentType when given the wrong type for prev"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setName(oldIdentifier=0.3, newName="New!")

    @raises(IndexError)
    def test_features_setName_exceptionPrevInvalidIndex(self):
        """ Test features.setName() for InvalidArgumentValue when given an invalid prev index"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setName(oldIdentifier=12, newName="New!")

    @raises(KeyError)
    def test_features_setName_exceptionPrevNotFound(self):
        """ Test features.setName() for InvalidArgumentValue when the prev featureName is not found"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setName(oldIdentifier="Previous!", newName="New!")

    @raises(InvalidArgumentType)
    def test_features_setName_exceptionNewInvalidType(self):
        """ Test features.setName() for InvalidArgumentValue when the new featureName is not a string"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setName(oldIdentifier="three", newName=4)

    @raises(InvalidArgumentValue)
    def test_features_setName_exceptionNonUnique(self):
        """ Test features.setName() for InvalidArgumentValue when a duplicate featureName is given"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setName(oldIdentifier="three", newName="two")

    @raises(ImproperObjectAction)
    def test_features_setName_exceptionNoFeatures(self):
        toTest = self.constructor()
        toTest.features.setName("hello", "2")

    @logCountAssertionFactory(2)
    def test_features_setName_handmade_viaIndex(self):
        """ Test features.setName() against handmade input when specifying the featureName by index """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setName(0, "ZERO")
        toTest.features.setName(3, "3")
        expectedFeatureNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'feature', expectedFeatureNames)

    def test_features_setName_handmade_viaFeatureName(self):
        """ Test features.setName() against handmade input when specifying the featureName by name """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setName("zero", "ZERO")
        toTest.features.setName("three", "3")
        expectedFeatureNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'feature', expectedFeatureNames)

    def test_features_setName_NoneOutput(self):
        """ Test features.setName() returns None as output """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        ret = toTest.features.setName("zero", "ZERO")
        assert ret is None

    #####################
    # points.setNames() #
    #####################

    @raises(InvalidArgumentValue)
    def test_points_setNames_exceptionWrongTypeObject(self):
        """ Test points.setNames() for InvalidArgumentValue a point in point Names is an unexpected type """
        # a list is the expected argument type, but the value is the incorrect type
        toTest = self.constructor(pointNames=['one'])
        toTest.points.setNames(12)

    @raises(InvalidArgumentValue)
    def test_points_setNames_exceptionNonStringPointNameInList(self):
        """ Test points.setNames() for InvalidArgumentValue when a list element is not a string """
        toTest = self.constructor(pointNames=['one', 'two', 'three'])
        nonStringName = ['one', 'two', 3]
        toTest.points.setNames(nonStringName)

    @raises(InvalidArgumentValue)
    def test_points_setNames_exceptionNonUniqueStringInList(self):
        """ Test points.setNames() for InvalidArgumentValue when a list element is not unique """
        toTest = self.constructor(pointNames=['one', 'two', 'three', 'four'])
        nonUnique = ['1', '2', '3', '1']
        toTest.points.setNames(nonUnique)

    @raises(InvalidArgumentValue)
    def test_points_setNames_exceptionNoPointsList(self):
        """ Test points.setNames() for ImproperObjectAction when there are no points to name """
        toTest = self.constructor()
        toAssign = ["hey", "gone", "none", "blank"]
        toTest.points.setNames(toAssign)

    @raises(CalledFunctionException)
    def test_points_setNames_calls_valuesToPythonList(self):
        toTest = self.constructor(pointNames=['one', 'two', 'three'])
        # need to use mock.patch as context manager after object creation
        # because Base.__init__ also calls valuesToPythonList
        with mock.patch('nimble.data.axis.valuesToPythonList', calledException):
            toTest.points.setNames(['a', 'b', 'c'])

    def test_points_setNames_emptyDataAndList(self):
        """ Test points.setNames() when both the data and the list are empty """
        toTest = self.constructor()
        toAssign = []
        toTest.points.setNames(toAssign)
        assert toTest.points.getNames() == []

    def test_points_setNames_addDefault(self):
        """ Test points.setNames() when given a default pointName """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        newNames = ["zero", "one", "two", DEFAULT_PREFIX + "17"]
        toTest.points.setNames(newNames)
        assert toTest._nextDefaultValuePoint > 17

    @oneLogEntryExpected
    def test_points_setNames_handmadeList(self):
        """ Test points.setNames() against handmade output """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        origNames = ["zero", "one", "two", "three"]
        toTest.points.setNames(origNames)
        confirmExpectedNames(toTest, 'point', origNames)

    def test_points_setNames_handmadeReplacingWithSameList(self):
        """ Test points.setNames() against handmade output when you're replacing the position of poitnNames """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        toAssign = ["hey", "gone", "none", "blank"]
        ret = toTest.points.setNames(toAssign)  # ret Check
        confirmExpectedNames(toTest, 'point', toAssign)
        assert ret is None

    @raises(InvalidArgumentValue)
    def test_points_setNames_exceptionNonStringPointNameInDict(self):
        """ Test points.setNames() for InvalidArgumentValue when a dict key is not a string """
        toTest = self.constructor(pointNames=['one'])
        nonStringNames = {1: 1}
        toTest.points.setNames(nonStringNames)

    @raises(InvalidArgumentValue)
    def test_points_setNames_exceptionNonIntIndexInDict(self):
        """ Test points.setNames() for InvalidArgumentValue when a dict value is not an int """
        toTest = self.constructor(pointNames=['one'])
        nonIntIndex = {"one": "one"}
        toTest.points.setNames(nonIntIndex)

    @raises(InvalidArgumentValue)
    def test_points_setNames_exceptionNoPointsDict(self):
        """ Test points.setNames() for ImproperObjectAction when there are no points to name """
        toTest = self.constructor(pointNames=[])
        toAssign = {"hey": 0, "gone": 1, "none": 2, "blank": 3}
        toTest.points.setNames(toAssign)

    def test_points_setNames_emptyDataAndDict(self):
        """ Test points.setNames() when both the data and the dict are empty """
        toTest = self.constructor()
        toAssign = {}
        toTest.points.setNames(toAssign)
        assert toTest.points.getNames() == []

    @oneLogEntryExpected
    def test_points_setNames_handmadeDict(self):
        """ Test points.setNames() against handmade output """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        origNames = {"zero": 0, "one": 1, "two": 2, "three": 3}
        toTest.points.setNames(origNames)
        confirmExpectedNames(toTest, 'point', origNames)

    def test_points_setNames_handmadeReplacingWithSameDict(self):
        """ Test points.setNames() against handmade output when you're replacing the position of pointNames """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        toAssign = {"hey": 0, "gone": 1, "none": 2, "blank": 3}
        ret = toTest.points.setNames(toAssign)
        confirmExpectedNames(toTest, 'point', toAssign)
        assert ret is None

    def test_points_setNames_list_mixedSpecifiedUnspecified_defaults(self):
        toTest = self.constructor(pointNames=([None] * 4))

        nextNum = toTest._nextDefaultValuePoint

        toAssign = [None] * 4
        toAssign[0] = DEFAULT_PREFIX + str(nextNum)
        toAssign[2] = DEFAULT_PREFIX + str(nextNum - 1)

        ret = toTest.points.setNames(toAssign)

        assert toTest.points.getName(0) == DEFAULT_PREFIX + str(nextNum)
        assert toTest.points.getName(1) == DEFAULT_PREFIX + str(nextNum + 1)
        assert toTest.points.getName(2) == DEFAULT_PREFIX + str(nextNum - 1)
        assert toTest.points.getName(3).startswith(DEFAULT_PREFIX)
        assert ret is None


    #######################
    # features.setNames() #
    #######################

    @raises(InvalidArgumentValue)
    def test_features_setNames_exceptionWrongTypeObject(self):
        """ Test features.setNames() for InvalidArgumentType when a feature in featureNames is an unexpected type """
        # a list is the expected argument type, but the value is the incorrect type
        toTest = self.constructor(featureNames=['one'])
        toTest.features.setNames(12)

    @raises(InvalidArgumentValue)
    def test_features_setNames_exceptionNonStringFeatureNameInDict(self):
        """ Test features.setNames() for InvalidArgumentValue when a dict key is not a string """
        toTest = self.constructor(featureNames=['one'])
        nonStringFeatureNames = {1: 1}
        toTest.features.setNames(nonStringFeatureNames)

    @raises(InvalidArgumentValue)
    def test_features_setNames_exceptionNonIntIndexInDict(self):
        """ Test features.setNames() for InvalidArgumentValue when a dict value is not an int """
        toTest = self.constructor(featureNames=['one'])
        nonIntIndex = {"one": "one"}
        toTest.features.setNames(nonIntIndex)

    @raises(InvalidArgumentValue)
    def test_features_setNames_exceptionNoFeaturesDict(self):
        """ Test features.setNames() for ImproperObjectAction when there are no features to name """
        toTest = self.constructor(featureNames=[])
        toAssign = {"hey": 0, "gone": 1, "none": 2, "blank": 3}
        toTest.features.setNames(toAssign)

    @raises(CalledFunctionException)
    def test_features_setNames_calls_valuesToPythonList(self):
        toTest = self.constructor(featureNames=['one', 'two', 'three'])
        # need to use mock.patch as context manager after object creation
        # because Base.__init__ also calls valuesToPythonList
        with mock.patch('nimble.data.axis.valuesToPythonList', calledException):
            toTest.features.setNames(['a', 'b', 'c'])

    def test_features_setNames_emptyDataAndDict(self):
        """ Test features.setNames() when both the data and the dict are empty """
        toTest = self.constructor()
        toAssign = {}
        toTest.features.setNames(toAssign)
        assert toTest.features.getNames() == []

    @oneLogEntryExpected
    def test_features_setNames_handmadeDict(self):
        """ Test features.setNames() against handmade output """
        toTest = self.constructor(featureNames=["blank", "none", "gone", "hey"])
        origFeatureNames = {"zero": 0, "one": 1, "two": 2, "three": 3}
        toTest.features.setNames(origFeatureNames)
        confirmExpectedNames(toTest, 'feature', origFeatureNames)

    def test_features_setNames_handmadeReplacingWithSameDict(self):
        """ Test features.setNames() against handmade output when you're replacing the position of featureNames """
        toTest = self.constructor(featureNames=["blank", "none", "gone", "hey"])
        toAssign = {"hey": 0, "gone": 1, "none": 2, "blank": 3}
        ret = toTest.features.setNames(toAssign)
        confirmExpectedNames(toTest, 'feature', toAssign)
        assert ret is None

    @raises(InvalidArgumentValue)
    def test_features_setNames_exceptionNonStringFeatureNameInList(self):
        """ Test features.setNames() for InvalidArgumentValue when a list element is not a string """
        toTest = self.constructor(featureNames=['one', 'two', 'three'])
        nonStringFeatureNames = ['one', 'two', 3]
        toTest.features.setNames(nonStringFeatureNames)

    @raises(InvalidArgumentValue)
    def test_features_setNames_exceptionNonUniqueStringInList(self):
        """ Test features.setNames() for InvalidArgumentValue when a list element is not unique """
        toTest = self.constructor(featureNames=['one', 'two', 'three', 'four'])
        nonUnique = ['1', '2', '3', '1']
        toTest.features.setNames(nonUnique)

    @raises(InvalidArgumentValue)
    def test_features_setNames_exceptionNoFeaturesList(self):
        """ Test features.setNames() for ImproperObjectAction when there are no features to name """
        toTest = self.constructor()
        toAssign = ["hey", "gone", "none", "blank"]
        toTest.features.setNames(toAssign)

    def test_features_setNames_emptyDataAndList(self):
        """ Test features.setNames() when both the data and the list are empty """
        toTest = self.constructor()
        toAssign = []
        toTest.features.setNames(toAssign)
        assert toTest.features.getNames() == []

    def test_features_setNames_addDefault(self):
        """ Test features.setNames() when given a default featureName """
        toTest = self.constructor(featureNames=["blank", "none", "gone", "hey"])
        newFeatureNames = ["zero", "one", "two", DEFAULT_PREFIX + "17"]
        toTest.features.setNames(newFeatureNames)
        assert toTest._nextDefaultValueFeature > 17

    @oneLogEntryExpected
    def test_features_setNames_handmadeList(self):
        """ Test features.setNames() against handmade output """
        toTest = self.constructor(featureNames=["blank", "none", "gone", "hey"])
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest.features.setNames(origFeatureNames)
        confirmExpectedNames(toTest, 'feature', origFeatureNames)

    def test_features_setNames_handmadeReplacingWithSameList(self):
        """ Test features.setNames() against handmade output when you're replacing the position of featureNames """
        toTest = self.constructor(featureNames=["blank", "none", "gone", "hey"])
        toAssign = ["hey", "gone", "none", "blank"]
        ret = toTest.features.setNames(toAssign)  # RET CHECK
        confirmExpectedNames(toTest, 'feature', toAssign)
        assert ret is None

    def test_features_setNames_list_mixedSpecifiedUnspecified_defaults(self):
        toTest = self.constructor(featureNames=([None] * 4))

        nextNum = toTest._nextDefaultValueFeature

        toAssign = [None] * 4
        toAssign[0] = DEFAULT_PREFIX + str(nextNum)
        toAssign[2] = DEFAULT_PREFIX + str(nextNum - 1)

        ret = toTest.features.setNames(toAssign)

        assert toTest.features.getName(0) == DEFAULT_PREFIX + str(nextNum)
        assert toTest.features.getName(1) == DEFAULT_PREFIX + str(nextNum + 1)
        assert toTest.features.getName(2) == DEFAULT_PREFIX + str(nextNum - 1)
        assert toTest.features.getName(3).startswith(DEFAULT_PREFIX)
        assert ret is None

    ##################################################################
    # points._adjustCountAndNames() / features._adjustCountAndNames()#
    ##################################################################

    def test_adjustCountAndNames_pointCountAndNames(self):
        origNames = ["zero", "one", "two", "three"]
        orig = self.constructor(pointNames=origNames)
        other = self.constructor(pointNames=["one", "two"])
        expNames = ["zero", "three"]
        orig.points._adjustCountAndNames(other)

        assert len(orig.points) == 2
        assert orig.points.getNames() == expNames

    def test_adjustCountAndNames_featureCountAndNames(self):
        origNames = ["zero", "one", "two", "three"]
        orig = self.constructor(featureNames=origNames)
        other = self.constructor(featureNames=["one", "two"])
        expNames = ["zero", "three"]
        orig.features._adjustCountAndNames(other)

        assert len(orig.features) == 2
        assert orig.features.getNames() == expNames

    ######################
    # _equalPointNames() #
    ######################

    def test__equalPointNames_False(self):
        """ Test _equalPointNames() against some non-equal input """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        assert not toTest._equalPointNames(self.constructor(pointNames=["short", "list"]))

        subset = self.constructor(pointNames=["zero", "one", "two"])
        superset = self.constructor(pointNames=["zero", "one", "two", "three", "four"])
        assert not toTest._equalPointNames(subset)
        assert not toTest._equalPointNames(superset)
        assert not subset._equalPointNames(toTest)
        assert not superset._equalPointNames(toTest)

    def test__equalPointNames_actuallyEqual(self):
        """ Test _equalPointNames() against some actually equal input """
        origNames = ["zero", "one", "two", "three"]
        namesDict = {"zero": 0, "one": 1, "two": 2, "three": 3}
        toTest1 = self.constructor(pointNames=origNames)
        toTest2 = self.constructor(pointNames=namesDict)
        assert toTest1._equalPointNames(toTest2)
        assert toTest2._equalPointNames(toTest1)

    def test__equalPointNames_noData(self):
        """ Test _equalPointNames() for empty objects """
        toTest1 = self.constructor(pointNames=None)
        toTest2 = self.constructor(pointNames=None)
        assert toTest1._equalPointNames(toTest2)
        assert toTest2._equalPointNames(toTest1)

    def test__equalPointNames_DefaultInequality(self):
        """ Test _equalPointNames() for inequality of some default named, different sized objects """
        toTest1 = self.constructor(pointNames=['1', '2'])
        toTest2 = self.constructor(pointNames=['1', '2', '3'])
        toTest1.points.setName(0, None)
        toTest1.points.setName(1, None)
        toTest2.points.setName(0, None)
        toTest2.points.setName(1, None)
        toTest2.points.setName(2, None)
        assert not toTest1._equalPointNames(toTest2)
        assert not toTest2._equalPointNames(toTest1)

    def test__equalPointNames_ignoresDefaults(self):
        """ Test _equalPointNames() for equality of default named objects """
        toTest1 = self.constructor(pointNames=['1', '2'])
        toTest2 = self.constructor(pointNames=['1', '2'])
        toTest1.points.setName(0, None)
        toTest1.points.setName(1, None)
        toTest2.points.setName(0, None)
        toTest2.points.setName(1, None)
        assert toTest1._equalPointNames(toTest2)
        assert toTest2._equalPointNames(toTest1)

    def test__equalPointNames_mixedDefaultsAndActual(self):
        toTest1 = self.constructor(pointNames=['1', '2'])
        toTest2 = self.constructor(pointNames=['1', '2'])
        toTest1.points.setName(0, None)
        toTest1.points.setName(1, '1')
        toTest2.points.setName(1, None)
        # have: test1 [Default, '1']
        # test2 ['1', Default]
        assert not toTest1._equalPointNames(toTest2)
        assert not toTest2._equalPointNames(toTest1)
        toTest1.points.setName(1, '2')
        toTest1.points.setName(1, None)
        # have: test1 [Default, '2']
        # test2: ['1', Default]
        assert not toTest1._equalPointNames(toTest2)
        assert not toTest2._equalPointNames(toTest1)


    ########################
    # _equalFeatureNames() #
    ########################

    def test_equalFeatureNames_False(self):
        """ Test _equalFeatureNames() against some non-equal input """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        assert not toTest._equalFeatureNames(self.constructor(featureNames=["short", "list"]))

        subset = self.constructor(featureNames=["zero", "one", "two"])
        superset = self.constructor(featureNames=["zero", "one", "two", "three", "four"])
        assert not toTest._equalFeatureNames(subset)
        assert not toTest._equalFeatureNames(superset)
        assert not subset._equalFeatureNames(toTest)
        assert not superset._equalFeatureNames(toTest)

    def test_equalFeatureNames_actuallyEqual(self):
        """ Test _equalFeatureNames() against some actually equal input """
        origFeatureNames = ["zero", "one", "two", "three"]
        featureNamesDict = {"zero": 0, "one": 1, "two": 2, "three": 3}
        toTest1 = self.constructor(featureNames=origFeatureNames)
        toTest2 = self.constructor(featureNames=featureNamesDict)
        assert toTest1._equalFeatureNames(toTest2)
        assert toTest2._equalFeatureNames(toTest1)

    def test_equalFeatureNames_noData(self):
        """ Test _equalFeatureNames() for empty objects """
        toTest1 = self.constructor(featureNames=None)
        toTest2 = self.constructor(featureNames=None)
        assert toTest1._equalFeatureNames(toTest2)
        assert toTest2._equalFeatureNames(toTest1)

    def test_equalFeatureNames_DefaultInequality(self):
        """ Test _equalFeatureNames() for inequality of some default named, different sized objects """
        toTest1 = self.constructor(featureNames=['1', '2'])
        toTest2 = self.constructor(featureNames=['1', '2', '3'])
        toTest1.features.setName(0, None)
        toTest1.features.setName(1, None)
        toTest2.features.setName(0, None)
        toTest2.features.setName(1, None)
        toTest2.features.setName(2, None)
        assert not toTest1._equalFeatureNames(toTest2)
        assert not toTest2._equalFeatureNames(toTest1)

    def test_equalFeatureNames_ignoresDefaults(self):
        """ Test _equalFeatureNames() for equality of default named objects """
        toTest1 = self.constructor(featureNames=['1', '2'])
        toTest2 = self.constructor(featureNames=['1', '2'])
        toTest1.features.setName(0, None)
        toTest1.features.setName(1, None)
        toTest2.features.setName(0, None)
        toTest2.features.setName(1, None)
        assert toTest1._equalFeatureNames(toTest2)
        assert toTest2._equalFeatureNames(toTest1)

    def test_equalFeatureNames_mixedDefaultsAndActual(self):
        toTest1 = self.constructor(featureNames=['1', '2'])
        toTest2 = self.constructor(featureNames=['1', '2'])
        toTest1.features.setName(0, None)
        toTest1.features.setName(1, '1')
        toTest2.features.setName(1, None)
        # have: test1 [Default, '1']
        # test2 ['1', Default]
        assert not toTest1._equalFeatureNames(toTest2)
        assert not toTest2._equalFeatureNames(toTest1)
        toTest1.features.setName(1, '2')
        toTest1.features.setName(1, None)
        # have: test1 [Default, '2']
        # test2: ['1', Default]
        assert not toTest1._equalFeatureNames(toTest2)
        assert not toTest2._equalFeatureNames(toTest1)


    ########################
    # default Object names #
    ########################

    def test_default_object_names(self):
        """ Test that default object names increment correctly """
        toTest0 = self.constructor(featureNames=['a', '2'])
        toTest1 = self.constructor(pointNames=['1b', '2'])
        toTest2 = self.constructor(featureNames=['c', '2'])

        firstNumber = int(toTest0.name[len(DEFAULT_NAME_PREFIX):])
        second = firstNumber + 1
        third = second + 1

        assert toTest1.name == DEFAULT_NAME_PREFIX + str(second)
        assert toTest2.name == DEFAULT_NAME_PREFIX + str(third)

    ###################
    # points.getNames #
    ###################

    def test_points_getNames_Empty(self):
        toTest = self.constructor(psize=0, fsize=2)

        ret = toTest.points.getNames()
        assert ret == []

    def test_points_getNames_basic(self):
        pnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(pointNames=pnames, fsize=2)

        ret = toTest.points.getNames()
        assert ret == ['zero', 'one', 'hello']

    def test_points_getNames_mixedDefault(self):
        pnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(pointNames=pnames, fsize=2)
        toTest.points.setName(0, None)

        ret = toTest.points.getNames()
        assert ret[0].startswith(DEFAULT_PREFIX)
        assert ret[1] == 'one'
        assert ret[2] == 'hello'

    def test_points_getNames_unmodifiable(self):
        pnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(pointNames=pnames, fsize=2)

        ret = toTest.points.getNames()

        ret[0] = 'modified'
        toTest.points.setName(1, 'modified')

        assert ret[1] == 'one'
        assert toTest.points.getIndex('zero') == 0
        assert toTest.points.getName(0) == 'zero'


    #####################
    # features.getNames #
    #####################

    def test_features_getNames_Empty(self):
        toTest = self.constructor(psize=2, fsize=0)

        ret = toTest.features.getNames()
        assert ret == []

    def test_features_getNames_basic(self):
        fnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(featureNames=fnames, psize=2)

        ret = toTest.features.getNames()
        assert ret == ['zero', 'one', 'hello']

    def test_features_getNames_mixedDefault(self):
        fnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(featureNames=fnames, psize=2)
        toTest.features.setName(0, None)

        ret = toTest.features.getNames()
        assert ret[0].startswith(DEFAULT_PREFIX)
        assert ret[1] == 'one'
        assert ret[2] == 'hello'

    def test_features_getNames_unmodifiable(self):
        fnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(featureNames=fnames, psize=2)

        ret = toTest.features.getNames()

        ret[0] = 'modified'
        toTest.features.setName(1, 'modified')

        assert ret[1] == 'one'
        assert toTest.features.getIndex('zero') == 0
        assert toTest.features.getName(0) == 'zero'


    ############################################################
    # features.getIndex, features.getIndices, features.getName #
    # points.getIndex, points.getIndices, points.getName       #
    ############################################################

    # consistency checks between all sources of axis name information
    @noLogEntryExpected
    def test_name_index_consistency(self):
        pnames = ['p0', 'p1', 'p2', 'p3', 'p4']
        fnames = ['fa', 'fb', 'fc']

        toTest = self.constructor(featureNames=fnames, pointNames=pnames)

        pByGetAll = toTest.points.getNames()
        pByIndex = [toTest.points.getName(i) for i in range(len(toTest.points))]
        assert pByIndex == pByGetAll

        pnamesShuffle = pythonRandom.sample(pnames, len(pnames))
        pByName = [toTest.points.getIndex(n) for n in pnamesShuffle]
        pByNames = toTest.points.getIndices([n for n in pnamesShuffle])
        pByPyIndex = [pnames.index(n) for n in pnamesShuffle]
        assert pByName == pByPyIndex
        assert pByNames == pByPyIndex

        fByGetAll = toTest.features.getNames()
        fByIndex = [toTest.features.getName(i) for i in range(len(toTest.features))]
        assert fByIndex == fByGetAll

        fnamesShuffle = pythonRandom.sample(fnames, len(fnames))
        fByName = [toTest.features.getIndex(n) for n in fnamesShuffle]
        fByNames = toTest.features.getIndices([n for n in fnamesShuffle])
        fByPyIndex = [fnames.index(n) for n in fnamesShuffle]
        assert fByName == fByPyIndex
        assert fByNames == fByPyIndex


    ###########################
    # points/features.hasName #
    ###########################
    @noLogEntryExpected
    def test_hasName(self):
        pnames = ['p0', 'p1', 'p2', 'p3', 'p4']
        fnames = ['fa', 'fb', 'fc']
        toTest = self.constructor(featureNames=fnames, pointNames=pnames)

        for name in pnames:
            assert toTest.points.hasName(name)
        for name in fnames:
            assert toTest.features.hasName(name)
        notNames = ['x', 'y', 'z']
        for name in notNames:
            assert not toTest.points.hasName(name)
            assert not toTest.features.hasName(name)

    ###########
    # __len__ #
    ###########
    @noLogEntryExpected
    def test_len_handmade(self):
        zeroZero = self.constructor(psize=0, fsize=0)
        assert len(zeroZero) == 0
        zeroOne = self.constructor(psize=0, fsize=1)
        assert len(zeroOne) == 0
        zeroN = self.constructor(psize=0, fsize=12)
        assert len(zeroN) == 0

        oneZero = self.constructor(psize=1, fsize=0)
        assert len(oneZero) == 0
        oneOne = self.constructor(psize=1, fsize=1)
        assert len(oneOne) == 1
        oneN = self.constructor(psize=1, fsize=13)
        assert len(oneN) == 13

        nZero = self.constructor(psize=15, fsize=0)
        assert len(nZero) == 0
        nOne = self.constructor(psize=11, fsize=1)
        assert len(nOne) == 11

    @raises(TypeError)
    def test_len_exception(self):
        nn = self.constructor(psize=11, fsize=33)
        len(nn)

    #####################################
    # points.__len__ / features.__len__ #
    #####################################
    @noLogEntryExpected
    def test_axis_len_handmade(self):
        zeroZero = self.constructor(psize=0, fsize=0)
        assert len(zeroZero.points) == 0
        assert len(zeroZero.features) == 0
        zeroOne = self.constructor(psize=0, fsize=1)
        assert len(zeroOne.points) == 0
        assert len(zeroOne.features) == 1
        zeroN = self.constructor(psize=0, fsize=12)
        assert len(zeroN.points) == 0
        assert len(zeroN.features) == 12

        oneZero = self.constructor(psize=1, fsize=0)
        assert len(oneZero.points) == 1
        assert len(oneZero.features) == 0
        oneOne = self.constructor(psize=1, fsize=1)
        assert len(oneOne.points) == 1
        assert len(oneOne.features) == 1
        oneN = self.constructor(psize=1, fsize=13)
        assert len(oneN.points) == 1
        assert len(oneN.features) == 13

        nZero = self.constructor(psize=15, fsize=0)
        assert len(nZero.points) == 15
        assert len(nZero.features) == 0
        nOne = self.constructor(psize=11, fsize=1)
        assert len(nOne.points) == 11
        assert len(nOne.features) == 1

        nn = self.constructor(psize=11, fsize=33)
        assert len(nn.points) == 11
        assert len(nn.features) == 33

    ############
    # __bool__ #
    ############
    @noLogEntryExpected
    def test_bool_handmade(self):
        bothEmpty = self.constructor(psize=0, fsize=0)
        assert not bool(bothEmpty)
        pointEmpty = self.constructor(psize=0, fsize=4)
        assert not bool(pointEmpty)
        featEmpty = self.constructor(psize=4, fsize=0)
        assert not bool(featEmpty)
        noEmpty = self.constructor(psize=4, fsize=4)
        assert bool(noEmpty)

    #######################################
    # points.__bool__ / features.__bool__ #
    #######################################
    @noLogEntryExpected
    def test_axis_bool_handmade(self):
        bothEmpty = self.constructor(psize=0, fsize=0)
        assert not bool(bothEmpty.points)
        assert not bool(bothEmpty.features)
        pointEmpty = self.constructor(psize=0, fsize=4)
        assert not bool(pointEmpty.points)
        assert bool(pointEmpty.features)
        featEmpty = self.constructor(psize=4, fsize=0)
        assert bool(featEmpty.points)
        assert not bool(featEmpty.features)
        noEmpty = self.constructor(psize=4, fsize=4)
        assert bool(noEmpty.points)
        assert bool(noEmpty.features)

    #########################
    # constructIndicesList #
    #########################

    def constructIndicesList_backend(self, container):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)
        expected = [1,2]

        # one-dimensional input
        intPts1D = container([1,2])
        strPts1D = container(['p2', 'p3'])
        mixPts1D = container([1, 'p3'])
        intFts1D = container([1,2])
        strFts1D = container(['f2', 'f3'])
        mixFts1D = container([1, 'f3'])

        assert constructIndicesList(toTest, 'point', intPts1D) == expected
        assert constructIndicesList(toTest, 'point', strPts1D) == expected
        assert constructIndicesList(toTest, 'point', mixPts1D) == expected
        assert constructIndicesList(toTest, 'feature', intFts1D) == expected
        assert constructIndicesList(toTest, 'feature', strFts1D) == expected
        assert constructIndicesList(toTest, 'feature', mixFts1D) == expected

    @raises(CalledFunctionException)
    def test_constructIndicesList_calls_valuesToPythonList(self):
        pointNames = ['p1','p2','p3']
        toTest = self.constructor(pointNames=pointNames)
        # need to use mock.patch as context manager after object creation
        # because Base.__init__ also calls valuesToPythonList
        with mock.patch('nimble.data.dataHelpers.valuesToPythonList', calledException):
            constructIndicesList(toTest, 'point', pointNames)

    def testconstructIndicesList_pythonList(self):
        self.constructIndicesList_backend(lambda lst: lst)

    def testconstructIndicesList_pythonTuple(self):
        self.constructIndicesList_backend(lambda lst: tuple(lst))

    def testconstructIndicesList_pythonGenerator(self):
        self.constructIndicesList_backend(lambda lst: (val for val in lst))

    def testconstructIndicesList_NimbleObjects(self):
        for retType in available:
            self.constructIndicesList_backend(
                lambda lst: createData(retType, lst, convertToType=object))

    def testconstructIndicesList_numpyArray(self):
        self.constructIndicesList_backend(lambda lst: numpy.array(lst,dtype=object))

    def testconstructIndicesList_pandasSeries(self):
        self.constructIndicesList_backend(lambda lst: pandas.Series(lst))

    def testconstructIndicesList_handmadeIterator(self):
        self.constructIndicesList_backend(lambda lst: SimpleIterator(*lst))

    def testconstructIndicesList_handmadeGetItemOnly(self):
        self.constructIndicesList_backend(lambda lst: GetItemOnly(*lst))

    def testconstructIndicesList_singleInteger(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)
        expected = [2]

        index = 2

        assert constructIndicesList(toTest, 'point', index) == expected
        assert constructIndicesList(toTest, 'feature', index) == expected

    def testconstructIndicesList_singleString(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)
        expected = [2]

        ptIndex = 'p3'
        ftIndex = 'f3'

        assert constructIndicesList(toTest, 'point', ptIndex) == expected
        assert constructIndicesList(toTest, 'feature', ftIndex) == expected

    def testconstructIndicesList_pythonRange(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)
        expected = [1, 2]

        testRange = range(1,3)

        assert constructIndicesList(toTest, 'point', testRange) == expected
        assert constructIndicesList(toTest, 'feature', testRange) == expected

    @raises(InvalidArgumentType)
    def testconstructIndicesList_singleFloat(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)

        ptIndex = 2.0

        constructIndicesList(toTest, 'point', ptIndex)

    @raises(InvalidArgumentType)
    def testconstructIndicesList_floatIteratable(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)

        ftIndex = [2.0]

        constructIndicesList(toTest, 'feature', ftIndex)

    @raises(InvalidArgumentType)
    def testconstructIndicesList_floatInList(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)

        ptIndex = [0, 'p2', 2.0]

        constructIndicesList(toTest, 'point', ptIndex)

    @raises(IndexError)
    def testconstructIndicesList_InvalidIndexInteger(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)

        ftIndex = [2, 3]

        constructIndicesList(toTest, 'feature', ftIndex)

    @raises(KeyError)
    def testconstructIndicesList_InvalidIndexString(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)

        ftIndex = ['f3', 'f4']

        constructIndicesList(toTest, 'feature', ftIndex)

    @raises(InvalidArgumentType)
    def testconstructIndicesList_handmadeNotIterable(self):
        self.constructIndicesList_backend(lambda lst: NotIterable(*lst))

    @raises(InvalidArgumentType)
    def testconstructIndicesList_numpyMatrix(self):
        self.constructIndicesList_backend(lambda lst: numpy2DArray(lst))

    @raises(InvalidArgumentType)
    def testconstructIndicesList_pandasDataFrame(self):
        self.constructIndicesList_backend(lambda lst: pandas.DataFrame(lst))

    @raises(InvalidArgumentType)
    def testconstructIndicesList_handmade2DOne(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)

        list2D = [['f1','f2']]

        constructIndicesList(toTest, 'feature', list2D)

    @raises(InvalidArgumentType)
    def testconstructIndicesList_handmade2DTwo(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)

        array2D = numpy.array([[1,2,3],[4,5,6]])

        constructIndicesList(toTest, 'feature', array2D)

    @raises(InvalidArgumentType)
    def testconstructIndicesList_handmade2DThree(self):
        pointNames = ['p1','p2','p3']
        featureNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(pointNames=pointNames, featureNames=featureNames)

        iter2D = SimpleIterator([1,'p2'])

        constructIndicesList(toTest, 'point', iter2D)

    ##################
    # High Dimension #
    ##################

    def test_highDimension_shapes(self):
        toTest3D = self.constructor((3, 3, 5))
        assert toTest3D._shape == [3, 3, 5]
        assert toTest3D.dimensions == (3, 3, 5)
        assert toTest3D.shape == (3, 15)
        assert len(toTest3D.points) == 3
        assert len(toTest3D.features) == 15

        toTest4D = self.constructor((4, 3, 3, 5))
        assert toTest4D._shape == [4, 3, 3, 5]
        assert toTest4D.dimensions == (4, 3, 3, 5)
        assert toTest4D.shape == (4, 45)
        assert len(toTest4D.points) == 4
        assert len(toTest4D.features) == 45

        toTest3DEmpty = self.constructor((0, 0, 0))
        assert toTest3DEmpty._shape == [0, 0, 0]
        assert toTest3DEmpty.dimensions == (0, 0, 0)
        assert toTest3DEmpty.shape == (0, 0)
        assert len(toTest3DEmpty.points) == 0
        assert len(toTest3DEmpty.features) == 0

    def test_highDimension_namesAndIndices(self):
        pNames = ['p1', 'p2', 'p3']
        fNames = ['f' + str(i) for i in range(1, 16)]
        toTest3D = self.constructor(shape=(3, 3, 5), pointNames=pNames,
                                    featureNames=fNames)
        assert toTest3D.nameIsDefault()
        assert toTest3D.points.getNames() == pNames
        assert toTest3D.features.getNames() == fNames

        newPNames = ['a', 'b', 'c']
        toTest3D.points.setNames(newPNames)
        assert toTest3D.points.getNames() == newPNames
        assert toTest3D.points.getIndices(['b', 'a']) == [1, 0]

        newFNames = ['ft_' + str(i) for i in range(15)]
        toTest3D.features.setNames(newFNames)
        assert toTest3D.features.getNames() == newFNames
        assert toTest3D.features.getIndices(['ft_1', 'ft_0']) == [1, 0]

        toTest3D.points.setName('a', 'z')
        assert not toTest3D.points.hasName('a')
        assert toTest3D.points.hasName('z')
        assert toTest3D.points.getName(0) == 'z'
        assert toTest3D.points.getIndex('z') == 0

        toTest3D.features.setName('ft_0', 'ft_first')
        assert not toTest3D.features.hasName('ft_0')
        assert toTest3D.features.hasName('ft_first')
        assert toTest3D.features.getName(0) == 'ft_first'
        assert toTest3D.features.getIndex('ft_first') == 0

    def test_highDimension_len(self):
        tensor3D = self.constructor((3, 3, 5))
        tensor4D = self.constructor((4, 3, 3, 5))
        tensor5D = self.constructor((5, 4, 3, 3, 5))
        for tensor in [tensor3D, tensor4D, tensor5D]:
            try:
                len(tensor)
                assert False # expected ImproperObjectAction
            except ImproperObjectAction:
                pass

    def test_highDimension_bool(self):
        tensor3D = self.constructor((3, 3, 5))
        tensor4D = self.constructor((4, 3, 3, 5))
        tensor5D = self.constructor((5, 4, 3, 3, 5))
        for tensor in [tensor3D, tensor4D, tensor5D]:
            assert bool(tensor)

        empty3D = self.constructor((0, 0, 0))
        empty4D = self.constructor((0, 0, 0, 0))
        empty5D = self.constructor((0, 0, 0, 0, 0))
        for tensor in [empty3D, empty4D, empty5D]:
            assert not bool(tensor)
