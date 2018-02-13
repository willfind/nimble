"""
Unit tests of the low level functions defined by the base representation class.

Since these functions are stand alone in the base class, they can be tested directly.
FeatureName equality is defined using _features_implementation, which is to be defined
in the derived class, so all of these tests make a call to self.constructor instead
of directly instantiating a Base object. This function temporarily fills in
that missing implementation.

Methods tested in this file (none modify the data):
_addPointName, _addFeatureName, _pointNameDifference, _featureNameDifference,
_pointNameIntersection, _featureNameIntersection, _pointNameSymmetricDifference,
_featureNameSymmetricDifference, _pointNameUnion, _featureNameUnion,
setPointName, setFeatureName, setPointNames, setFeatureNames,
_removePointNameAndShift, _removeFeatureNameAndShift, _equalPointNames,
_equalFeatureNames, getPointNames, getFeatureNames, __len__,
getFeatureIndex, getFeatureName, getPointIndex, getPointName

"""

from __future__ import absolute_import
from UML.data import Base
from UML.data.dataHelpers import DEFAULT_PREFIX
from UML.data.dataHelpers import DEFAULT_NAME_PREFIX
from nose.tools import *
from UML.exceptions import ArgumentException
from UML.exceptions import ImproperActionException

from six.moves import range
from UML.randomness import pythonRandom



###########
# helpers #
###########


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
    ###################
    # _addPointName() #
    ###################

    @raises(ArgumentException)
    def test__addPName_exceptionPointNameWrongType(self):
        """ Test _addPointName() for ArgumentException when given a non string  pointName """
        toTest = self.constructor(pointNames=["hello"])
        toTest._addPointName(34)

    @raises(ArgumentException)
    def test__addPointName_exceptionNonUnique(self):
        """ Test _addPointName() for ArgumentException when given a duplicate pointName """
        toTest = self.constructor(pointNames=["hello"])
        toTest._addPointName("hello")

    def test__addPointName_handmadeDefaultCounter(self):
        """ Test _addPointName() changes nextDefault counter correctly """
        toTest = self.constructor(pointNames=["hello"])
        assert toTest._nextDefaultValuePoint == 1
        toTest._addPointName(DEFAULT_PREFIX + "1")
        assert toTest._nextDefaultValuePoint == 2

    def test__addPointName_handmade(self):
        """ Test _addPointName() against handmade output """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest._addPointName("four")
        toTest._addPointName("five")

        expected = ["zero", "one", "two", "three", "four", "five"]
        confirmExpectedNames(toTest, 'point', expected)

    def test__addPointName_handmade_addDefault(self):
        """ Test _addPointName() against handmade output """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest._addPointName(None)
        toTest._addPointName(None)

        expected = ["zero", "one", "two", "three", DEFAULT_PREFIX, DEFAULT_PREFIX]

        confirmExpectedNames(toTest, 'point', expected)

    ###############
    # _addFeatureName() #
    ###############
    @raises(ArgumentException)
    def test_addFeatureName_exceptionFeatureNameWrongType(self):
        """ Test _addFeatureName() for ArgumentException when given a non string featureName """
        toTest = self.constructor(featureNames=["hello"])
        toTest._addFeatureName(34)

    @raises(ArgumentException)
    def test_addFeatureName_exceptionNonUnique(self):
        """ Test _addFeatureName() for ArgumentException when given a duplicate featureName """
        toTest = self.constructor(featureNames=["hello"])
        toTest._addFeatureName("hello")

    def test_addFeatureName_handmadeDefaultCounter(self):
        """ Test _addFeatureName() changes nextDefault counter correctly """
        toTest = self.constructor(featureNames=["hello"])
        assert toTest._nextDefaultValueFeature == 1
        toTest._addFeatureName(DEFAULT_PREFIX + "1")
        assert toTest._nextDefaultValueFeature == 2

    def test_addFeatureName_handmade(self):
        """ Test _addFeatureName() against handmade output """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest._addFeatureName("four")
        toTest._addFeatureName("five")

        expected = ["zero", "one", "two", "three", "four", "five"]
        confirmExpectedNames(toTest, 'feature', expected)

    def test_addFeatureName_handmade_addDefault(self):
        """ Test _addFeatureName() against handmade output """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest._addFeatureName(None)
        toTest._addFeatureName(None)

        expected = ["zero", "one", "two", "three", DEFAULT_PREFIX, DEFAULT_PREFIX]

        confirmExpectedNames(toTest, 'feature', expected)


    ##########################
    # _pointNameDifference() #
    ##########################

    @raises(ArgumentException)
    def test__pointNameDifference_exceptionOtherNone(self):
        """ Test _pointNameDifference() for ArgumentException when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameDifference(None)

    @raises(ArgumentException)
    def test__pointNameDifference_exceptionWrongType(self):
        """ Test _pointNameDifference() for ArgumentException when the other object is not the right type """
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

    @raises(ArgumentException)
    def test_featureNameDifference_exceptionOtherNone(self):
        """ Test _featureNameDifference() for ArgumentException when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameDifference(None)

    @raises(ArgumentException)
    def test_featureNameDifference_exceptionWrongType(self):
        """ Test _featureNameDifference() for ArgumentException when the other object is not the right type """
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


    ############################
    # _pointNameIntersection() #
    ############################

    @raises(ArgumentException)
    def test__pointNameIntersection_exceptionOtherNone(self):
        """ Test _pointNameIntersection() for ArgumentException when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameIntersection(None)

    @raises(ArgumentException)
    def test__pointNameIntersection_exceptionWrongType(self):
        """ Test _pointNameIntersection() for ArgumentException when the other object is not the right type """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameIntersection("wrong")

    def test__pointNameIntersection_handmade(self):
        """ Test _pointNameIntersection() against handmade output """
        toTest1 = self.constructor(pointNames=["one", "two", "three"])
        toTest2 = self.constructor(pointNames=["two", "four"])
        results = toTest1._pointNameIntersection(toTest2)
        assert "one" not in results
        assert "two" in results
        assert "three" not in results
        assert "four" not in results

    #######################
    # _featureNameIntersection() #
    #######################

    @raises(ArgumentException)
    def test_featureNameIntersection_exceptionOtherNone(self):
        """ Test _featureNameIntersection() for ArgumentException when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameIntersection(None)

    @raises(ArgumentException)
    def test_featureNameIntersection_exceptionWrongType(self):
        """ Test _featureNameIntersection() for ArgumentException when the other object is not the right type """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameIntersection("wrong")

    def test_featureNameIntersection_handmade(self):
        """ Test _featureNameIntersection() against handmade output """
        toTest1 = self.constructor(featureNames=["one", "two", "three"])
        toTest2 = self.constructor(featureNames=["two", "four"])
        results = toTest1._featureNameIntersection(toTest2)
        assert "one" not in results
        assert "two" in results
        assert "three" not in results
        assert "four" not in results

    ##############################
    # _pointNameSymmetricDifference() #
    ##############################

    @raises(ArgumentException)
    def test__pointNameSymmetricDifference_exceptionOtherNone(self):
        """ Test _pointNameSymmetricDifference() for ArgumentException when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameSymmetricDifference(None)

    @raises(ArgumentException)
    def test__pointNameSymmetricDifference_exceptionWrongType(self):
        """ Test _pointNameSymmetricDifference() for ArgumentException when the other object is not the right type """
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

    @raises(ArgumentException)
    def test_featureNameSymmetricDifference_exceptionOtherNone(self):
        """ Test _featureNameSymmetricDifference() for ArgumentException when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameSymmetricDifference(None)

    @raises(ArgumentException)
    def test_featureNameSymmetricDifference_exceptionWrongType(self):
        """ Test _featureNameSymmetricDifference() for ArgumentException when the other object is not the right type """
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

    @raises(ArgumentException)
    def test__pointNameUnion_exceptionOtherNone(self):
        """ Test _pointNameUnion() for ArgumentException when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest._pointNameUnion(None)

    @raises(ArgumentException)
    def test__pointNameUnion_exceptionWrongType(self):
        """ Test _pointNameUnion() for ArgumentException when the other object is not the right type """
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

    @raises(ArgumentException)
    def test_featureNameUnion_exceptionOtherNone(self):
        """ Test _featureNameUnion() for ArgumentException when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest._featureNameUnion(None)

    @raises(ArgumentException)
    def test_featureNameUnion_exceptionWrongType(self):
        """ Test _featureNameUnion() for ArgumentException when the other object is not the right type """
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

    #################
    # setPointName() #
    #################

    @raises(ArgumentException)
    def test_setPointName_exceptionPrevWrongType(self):
        """ Test setPointName() for ArgumentException when given the wrong type for prev"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.setPointName(oldIdentifier=0.3, newName="New!")

    @raises(ArgumentException)
    def test_setPointName_exceptionPrevInvalidIndex(self):
        """ Test setPointName() for ArgumentException when given an invalid prev index"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.setPointName(oldIdentifier=12, newName="New!")

    @raises(ArgumentException)
    def test_setPointName_exceptionPrevNotFound(self):
        """ Test setPointName() for ArgumentException when the prev pointName is not found"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.setPointName(oldIdentifier="Previous!", newName="New!")

    @raises(ArgumentException)
    def test_setPointName_exceptionNewInvalidType(self):
        """ Test setPointName() for ArgumentException when the new pointName is not a string"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.setPointName(oldIdentifier="three", newName=4)

    @raises(ArgumentException)
    def test_setPointName_exceptionNonUnique(self):
        """ Test setPointName() for ArgumentException when a duplicate pointName is given"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.setPointName(oldIdentifier="three", newName="two")

    @raises(ArgumentException)
    def test_setPointName_exceptionNoPoints(self):
        toTest = self.constructor()
        toTest.setPointName("hello", "2")

    def test_setPointName_handmade_viaIndex(self):
        """ Test setPointName() against handmade input when specifying the pointName by index """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.setPointName(0, "ZERO")
        toTest.setPointName(3, "3")
        expectedNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'point', expectedNames)

    def test_setPointName_handmade_viaPointName(self):
        """ Test setPointName() against handmade input when specifying the pointName by name """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.setPointName("zero", "ZERO")
        toTest.setPointName("three", "3")
        expectedNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'point', expectedNames)

    def test_setPointName_NoneOutput(self):
        """ Test setPointName() return None as output """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        ret = toTest.setPointName("zero", "ZERO")
        assert ret is None


    #################
    # setFeatureName() #
    #################

    @raises(ArgumentException)
    def test_setFeatureName_exceptionPrevWrongType(self):
        """ Test setFeatureName() for ArgumentException when given the wrong type for prev"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.setFeatureName(oldIdentifier=0.3, newName="New!")

    @raises(ArgumentException)
    def test_setFeatureName_exceptionPrevInvalidIndex(self):
        """ Test setFeatureName() for ArgumentException when given an invalid prev index"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.setFeatureName(oldIdentifier=12, newName="New!")

    @raises(ArgumentException)
    def test_setFeatureName_exceptionPrevNotFound(self):
        """ Test setFeatureName() for ArgumentException when the prev featureName is not found"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.setFeatureName(oldIdentifier="Previous!", newName="New!")

    @raises(ArgumentException)
    def test_setFeatureName_exceptionNewInvalidType(self):
        """ Test setFeatureName() for ArgumentException when the new featureName is not a string"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.setFeatureName(oldIdentifier="three", newName=4)

    @raises(ArgumentException)
    def test_setFeatureName_exceptionNonUnique(self):
        """ Test setFeatureName() for ArgumentException when a duplicate featureName is given"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.setFeatureName(oldIdentifier="three", newName="two")

    #@raises(ArgumentException)
    #def test_setFeatureName_exceptionManualAddDefault(self):
    #	""" Test setFeatureName() for ArgumentException when given a default featureName """
    #	toTest = self.constructor(featureNames=["hello"])
    #	toTest.setFeatureName("hello",DEFAULT_PREFIX + "2")

    @raises(ArgumentException)
    def test_setFeatureName_exceptionNoFeatures(self):
        toTest = self.constructor()
        toTest.setFeatureName("hello", "2")

    def test_setFeatureName_handmade_viaIndex(self):
        """ Test setFeatureName() against handmade input when specifying the featureName by index """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.setFeatureName(0, "ZERO")
        toTest.setFeatureName(3, "3")
        expectedFeatureNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'feature', expectedFeatureNames)

    def test_setFeatureName_handmade_viaFeatureName(self):
        """ Test setFeatureName() against handmade input when specifying the featureName by name """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.setFeatureName("zero", "ZERO")
        toTest.setFeatureName("three", "3")
        expectedFeatureNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'feature', expectedFeatureNames)

    def test_setFeatureName_NoneOutput(self):
        """ Test setFeatureName() returns None as output """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        ret = toTest.setFeatureName("zero", "ZERO")
        assert ret is None

    ###################
    # setPointNames() #
    ###################

    @raises(ArgumentException)
    def test_setPointNames_exceptionWrongTypeObject(self):
        """ Test setPointNames() for ArgumentException when pointNames is an unexpected type """
        toTest = self.constructor(pointNames=['one'])
        toTest.setPointNames(12)

    @raises(ArgumentException)
    def test_setPointNames_exceptionNonStringPointNameInList(self):
        """ Test setPointNames() for ArgumentException when a list element is not a string """
        toTest = self.constructor(pointNames=['one'])
        nonStringNames = [1, 2, 3]
        toTest.setPointNames(nonStringNames)

    @raises(ArgumentException)
    def test_setPointNames_exceptionNonUniqueStringInList(self):
        """ Test setPointNames() for ArgumentException when a list element is not unique """
        toTest = self.constructor(pointNames=['one'])
        nonUnique = ['1', '2', '3', '1']
        toTest.setPointNames(nonUnique)

    @raises(ArgumentException)
    def test_setPointNames_exceptionNoPointsList(self):
        """ Test setPointNames() for ArgumentException when there are no points to name """
        toTest = self.constructor()
        toAssign = ["hey", "gone", "none", "blank"]
        toTest.setPointNames(toAssign)

    def test_setPointNames_emptyDataAndList(self):
        """ Test setPointNames() when both the data and the list are empty """
        toTest = self.constructor()
        toAssign = []
        toTest.setPointNames(toAssign)
        assert toTest.getPointNames() == []

    def test_setPointNames_addDefault(self):
        """ Test setPointNames() when given a default pointName """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        newNames = ["zero", "one", "two", DEFAULT_PREFIX + "17"]
        toTest.setPointNames(newNames)
        assert toTest._nextDefaultValuePoint > 17

    def test_setPointNames_handmadeList(self):
        """ Test setPointNames() against handmade output """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        origNames = ["zero", "one", "two", "three"]
        toTest.setPointNames(origNames)
        confirmExpectedNames(toTest, 'point', origNames)

    def test_setPointNames_handmadeReplacingWithSameList(self):
        """ Test setPointNames() against handmade output when you're replacing the position of poitnNames """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        toAssign = ["hey", "gone", "none", "blank"]
        ret = toTest.setPointNames(toAssign)  # ret Check
        confirmExpectedNames(toTest, 'point', toAssign)
        assert ret is None

    @raises(ArgumentException)
    def test_setPointNames_exceptionNonStringPointNameInDict(self):
        """ Test setPointNames() for ArgumentException when a dict key is not a string """
        toTest = self.constructor(pointNames=['one'])
        nonStringNames = {1: 1}
        toTest.setPointNames(nonStringNames)

    @raises(ArgumentException)
    def test_setPointNames_exceptionNonIntIndexInDict(self):
        """ Test setPointNames() for ArgumentException when a dict value is not an int """
        toTest = self.constructor(pointNames=['one'])
        nonIntIndex = {"one": "one"}
        toTest.setPointNames(nonIntIndex)

    @raises(ArgumentException)
    def test_setPointNames_exceptionNoPointsDict(self):
        """ Test setPointNames() for ArgumentException when there are no points to name """
        toTest = self.constructor(pointNames=[])
        toAssign = {"hey": 0, "gone": 1, "none": 2, "blank": 3}
        toTest.setPointNames(toAssign)

    def test_setPointNames_emptyDataAndDict(self):
        """ Test setPointNames() when both the data and the dict are empty """
        toTest = self.constructor()
        toAssign = {}
        toTest.setPointNames(toAssign)
        assert toTest.getPointNames() == []

    def test_setPointNames_handmadeDict(self):
        """ Test setPointNames() against handmade output """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        origNames = {"zero": 0, "one": 1, "two": 2, "three": 3}
        toTest.setPointNames(origNames)
        confirmExpectedNames(toTest, 'point', origNames)

    def test_setPointNames_handmadeReplacingWithSameDict(self):
        """ Test setPointNames() against handmade output when you're replacing the position of pointNames """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        toAssign = {"hey": 0, "gone": 1, "none": 2, "blank": 3}
        ret = toTest.setPointNames(toAssign)
        confirmExpectedNames(toTest, 'point', toAssign)
        assert ret is None

    def test_setPointNames_list_mixedSpecifiedUnspecified_defaults(self):
        toTest = self.constructor(pointNames=([None] * 4))

        nextNum = toTest._nextDefaultValuePoint

        toAssign = [None] * 4
        toAssign[0] = DEFAULT_PREFIX + str(nextNum)
        toAssign[2] = DEFAULT_PREFIX + str(nextNum - 1)

        ret = toTest.setPointNames(toAssign)

        assert toTest.getPointName(0) == DEFAULT_PREFIX + str(nextNum)
        assert toTest.getPointName(1) == DEFAULT_PREFIX + str(nextNum + 1)
        assert toTest.getPointName(2) == DEFAULT_PREFIX + str(nextNum - 1)
        assert toTest.getPointName(3).startswith(DEFAULT_PREFIX)
        assert ret is None


    #####################
    # setFeatureNames() #
    #####################

    @raises(ArgumentException)
    def test_setFeatureNames_exceptionWrongTypeObject(self):
        """ Test setFeatureNames() for ArgumentException when featureNames is an unexpected type """
        toTest = self.constructor(featureNames=['one'])
        toTest.setFeatureNames(12)

    @raises(ArgumentException)
    def test_setFeatureNames_exceptionNonStringFeatureNameInDict(self):
        """ Test setFeatureNames() for ArgumentException when a dict key is not a string """
        toTest = self.constructor(featureNames=['one'])
        nonStringFeatureNames = {1: 1}
        toTest.setFeatureNames(nonStringFeatureNames)

    @raises(ArgumentException)
    def test_setFeatureNames_exceptionNonIntIndexInDict(self):
        """ Test setFeatureNames() for ArgumentException when a dict value is not an int """
        toTest = self.constructor(featureNames=['one'])
        nonIntIndex = {"one": "one"}
        toTest.setFeatureNames(nonIntIndex)

    @raises(ArgumentException)
    def test_setFeatureNames_exceptionNoFeaturesDict(self):
        """ Test setFeatureNames() for ArgumentException when there are no features to name """
        toTest = self.constructor(featureNames=[])
        toAssign = {"hey": 0, "gone": 1, "none": 2, "blank": 3}
        toTest.setFeatureNames(toAssign)

    def test_setFeatureNames_emptyDataAndDict(self):
        """ Test setFeatureNames() when both the data and the dict are empty """
        toTest = self.constructor()
        toAssign = {}
        toTest.setFeatureNames(toAssign)
        assert toTest.getFeatureNames() == []

    def test_setFeatureNames_handmadeDict(self):
        """ Test setFeatureNames() against handmade output """
        toTest = self.constructor(featureNames=["blank", "none", "gone", "hey"])
        origFeatureNames = {"zero": 0, "one": 1, "two": 2, "three": 3}
        toTest.setFeatureNames(origFeatureNames)
        confirmExpectedNames(toTest, 'feature', origFeatureNames)

    def test_setFeatureNames_handmadeReplacingWithSameDict(self):
        """ Test setFeatureNames() against handmade output when you're replacing the position of featureNames """
        toTest = self.constructor(featureNames=["blank", "none", "gone", "hey"])
        toAssign = {"hey": 0, "gone": 1, "none": 2, "blank": 3}
        ret = toTest.setFeatureNames(toAssign)
        confirmExpectedNames(toTest, 'feature', toAssign)
        assert ret is None

    @raises(ArgumentException)
    def test_setFeatureNames_exceptionNonStringFeatureNameInList(self):
        """ Test setFeatureNames() for ArgumentException when a list element is not a string """
        toTest = self.constructor(featureNames=['one'])
        nonStringFeatureNames = [1, 2, 3]
        toTest.setFeatureNames(nonStringFeatureNames)

    @raises(ArgumentException)
    def test_setFeatureNames_exceptionNonUniqueStringInList(self):
        """ Test setFeatureNames() for ArgumentException when a list element is not unique """
        toTest = self.constructor(featureNames=['one'])
        nonUnique = ['1', '2', '3', '1']
        toTest.setFeatureNames(nonUnique)

    @raises(ArgumentException)
    def test_setFeatureNames_exceptionNoFeaturesList(self):
        """ Test setFeatureNames() for ArgumentException when there are no features to name """
        toTest = self.constructor()
        toAssign = ["hey", "gone", "none", "blank"]
        toTest.setFeatureNames(toAssign)

    def test_setFeatureNames_emptyDataAndList(self):
        """ Test setFeatureNames() when both the data and the list are empty """
        toTest = self.constructor()
        toAssign = []
        toTest.setFeatureNames(toAssign)
        assert toTest.getFeatureNames() == []

    def test_setFeatureNames_addDefault(self):
        """ Test setFeatureNames() when given a default featureName """
        toTest = self.constructor(featureNames=["blank", "none", "gone", "hey"])
        newFeatureNames = ["zero", "one", "two", DEFAULT_PREFIX + "17"]
        toTest.setFeatureNames(newFeatureNames)
        assert toTest._nextDefaultValueFeature > 17

    def test_setFeatureNames_handmadeList(self):
        """ Test setFeatureNames() against handmade output """
        toTest = self.constructor(featureNames=["blank", "none", "gone", "hey"])
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest.setFeatureNames(origFeatureNames)
        confirmExpectedNames(toTest, 'feature', origFeatureNames)

    def test_setFeatureNames_handmadeReplacingWithSameList(self):
        """ Test setFeatureNames() against handmade output when you're replacing the position of featureNames """
        toTest = self.constructor(featureNames=["blank", "none", "gone", "hey"])
        toAssign = ["hey", "gone", "none", "blank"]
        ret = toTest.setFeatureNames(toAssign)  # RET CHECK
        confirmExpectedNames(toTest, 'feature', toAssign)
        assert ret is None

    def test_setFeatureNames_list_mixedSpecifiedUnspecified_defaults(self):
        toTest = self.constructor(featureNames=([None] * 4))

        nextNum = toTest._nextDefaultValueFeature

        toAssign = [None] * 4
        toAssign[0] = DEFAULT_PREFIX + str(nextNum)
        toAssign[2] = DEFAULT_PREFIX + str(nextNum - 1)

        ret = toTest.setFeatureNames(toAssign)

        assert toTest.getFeatureName(0) == DEFAULT_PREFIX + str(nextNum)
        assert toTest.getFeatureName(1) == DEFAULT_PREFIX + str(nextNum + 1)
        assert toTest.getFeatureName(2) == DEFAULT_PREFIX + str(nextNum - 1)
        assert toTest.getFeatureName(3).startswith(DEFAULT_PREFIX)
        assert ret is None


    ##########################
    # _removePointNameAndShift() #
    ##########################

    @raises(ArgumentException)
    def test__removePointNameAndShift_exceptionNoneInput(self):
        """ Test _removePointNameAndShift() for ArgumentException when the identifier is None"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest._removePointNameAndShift(None)

    @raises(ArgumentException)
    def test__removePointNameAndShift_exceptionWrongTypeInput(self):
        """ Test _removePointNameAndShift() for ArgumentException when given the wrong type"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest._removePointNameAndShift(0.3)

    @raises(ArgumentException)
    def test__removePointNameAndShift_exceptionInvalidIndex(self):
        """ Test _removePointNameAndShift() for ArgumentException when given an invalid index"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest._removePointNameAndShift(12)

    @raises(ArgumentException)
    def test__removePointNameAndShift_exceptionFeatureNameNotFound(self):
        """ Test _removePointNameAndShift() for ArgumentException when the featureName is not found"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest._removePointNameAndShift("bogus")

    def test__removePointNameAndShift_handmade_viaIndex(self):
        """ Test _removePointNameAndShift() against handmade output """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest._removePointNameAndShift(0)
        toTest._removePointNameAndShift(2)
        expectedNames = ["one", "two"]
        confirmExpectedNames(toTest, 'point', expectedNames)

    def test__removePointNameAndShift_handmade_viaFeatureName(self):
        """ Test _removePointNameAndShift() against handmade output """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest._removePointNameAndShift('zero')
        toTest._removePointNameAndShift('two')
        expectedNames = ["one", "three"]
        confirmExpectedNames(toTest, 'point', expectedNames)


    ##########################
    # _removeFeatureNameAndShift() #
    ##########################

    @raises(ArgumentException)
    def test_removeFeatureNameAndShift_exceptionNoneInput(self):
        """ Test _removeFeatureNameAndShift() for ArgumentException when the identifier is None"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest._removeFeatureNameAndShift(None)

    @raises(ArgumentException)
    def test_removeFeatureNameAndShift_exceptionWrongTypeInput(self):
        """ Test _removeFeatureNameAndShift() for ArgumentException when given the wrong type"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest._removeFeatureNameAndShift(0.3)

    @raises(ArgumentException)
    def test_removeFeatureNameAndShift_exceptionInvalidIndex(self):
        """ Test _removeFeatureNameAndShift() for ArgumentException when given an invalid index"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest._removeFeatureNameAndShift(12)

    @raises(ArgumentException)
    def test_removeFeatureNameAndShift_exceptionFeatureNameNotFound(self):
        """ Test _removeFeatureNameAndShift() for ArgumentException when the featureName is not found"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest._removeFeatureNameAndShift("bogus")

    def test_removeFeatureNameAndShift_handmade_viaIndex(self):
        """ Test _removeFeatureNameAndShift() against handmade output """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest._removeFeatureNameAndShift(0)
        toTest._removeFeatureNameAndShift(2)
        expectedFeatureNames = ["one", "two"]
        confirmExpectedNames(toTest, 'feature', expectedFeatureNames)

    def test_removeFeatureNameAndShift_handmade_viaFeatureName(self):
        """ Test _removeFeatureNameAndShift() against handmade output """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest._removeFeatureNameAndShift('zero')
        toTest._removeFeatureNameAndShift('two')
        expectedFeatureNames = ["one", "three"]
        confirmExpectedNames(toTest, 'feature', expectedFeatureNames)


    ##################
    # _equalPointNames() #
    ##################

    def test__equalPointNames_False(self):
        """ Test _equalPointNames() against some non-equal input """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        assert not toTest._equalPointNames(None)
        assert not toTest._equalPointNames(5)
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
        toTest1.setPointName(0, None)
        toTest1.setPointName(1, None)
        toTest2.setPointName(0, None)
        toTest2.setPointName(1, None)
        toTest2.setPointName(2, None)
        assert not toTest1._equalPointNames(toTest2)
        assert not toTest2._equalPointNames(toTest1)

    def test__equalPointNames_ignoresDefaults(self):
        """ Test _equalPointNames() for equality of default named objects """
        toTest1 = self.constructor(pointNames=['1', '2'])
        toTest2 = self.constructor(pointNames=['1', '2'])
        toTest1.setPointName(0, None)
        toTest1.setPointName(1, None)
        toTest2.setPointName(0, None)
        toTest2.setPointName(1, None)
        assert toTest1._equalPointNames(toTest2)
        assert toTest2._equalPointNames(toTest1)

    def test__equalPointNames_mixedDefaultsAndActual(self):
        toTest1 = self.constructor(pointNames=['1', '2'])
        toTest2 = self.constructor(pointNames=['1', '2'])
        toTest1.setPointName(0, None)
        toTest1.setPointName(1, '1')
        toTest2.setPointName(1, None)
        # have: test1 [Default, '1']
        # test2 ['1', Default]
        assert not toTest1._equalPointNames(toTest2)
        assert not toTest2._equalPointNames(toTest1)
        toTest1.setPointName(1, '2')
        toTest1.setPointName(1, None)
        # have: test1 [Default, '2']
        # test2: ['1', Default]
        assert not toTest1._equalPointNames(toTest2)
        assert not toTest2._equalPointNames(toTest1)


    ##################
    # _equalFeatureNames() #
    ##################

    def test_equalFeatureNames_False(self):
        """ Test _equalFeatureNames() against some non-equal input """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        assert not toTest._equalFeatureNames(None)
        assert not toTest._equalFeatureNames(5)
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
        toTest1.setFeatureName(0, None)
        toTest1.setFeatureName(1, None)
        toTest2.setFeatureName(0, None)
        toTest2.setFeatureName(1, None)
        toTest2.setFeatureName(2, None)
        assert not toTest1._equalFeatureNames(toTest2)
        assert not toTest2._equalFeatureNames(toTest1)

    def test_equalFeatureNames_ignoresDefaults(self):
        """ Test _equalFeatureNames() for equality of default named objects """
        toTest1 = self.constructor(featureNames=['1', '2'])
        toTest2 = self.constructor(featureNames=['1', '2'])
        toTest1.setFeatureName(0, None)
        toTest1.setFeatureName(1, None)
        toTest2.setFeatureName(0, None)
        toTest2.setFeatureName(1, None)
        assert toTest1._equalFeatureNames(toTest2)
        assert toTest2._equalFeatureNames(toTest1)

    def test_equalFeatureNames_mixedDefaultsAndActual(self):
        toTest1 = self.constructor(featureNames=['1', '2'])
        toTest2 = self.constructor(featureNames=['1', '2'])
        toTest1.setFeatureName(0, None)
        toTest1.setFeatureName(1, '1')
        toTest2.setFeatureName(1, None)
        # have: test1 [Default, '1']
        # test2 ['1', Default]
        assert not toTest1._equalFeatureNames(toTest2)
        assert not toTest2._equalFeatureNames(toTest1)
        toTest1.setFeatureName(1, '2')
        toTest1.setFeatureName(1, None)
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


    #################
    # getPointNames #
    #################

    def test_getPointNames_Empty(self):
        toTest = self.constructor(psize=0, fsize=2)

        ret = toTest.getPointNames()
        assert ret == []

    def test_getPointNames_basic(self):
        pnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(pointNames=pnames, fsize=2)

        ret = toTest.getPointNames()
        assert ret == ['zero', 'one', 'hello']

    def test_getPointNames_mixedDefault(self):
        pnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(pointNames=pnames, fsize=2)
        toTest.setPointName(0, None)

        ret = toTest.getPointNames()
        assert ret[0].startswith(DEFAULT_PREFIX)
        assert ret[1] == 'one'
        assert ret[2] == 'hello'

    def test_getPointNames_unmodifiable(self):
        pnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(pointNames=pnames, fsize=2)

        ret = toTest.getPointNames()

        ret[0] = 'modified'
        toTest.setPointName(1, 'modified')

        assert ret[1] == 'one'
        assert toTest.getPointIndex('zero') == 0
        assert toTest.getPointName(0) == 'zero'


    ###################
    # getFeatureNames #
    ###################

    def test_getFeatureNames_Empty(self):
        toTest = self.constructor(psize=2, fsize=0)

        ret = toTest.getFeatureNames()
        assert ret == []

    def test_getFeatureNames_basic(self):
        fnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(featureNames=fnames, psize=2)

        ret = toTest.getFeatureNames()
        assert ret == ['zero', 'one', 'hello']

    def test_getFeatureNames_mixedDefault(self):
        fnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(featureNames=fnames, psize=2)
        toTest.setFeatureName(0, None)

        ret = toTest.getFeatureNames()
        assert ret[0].startswith(DEFAULT_PREFIX)
        assert ret[1] == 'one'
        assert ret[2] == 'hello'

    def test_getFeatureNames_unmodifiable(self):
        fnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(featureNames=fnames, psize=2)

        ret = toTest.getFeatureNames()

        ret[0] = 'modified'
        toTest.setFeatureName(1, 'modified')

        assert ret[1] == 'one'
        assert toTest.getFeatureIndex('zero') == 0
        assert toTest.getFeatureName(0) == 'zero'


    ################################################################
    # getFeatureIndex, getFeatureName, getPointIndex, getPointName #
    ################################################################

    # consistency checks between all sources of axis name information
    def test_name_index_consistency(self):
        pnames = ['p0', 'p1', 'p2', 'p3', 'p4']
        fnames = ['fa', 'fb', 'fc']

        toTest = self.constructor(featureNames=fnames, pointNames=pnames)

        pByGetAll = toTest.getPointNames()
        pByIndex = [toTest.getPointName(i) for i in range(toTest.points)]
        assert pByIndex == pByGetAll

        pnamesShuffle = pythonRandom.sample(pnames, len(pnames))
        pByName = [toTest.getPointIndex(n) for n in pnamesShuffle]
        pByPyIndex = [pnames.index(n) for n in pnamesShuffle]
        assert pByName == pByPyIndex

        fByGetAll = toTest.getFeatureNames()
        fByIndex = [toTest.getFeatureName(i) for i in range(toTest.features)]
        assert fByIndex == fByGetAll

        fnamesShuffle = pythonRandom.sample(fnames, len(fnames))
        fByName = [toTest.getFeatureIndex(n) for n in fnamesShuffle]
        fByPyIndex = [fnames.index(n) for n in fnamesShuffle]
        assert fByName == fByPyIndex


    ###########
    # __len__ #
    ###########

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

    @raises(ImproperActionException)
    def test_len_exception(self):
        nn = self.constructor(psize=11, fsize=33)
        len(nn)
