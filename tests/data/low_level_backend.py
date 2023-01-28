"""
Unit tests of the low level functions defined by the base representation class.

Since these functions are stand alone in the base class, they can be tested directly.
FeatureName equality is defined using _features_implementation, which is to be defined
in the derived class, so all of these tests make a call to self.constructor instead
of directly instantiating a Base object. This function temporarily fills in
that missing implementation.

Methods tested in this file (none modify the data):
ID, points._nameDifference, features._nameDifference, points._nameIntersection,
features._nameIntersection, points._nameSymmetricDifference,
features._nameSymmetricDifference, points._nameUnion, features._nameUnion,
points.setNames, features.setNames, _removePointNameAndShift, _removeFeatureNameAndShift, 
_equalPointNames, _equalFeatureNames, points.getNames, features.getNames, __len__,
features.getIndex, features.getName, points.getIndex, points.getName,
points.getIndices, features.getIndices, constructIndicesList, copy
features.hasName, points.hasName, __bool__, _treatAs2D
"""

import numpy as np

import nimble
from nimble.core.data import Base
from nimble.core.data import available
from nimble._utility import numpy2DArray
from nimble._utility import pd
from nimble.core.data._dataHelpers import constructIndicesList
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import ImproperObjectAction
from nimble.random import pythonRandom
from tests.helpers import raises, assertCalled
from tests.helpers import logCountAssertionFactory
from tests.helpers import noLogEntryExpected, oneLogEntryExpected

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
        names = toTest.points.names
        namesInv = toTest.points.namesInverse
    else:
        names = toTest.features.names
        namesInv = toTest.features.namesInverse
    if isinstance(expected, list):
        for i in range(len(expected)):
            expectedFeatureName = expected[i]
            if not expectedFeatureName is None:
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

    ############################
    # points._nameDifference() #
    ############################

    @raises(InvalidArgumentType)
    def test_points_nameDifference_exceptionOtherNone(self):
        """ Test points._nameDifference() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest.points._nameDifference(None)

    @raises(InvalidArgumentType)
    def test_points_nameDifference_exceptionWrongType(self):
        """ Test points._nameDifference() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(pointNames=["hello"])
        toTest.points._nameDifference("wrong")

    def test_points_nameDifference_handmade(self):
        """ Test points._nameDifference() against handmade output """
        toTest1 = self.constructor(pointNames=["one", "two", "three"])
        toTest2 = self.constructor(pointNames=["two", "four"])
        results = toTest1.points._nameDifference(toTest2)
        assert "one" in results
        assert "two" not in results
        assert "three" in results
        assert "four" not in results

    ##############################
    # features._nameDifference() #
    ##############################

    @raises(InvalidArgumentType)
    def testfeatures_nameDifference_exceptionOtherNone(self):
        """ Test features._nameDifference() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest.features._nameDifference(None)

    @raises(InvalidArgumentType)
    def testfeatures_nameDifference_exceptionWrongType(self):
        """ Test features._nameDifference() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(featureNames=["hello"])
        toTest.features._nameDifference("wrong")

    def testfeatures_nameDifference_handmade(self):
        """ Test features._nameDifference() against handmade output """
        toTest1 = self.constructor(featureNames=["one", "two", "three"])
        toTest2 = self.constructor(featureNames=["two", "four"])
        results = toTest1.features._nameDifference(toTest2)
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

    #####################################
    # points._nameSymmetricDifference() #
    #####################################

    @raises(InvalidArgumentType)
    def test_points_nameSymmetricDifference_exceptionOtherNone(self):
        """ Test points._nameSymmetricDifference() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest.points._nameSymmetricDifference(None)

    @raises(InvalidArgumentType)
    def test_points_nameSymmetricDifference_exceptionWrongType(self):
        """ Test points._nameSymmetricDifference() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(pointNames=["hello"])
        toTest.points._nameSymmetricDifference("wrong")

    def test_points_nameSymmetricDifference_handmade(self):
        """ Test points._nameSymmetricDifference() against handmade output """
        toTest1 = self.constructor(pointNames=["one", "two", "three"])
        toTest2 = self.constructor(pointNames=["two", "four"])
        results = toTest1.points._nameSymmetricDifference(toTest2)
        assert "one" in results
        assert "two" not in results
        assert "three" in results
        assert "four" in results

    #######################################
    # features._nameSymmetricDifference() #
    #######################################

    @raises(InvalidArgumentType)
    def testfeatures_nameSymmetricDifference_exceptionOtherNone(self):
        """ Test features._nameSymmetricDifference() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest.features._nameSymmetricDifference(None)

    @raises(InvalidArgumentType)
    def testfeatures_nameSymmetricDifference_exceptionWrongType(self):
        """ Test features._nameSymmetricDifference() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(featureNames=["hello"])
        toTest.features._nameSymmetricDifference("wrong")

    def testfeatures_nameSymmetricDifference_handmade(self):
        """ Test features._nameSymmetricDifference() against handmade output """
        toTest1 = self.constructor(featureNames=["one", "two", "three"])
        toTest2 = self.constructor(featureNames=["two", "four"])
        results = toTest1.features._nameSymmetricDifference(toTest2)
        assert "one" in results
        assert "two" not in results
        assert "three" in results
        assert "four" in results

    #######################
    # points._nameUnion() #
    #######################

    @raises(InvalidArgumentType)
    def test_points_nameUnion_exceptionOtherNone(self):
        """ Test points._nameUnion() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(pointNames=["hello"])
        toTest.points._nameUnion(None)

    @raises(InvalidArgumentType)
    def test_points_nameUnion_exceptionWrongType(self):
        """ Test points._nameUnion() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(pointNames=["hello"])
        toTest.points._nameUnion("wrong")

    def test_points_nameUnion_handmade(self):
        """ Test points._nameUnion() against handmade output """
        toTest1 = self.constructor(pointNames=["one", "two", "three"])
        toTest2 = self.constructor(pointNames=["two", "four"])
        results = toTest1.points._nameUnion(toTest2)
        assert "one" in results
        assert "two" in results
        assert "three" in results
        assert "four" in results

    #########################
    # features._nameUnion() #
    #########################

    @raises(InvalidArgumentType)
    def testfeatures_nameUnion_exceptionOtherNone(self):
        """ Test features._nameUnion() for InvalidArgumentType when the other object is None """
        toTest = self.constructor(featureNames=["hello"])
        toTest.features._nameUnion(None)

    @raises(InvalidArgumentType)
    def testfeatures_nameUnion_exceptionWrongType(self):
        """ Test features._nameUnion() for InvalidArgumentType when the other object is not the right type """
        toTest = self.constructor(featureNames=["hello"])
        toTest.features._nameUnion("wrong")

    def testfeatures_nameUnion_handmade(self):
        """ Test features._nameUnion() against handmade output """
        toTest1 = self.constructor(featureNames=["one", "two", "three"])
        toTest2 = self.constructor(featureNames=["two", "four"])
        results = toTest1.features._nameUnion(toTest2)
        assert "one" in results
        assert "two" in results
        assert "three" in results
        assert "four" in results


    #####################
    # points.setNames() #
    #####################
    
    @raises(InvalidArgumentType)
    def test_points_setNames_exceptionPrevWrongType(self):
        """ Test points.setNames() for InvalidArgumentType when given the wrong type for prev"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setNames("New!", oldIdentifiers=0.3)

    @raises(IndexError)
    def test_points_setNames_exceptionPrevInvalidIndex(self):
        """ Test points.setNames() for InvalidArgumentValue when given an invalid prev index"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setNames("New!", oldIdentifiers=12)

    @raises(KeyError)
    def test_points_setNames_exceptionPrevNotFound(self):
        """ Test points.setNames() for InvalidArgumentValue when the prev pointName is not found"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setNames( "New!", oldIdentifiers="Previous!")

    @raises(InvalidArgumentType)
    def test_points_setNames_exceptionNewInvalidType(self):
        """ Test points.setNames() for InvalidArgumentValue when the new pointName is not a string"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setNames(4, oldIdentifiers="three")

    @raises(InvalidArgumentValue)
    def test_points_setNames_exceptionNonUnique(self):
        """ Test points.setNames() for InvalidArgumentValue when a duplicate pointName is given"""
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setNames( "two", oldIdentifiers="three")

    @raises(ImproperObjectAction)
    def test_points_setNames_exceptionNoPoints(self):
        toTest = self.constructor()
        toTest.points.setNames( "2", oldIdentifiers="hello")

    @logCountAssertionFactory(2)
    def test_points_setNames_handmade_viaIndex(self):
        """ Test points.setNames() against handmade input when specifying the pointName by index """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setNames("ZERO", oldIdentifiers=0)
        toTest.points.setNames( "3", oldIdentifiers=3)
        expectedNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'point', expectedNames)
    
    @logCountAssertionFactory(1)
    def test_points_setNames_handmade_multiAssign_viaIndex(self):
        """ Test points.setNames() against handmade input when specifying the pointName by index """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setNames(["ZERO", "3"], oldIdentifiers=[0,3])
        expectedNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'point', expectedNames)
        
    def test_points_setNames_handmade_viaPointName(self):
        """ Test points.setNames() against handmade input when specifying the pointName by name """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setNames("ZERO", oldIdentifiers="zero")
        toTest.points.setNames("3", oldIdentifiers="three")
        expectedNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'point', expectedNames)

    def test_points_setNames_handmade_multiAssign_viaPointName(self):
        """ Test points.setNames() against handmade input when specifying the pointNames by name """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        toTest.points.setNames(["ZERO", "3"], oldIdentifiers=["zero", "three"])
        expectedNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'point', expectedNames)

    def test_points_setNames_NoneOutput(self):
        """ Test points.setNames() return None as output """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(pointNames=origNames)
        ret = toTest.points.setNames("ZERO", oldIdentifiers="zero")

        assert ret is None

    @raises(InvalidArgumentType)
    def test_points_setNames_exceptionWrongTypeObject(self):
        """ Test points.setNames() for InvalidArgumentType a point in point Names is an unexpected type """
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

    def test_points_setNames_calls_valuesToPythonList(self):
        toTest = self.constructor(pointNames=['one', 'two', 'three'])
        # need to use context manager after object creation because
        # Axis.__init__ also calls valuesToPythonList
        with assertCalled(nimble.core.data.axis, 'valuesToPythonList'):
            toTest.points.setNames(('a', 'b', 'c'))

    def test_points_setNames_emptyDataAndList(self):
        """ Test points.setNames() when both the data and the list are empty """
        toTest = self.constructor()
        toAssign = []
        toTest.points.setNames(toAssign)
        assert toTest.points.getNames() == []

    def test_points_setNames_addDefault(self):
        """ Test points.setNames() when given a default pointName """
        toTest = self.constructor(pointNames=["blank", "none", "gone", "hey"])
        newNames = ["zero", "one", "two", None]
        toTest.points.setNames(newNames)
        assert toTest.points.getName(3) is None
        assert toTest.points.getNames()[3] is None

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

        toAssign = [None] * 4
        toAssign[1] = 'one'
        toAssign[3] = 'three'
        ret = toTest.points.setNames(toAssign)

        assert toTest.points.getName(0) is None
        assert toTest.points.getName(1) == 'one'
        assert toTest.points.getName(2) is None
        assert toTest.points.getName(3) == 'three'
        assert ret is None


    #######################
    # features.setNames() #
    #######################
    
    @raises(InvalidArgumentType)
    def test_features_setNames_exceptionPrevWrongType(self):
        """ Test features.setNames() for InvalidArgumentType when given the wrong type for prev"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setNames("New!", oldIdentifiers=0.3)

    @raises(IndexError)
    def test_features_setNames_exceptionPrevInvalidIndex(self):
        """ Test features.setNames() for InvalidArgumentValue when given an invalid prev index"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setNames("New!", oldIdentifiers=12)

    @raises(KeyError)
    def test_features_setNames_exceptionPrevNotFound(self):
        """ Test features.setNames() for InvalidArgumentValue when the prev featureName is not found"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setNames("New!", oldIdentifiers="Previous!")

    @raises(InvalidArgumentType)
    def test_features_setNames_exceptionNewInvalidType(self):
        """ Test features.setNames() for InvalidArgumentValue when the new featureName is not a string"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setNames(4, oldIdentifiers="three")

    @raises(InvalidArgumentValue)
    def test_features_setNames_exceptionNonUnique(self):
        """ Test features.setNames() for InvalidArgumentValue when a duplicate featureName is given"""
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setNames("two", oldIdentifiers="three")

    @raises(ImproperObjectAction)
    def test_features_setNames_exceptionNoFeatures(self):
        toTest = self.constructor()
        toTest.features.setNames("2", oldIdentifiers="hello")

    @logCountAssertionFactory(2)
    def test_features_setNames_handmade_viaIndex(self):
        """ Test features.setNames() against handmade input when specifying the featureName by index """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setNames( "ZERO", oldIdentifiers=0)
        toTest.features.setNames("3", oldIdentifiers=3)
        expectedFeatureNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'feature', expectedFeatureNames)
    
    @logCountAssertionFactory(1)
    def test_features_setNames_handmade_multiAssign_viaIndex(self):
        """ Test features.setNames() against handmade input when specifying the featureName by index """
        origNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origNames)
        toTest.features.setNames(["ZERO", "3"], oldIdentifiers=[0,3])
        expectedNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'feature', expectedNames)
        
    def test_features_setName_handmade_viaFeatureName(self):
        """ Test features.setNames() against handmade input when specifying the featureName by name """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setNames("ZERO", oldIdentifiers="zero")
        toTest.features.setNames("3", oldIdentifiers="three")
        expectedFeatureNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'feature', expectedFeatureNames)

    def test_features_setNames_handmade_multiAssign_viaFeatureName(self):
        """ Test features.setNames() against handmade input when specifying the featureNames by name """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        toTest.features.setNames(["ZERO", "3"], oldIdentifiers=["zero", "three"])
        expectedFeatureNames = ["ZERO", "one", "two", "3"]
        confirmExpectedNames(toTest, 'feature', expectedFeatureNames)

    def test_features_setNames_NoneOutput(self):
        """ Test features.setNames() returns None as output """
        origFeatureNames = ["zero", "one", "two", "three"]
        toTest = self.constructor(featureNames=origFeatureNames)
        ret = toTest.features.setNames("ZERO", oldIdentifiers="zero")
        assert ret is None

    @raises(InvalidArgumentType)
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

    def test_features_setNames_calls_valuesToPythonList(self):
        toTest = self.constructor(featureNames=['one', 'two', 'three'])
        # need to use context manager after object creation because
        # Axis.__init__ also calls valuesToPythonList
        with assertCalled(nimble.core.data.axis, 'valuesToPythonList'):
            toTest.features.setNames(('a', 'b', 'c'))

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
        newFeatureNames = ["zero", "one", "two", None]
        toTest.features.setNames(newFeatureNames)
        assert toTest.features.getName(3) is None
        assert toTest.features.getNames()[3] is None

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

        toAssign = [None] * 4
        toAssign[0] = 'zero'
        toAssign[2] = 'two'

        ret = toTest.features.setNames(toAssign)

        assert toTest.features.getName(0) == 'zero'
        assert toTest.features.getName(1) is None
        assert toTest.features.getName(2) == 'two'
        assert toTest.features.getName(3) is None
        assert ret is None

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
        toTest1.points.setNames(None, oldIdentifiers=0)
        toTest1.points.setNames(None, oldIdentifiers=1)
        toTest2.points.setNames(None, oldIdentifiers=0)
        toTest2.points.setNames(None, oldIdentifiers=1)
        toTest2.points.setNames(None, oldIdentifiers=2)
        assert not toTest1._equalPointNames(toTest2)
        assert not toTest2._equalPointNames(toTest1)

    def test__equalPointNames_ignoresDefaults(self):
        """ Test _equalPointNames() for equality of default named objects """
        toTest1 = self.constructor(pointNames=['1', '2'])
        toTest2 = self.constructor(pointNames=['1', '2'])
        toTest1.points.setNames(None, oldIdentifiers=0)
        toTest1.points.setNames(None, oldIdentifiers=1)
        toTest2.points.setNames(None, oldIdentifiers=0)
        toTest2.points.setNames(None, oldIdentifiers=1)
        assert toTest1._equalPointNames(toTest2)
        assert toTest2._equalPointNames(toTest1)

    def test__equalPointNames_mixedDefaultsAndActual(self):
        toTest1 = self.constructor(pointNames=['1', '2'])
        toTest2 = self.constructor(pointNames=['1', '2'])
        toTest1.points.setNames(None, oldIdentifiers=0)
        toTest1.points.setNames('1', oldIdentifiers=1)
        toTest2.points.setNames(None, oldIdentifiers=1)
        # have: test1 [Default, '1']
        # test2 ['1', Default]
        assert not toTest1._equalPointNames(toTest2)
        assert not toTest2._equalPointNames(toTest1)
        toTest1.points.setNames('2', oldIdentifiers=1)
        toTest1.points.setNames(None, oldIdentifiers=1)
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
        toTest1.features.setNames(None, oldIdentifiers=0)
        toTest1.features.setNames(None, oldIdentifiers=1)
        toTest2.features.setNames(None, oldIdentifiers=0)
        toTest2.features.setNames(None, oldIdentifiers=1)
        toTest2.features.setNames(None, oldIdentifiers=2)
        assert not toTest1._equalFeatureNames(toTest2)
        assert not toTest2._equalFeatureNames(toTest1)

    def test_equalFeatureNames_ignoresDefaults(self):
        """ Test _equalFeatureNames() for equality of default named objects """
        toTest1 = self.constructor(featureNames=['1', '2'])
        toTest2 = self.constructor(featureNames=['1', '2'])
        toTest1.features.setNames(None, oldIdentifiers=0)
        toTest1.features.setNames(None, oldIdentifiers=1)
        toTest2.features.setNames(None, oldIdentifiers=0)
        toTest2.features.setNames(None, oldIdentifiers=1)
        assert toTest1._equalFeatureNames(toTest2)
        assert toTest2._equalFeatureNames(toTest1)

    def test_equalFeatureNames_mixedDefaultsAndActual(self):
        toTest1 = self.constructor(featureNames=['1', '2'])
        toTest2 = self.constructor(featureNames=['1', '2'])
        toTest1.features.setNames(None, oldIdentifiers=0)
        toTest1.features.setNames('1', oldIdentifiers=1)
        toTest2.features.setNames(None, oldIdentifiers=1)
        # have: test1 [Default, '1']
        # test2 ['1', Default]
        assert not toTest1._equalFeatureNames(toTest2)
        assert not toTest2._equalFeatureNames(toTest1)
        toTest1.features.setNames('2', oldIdentifiers=1)
        toTest1.features.setNames(None, oldIdentifiers=1)
        # have: test1 [Default, '2']
        # test2: ['1', Default]
        assert not toTest1._equalFeatureNames(toTest2)
        assert not toTest2._equalFeatureNames(toTest1)


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
        toTest.points.setNames(None, oldIdentifiers=0)

        ret = toTest.points.getNames()
        assert ret[0] is None
        assert ret[1] == 'one'
        assert ret[2] == 'hello'

    def test_points_getNames_unmodifiable(self):
        pnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(pointNames=pnames, fsize=2)

        ret = toTest.points.getNames()

        ret[0] = 'modified'
        toTest.points.setNames('modified', oldIdentifiers=1)

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
        toTest.features.setNames(None, oldIdentifiers=0)

        ret = toTest.features.getNames()
        assert ret[0] is None
        assert ret[1] == 'one'
        assert ret[2] == 'hello'

    def test_features_getNames_unmodifiable(self):
        fnames = {'zero': 0, 'one': 1, 'hello': 2}
        toTest = self.constructor(featureNames=fnames, psize=2)

        ret = toTest.features.getNames()

        ret[0] = 'modified'
        toTest.features.setNames('modified', oldIdentifiers=1)

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

    def test_constructIndicesList_calls_valuesToPythonList(self):
        pointNames = ['p1','p2','p3']
        toTest = self.constructor(pointNames=pointNames)
        # need to use context manager after object creation because
        # Axis.__init__ also calls valuesToPythonList
        with assertCalled(nimble.core.data._dataHelpers, 'valuesToPythonList'):
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
                lambda lst: nimble.data(lst, convertToType=object))

    def testconstructIndicesList_numpyArray(self):
        self.constructIndicesList_backend(lambda lst: np.array(lst,dtype=object))

    def testconstructIndicesList_pandasSeries(self):
        self.constructIndicesList_backend(lambda lst: pd.Series(lst))

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
        self.constructIndicesList_backend(lambda lst: pd.DataFrame(lst))

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

        array2D = np.array([[1,2,3],[4,5,6]])

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
        assert toTest3D._dims == [3, 3, 5]
        assert toTest3D.dimensions == (3, 3, 5)
        assert toTest3D.shape == (3, 15)
        assert len(toTest3D.points) == 3
        assert len(toTest3D.features) == 15

        toTest4D = self.constructor((4, 3, 3, 5))
        assert toTest4D._dims == [4, 3, 3, 5]
        assert toTest4D.dimensions == (4, 3, 3, 5)
        assert toTest4D.shape == (4, 45)
        assert len(toTest4D.points) == 4
        assert len(toTest4D.features) == 45

        toTest3DEmpty = self.constructor((0, 0, 0))
        assert toTest3DEmpty._dims == [0, 0, 0]
        assert toTest3DEmpty.dimensions == (0, 0, 0)
        assert toTest3DEmpty.shape == (0, 0)
        assert len(toTest3DEmpty.points) == 0
        assert len(toTest3DEmpty.features) == 0

    def test_highDimension_namesAndIndices(self):
        pNames = ['p1', 'p2', 'p3']
        fNames = ['f' + str(i) for i in range(1, 16)]
        toTest3D = self.constructor(shape=(3, 3, 5), pointNames=pNames,
                                    featureNames=fNames)
        assert toTest3D.name is None
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

        toTest3D.points.setNames('z', oldIdentifiers='a')
        assert not toTest3D.points.hasName('a')
        assert toTest3D.points.hasName('z')
        assert toTest3D.points.getName(0) == 'z'
        assert toTest3D.points.getIndex('z') == 0

        toTest3D.features.setNames('ft_first', oldIdentifiers='ft_0')
        assert not toTest3D.features.hasName('ft_0')
        assert toTest3D.features.hasName('ft_first')
        assert toTest3D.features.getName(0) == 'ft_first'
        assert toTest3D.features.getIndex('ft_first') == 0

    def test_highDimension_len(self):
        tensor3D = self.constructor((3, 3, 5))
        tensor4D = self.constructor((4, 3, 3, 5))
        tensor5D = self.constructor((5, 4, 3, 3, 5))
        for tensor in [tensor3D, tensor4D, tensor5D]:
            with raises(ImproperObjectAction):
                len(tensor)

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

    def test_treatAs2D(self):
        tensor2D = self.constructor((3, 5))
        origShape = tensor2D._dims
        # 2D _treatAs2D context manager without as
        assert len(tensor2D._dims) == 2
        with tensor2D._treatAs2D():
            assert len(tensor2D._dims) == 2
            assert tensor2D._dims == origShape
        assert len(tensor2D._dims) == 2
        assert tensor2D._dims == origShape

        # 2D _treatAs2D context manager with as
        with tensor2D._treatAs2D() as tensor:
            assert len(tensor._dims) == 2
            assert tensor._dims == origShape
        assert len(tensor2D._dims) == 2
        assert tensor2D._dims == origShape

        # 2D _treatAs2D context manager encounters error
        try:
            with tensor2D._treatAs2D():
                raise TypeError()
        except TypeError:
            assert len(tensor2D._dims) == 2
            assert tensor2D._dims == origShape


        tensor3D = self.constructor((3, 3, 5))
        tensor4D = self.constructor((4, 3, 3, 5))
        tensor5D = self.constructor((5, 4, 3, 3, 5))
        for tensor in [tensor3D, tensor4D, tensor5D]:
            origShape = tensor._dims
            shape2D = tensor.shape
            # 3D+ _treatAs2D context manager without as
            assert len(tensor._dims) > 2
            with tensor._treatAs2D():
                assert len(tensor._dims) == 2
                assert tensor.shape == tuple(tensor._dims)
                assert tensor._dims != origShape
            assert len(tensor._dims) > 2
            assert tensor._dims == origShape

            # 3D+ _treatAs2D context manager with as
            with tensor._treatAs2D() as tensor2D:
                assert len(tensor._dims) == 2
                assert tensor.shape == tuple(tensor._dims)
                assert tensor._dims != origShape
            assert len(tensor._dims) > 2
            assert tensor._dims == origShape

            # 3D+ _treatAs2D context manager encounters error
            try:
                with tensor._treatAs2D():
                    raise TypeError()
            except TypeError:
                assert len(tensor._dims) > 2
                assert tensor._dims == origShape

    ########
    # Name #
    ########

    def test_name_change_and_lookup(self):
        origName = 'one'
        changeName = 'two'

        test = self.constructor(name=origName)
        assert test.name == origName

        test.name = changeName
        assert test.name == changeName
