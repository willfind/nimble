"""
Unit tests of the low level functions defined by the base representation class.

Since these functions are stand alone in the base class, they can be tested directly.
FeatureName equality is defined using _features_implementation, which is to be defined
in the derived class, so all of these tests make a call to self.constructor instead
of directly instantiating a Base object. This function temporarily fills in
that missing implementation.

"""

from UML.data import Base
from UML.data.dataHelpers import DEFAULT_PREFIX
from UML.data.dataHelpers import DEFAULT_NAME_PREFIX
from nose.tools import *
from UML.exceptions import ArgumentException
from UML.exceptions import ImproperActionException



###########
# helpers #
###########


def confirmExpectedFeatureNames(toTest, expected):
	if isinstance(expected, list):
		for i in range(len(expected)):
			expectedFeatureName = expected[i]
			if not expectedFeatureName.startswith(DEFAULT_PREFIX):	
				actualIndex = toTest.featureNames[expectedFeatureName]
				actualFeatureName = toTest.featureNamesInverse[i]
				assert (actualIndex == i)
				assert (actualFeatureName == expectedFeatureName)
	else:
		for k in expected.keys():
			kIndex = expected[k]
			assert k in toTest.featureNames
			actualIndex = toTest.featureNames[k]
			actualFeatureName = toTest.featureNamesInverse[kIndex]
			assert (actualIndex == kIndex)
			assert (actualFeatureName == k)



class LowLevelBackend(object):

	def __init__(self, constructor):
		self.constructor = constructor
#		super(LowLevelBackend, self).__init__()

	###############
	# _addFeatureName() #
	###############

	@raises(ArgumentException)
	def test_addFeatureName_exceptionFeatureNameWrongType(self):
		""" Test _addFeatureName() for ArgumentException when given a non string featureName """
		toTest = self.constructor(["hello"])
		toTest._addFeatureName(featureName=34)

	@raises(ArgumentException)
	def test_addFeatureName_exceptionNonUnique(self):
		""" Test _addFeatureName() for ArgumentException when given a duplicate featureName """
		toTest = self.constructor(["hello"])
		toTest._addFeatureName("hello")

	def test_addFeatureName_handmadeDefaultCounter(self):
		""" Test _addFeatureName() changes nextDefault counter correctly """
		toTest = self.constructor(["hello"])
		assert toTest._nextDefaultValue == 1
		toTest._addFeatureName(DEFAULT_PREFIX +"1")
		assert toTest._nextDefaultValue == 2

	def test_addFeatureName_handmade(self):
		""" Test _addFeatureName() against handmade output """
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest._addFeatureName("four")
		toTest._addFeatureName("five")

		expected = ["zero","one","two","three","four","five"]
		confirmExpectedFeatureNames(toTest,expected)

	def test_addFeatureName_handmade_addDefault(self):
		""" Test _addFeatureName() against handmade output """
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest._addFeatureName(None)
		toTest._addFeatureName(None)

		expected = ["zero","one","two","three",DEFAULT_PREFIX,DEFAULT_PREFIX]

		confirmExpectedFeatureNames(toTest,expected)	

	#####################
	# _featureNameDifference() #
	#####################

	@raises(ArgumentException)
	def test_featureNameDifference_exceptionOtherNone(self):
		""" Test _featureNameDifference() for ArgumentException when the other object is None """
		toTest = self.constructor(["hello"])
		toTest._featureNameDifference(None)

	@raises(ArgumentException)
	def test_featureNameDifference_exceptionWrongType(self):
		""" Test _featureNameDifference() for ArgumentException when the other object is not the right type """
		toTest = self.constructor(["hello"])
		toTest._featureNameDifference("wrong")

	def test_featureNameDifference_handmade(self):
		""" Test _featureNameDifference() against handmade output """
		toTest1 = self.constructor(["one","two","three"])
		toTest2 = self.constructor(["two","four"])
		results = toTest1._featureNameDifference(toTest2)
		assert "one" in results
		assert "two" not in results
		assert "three" in results
		assert "four" not in results


	#######################
	# _featureNameIntersection() #
	#######################

	@raises(ArgumentException)
	def test_featureNameIntersection_exceptionOtherNone(self):
		""" Test _featureNameIntersection() for ArgumentException when the other object is None """
		toTest = self.constructor(["hello"])
		toTest._featureNameIntersection(None)

	@raises(ArgumentException)
	def test_featureNameIntersection_exceptionWrongType(self):
		""" Test _featureNameIntersection() for ArgumentException when the other object is not the right type """
		toTest = self.constructor(["hello"])
		toTest._featureNameIntersection("wrong")

	def test_featureNameIntersection_handmade(self):
		""" Test _featureNameIntersection() against handmade output """
		toTest1 = self.constructor(["one","two","three"])
		toTest2 = self.constructor(["two","four"])
		results = toTest1._featureNameIntersection(toTest2)
		assert "one" not in results
		assert "two" in results
		assert "three" not in results
		assert "four" not in results


	##############################
	# _featureNameSymmetricDifference() #
	##############################

	@raises(ArgumentException)
	def test_featureNameSymmetricDifference_exceptionOtherNone(self):
		""" Test _featureNameSymmetricDifference() for ArgumentException when the other object is None """
		toTest = self.constructor(["hello"])
		toTest._featureNameSymmetricDifference(None)

	@raises(ArgumentException)
	def test_featureNameSymmetricDifference_exceptionWrongType(self):
		""" Test _featureNameSymmetricDifference() for ArgumentException when the other object is not the right type """
		toTest = self.constructor(["hello"])
		toTest._featureNameSymmetricDifference("wrong")

	def test_featureNameSymmetricDifference_handmade(self):
		""" Test _featureNameSymmetricDifference() against handmade output """
		toTest1 = self.constructor(["one","two","three"])
		toTest2 = self.constructor(["two","four"])
		results = toTest1._featureNameSymmetricDifference(toTest2)
		assert "one" in results
		assert "two" not in results
		assert "three" in results
		assert "four" in results

	################
	# _featureNameUnion() #
	################

	@raises(ArgumentException)
	def test_featureNameUnion_exceptionOtherNone(self):
		""" Test _featureNameUnion() for ArgumentException when the other object is None """
		toTest = self.constructor(["hello"])
		toTest._featureNameUnion(None)

	@raises(ArgumentException)
	def test_featureNameUnion_exceptionWrongType(self):
		""" Test _featureNameUnion() for ArgumentException when the other object is not the right type """
		toTest = self.constructor(["hello"])
		toTest._featureNameUnion("wrong")

	def test_featureNameUnion_handmade(self):
		""" Test _featureNameUnion() against handmade output """
		toTest1 = self.constructor(["one","two","three"])
		toTest2 = self.constructor(["two","four"])
		results = toTest1._featureNameUnion(toTest2)
		assert "one" in results
		assert "two" in results
		assert "three" in results
		assert "four" in results

	#################
	# setFeatureName() #
	#################

	@raises(ArgumentException)
	def test_setFeatureName_exceptionPrevWrongType(self):
		""" Test setFeatureName() for ArgumentException when given the wrong type for prev"""
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest.setFeatureName(oldIdentifier=0.3,newFeatureName="New!")

	@raises(ArgumentException)
	def test_setFeatureName_exceptionPrevInvalidIndex(self):
		""" Test setFeatureName() for ArgumentException when given an invalid prev index"""
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest.setFeatureName(oldIdentifier=12,newFeatureName="New!")

	@raises(ArgumentException)
	def test_setFeatureName_exceptionPrevNotFound(self):
		""" Test setFeatureName() for ArgumentException when the prev featureName is not found"""
		origFeatureNames = ["zero","one","two","three"]	
		toTest =self.constructor(origFeatureNames)
		toTest.setFeatureName(oldIdentifier="Previous!",newFeatureName="New!")

	@raises(ArgumentException)
	def test_setFeatureName_exceptionNewInvalidType(self):
		""" Test setFeatureName() for ArgumentException when the new featureName is not a string"""
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest.setFeatureName(oldIdentifier="three",newFeatureName=4)

	@raises(ArgumentException)
	def test_setFeatureName_exceptionNonUnique(self):
		""" Test setFeatureName() for ArgumentException when a duplicate featureName is given"""
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest.setFeatureName(oldIdentifier="three",newFeatureName="two")

	#@raises(ArgumentException)
	#def test_setFeatureName_exceptionManualAddDefault(self):
	#	""" Test setFeatureName() for ArgumentException when given a default featureName """
	#	toTest = self.constructor(["hello"])
	#	toTest.setFeatureName("hello",DEFAULT_PREFIX + "2")

	@raises(ArgumentException)
	def test_setFeatureName_exceptionNoFeatures(self):
		toTest = self.constructor()
		toTest.setFeatureName("hello","2")

	def test_setFeatureName_handmade_viaIndex(self):
		""" Test setFeatureName() against handmade input when specifying the featureName by index """
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		ret0 = toTest.setFeatureName(0,"ZERO")
		ret3 = toTest.setFeatureName(3,"3")
		expectedFeatureNames = ["ZERO","one","two","3"]
		confirmExpectedFeatureNames(toTest,expectedFeatureNames)
		assert toTest == ret0
		assert toTest == ret3

	def test_setFeatureName_handmade_viaFeatureName(self):
		""" Test setFeatureName() against handmade input when specifying the featureName by name """
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		ret0 = toTest.setFeatureName("zero","ZERO")
		ret3 = toTest.setFeatureName("three","3")
		expectedFeatureNames = ["ZERO","one","two","3"]
		confirmExpectedFeatureNames(toTest,expectedFeatureNames)
		assert toTest == ret0
		assert toTest == ret3


	##########################
	# setFeatureNamesFromList() #
	##########################

	@raises(ArgumentException)
	def test_setFeatureNamesFromList_exceptionWrongTypeObject(self):
		""" Test setFeatureNamesFromList() for ArgumentException when featureNames is an unexpected type """
		toTest = self.constructor(['one'])
		toTest.setFeatureNamesFromList(12)

	@raises(ArgumentException)
	def test_setFeatureNamesFromList_exceptionNonStringFeatureNameInList(self):
		""" Test setFeatureNamesFromList() for ArgumentException when a list element is not a string """
		toTest = self.constructor(['one'])
		nonStringFeatureNames = [1,2,3]
		toTest.setFeatureNamesFromList(nonStringFeatureNames)

	@raises(ArgumentException)
	def test_setFeatureNamesFromList_exceptionNonUniqueStringInList(self):
		""" Test setFeatureNamesFromList() for ArgumentException when a list element is not unique """
		toTest = self.constructor(['one'])
		nonUnique = ['1','2','3','1']
		toTest.setFeatureNamesFromList(nonUnique)

	@raises(ArgumentException)
	def test_setFeatureNamesFromList_exceptionNoFeatures(self):
		""" Test setFeatureNamesFromList() for ArgumentException when there are no features to name """
		toTest = self.constructor()
		toAssign = ["hey","gone","none","blank"]
		toTest.setFeatureNamesFromList(toAssign)

	def test_setFeatureNamesFromList_emptyDataAndList(self):
		""" Test setFeatureNamesFromList() when both the data and the list are empty """
		toTest = self.constructor()
		toAssign = []
		ret = toTest.setFeatureNamesFromList(toAssign)
		assert toTest == ret

	def test_setFeatureNamesFromList_addDefault(self):
		""" Test setFeatureNamesFromList() when given a default featureName """
		toTest = self.constructor(["blank","none","gone","hey"])
		newFeatureNames = ["zero","one","two",DEFAULT_PREFIX + "17"]
		ret = toTest.setFeatureNamesFromList(newFeatureNames)
		assert toTest._nextDefaultValue > 17
		assert toTest == ret

	def test_setFeatureNamesFromList_handmade(self):
		""" Test setFeatureNamesFromList() against handmade output """
		toTest = self.constructor(["blank","none","gone","hey"])
		origFeatureNames = ["zero","one","two","three"]	
		ret = toTest.setFeatureNamesFromList(origFeatureNames)
		confirmExpectedFeatureNames(toTest,origFeatureNames)
		assert toTest == ret

	def test_setFeatureNamesFromList_handmadeReplacingWithSame(self):
		""" Test setFeatureNamesFromList() against handmade output when you're replacing the position of featureNames """
		toTest = self.constructor(["blank","none","gone","hey"])
		toAssign = ["hey","gone","none","blank"]
		ret = toTest.setFeatureNamesFromList(toAssign)
		confirmExpectedFeatureNames(toTest,toAssign)
		assert toTest == ret

	##########################
	# setFeatureNamesFromDict() #
	##########################

	@raises(ArgumentException)
	def test_setFeatureNamesFromDict_exceptionWrongTypeObject(self):
		""" Test setFeatureNamesFromDict() for ArgumentException when featureNames is an unexpected type """
		toTest = self.constructor(['one'])
		toTest.setFeatureNamesFromDict(12)

	@raises(ArgumentException)
	def test_setFeatureNamesFromDict_exceptionNonStringFeatureNameInDict(self):
		""" Test setFeatureNamesFromDict() for ArgumentException when a dict key is not a string """
		toTest = self.constructor(['one'])
		nonStringFeatureNames = {1:1}
		toTest.setFeatureNamesFromDict(nonStringFeatureNames)

	@raises(ArgumentException)
	def test_setFeatureNamesFromDict_exceptionNonIntIndexInDict(self):
		""" Test setFeatureNamesFromDict() for ArgumentException when a dict value is not an int """
		toTest = self.constructor(['one'])
		nonIntIndex = {"one":"one"}
		toTest.setFeatureNamesFromDict(nonIntIndex)

	@raises(ArgumentException)
	def test_setFeatureNamesFromDict_exceptionNoFeatures(self):
		""" Test setFeatureNamesFromDict() for ArgumentException when there are no features to name """
		toTest = self.constructor([])
		toAssign = {"hey":0,"gone":1,"none":2,"blank":3}
		toTest.setFeatureNamesFromDict(toAssign)

	def test_setFeatureNamesFromDict_emptyDataAndList(self):
		""" Test setFeatureNamesFromDict() when both the data and the list are empty """
		toTest = self.constructor()
		toAssign = {}
		ret = toTest.setFeatureNamesFromDict(toAssign)	
		assert toTest == ret

	def test_setFeatureNamesFromDict_handmade(self):
		""" Test setFeatureNamesFromDict() against handmade output """
		toTest = self.constructor(["blank","none","gone","hey"])
		origFeatureNames = {"zero":0,"one":1,"two":2,"three":3}	
		ret = toTest.setFeatureNamesFromDict(origFeatureNames)
		confirmExpectedFeatureNames(toTest,origFeatureNames)
		assert toTest == ret 

	def test_setFeatureNamesFromDict_handmadeReplacingWithSame(self):
		""" Test setFeatureNamesFromDict() against handmade output when you're replacing the position of featureNames """
		toTest = self.constructor(["blank","none","gone","hey"])
		toAssign = {"hey":0,"gone":1,"none":2,"blank":3}
		ret = toTest.setFeatureNamesFromDict(toAssign)
		confirmExpectedFeatureNames(toTest,toAssign)
		assert toTest == ret 


	##########################
	# _removeFeatureNameAndShift() #
	##########################

	@raises(ArgumentException)
	def test_removeFeatureNameAndShift_exceptionNoneInput(self):
		""" Test _removeFeatureNameAndShift() for ArgumentException when the identifier is None"""
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest._removeFeatureNameAndShift(None)

	@raises(ArgumentException)
	def test_removeFeatureNameAndShift_exceptionWrongTypeInput(self):
		""" Test _removeFeatureNameAndShift() for ArgumentException when given the wrong type"""
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest._removeFeatureNameAndShift(0.3)

	@raises(ArgumentException)
	def test_removeFeatureNameAndShift_exceptionInvalidIndex(self):
		""" Test _removeFeatureNameAndShift() for ArgumentException when given an invalid index"""
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest._removeFeatureNameAndShift(12)

	@raises(ArgumentException)
	def test_removeFeatureNameAndShift_exceptionFeatureNameNotFound(self):
		""" Test _removeFeatureNameAndShift() for ArgumentException when the featureName is not found"""
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest._removeFeatureNameAndShift("bogus")

	def test_removeFeatureNameAndShift_handmade_viaIndex(self):
		""" Test _removeFeatureNameAndShift() against handmade output """
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest._removeFeatureNameAndShift(0)
		toTest._removeFeatureNameAndShift(2)
		expectedFeatureNames = ["one","two"]
		confirmExpectedFeatureNames(toTest,expectedFeatureNames)

	def test_removeFeatureNameAndShift_handmade_viaFeatureName(self):
		""" Test _removeFeatureNameAndShift() against handmade output """
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		toTest._removeFeatureNameAndShift('zero')
		toTest._removeFeatureNameAndShift('two')
		expectedFeatureNames = ["one","three"]
		confirmExpectedFeatureNames(toTest,expectedFeatureNames)



	##################
	# _equalFeatureNames() #
	##################

	def test_equalFeatureNames_False(self):
		""" Test _equalFeatureNames() against some non-equal input """
		origFeatureNames = ["zero","one","two","three"]	
		toTest = self.constructor(origFeatureNames)
		assert not toTest._equalFeatureNames(None)
		assert not toTest._equalFeatureNames(5)
		assert not toTest._equalFeatureNames(self.constructor(["short","list"]))

		subset = self.constructor(["zero","one","two"])
		superset = self.constructor(["zero","one","two","three","four"])
		assert not toTest._equalFeatureNames(subset)
		assert not toTest._equalFeatureNames(superset)
		assert not subset._equalFeatureNames(toTest)
		assert not superset._equalFeatureNames(toTest)

	def test_equalFeatureNames_actuallyEqual(self):
		""" Test _equalFeatureNames() against some actually equal input """
		origFeatureNames = ["zero","one","two","three"]
		featureNamesDict = {"zero":0,"one":1,"two":2,"three":3}
		toTest1 = self.constructor(origFeatureNames)
		toTest2 = self.constructor(featureNamesDict)
		assert toTest1._equalFeatureNames(toTest2)
		assert toTest2._equalFeatureNames(toTest1)

	def test_equalFeatureNames_noData(self):
		""" Test _equalFeatureNames() for empty objects """
		toTest1 = self.constructor(None)
		toTest2 = self.constructor(None)
		assert toTest1._equalFeatureNames(toTest2)
		assert toTest2._equalFeatureNames(toTest1)

	def test_equalFeatureNames_DefaultInequality(self):
		""" Test _equalFeatureNames() for inequality of some default named, different sized objects """
		toTest1 = self.constructor(['1','2'])
		toTest2 = self.constructor(['1','2','3'])
		toTest1.setFeatureName(0,None)
		toTest1.setFeatureName(1,None)
		toTest2.setFeatureName(0,None)
		toTest2.setFeatureName(1,None)
		toTest2.setFeatureName(2,None)
		assert not toTest1._equalFeatureNames(toTest2)
		assert not toTest2._equalFeatureNames(toTest1)

	def test_equalFeatureNames_ignoresDefaults(self):
		""" Test _equalFeatureNames() for equality of default named objects """
		toTest1 = self.constructor(['1','2'])
		toTest2 = self.constructor(['1','2'])
		toTest1.setFeatureName(0,None)
		toTest1.setFeatureName(1,None)
		toTest2.setFeatureName(0,None)
		toTest2.setFeatureName(1,None)
		assert toTest1._equalFeatureNames(toTest2)
		assert toTest2._equalFeatureNames(toTest1)

	def test_equalFeatureNames_mixedDefaultsAndActual(self):
		""" Test _equalFeatureNames() for equality of default named objects mixed with actual featureNames """
		toTest1 = self.constructor(['1','2'])
		toTest2 = self.constructor(['1','2'])
		toTest1.setFeatureName(0,None)
		toTest2.setFeatureName(1,None)
		assert not toTest1._equalFeatureNames(toTest2)
		assert not toTest2._equalFeatureNames(toTest1)
		toTest1.setFeatureName(0,'1')
		toTest1.setFeatureName(1,None)
		print toTest1.featureNames
		print toTest2.featureNames
		assert toTest1._equalFeatureNames(toTest2)
		assert toTest2._equalFeatureNames(toTest1)


	########################
	# default Object names #
	########################

	def test_default_object_names(self):
		""" Test that default object names increment correctly """
		toTest0 = self.constructor(['a','2'])
		toTest1 = self.constructor(['1b','2'])
		toTest2 = self.constructor(['c','2'])

		firstNumber = int(toTest0.name[len(DEFAULT_NAME_PREFIX):])
		second = firstNumber + 1
		third = second + 1

		assert toTest1.name == DEFAULT_NAME_PREFIX + str(second)
		assert toTest2.name == DEFAULT_NAME_PREFIX + str(third)
