"""
Unit tests of the low level functions defined by the base representation class.

Since these functions are stand alone in the base class, they can be tested directly.
FeatureName equality is defined using _features_implementation, which is to be defined
in the derived class, so all of these tests make a call to makeAndDefine instead
of directly instantiating a BaseData object. This function temporarily fills in
that missing implementation.

"""

from UML.processing.base_data import *
from nose.tools import *
from UML.exceptions import ArgumentException

def makeConst(num):
	def const(dummy=2):
		return num
	return const

def makeAndDefine(featureNames=None, size=0):
	""" Make a base data object that will think it has as many features as it has featureNames,
	even though it has no actual data """
	cols = size if featureNames is None else len(featureNames)
	specificImp = makeConst(cols)
	BaseData._features_implementation = specificImp
	ret = BaseData(featureNames)
	ret._features_implementation = specificImp
	return ret


########
# init #
########

#@raises(ArgumentException)
#def test_init_exceptionNoFeatureNamesOrCount():
#	""" Test init() for ArgumentException when called without featureNames or a count of features """
#	toTest= BaseData(None)

###############
# _addFeatureName() #
###############

@raises(ArgumentException)
def test_addFeatureName_exceptionFeatureNameWrongType():
	""" Test _addFeatureName() for ArgumentException when given a non string featureName """
	toTest = makeAndDefine(["hello"])
	toTest._addFeatureName(featureName=34)

@raises(ArgumentException)
def test_addFeatureName_exceptionNonUnique():
	""" Test _addFeatureName() for ArgumentException when given a duplicate featureName """
	toTest = makeAndDefine(["hello"])
	toTest._addFeatureName("hello")

def test_addFeatureName_handmadeDefaultCounter():
	""" Test _addFeatureName() changes nextDefault counter correctly """
	toTest = makeAndDefine(["hello"])
	assert toTest._nextDefaultValue == 1
	toTest._addFeatureName(DEFAULT_PREFIX +"1")
	assert toTest._nextDefaultValue == 2

def test_addFeatureName_handmade():
	""" Test _addFeatureName() against handmade output """
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest._addFeatureName("four")
	toTest._addFeatureName("five")

	expected = ["zero","one","two","three","four","five"]
	confirmExpectedFeatureNames(toTest,expected)

def test_addFeatureName_handmade_addDefault():
	""" Test _addFeatureName() against handmade output """
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest._addFeatureName(None)
	toTest._addFeatureName(None)

	expected = ["zero","one","two","three",DEFAULT_PREFIX,DEFAULT_PREFIX]

	confirmExpectedFeatureNames(toTest,expected)	

#####################
# featureNameDifference() #
#####################

@raises(ArgumentException)
def test_featureNameDifference_exceptionOtherNone():
	""" Test featureNameDifference() for ArgumentException when the other object is None """
	toTest = makeAndDefine(["hello"])
	toTest.featureNameDifference(None)

@raises(ArgumentException)
def test_featureNameDifference_exceptionWrongType():
	""" Test featureNameDifference() for ArgumentException when the other object is not the right type """
	toTest = makeAndDefine(["hello"])
	toTest.featureNameDifference("wrong")

def test_featureNameDifference_handmade():
	""" Test featureNameDifference() against handmade output """
	toTest1 = makeAndDefine(["one","two","three"])
	toTest2 = makeAndDefine(["two","four"])
	results = toTest1.featureNameDifference(toTest2)
	assert "one" in results
	assert "two" not in results
	assert "three" in results
	assert "four" not in results


#######################
# featureNameIntersection() #
#######################

@raises(ArgumentException)
def test_featureNameIntersection_exceptionOtherNone():
	""" Test featureNameIntersection() for ArgumentException when the other object is None """
	toTest = makeAndDefine(["hello"])
	toTest.featureNameIntersection(None)

@raises(ArgumentException)
def test_featureNameIntersection_exceptionWrongType():
	""" Test featureNameIntersection() for ArgumentException when the other object is not the right type """
	toTest = makeAndDefine(["hello"])
	toTest.featureNameIntersection("wrong")

def test_featureNameIntersection_handmade():
	""" Test featureNameIntersection() against handmade output """
	toTest1 = makeAndDefine(["one","two","three"])
	toTest2 = makeAndDefine(["two","four"])
	results = toTest1.featureNameIntersection(toTest2)
	assert "one" not in results
	assert "two" in results
	assert "three" not in results
	assert "four" not in results


##############################
# featureNameSymmetricDifference() #
##############################

@raises(ArgumentException)
def test_featureNameSymmetricDifference_exceptionOtherNone():
	""" Test featureNameSymmetricDifference() for ArgumentException when the other object is None """
	toTest = makeAndDefine(["hello"])
	toTest.featureNameSymmetricDifference(None)

@raises(ArgumentException)
def test_featureNameSymmetricDifference_exceptionWrongType():
	""" Test featureNameSymmetricDifference() for ArgumentException when the other object is not the right type """
	toTest = makeAndDefine(["hello"])
	toTest.featureNameSymmetricDifference("wrong")

def test_featureNameSymmetricDifference_handmade():
	""" Test featureNameSymmetricDifference() against handmade output """
	toTest1 = makeAndDefine(["one","two","three"])
	toTest2 = makeAndDefine(["two","four"])
	results = toTest1.featureNameSymmetricDifference(toTest2)
	assert "one" in results
	assert "two" not in results
	assert "three" in results
	assert "four" in results

################
# featureNameUnion() #
################

@raises(ArgumentException)
def test_featureNameUnion_exceptionOtherNone():
	""" Test featureNameUnion() for ArgumentException when the other object is None """
	toTest = makeAndDefine(["hello"])
	toTest.featureNameUnion(None)

@raises(ArgumentException)
def test_featureNameUnion_exceptionWrongType():
	""" Test featureNameUnion() for ArgumentException when the other object is not the right type """
	toTest = makeAndDefine(["hello"])
	toTest.featureNameUnion("wrong")

def test_featureNameUnion_handmade():
	""" Test featureNameUnion() against handmade output """
	toTest1 = makeAndDefine(["one","two","three"])
	toTest2 = makeAndDefine(["two","four"])
	results = toTest1.featureNameUnion(toTest2)
	assert "one" in results
	assert "two" in results
	assert "three" in results
	assert "four" in results

#################
# renameFeatureName() #
#################

@raises(ArgumentException)
def test_renameFeatureName_exceptionPrevWrongType():
	""" Test renameFeatureName() for ArgumentException when given the wrong type for prev"""
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest.renameFeatureName(oldIdentifier=0.3,newFeatureName="New!")

@raises(ArgumentException)
def test_renameFeatureName_exceptionPrevInvalidIndex():
	""" Test renameFeatureName() for ArgumentException when given an invalid prev index"""
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest.renameFeatureName(oldIdentifier=12,newFeatureName="New!")

@raises(ArgumentException)
def test_renameFeatureName_exceptionPrevNotFound():
	""" Test renameFeatureName() for ArgumentException when the prev featureName is not found"""
	origFeatureNames = ["zero","one","two","three"]	
	toTest =makeAndDefine(origFeatureNames)
	toTest.renameFeatureName(oldIdentifier="Previous!",newFeatureName="New!")

@raises(ArgumentException)
def test_renameFeatureName_exceptionNewInvalidType():
	""" Test renameFeatureName() for ArgumentException when the new featureName is not a string"""
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest.renameFeatureName(oldIdentifier="three",newFeatureName=4)

@raises(ArgumentException)
def test_renameFeatureName_exceptionNonUnique():
	""" Test renameFeatureName() for ArgumentException when a duplicate featureName is given"""
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest.renameFeatureName(oldIdentifier="three",newFeatureName="two")

@raises(ArgumentException)
def test_renameFeatureName_exceptionManualAddDefault():
	""" Test renameFeatureName() for ArgumentException when given a default featureName """
	toTest = makeAndDefine(["hello"])
	toTest.renameFeatureName("hello",DEFAULT_PREFIX + "2")

@raises(ImproperActionException)
def test_renameFeatureName_exceptionNoFeatures():
	toTest = makeAndDefine([])
	toTest.renameFeatureName("hello","2")

def test_renameFeatureName_handmade_viaIndex():
	""" Test renameFeatureName() against handmade input when specifying the featureName by index """
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest.renameFeatureName(0,"ZERO")
	toTest.renameFeatureName(3,"3")
	expectedFeatureNames = ["ZERO","one","two","3"]
	confirmExpectedFeatureNames(toTest,expectedFeatureNames)

def test_renameFeatureName_handmade_viaFeatureName():
	""" Test renameFeatureName() against handmade input when specifying the featureName by name """
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest.renameFeatureName("zero","ZERO")
	toTest.renameFeatureName("three","3")
	expectedFeatureNames = ["ZERO","one","two","3"]
	confirmExpectedFeatureNames(toTest,expectedFeatureNames)



##########################
# renameMultipleFeatureNames() #
##########################


@raises(ArgumentException)
def test_renameMultipleFeatureNames_exceptionWrongTypeObject():
	""" Test renameMultipleFeatureNames() for ArgumentException when featureNames is an unexpected type """
	toTest = makeAndDefine(['one'])
	toTest.renameMultipleFeatureNames(12)

@raises(ArgumentException)
def test_renameMultipleFeatureNames_exceptionNonStringFeatureNameInList():
	""" Test renameMultipleFeatureNames() for ArgumentException when a list element is not a string """
	toTest = makeAndDefine(['one'])
	nonStringFeatureNames = [1,2,3]
	toTest.renameMultipleFeatureNames(nonStringFeatureNames)

@raises(ArgumentException)
def test_renameMultipleFeatureNames_exceptionNonUniqueStringInList():
	""" Test renameMultipleFeatureNames() for ArgumentException when a list element is not unique """
	toTest = makeAndDefine(['one'])
	nonUnique = [1,2,3,1]
	toTest.renameMultipleFeatureNames(nonUnique)

@raises(ArgumentException)
def test_renameMultipleFeatureNames_exceptionNonStringFeatureNameInDict():
	""" Test renameMultipleFeatureNames() for ArgumentException when a dict key is not a string """
	toTest = makeAndDefine(['one'])
	nonStringFeatureNames = {1:1}
	toTest.renameMultipleFeatureNames(nonStringFeatureNames)

@raises(ArgumentException)
def test_renameMultipleFeatureNames_exceptionNonIntIndexInDict():
	""" Test renameMultipleFeatureNames() for ArgumentException when a dict value is not an int """
	toTest = makeAndDefine(['one'])
	nonIntIndex = {"one":"one"}
	toTest.renameMultipleFeatureNames(nonIntIndex)


@raises(ArgumentException)
def test_renameMultipleFeatureNames_exceptionManualAddDefault():
	""" Test renameMultipleFeatureNames() for ArgumentException when given a default featureName """
	toTest = makeAndDefine(["blank","none","gone","hey"])
	newFeatureNames = ["zero","one","two",DEFAULT_PREFIX + "2"]
	toTest.renameMultipleFeatureNames(newFeatureNames)

@raises(ImproperActionException)
def test_renameMultipleFeatureName_exceptionNoFeatures():
	toTest = makeAndDefine([])
	toAssign = ["hey","gone","none","blank"]
	toTest.renameMultipleFeatureNames(toAssign)

def test_renameMultipleFeatureNames_handmade():
	""" Test renameMultipleFeatureNames() against handmade output """
	toTest = makeAndDefine(["blank","none","gone","hey"])
	origFeatureNames = ["zero","one","two","three"]	
	toTest.renameMultipleFeatureNames(origFeatureNames)
	confirmExpectedFeatureNames(toTest,origFeatureNames)

def test_renameMultipleFeatureNames_handmadeReplacingWithSame():
	""" Test renameMultipleFeatureNames() against handmade output when you're replacing the position of featureNames """
	toTest = makeAndDefine(["blank","none","gone","hey"])
	toAssign = ["hey","gone","none","blank"]
	toTest.renameMultipleFeatureNames(toAssign)
	confirmExpectedFeatureNames(toTest,toAssign)


##########################
# _removeFeatureNameAndShift() #
##########################

@raises(ArgumentException)
def test_removeFeatureNameAndShift_exceptionNoneInput():
	""" Test _removeFeatureNameAndShift() for ArgumentException when the identifier is None"""
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest._removeFeatureNameAndShift(None)

@raises(ArgumentException)
def test_removeFeatureNameAndShift_exceptionWrongTypeInput():
	""" Test _removeFeatureNameAndShift() for ArgumentException when given the wrong type"""
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest._removeFeatureNameAndShift(0.3)

@raises(ArgumentException)
def test_removeFeatureNameAndShift_exceptionInvalidIndex():
	""" Test _removeFeatureNameAndShift() for ArgumentException when given an invalid index"""
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest._removeFeatureNameAndShift(12)

@raises(ArgumentException)
def test_removeFeatureNameAndShift_exceptionFeatureNameNotFound():
	""" Test _removeFeatureNameAndShift() for ArgumentException when the featureName is not found"""
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest._removeFeatureNameAndShift("bogus")

def test_removeFeatureNameAndShift_handmade_viaIndex():
	""" Test _removeFeatureNameAndShift() against handmade output """
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest._removeFeatureNameAndShift(0)
	toTest._removeFeatureNameAndShift(2)
	expectedFeatureNames = ["one","two"]
	confirmExpectedFeatureNames(toTest,expectedFeatureNames)

def test_removeFeatureNameAndShift_handmade_viaFeatureName():
	""" Test _removeFeatureNameAndShift() against handmade output """
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	toTest._removeFeatureNameAndShift('zero')
	toTest._removeFeatureNameAndShift('two')
	expectedFeatureNames = ["one","three"]
	confirmExpectedFeatureNames(toTest,expectedFeatureNames)



##################
# _equalFeatureNames() #
##################

def test_equalFeatureNames_False():
	""" Test _equalFeatureNames() against some non-equal input """
	origFeatureNames = ["zero","one","two","three"]	
	toTest = makeAndDefine(origFeatureNames)
	assert not toTest._equalFeatureNames(None)
	assert not toTest._equalFeatureNames(5)
	assert not toTest._equalFeatureNames(makeAndDefine(["short","list"]))

	subset = makeAndDefine(["zero","one","two"])
	superset = makeAndDefine(["zero","one","two","three","four"])
	assert not toTest._equalFeatureNames(subset)
	assert not toTest._equalFeatureNames(superset)
	assert not subset._equalFeatureNames(toTest)
	assert not superset._equalFeatureNames(toTest)

def test_equalFeatureNames_actuallyEqual():
	""" Test _equalFeatureNames() against some actually equal input """
	origFeatureNames = ["zero","one","two","three"]
	featureNamesDict = {"zero":0,"one":1,"two":2,"three":3}
	toTest1 = makeAndDefine(origFeatureNames)
	toTest2 = makeAndDefine(featureNamesDict)
	assert toTest1._equalFeatureNames(toTest2)
	assert toTest2._equalFeatureNames(toTest1)

def test_equalFeatureNames_noData():
	""" Test _equalFeatureNames() for empty objects """
	toTest1 = makeAndDefine(None)
	toTest2 = makeAndDefine(None)
	assert toTest1._equalFeatureNames(toTest2)
	assert toTest2._equalFeatureNames(toTest1)

def test_equalFeatureNames_DefaultInequality():
	""" Test _equalFeatureNames() for inequality of some default named, different sized objects """
	toTest1 = makeAndDefine(['1','2'])
	toTest2 = makeAndDefine(['1','2','3'])
	toTest1.renameFeatureName(0,None)
	toTest1.renameFeatureName(1,None)
	toTest2.renameFeatureName(0,None)
	toTest2.renameFeatureName(1,None)
	toTest2.renameFeatureName(2,None)
	assert not toTest1._equalFeatureNames(toTest2)
	assert not toTest2._equalFeatureNames(toTest1)

def test_equalFeatureNames_ignoresDefaults():
	""" Test _equalFeatureNames() for equality of default named objects """
	toTest1 = makeAndDefine(['1','2'])
	toTest2 = makeAndDefine(['1','2'])
	toTest1.renameFeatureName(0,None)
	toTest1.renameFeatureName(1,None)
	toTest2.renameFeatureName(0,None)
	toTest2.renameFeatureName(1,None)
	assert toTest1._equalFeatureNames(toTest2)
	assert toTest2._equalFeatureNames(toTest1)

def test_equalFeatureNames_mixedDefaultsAndActual():
	""" Test _equalFeatureNames() for equality of default named objects mixed with actual featureNames """
	toTest1 = makeAndDefine(['1','2'])
	toTest2 = makeAndDefine(['1','2'])
	toTest1.renameFeatureName(0,None)
	toTest2.renameFeatureName(1,None)
	assert not toTest1._equalFeatureNames(toTest2)
	assert not toTest2._equalFeatureNames(toTest1)
	toTest1.renameFeatureName(0,'1')
	toTest1.renameFeatureName(1,None)
	print toTest1.featureNames
	print toTest2.featureNames
	assert toTest1._equalFeatureNames(toTest2)
	assert toTest2._equalFeatureNames(toTest1)

###########
# helpers #
###########


def confirmExpectedFeatureNames(toTest, expected):
	for i in range(len(expected)):
		expectedFeatureName = expected[i]
		if not expectedFeatureName.startswith(DEFAULT_PREFIX):	
			actualIndex = toTest.featureNames[expectedFeatureName]
			actualFeatureName = toTest.featureNamesInverse[i]
			assert (actualIndex == i)
			assert (actualFeatureName == expectedFeatureName)

		

