"""
Unit tests of the low level functions defined by the base representation class.

Since these functions are stand alone in the base class, they can be tested directly.
Label equality is defined using _numColumns_implementation, which is to be defined
in the derived class, so all of these tests make a call to makeAndDefine instead
of directly instantiating a BaseData object. This function temporarily fills in
that missing implementation.

"""

from ..base_data import *
from nose.tools import *

def makeConst(num):
	def const(dummy):
		return num
	return const

def makeAndDefine(labels=None, size=0):
	""" Make a base data object that will think it has as many columns as it has labels,
	even though it has no actual data """
	cols = size if labels is None else len(labels)
	specificImp = makeConst(cols)
	BaseData._numColumns_implementation = specificImp
	ret = BaseData(labels)
	ret._numColumns_implementation = specificImp
	return ret


########
# init #
########

#@raises(ArgumentException)
#def test_init_exceptionNoLabelsOrCount():
#	""" Test init() for ArgumentException when called without labels or a count of columns """
#	toTest= BaseData(None)

###############
# _addLabel() #
###############

@raises(ArgumentException)
def test_addLabel_exceptionLabelWrongType():
	""" Test _addLabel() for ArgumentException when given a non string label """
	toTest = makeAndDefine(["hello"])
	toTest._addLabel(label=34)

@raises(ArgumentException)
def test_addLabel_exceptionNonUnique():
	""" Test _addLabel() for ArgumentException when given a duplicate label """
	toTest = makeAndDefine(["hello"])
	toTest._addLabel("hello")

def test_addLabel_handmadeDefaultCounter():
	""" Test _addLabel() changes nextDefault counter correctly """
	toTest = makeAndDefine(["hello"])
	assert toTest._nextDefaultValue == 1
	toTest._addLabel(DEFAULT_PREFIX +"1")
	assert toTest._nextDefaultValue == 2

def test_addLabel_handmade():
	""" Test _addLabel() against handmade output """
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest._addLabel("four")
	toTest._addLabel("five")

	expected = ["zero","one","two","three","four","five"]
	confirmExpectedLabels(toTest,expected)

def test_addLabel_handmade_addDefault():
	""" Test _addLabel() against handmade output """
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest._addLabel(None)
	toTest._addLabel(None)

	expected = ["zero","one","two","three",DEFAULT_PREFIX,DEFAULT_PREFIX]

	confirmExpectedLabels(toTest,expected)	

#####################
# labelDifference() #
#####################

@raises(ArgumentException)
def test_labelDifference_exceptionOtherNone():
	""" Test labelDifference() for ArgumentException when the other object is None """
	toTest = makeAndDefine(["hello"])
	toTest.labelDifference(None)

@raises(ArgumentException)
def test_labelDifference_exceptionWrongType():
	""" Test labelDifference() for ArgumentException when the other object is not the right type """
	toTest = makeAndDefine(["hello"])
	toTest.labelDifference("wrong")

def test_labelDifference_handmade():
	""" Test labelDifference() against handmade output """
	toTest1 = makeAndDefine(["one","two","three"])
	toTest2 = makeAndDefine(["two","four"])
	results = toTest1.labelDifference(toTest2)
	assert "one" in results
	assert "two" not in results
	assert "three" in results
	assert "four" not in results


#######################
# labelIntersection() #
#######################

@raises(ArgumentException)
def test_labelIntersection_exceptionOtherNone():
	""" Test labelIntersection() for ArgumentException when the other object is None """
	toTest = makeAndDefine(["hello"])
	toTest.labelIntersection(None)

@raises(ArgumentException)
def test_labelIntersection_exceptionWrongType():
	""" Test labelIntersection() for ArgumentException when the other object is not the right type """
	toTest = makeAndDefine(["hello"])
	toTest.labelIntersection("wrong")

def test_labelIntersection_handmade():
	""" Test labelIntersection() against handmade output """
	toTest1 = makeAndDefine(["one","two","three"])
	toTest2 = makeAndDefine(["two","four"])
	results = toTest1.labelIntersection(toTest2)
	assert "one" not in results
	assert "two" in results
	assert "three" not in results
	assert "four" not in results


##############################
# labelSymmetricDifference() #
##############################

@raises(ArgumentException)
def test_labelSymmetricDifference_exceptionOtherNone():
	""" Test labelSymmetricDifference() for ArgumentException when the other object is None """
	toTest = makeAndDefine(["hello"])
	toTest.labelSymmetricDifference(None)

@raises(ArgumentException)
def test_labelSymmetricDifference_exceptionWrongType():
	""" Test labelSymmetricDifference() for ArgumentException when the other object is not the right type """
	toTest = makeAndDefine(["hello"])
	toTest.labelSymmetricDifference("wrong")

def test_labelSymmetricDifference_handmade():
	""" Test labelSymmetricDifference() against handmade output """
	toTest1 = makeAndDefine(["one","two","three"])
	toTest2 = makeAndDefine(["two","four"])
	results = toTest1.labelSymmetricDifference(toTest2)
	assert "one" in results
	assert "two" not in results
	assert "three" in results
	assert "four" in results

################
# labelUnion() #
################

@raises(ArgumentException)
def test_labelUnion_exceptionOtherNone():
	""" Test labelUnion() for ArgumentException when the other object is None """
	toTest = makeAndDefine(["hello"])
	toTest.labelUnion(None)

@raises(ArgumentException)
def test_labelUnion_exceptionWrongType():
	""" Test labelUnion() for ArgumentException when the other object is not the right type """
	toTest = makeAndDefine(["hello"])
	toTest.labelUnion("wrong")

def test_labelUnion_handmade():
	""" Test labelUnion() against handmade output """
	toTest1 = makeAndDefine(["one","two","three"])
	toTest2 = makeAndDefine(["two","four"])
	results = toTest1.labelUnion(toTest2)
	assert "one" in results
	assert "two" in results
	assert "three" in results
	assert "four" in results

#################
# renameLabel() #
#################

@raises(ArgumentException)
def test_renameLabel_exceptionPrevWrongType():
	""" Test renameLabel() for ArgumentException when given the wrong type for prev"""
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest.renameLabel(oldIdentifier=0.3,newLabel="New!")

@raises(ArgumentException)
def test_renameLabel_exceptionPrevInvalidIndex():
	""" Test renameLabel() for ArgumentException when given an invalid prev index"""
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest.renameLabel(oldIdentifier=12,newLabel="New!")

@raises(ArgumentException)
def test_renameLabel_exceptionPrevNotFound():
	""" Test renameLabel() for ArgumentException when the prev label is not found"""
	origLabels = ["zero","one","two","three"]	
	toTest =makeAndDefine(origLabels)
	toTest.renameLabel(oldIdentifier="Previous!",newLabel="New!")

@raises(ArgumentException)
def test_renameLabel_exceptionNewInvalidType():
	""" Test renameLabel() for ArgumentException when the new label is not a string"""
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest.renameLabel(oldIdentifier="three",newLabel=4)

@raises(ArgumentException)
def test_renameLabel_exceptionNonUnique():
	""" Test renameLabel() for ArgumentException when a duplicate label is given"""
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest.renameLabel(oldIdentifier="three",newLabel="two")

@raises(ArgumentException)
def test_renameLabel_exceptionManualAddDefault():
	""" Test renameLabel() for ArgumentException when given a default label """
	toTest = makeAndDefine(["hello"])
	toTest.renameLabel("hello",DEFAULT_PREFIX + "2")


def test_renameLabel_handmade_viaIndex():
	""" Test renameLabel() against handmade input when specifying the label by index """
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest.renameLabel(0,"ZERO")
	toTest.renameLabel(3,"3")
	expectedLabels = ["ZERO","one","two","3"]
	confirmExpectedLabels(toTest,expectedLabels)

def test_renameLabel_handmade_viaLabel():
	""" Test renameLabel() against handmade input when specifying the label by name """
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest.renameLabel("zero","ZERO")
	toTest.renameLabel("three","3")
	expectedLabels = ["ZERO","one","two","3"]
	confirmExpectedLabels(toTest,expectedLabels)



##########################
# renameMultipleLabels() #
##########################


@raises(ArgumentException)
def test_renameMultipleLabels_exceptionWrongTypeObject():
	""" Test renameMultipleLabels() for ArgumentException when labels is an unexpected type """
	toTest = makeAndDefine(['one'])
	toTest.renameMultipleLabels(12)

@raises(ArgumentException)
def test_renameMultipleLabels_exceptionNonStringLabelInList():
	""" Test renameMultipleLabels() for ArgumentException when a list element is not a string """
	toTest = makeAndDefine(['one'])
	nonStringLabels = [1,2,3]
	toTest.renameMultipleLabels(nonStringLabels)

@raises(ArgumentException)
def test_renameMultipleLabels_exceptionNonUniqueStringInList():
	""" Test renameMultipleLabels() for ArgumentException when a list element is not unique """
	toTest = makeAndDefine(['one'])
	nonUnique = [1,2,3,1]
	toTest.renameMultipleLabels(nonUnique)

@raises(ArgumentException)
def test_renameMultipleLabels_exceptionNonStringLabelInDict():
	""" Test renameMultipleLabels() for ArgumentException when a dict key is not a string """
	toTest = makeAndDefine(['one'])
	nonStringLabels = {1:1}
	toTest.renameMultipleLabels(nonStringLabels)

@raises(ArgumentException)
def test_renameMultipleLabels_exceptionNonIntIndexInDict():
	""" Test renameMultipleLabels() for ArgumentException when a dict value is not an int """
	toTest = makeAndDefine(['one'])
	nonIntIndex = {"one":"one"}
	toTest.renameMultipleLabels(nonIntIndex)


@raises(ArgumentException)
def test_renameMultipleLabels_exceptionManualAddDefault():
	""" Test renameMultipleLabels() for ArgumentException when given a default label """
	toTest = makeAndDefine(["blank","none","gone","hey"])
	newLabels = ["zero","one","two",DEFAULT_PREFIX + "2"]
	toTest.renameMultipleLabels(newLabels)


def test_renameMultipleLabels_handmade():
	""" Test renameMultipleLabels() against handmade output """
	toTest = makeAndDefine(["blank","none","gone","hey"])
	origLabels = ["zero","one","two","three"]	
	toTest.renameMultipleLabels(origLabels)
	confirmExpectedLabels(toTest,origLabels)

def test_renameMultipleLabels_handmadeReplacingWithSame():
	""" Test renameMultipleLabels() against handmade output when you're replacing the position of labels """
	toTest = makeAndDefine(["blank","none","gone","hey"])
	toAssign = ["hey","gone","none","blank"]
	toTest.renameMultipleLabels(toAssign)
	confirmExpectedLabels(toTest,toAssign)


##########################
# _removeLabelAndShift() #
##########################

@raises(ArgumentException)
def test_removeLabelAndShift_exceptionNoneInput():
	""" Test _removeLabelAndShift() for ArgumentException when the identifier is None"""
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest._removeLabelAndShift(None)

@raises(ArgumentException)
def test_removeLabelAndShift_exceptionWrongTypeInput():
	""" Test _removeLabelAndShift() for ArgumentException when given the wrong type"""
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest._removeLabelAndShift(0.3)

@raises(ArgumentException)
def test_removeLabelAndShift_exceptionInvalidIndex():
	""" Test _removeLabelAndShift() for ArgumentException when given an invalid index"""
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest._removeLabelAndShift(12)

@raises(ArgumentException)
def test_removeLabelAndShift_exceptionLabelNotFound():
	""" Test _removeLabelAndShift() for ArgumentException when the label is not found"""
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest._removeLabelAndShift("bogus")

def test_removeLabelAndShift_handmade_viaIndex():
	""" Test _removeLabelAndShift() against handmade output """
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest._removeLabelAndShift(0)
	toTest._removeLabelAndShift(2)
	expectedLabels = ["one","two"]
	confirmExpectedLabels(toTest,expectedLabels)

def test_removeLabelAndShift_handmade_viaLabel():
	""" Test _removeLabelAndShift() against handmade output """
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	toTest._removeLabelAndShift('zero')
	toTest._removeLabelAndShift('two')
	expectedLabels = ["one","three"]
	confirmExpectedLabels(toTest,expectedLabels)



##################
# _equalLabels() #
##################

def test_equalLabels_False():
	""" Test _equalLabels() against some non-equal input """
	origLabels = ["zero","one","two","three"]	
	toTest = makeAndDefine(origLabels)
	assert not toTest._equalLabels(None)
	assert not toTest._equalLabels(5)
	assert not toTest._equalLabels(makeAndDefine(["short","list"]))

	subset = makeAndDefine(["zero","one","two"])
	superset = makeAndDefine(["zero","one","two","three","four"])
	assert not toTest._equalLabels(subset)
	assert not toTest._equalLabels(superset)
	assert not subset._equalLabels(toTest)
	assert not superset._equalLabels(toTest)

def test_equalLabels_actuallyEqual():
	""" Test _equalLabels() against some actually equal input """
	origLabels = ["zero","one","two","three"]
	labelsDict = {"zero":0,"one":1,"two":2,"three":3}
	toTest1 = makeAndDefine(origLabels)
	toTest2 = makeAndDefine(labelsDict)
	assert toTest1._equalLabels(toTest2)
	assert toTest2._equalLabels(toTest1)

def test_equalLabels_noData():
	""" Test _equalLabels() for empty objects """
	toTest1 = makeAndDefine(None)
	toTest2 = makeAndDefine(None)
	assert toTest1._equalLabels(toTest2)
	assert toTest2._equalLabels(toTest1)

def test_equalLabels_DefaultInequality():
	""" Test _equalLabels() for inequality of some default labeled different sized objects """
	toTest1 = makeAndDefine(['1','2'])
	toTest2 = makeAndDefine(['1','2','3'])
	toTest1.renameLabel(0,None)
	toTest1.renameLabel(1,None)
	toTest2.renameLabel(0,None)
	toTest2.renameLabel(1,None)
	toTest2.renameLabel(2,None)
	assert not toTest1._equalLabels(toTest2)
	assert not toTest2._equalLabels(toTest1)

def test_equalLabels_ignoresDefaults():
	""" Test _equalLabels() for equality of default labeled objects """
	toTest1 = makeAndDefine(['1','2'])
	toTest2 = makeAndDefine(['1','2'])
	toTest1.renameLabel(0,None)
	toTest1.renameLabel(1,None)
	toTest2.renameLabel(0,None)
	toTest2.renameLabel(1,None)
	assert toTest1._equalLabels(toTest2)
	assert toTest2._equalLabels(toTest1)

def test_equalLabels_mixedDefaultsAndActual():
	""" Test _equalLabels() for equality of default labeled objects mixed with actual labels """
	toTest1 = makeAndDefine(['1','2'])
	toTest2 = makeAndDefine(['1','2'])
	toTest1.renameLabel(0,None)
	toTest2.renameLabel(1,None)
	assert not toTest1._equalLabels(toTest2)
	assert not toTest2._equalLabels(toTest1)
	toTest1.renameLabel(0,'1')
	toTest1.renameLabel(1,None)
	print toTest1.labels
	print toTest2.labels
	assert toTest1._equalLabels(toTest2)
	assert toTest2._equalLabels(toTest1)

###########
# helpers #
###########


def confirmExpectedLabels(toTest, expected):
	for i in range(len(expected)):
		expectedLabel = expected[i]
		if not expectedLabel.startswith(DEFAULT_PREFIX):	
			actualIndex = toTest.labels[expectedLabel]
			actualLabel = toTest.labelsInverse[i]
			assert (actualIndex == i)
			assert (actualLabel == expectedLabel)

		

