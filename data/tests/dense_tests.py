"""
Unit tests for the Matrix object. Calls upon the functions defined
in derived_backend.py using appropriate input


"""

from UML.data.tests.derived_backend import *
from nose.tools import *
from UML import createData
from UML.exceptions import ArgumentException

def constructor(data=None, featureNames=None):
	return createData('Matrix', data, featureNames)


##############
# __init__() #
##############

def test_init_allEqual():
	""" Test Matrix __init__() that every way to instantiate produces equal objects """
	init_allEqual(constructor)

def test_init_allEqualWithFeatureNames():
	""" Test Matrix __init__() that every way to instantiate produces equal objects, with featureNames """
	init_allEqualWithFeatureNames(constructor)

############
# isIdentical() #
############

def test_isIdentical_False():
	""" Test Matrix isIdentical() against some non-equal input """
	isIdentical_False(constructor)

def test_isIdentical_True():
	""" Test Matrix isIdentical() against some actually equal input """
	isIdentical_True(constructor)


###############
# transpose() #
###############

def test_transpose_handmade():
	""" Test Matrix transpose() function against handmade output """
	transpose_handmade(constructor)


#############
# appendPoints() #
#############

@raises(ArgumentException)
def test_appendPoints_exceptionNone():
	""" Test Matrix appendPoints() for ArgumentException when toAppend is None"""
	appendPoints_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendPoints_exceptionWrongSize():
	""" Test Matrix appendPoints() for ArgumentException when toAppend has too many features """
	appendPoints_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_appendPoints_exceptionMismatchedFeatureNames():
	""" Test Matrix appendPoints() for ArgumentException when toAppend and self's feature names do not match"""
	appendPoints_exceptionMismatchedFeatureNames(constructor)

def test_appendPoints_handmadeSingle():
	""" Test Matrix appendPoints() against handmade output for a single added point """
	appendPoints_handmadeSingle(constructor)

def test_appendPoints_handmadeSequence():
	""" Test Matrix appendPoints() against handmade output for a sequence of additions"""
	appendPoints_handmadeSequence(constructor)


################
# appendFeatures() #
################

@raises(ArgumentException)
def test_appendFeatures_exceptionNone():
	""" Test Matrix appendFeatures() for ArgumentException when toAppend is None """
	appendFeatures_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendFeatures_exceptionWrongSize():
	""" Test Matrix appendFeatures() for ArgumentException when toAppend has too many points """
	appendFeatures_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_appendFeatures_exceptionSameFeatureName():
	""" Test Matrix appendFeatures() for ArgumentException when toAppend and self have a featureName in common """
	appendFeatures_exceptionSameFeatureName(constructor)

def test_appendFeatures_handmadeSingle():
	""" Test Matrix appendFeatures() against handmade output for a single added feature"""
	appendFeatures_handmadeSingle(constructor)

def test_appendFeatures_handmadeSequence():
	""" Test Matrix appendFeatures() against handmade output for a sequence of additions"""
	appendFeatures_handmadeSequence(constructor)



##############
# sortPoints() #
##############

@raises(ArgumentException)
def test_sortPoints_exceptionAtLeastOne():
	""" Test Matrix sortPoints() has at least one paramater """
	sortPoints_exceptionAtLeastOne(constructor)

def test_sortPoints_naturalByFeature():
	""" Test Matrix sortPoints() when we specify a feature to sort by """	
	sortPoints_naturalByFeature(constructor)

def test_sortPoints_scorer():
	""" Test Matrix sortPoints() when we specify a scoring function """
	sortPoints_scorer(constructor)

def test_sortPoints_comparator():
	""" Test Matrix sortPoints() when we specify a comparator function """
	sortPoints_comparator(constructor)


#################
# sortFeatures() #
#################


@raises(ArgumentException)
def test_sortFeatures_exceptionAtLeastOne():
	""" Test Matrix sortFeatures() has at least one paramater """
	sortFeatures_exceptionAtLeastOne(constructor)

def test_sortFeatures_naturalByPointWithNames():
	""" Test Matrix sortFeatures() when we specify a point to sort by; includes featureNames """	
	sortFeatures_naturalByPointWithNames(constructor)

def test_sortFeatures_scorer():
	""" Test Matrix sortFeatures() when we specify a scoring function """
	sortFeatures_scorer(constructor)

def test_sortFeatures_comparator():
	""" Test Matrix sortFeatures() when we specify a comparator function """
	sortFeatures_comparator(constructor)



#################
# extractPoints() #
#################

def test_extractPoints_emptyInput(): 
	""" Test Matrix extractPoints() does nothing when not provided with any input """
	extractPoints_emptyInput(constructor)

def test_extractPoints_handmadeSingle():
	""" Test Matrix extractPoints() against handmade output when extracting one point """
	extractPoints_handmadeSingle(constructor)

def test_extractPoints_handmadeListSequence():
	""" Test Matrix extractPoints() against handmade output for several list extractions """
	extractPoints_handmadeListSequence(constructor)

def test_extractPoints_handmadeListOrdering():
	""" Test Matrix extractPoints() against handmade output for out of order extraction """
	extractPoints_handmadeListOrdering(constructor)

def test_extractPoints_handmadeFunction():
	""" Test Matrix extractPoints() against handmade output for function extraction """
	extractPoints_handmadeFunction(constructor)

def test_extractPoints_handmadeFuncionWithFeatureNames():
	""" Test Matrix extractPoints() against handmade output for function extraction with featureNames"""
	extractPoints_handmadeFuncionWithFeatureNames(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionStartInvalid():
	""" Test Matrix extractPoints() for ArgumentException when start is not a valid point index """
	extractPoints_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionEndInvalid():
	""" Test Matrix extractPoints() for ArgumentException when start is not a valid feature index """
	extractPoints_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionInversion():
	""" Test Matrix extractPoints() for ArgumentException when start comes after end """
	extractPoints_exceptionInversion(constructor)

def test_extractPoints_handmadeRange():
	""" Test Matrix extractPoints() against handmade output for range extraction """
	extractPoints_handmadeRange(constructor)

def test_extractPoints_handmadeRangeWithFeatureNames():
	""" Test Matrix extractPoints() against handmade output for range extraction with featureNames """
	extractPoints_handmadeRangeWithFeatureNames(constructor)

def test_extractPoints_handmadeRangeRand_FM():
	""" Test Matrix extractPoints() against handmade output for randomized range extraction with featureNames """
	extractPoints_handmadeRangeRand_FM(constructor)

def test_extractPoints_handmadeRangeDefaults():
	""" Test Matrix extractPoints() uses the correct defaults in the case of range based extraction """
	extractPoints_handmadeRangeDefaults(constructor)

####################
# extractFeatures() #
####################


def test_extractFeatures_handmadeSingle():
	""" Test Matrix extractFeatures() against handmade output when extracting one feature """
	extractFeatures_handmadeSingle(constructor)

def test_extractFeatures_handmadeListSequence():
	""" Test Matrix extractFeatures() against handmade output for several extractions by list """
	extractFeatures_handmadeListSequence(constructor)

def test_extractFeatures_handmadeListWithFeatureName():
	""" Test Matrix extractFeatures() against handmade output for list extraction when specifying featureNames """
	extractFeatures_handmadeListWithFeatureName(constructor)

def test_extractFeatures_handmadeFunction():
	""" Test Matrix extractFeatures() against handmade output for function extraction """
	extractFeatures_handmadeFunction(constructor)

def test_extractFeatures_handmadeFunctionWithFeatureName():
	""" Test Matrix extractFeatures() against handmade output for function extraction with featureNames """
	extractFeatures_handmadeFunctionWithFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionStartInvalid():
	""" Test Matrix extractFeatures() for ArgumentException when start is not a valid feature index """
	extractFeatures_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionStartInvalidFeatureName():
	""" Test Matrix extractFeatures() for ArgumentException when start is not a valid featureName """
	extractFeatures_exceptionStartInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionEndInvalid():
	""" Test Matrix extractFeatures() for ArgumentException when start is not a valid feature index """
	extractFeatures_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionEndInvalidFeatureName():
	""" Test Matrix extractFeatures() for ArgumentException when start is not a valid featureName """
	extractFeatures_exceptionEndInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionInversion():
	""" Test Matrix extractFeatures() for ArgumentException when start comes after end """
	extractFeatures_exceptionInversion(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionInversionFeatureName():
	""" Test Matrix extractFeatures() for ArgumentException when start comes after end as FeatureNames"""
	extractFeatures_exceptionInversionFeatureName(constructor)

def test_extractFeatures_handmadeRange():
	""" Test Matrix extractFeatures() against handmade output for range extraction """
	extractFeatures_handmadeRange(constructor)

def test_extractFeatures_handmadeWithFeatureNames():
	""" Test Matrix extractFeatures() against handmade output for range extraction with FeatureNames """
	extractFeatures_handmadeWithFeatureNames(constructor)


##########################
# toList() #
##########################


def test_toList_handmade_defaultFeatureNames():
	""" Test Matrix toList with default featureNames """
	toList_handmade_defaultFeatureNames(constructor)

	
def test_toList_handmade_assignedFeatureNames():
	""" Test Matrix toList with assigned featureNames """
	toList_handmade_assignedFeatureNames(constructor)



##############################
# toMatrix() #
##############################


def test_toMatrix_handmade_defaultFeatureNames():
	""" Test Matrix toMatrix with default featureNames """
	toMatrix_handmade_defaultFeatureNames(constructor)

	
def test_toMatrix_handmade_assignedFeatureNames():
	""" Test Matrix toMatrix with assigned featureNames """
	toMatrix_handmade_assignedFeatureNames(constructor)



############
# writeFile #
############

def test_writeFileCSV_handmade():
	""" Test Matrix writeFile() for csv extension with both data and featureNames """
	writeFileCSV_handmade(constructor)

def test_writeFileMTX_handmade():
	""" Test Matrix writeFile() for mtx extension with both data and featureNames """
	writeFileMTX_handmade(constructor)


#####################
# referenceDataFrom #
#####################


@raises(ArgumentException)
def test_referenceDataFrom_exceptionWrongType():
	""" Test Matrix referenceDataFrom() throws exception when other is not the same type """
	referenceDataFrom_exceptionWrongType(constructor)

def test_referenceDataFrom_sameReference():
	""" Test Matrix referenceDataFrom() successfully records the same reference """
	referenceDataFrom_sameReference(constructor)


#############
# copy #
#############

def test_copy_withZeros():
	""" Test Matrix copy() produces an equal object and doesn't just copy the references """
	copy_withZeros(constructor)


###################
# copyPoints #
###################

@raises(ArgumentException)
def test_copyPoints_exceptionNone():
	""" Test Matrix copyPoints() for exception when argument is None """
	copyPoints_exceptionNone(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionNonIndex():
	""" Test Matrix copyPoints() for exception when a value in the input is not a valid index """
	copyPoints_exceptionNonIndex(constructor)

def test_copyPoints_handmadeContents():
	""" Test Matrix copyPoints() returns the correct data """
	copyPoints_handmadeContents(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionStartInvalid():
	""" Test Matrix copyPoints() for ArgumentException when start is not a valid point index """
	copyPoints_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionEndInvalid():
	""" Test Matrix copyPoints() for ArgumentException when start is not a valid feature index """
	copyPoints_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionInversion():
	""" Test Matrix copyPoints() for ArgumentException when start comes after end """
	copyPoints_exceptionInversion(constructor)

def test_copyPoints_handmadeRange():
	""" Test Matrix copyPoints() against handmade output for range copying """
	copyPoints_handmadeRange(constructor)

def test_copyPoints_handmadeRangeWithFeatureNames():
	""" Test Matrix copyPoints() against handmade output for range copying with featureNames """
	copyPoints_handmadeRangeWithFeatureNames(constructor)

def test_copyPoints_handmadeRangeDefaults():
	""" Test Matrix copyPoints uses the correct defaults in the case of range based copying """
	copyPoints_handmadeRangeDefaults(constructor)

#####################
# copyFeatures #
#####################

@raises(ArgumentException)
def test_copyFeatures_exceptionNone():
	""" Test Matrix copyFeatures() for exception when argument is None """
	copyFeatures_exceptionNone(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionNonIndex():
	""" Test Matrix copyFeatures() for exception when a value in the input is not a valid index """
	copyFeatures_exceptionNonIndex(constructor)


def test_copyFeatures_handmadeContents():
	""" Test Matrix copyFeatures() returns the correct data """
	copyFeatures_handmadeContents(constructor)


@raises(ArgumentException)
def test_copyFeatures_exceptionStartInvalid():
	""" Test Matrix copyFeatures() for ArgumentException when start is not a valid feature index """
	copyFeatures_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionStartInvalidFeatureName():
	""" Test Matrix copyFeatures() for ArgumentException when start is not a valid feature FeatureName """
	copyFeatures_exceptionStartInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionEndInvalid():
	""" Test Matrix copyFeatures() for ArgumentException when start is not a valid feature index """
	copyFeatures_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionEndInvalidFeatureName():
	""" Test Matrix copyFeatures() for ArgumentException when start is not a valid featureName """
	copyFeatures_exceptionEndInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionInversion():
	""" Test Matrix copyFeatures() for ArgumentException when start comes after end """
	copyFeatures_exceptionInversion(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionInversionFeatureName():
	""" Test Matrix copyFeatures() for ArgumentException when start comes after end as FeatureNames"""
	copyFeatures_exceptionInversionFeatureName(constructor)


def test_copyFeatures_handmadeRange():
	""" Test Matrix copyFeatures() against handmade output for range copying """
	copyFeatures_handmadeRange(constructor)

def test_copyFeatures_handmadeWithFeatureNames():
	""" Test Matrix copyFeatures() against handmade output for range copying with FeatureNames """
	copyFeatures_handmadeWithFeatureNames(constructor)


##############
# __getitem__#
##############

def test_getitem_simpleExampeWithZeroes():
	""" Test Matrix __getitem__ returns the correct output for a number of simple queries """
	getitem_simpleExampeWithZeroes(constructor)



################
# pointView #
################

def test_pointView_isinstance():
	""" Test Matrix pointView returns an instance of the View in dataHelpers """
	pointView_isinstance(constructor)


##################
# featureView #
##################

def test_featureView_isinstance():
	""" Test Matrix featureView() returns an instance of the View in dataHelpers """
	featureView_isinstance(constructor)


############
# points() #
############

def test_points_vectorTest():
	""" Test Matrix points() when we only have row or column vectors of data """
	points_vectorTest(constructor)

############
# features() #
############

def test_features_vectorTest():
	""" Test Matrix features() when we only have row or column vectors of data """
	features_vectorTest(constructor)


