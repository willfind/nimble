"""
Unit tests for the List object. Calls upon the functions defined
in derived_backend.py using appropriate input


"""
from nose.tools import *

from UML.data.tests.derived_backend import *
from UML import create
from UML.exceptions import ArgumentException

def constructor(data=None, featureNames=None):
	return create('List', data, featureNames)


##############
# __init__() #
##############

def test_init_allEqual():
	""" Test List __init__() that every way to instantiate produces equal objects """
	init_allEqual(constructor)

def test_init_allEqualWithFeatureNames():
	""" Test List __init__() that every way to instantiate produces equal objects, with featureNames """
	init_allEqualWithFeatureNames(constructor)


############
# equals() #
############

def test_equals_False():
	""" Test List equals() against some non-equal input """
	equals_False(constructor)

def test_equals_True():
	""" Test List equals() against some actually equal input """
	equals_True(constructor)


###############
# transpose() #
###############

def test_transpose_handmade():
	""" Test List transpose() function against handmade output """
	transpose_handmade(constructor)


#############
# appendPoints() #
#############

@raises(ArgumentException)
def test_appendPoints_exceptionNone():
	""" Test List appendPoints() for ArgumentException when toAppend is None"""
	appendPoints_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendPoints_exceptionWrongSize():
	""" Test List appendPoints() for ArgumentException when toAppend has too many features """
	appendPoints_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_appendPoints_exceptionMismatchedFeatureNames():
	""" Test List appendPoints() for ArgumentException when toAppend and self's feature names do not match"""
	appendPoints_exceptionMismatchedFeatureNames(constructor)

def test_appendPoints_handmadeSingle():
	""" Test List appendPoints() against handmade output for a single added point """
	appendPoints_handmadeSingle(constructor)

def test_appendPoints_handmadeSequence():
	""" Test List appendPoints() against handmade output for a sequence of additions"""
	appendPoints_handmadeSequence(constructor)


################
# appendFeatures() #
################

@raises(ArgumentException)
def test_appendFeatures_exceptionNone():
	""" Test List appendFeatures() for ArgumentException when toAppend is None """
	appendFeatures_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendFeatures_exceptionWrongSize():
	""" Test List appendFeatures() for ArgumentException when toAppend has too many points """
	appendFeatures_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_appendFeatures_exceptionSameFeatureName():
	""" Test List appendFeatures() for ArgumentException when toAppend and self have a featureName in common """
	appendFeatures_exceptionSameFeatureName(constructor)

def test_appendFeatures_handmadeSingle():
	""" Test List appendFeatures() against handmade output for a single added feature"""
	appendFeatures_handmadeSingle(constructor)

def test_appendFeatures_handmadeSequence():
	""" Test List appendFeatures() against handmade output for a sequence of additions"""
	appendFeatures_handmadeSequence(constructor)



##############
# sortPoints() #
##############

@raises(ArgumentException)
def test_sortPoints_exceptionAtLeastOne():
	""" Test List sortPoints() has at least one paramater """
	sortPoints_exceptionAtLeastOne(constructor)

def test_sortPoints_naturalByFeature():
	""" Test List sortPoints() when we specify a feature to sort by """	
	sortPoints_naturalByFeature(constructor)

def test_sortPoints_scorer():
	""" Test List sortPoints() when we specify a scoring function """
	sortPoints_scorer(constructor)

def test_sortPoints_comparator():
	""" Test List sortPoints() when we specify a comparator function """
	sortPoints_comparator(constructor)

#################
# sortFeatures() #
#################


@raises(ArgumentException)
def test_sortFeatures_exceptionAtLeastOne():
	""" Test List sortFeatures() has at least one paramater """
	sortFeatures_exceptionAtLeastOne(constructor)

def test_sortFeatures_naturalByPointWithNames():
	""" Test List sortFeatures() when we specify a point to sort by; includes featureNames """	
	sortFeatures_naturalByPointWithNames(constructor)

def test_sortFeatures_scorer():
	""" Test List sortFeatures() when we specify a scoring function """
	sortFeatures_scorer(constructor)

def test_sortFeatures_comparator():
	""" Test List sortFeatures() when we specify a comparator function """
	sortFeatures_comparator(constructor)



#################
# extractPoints() #
#################

def test_extractPoints_emptyInput(): 
	""" Test List extractPoints() does nothing when not provided with any input """
	extractPoints_emptyInput(constructor)

def test_extractPoints_handmadeSingle():
	""" Test List extractPoints() against handmade output when extracting one point """
	extractPoints_handmadeSingle(constructor)

def test_extractPoints_handmadeListSequence():
	""" Test List extractPoints() against handmade output for several list extractions """
	extractPoints_handmadeListSequence(constructor)

def test_extractPoints_handmadeListOrdering():
	""" Test List extractPoints() against handmade output for out of order extraction """
	extractPoints_handmadeListOrdering(constructor)

def test_extractPoints_handmadeFunction():
	""" Test List extractPoints() against handmade output for function extraction """
	extractPoints_handmadeFunction(constructor)

def test_extractPoints_handmadeFuncionWithFeatureNames():
	""" Test List extractPoints() against handmade output for function extraction with featureNames"""
	extractPoints_handmadeFuncionWithFeatureNames(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionStartInvalid():
	""" Test List extractPoints() for ArgumentException when start is not a valid point index """
	extractPoints_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionEndInvalid():
	""" Test List extractPoints() for ArgumentException when start is not a valid feature index """
	extractPoints_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionInversion():
	""" Test List extractPoints() for ArgumentException when start comes after end """
	extractPoints_exceptionInversion(constructor)

def test_extractPoints_handmadeRange():
	""" Test List extractPoints() against handmade output for range extraction """
	extractPoints_handmadeRange(constructor)

def test_extractPoints_handmadeRangeWithFeatureNames():
	""" Test List extractPoints() against handmade output for range extraction with featureNames """
	extractPoints_handmadeRangeWithFeatureNames(constructor)

def test_extractPoints_handmadeRangeRand_FM():
	""" Test List extractPoints() against handmade output for randomized range extraction with featureNames """
	extractPoints_handmadeRangeRand_FM(constructor)

def test_extractPoints_handmadeRangeDefaults():
	""" Test List extractPoints() uses the correct defaults in the case of range based extraction """
	extractPoints_handmadeRangeDefaults(constructor)


####################
# extractFeatures() #
####################

def test_extractFeatures_handmadeSingle():
	""" Test List extractFeatures() against handmade output when extracting one feature """
	extractFeatures_handmadeSingle(constructor)

def test_extractFeatures_handmadeListSequence():
	""" Test List extractFeatures() against handmade output for several extractions by list """
	extractFeatures_handmadeListSequence(constructor)

def test_extractFeatures_handmadeListWithFeatureName():
	""" Test List extractFeatures() against handmade output for list extraction when specifying featureNames """
	extractFeatures_handmadeListWithFeatureName(constructor)

def test_extractFeatures_handmadeFunction():
	""" Test List extractFeatures() against handmade output for function extraction """
	extractFeatures_handmadeFunction(constructor)

def test_extractFeatures_handmadeFunctionWithFeatureName():
	""" Test List extractFeatures() against handmade output for function extraction with featureNames """
	extractFeatures_handmadeFunctionWithFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionStartInvalid():
	""" Test List extractFeatures() for ArgumentException when start is not a valid feature index """
	extractFeatures_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionStartInvalidFeatureName():
	""" Test List extractFeatures() for ArgumentException when start is not a valid featureName """
	extractFeatures_exceptionStartInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionEndInvalid():
	""" Test List extractFeatures() for ArgumentException when start is not a valid feature index """
	extractFeatures_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionEndInvalidFeatureName():
	""" Test List extractFeatures() for ArgumentException when start is not a valid featureName """
	extractFeatures_exceptionEndInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionInversion():
	""" Test List extractFeatures() for ArgumentException when start comes after end """
	extractFeatures_exceptionInversion(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionInversionFeatureName():
	""" Test List extractFeatures() for ArgumentException when start comes after end as FeatureNames"""
	extractFeatures_exceptionInversionFeatureName(constructor)

def test_extractFeatures_handmadeRange():
	""" Test List extractFeatures() against handmade output for range extraction """
	extractFeatures_handmadeRange(constructor)

def test_extractFeatures_handmadeWithFeatureNames():
	""" Test List extractFeatures() against handmade output for range extraction with FeatureNames """
	extractFeatures_handmadeWithFeatureNames(constructor)


##########################
# toList() #
##########################


def test_toList_handmade_defaultFeatureNames():
	""" Test List toList with default featureNames """
	toList_handmade_defaultFeatureNames(constructor)

	
def test_toList_handmade_assignedFeatureNames():
	""" Test List toList with assigned featureNames """
	toList_handmade_assignedFeatureNames(constructor)



##############################
# toDense() #
##############################


def test_toDense_handmade_defaultFeatureNames():
	""" Test List toDense with default featureNames """
	toDense_handmade_defaultFeatureNames(constructor)

	
def test_toDense_handmade_assignedFeatureNames():
	""" Test List toDense with assigned featureNames """
	toDense_handmade_assignedFeatureNames(constructor)



############
# writeFile #
############

def test_writeFileCSV_handmade():
	""" Test List writeFile() for csv extension with both data and featureNames """
	writeFileCSV_handmade(constructor)

def test_writeFileMTX_handmade():
	""" Test List writeFile() for mtx extension with both data and featureNames """
	writeFileMTX_handmade(constructor)


#####################
# copyReferences #
#####################


@raises(ArgumentException)
def test_copyReferences_exceptionWrongType():
	""" Test List copyReferences() throws exception when other is not the same type """
	copyReferences_exceptionWrongType(constructor)

def test_copyReferences_sameReference():
	""" Test List copyReferences() successfully records the same reference """
	copyReferences_sameReference(constructor)


#############
# duplicate #
#############

def test_duplicate_withZeros():
	""" Test List duplicate() produces an equal object and doesn't just copy the references """
	duplicate_withZeros(constructor)

###################
# copyPoints #
###################

@raises(ArgumentException)
def test_copyPoints_exceptionNone():
	""" Test List copyPoints() for exception when argument is None """
	copyPoints_exceptionNone(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionNonIndex():
	""" Test List copyPoints() for exception when a value in the input is not a valid index """
	copyPoints_exceptionNonIndex(constructor)

def test_copyPoints_handmadeContents():
	""" Test List copyPoints() returns the correct data """
	copyPoints_handmadeContents(constructor)



@raises(ArgumentException)
def test_copyPoints_exceptionStartInvalid():
	""" Test List copyPoints() for ArgumentException when start is not a valid point index """
	copyPoints_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionEndInvalid():
	""" Test List copyPoints() for ArgumentException when start is not a valid feature index """
	copyPoints_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionInversion():
	""" Test List copyPoints() for ArgumentException when start comes after end """
	copyPoints_exceptionInversion(constructor)

def test_copyPoints_handmadeRange():
	""" Test List copyPoints() against handmade output for range copying """
	copyPoints_handmadeRange(constructor)

def test_copyPoints_handmadeRangeWithFeatureNames():
	""" Test List copyPoints() against handmade output for range copying with featureNames """
	copyPoints_handmadeRangeWithFeatureNames(constructor)

def test_copyPoints_handmadeRangeDefaults():
	""" Test List copyPoints uses the correct defaults in the case of range based copying """
	copyPoints_handmadeRangeDefaults(constructor)

#####################
# copyFeatures #
#####################

@raises(ArgumentException)
def test_copyFeatures_exceptionNone():
	""" Test List copyFeatures() for exception when argument is None """
	copyFeatures_exceptionNone(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionNonIndex():
	""" Test List copyFeatures() for exception when a value in the input is not a valid index """
	copyFeatures_exceptionNonIndex(constructor)


def test_copyFeatures_handmadeContents():
	""" Test List copyFeatures() returns the correct data """
	copyFeatures_handmadeContents(constructor)


####

@raises(ArgumentException)
def test_copyFeatures_exceptionStartInvalid():
	""" Test List copyFeatures() for ArgumentException when start is not a valid feature index """
	copyFeatures_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionStartInvalidFeatureName():
	""" Test List copyFeatures() for ArgumentException when start is not a valid feature FeatureName """
	copyFeatures_exceptionStartInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionEndInvalid():
	""" Test List copyFeatures() for ArgumentException when start is not a valid feature index """
	copyFeatures_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionEndInvalidFeatureName():
	""" Test List copyFeatures() for ArgumentException when start is not a valid featureName """
	copyFeatures_exceptionEndInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionInversion():
	""" Test List copyFeatures() for ArgumentException when start comes after end """
	copyFeatures_exceptionInversion(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionInversionFeatureName():
	""" Test List copyFeatures() for ArgumentException when start comes after end as FeatureNames"""
	copyFeatures_exceptionInversionFeatureName(constructor)


def test_copyFeatures_handmadeRange():
	""" Test List copyFeatures() against handmade output for range copying """
	copyFeatures_handmadeRange(constructor)

def test_copyFeatures_handmadeWithFeatureNames():
	""" Test List copyFeatures() against handmade output for range copying with FeatureNames """
	copyFeatures_handmadeWithFeatureNames(constructor)


##############
# __getitem__#
##############


def test_getitem_simpleExampeWithZeroes():
	""" Test List __getitem__ returns the correct output for a number of simple queries """
	getitem_simpleExampeWithZeroes(constructor)


################
# getPointView #
################

def test_getPointView_isinstance():
	""" Test List getPointView returns an instance of the View in dataHelpers """
	getPointView_isinstance(constructor)


##################
# getFeatureView #
##################

def test_getFeatureView_isinstance():
	""" Test List getFeatureView() returns an instance of the View in dataHelpers """
	getFeatureView_isinstance(constructor)



############
# points() #
############

def test_points_vectorTest():
	""" Test List points() when we only have row or column vectors of data """
	points_vectorTest(constructor)

############
# features() #
############

def test_features_vectorTest():
	""" Test List features() when we only have row or column vectors of data """
	features_vectorTest(constructor)
