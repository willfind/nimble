"""
Unit tests for the DenseMatrixData object. Calls upon the functions defined
in derived_backend.py using appropriate input


"""

from derived_backend import *
from nose.tools import *
from ... import data as instantiate

def constructor(data=None, featureNames=None):
	return instantiate('DenseMatrixData', data, featureNames)


##############
# __init__() #
##############

def test_init_allEqual():
	""" Test DMD __init__() that every way to instantiate produces equal objects """
	init_allEqual(constructor)

def test_init_allEqualWithFeatureNames():
	""" Test DMD __init__() that every way to instantiate produces equal objects, with featureNames """
	init_allEqualWithFeatureNames(constructor)

############
# equals() #
############

def test_equals_False():
	""" Test DMD equals() against some non-equal input """
	equals_False(constructor)

def test_equals_True():
	""" Test DMD equals() against some actually equal input """
	equals_True(constructor)


###############
# transpose() #
###############

def test_transpose_handmade():
	""" Test DMD transpose() function against handmade output """
	transpose_handmade(constructor)


#############
# appendPoints() #
#############

@raises(ArgumentException)
def test_appendPoints_exceptionNone():
	""" Test DMD appendPoints() for ArgumentException when toAppend is None"""
	appendPoints_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendPoints_exceptionWrongSize():
	""" Test DMD appendPoints() for ArgumentException when toAppend has too many features """
	appendPoints_exceptionWrongSize(constructor)

def test_appendPoints_handmadeSingle():
	""" Test DMD appendPoints() against handmade output for a single added point """
	appendPoints_handmadeSingle(constructor)

def test_appendPoints_handmadeSequence():
	""" Test DMD appendPoints() against handmade output for a sequence of additions"""
	appendPoints_handmadeSequence(constructor)


################
# appendFeatures() #
################

@raises(ArgumentException)
def test_appendFeatures_exceptionNone():
	""" Test DMD appendFeatures() for ArgumentException when toAppend is None """
	appendFeatures_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendFeatures_exceptionWrongSize():
	""" Test DMD appendFeatures() for ArgumentException when toAppend has too many points """
	appendFeatures_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_appendFeatures_exceptionSameFeatureName():
	""" Test DMD appendFeatures() for ArgumentException when toAppend and self have a featureName in common """
	appendFeatures_exceptionSameFeatureName(constructor)

def test_appendFeatures_handmadeSingle():
	""" Test DMD appendFeatures() against handmade output for a single added feature"""
	appendFeatures_handmadeSingle(constructor)

def test_appendFeatures_handmadeSequence():
	""" Test DMD appendFeatures() against handmade output for a sequence of additions"""
	appendFeatures_handmadeSequence(constructor)



##############
# sortPoints() #
##############

@raises(ArgumentException)
def test_sortPoints_exceptionAtLeastOne():
	""" Test DMD sortPoints() has at least one paramater """
	sortPoints_exceptionAtLeastOne(constructor)

def test_sortPoints_naturalByFeature():
	""" Test DMD sortPoints() when we specify a feature to sort by """	
	sortPoints_naturalByFeature(constructor)

def test_sortPoints_scorer():
	""" Test DMD sortPoints() when we specify a scoring function """
	sortPoints_scorer(constructor)

def test_sortPoints_comparator():
	""" Test DMD sortPoints() when we specify a comparator function """
	sortPoints_comparator(constructor)


#################
# sortFeatures() #
#################


#def test_sortFeatures_handmadeWithFcn():
#	""" Test DMD sortFeatures() against handmade output when given cmp and key functions """	
#	sortFeatures_handmadeWithFcn(constructor)



#################
# extractPoints() #
#################

def test_extractPoints_emptyInput(): 
	""" Test DMD extractPoints() does nothing when not provided with any input """
	extractPoints_emptyInput(constructor)

def test_extractPoints_handmadeSingle():
	""" Test DMD extractPoints() against handmade output when extracting one point """
	extractPoints_handmadeSingle(constructor)

def test_extractPoints_handmadeListSequence():
	""" Test DMD extractPoints() against handmade output for several list extractions """
	extractPoints_handmadeListSequence(constructor)

def test_extractPoints_handmadeFunction():
	""" Test DMD extractPoints() against handmade output for function extraction """
	extractPoints_handmadeFunction(constructor)

def test_extractPoints_handmadeFuncionWithFeatureNames():
	""" Test DMD extractPoints() against handmade output for function extraction with featureNames"""
	extractPoints_handmadeFuncionWithFeatureNames(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionStartInvalid():
	""" Test DMD extractPoints() for ArgumentException when start is not a valid point index """
	extractPoints_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionEndInvalid():
	""" Test DMD extractPoints() for ArgumentException when start is not a valid feature index """
	extractPoints_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionInversion():
	""" Test DMD extractPoints() for ArgumentException when start comes after end """
	extractPoints_exceptionInversion(constructor)

def test_extractPoints_handmadeRange():
	""" Test DMD extractPoints() against handmade output for range extraction """
	extractPoints_handmadeRange(constructor)

def test_extractPoints_handmadeRangeWithFeatureNames():
	""" Test DMD extractPoints() against handmade output for range extraction with featureNames """
	extractPoints_handmadeRangeWithFeatureNames(constructor)

def test_extractPoints_handmadeRangeRand_FM():
	""" Test DMD extractPoints() against handmade output for randomized range extraction with featureNames """
	extractPoints_handmadeRangeRand_FM(constructor)

def test_extractPoints_handmadeRangeDefaults():
	""" Test DMD extractPoints() uses the correct defaults in the case of range based extraction """
	extractPoints_handmadeRangeDefaults(constructor)

####################
# extractFeatures() #
####################


def test_extractFeatures_handmadeSingle():
	""" Test DMD extractFeatures() against handmade output when extracting one feature """
	extractFeatures_handmadeSingle(constructor)

def test_extractFeatures_handmadeListSequence():
	""" Test DMD extractFeatures() against handmade output for several extractions by list """
	extractFeatures_handmadeListSequence(constructor)

def test_extractFeatures_handmadeListWithFeatureName():
	""" Test DMD extractFeatures() against handmade output for list extraction when specifying featureNames """
	extractFeatures_handmadeListWithFeatureName(constructor)

def test_extractFeatures_handmadeFunction():
	""" Test DMD extractFeatures() against handmade output for function extraction """
	extractFeatures_handmadeFunction(constructor)

def test_extractFeatures_handmadeFunctionWithFeatureName():
	""" Test DMD extractFeatures() against handmade output for function extraction with featureNames """
	extractFeatures_handmadeFunctionWithFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionStartInvalid():
	""" Test DMD extractFeatures() for ArgumentException when start is not a valid feature index """
	extractFeatures_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionStartInvalidFeatureName():
	""" Test DMD extractFeatures() for ArgumentException when start is not a valid featureName """
	extractFeatures_exceptionStartInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionEndInvalid():
	""" Test DMD extractFeatures() for ArgumentException when start is not a valid feature index """
	extractFeatures_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionEndInvalidFeatureName():
	""" Test DMD extractFeatures() for ArgumentException when start is not a valid featureName """
	extractFeatures_exceptionEndInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionInversion():
	""" Test DMD extractFeatures() for ArgumentException when start comes after end """
	extractFeatures_exceptionInversion(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionInversionFeatureName():
	""" Test DMD extractFeatures() for ArgumentException when start comes after end as FeatureNames"""
	extractFeatures_exceptionInversionFeatureName(constructor)

def test_extractFeatures_handmadeRange():
	""" Test DMD extractFeatures() against handmade output for range extraction """
	extractFeatures_handmadeRange(constructor)

def test_extractFeatures_handmadeWithFeatureNames():
	""" Test DMD extractFeatures() against handmade output for range extraction with FeatureNames """
	extractFeatures_handmadeWithFeatureNames(constructor)



####################
# applyFunctionToEachPoint() #
####################

@raises(ArgumentException)
def test_applyFunctionToEachPoint_exceptionInputNone():
	""" Test DMD applyFunctionToEachPoint() for ArgumentException when function is None """
	applyFunctionToEachPoint_exceptionInputNone(constructor)

def test_applyFunctionToEachPoint_Handmade():
	""" Test DMD applyFunctionToEachPoint() with handmade output """
	applyFunctionToEachPoint_Handmade(constructor)

def test_applyFunctionToEachPoint_nonZeroItAndLen():
	""" Test DMD applyFunctionToEachPoint() for the correct usage of the nonzero iterator """
	applyFunctionToEachPoint_nonZeroItAndLen(constructor)

#######################
# applyFunctionToEachFeature() #
#######################

@raises(ArgumentException)
def test_applyFunctionToEachFeature_exceptionInputNone():
	""" Test DMD applyFunctionToEachFeature() for ArgumentException when function is None """
	applyFunctionToEachFeature_exceptionInputNone(constructor)

def test_applyFunctionToEachFeature_Handmade():
	""" Test DMD applyFunctionToEachFeature() with handmade output """
	applyFunctionToEachFeature_Handmade(constructor)

def test_applyFunctionToEachFeature_nonZeroItAndLen():
	""" Test DMD applyFunctionToEachFeature() for the correct usage of the nonzero iterator """
	applyFunctionToEachFeature_nonZeroItAndLen(constructor)


#####################
# mapReduceOnPoints() #
#####################

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionNoneMap():
	""" Test DMD mapReduceOnPoints() for ArgumentException when mapper is None """
	mapReduceOnPoints_argumentExceptionNoneMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionNoneReduce():
	""" Test DMD mapReduceOnPoints() for ArgumentException when reducer is None """
	mapReduceOnPoints_argumentExceptionNoneReduce(constructor)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionUncallableMap():
	""" Test DMD mapReduceOnPoints() for ArgumentException when mapper is not callable """
	mapReduceOnPoints_argumentExceptionUncallableMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionUncallableReduce():
	""" Test DMD mapReduceOnPoints() for ArgumentException when reducer is not callable """
	mapReduceOnPoints_argumentExceptionUncallableReduce(constructor)



def test_mapReduceOnPoints_handmade():
	""" Test DMD mapReduceOnPoints() against handmade output """
	mapReduceOnPoints_handmade(constructor)

def test_mapReduceOnPoints_handmadeNoneReturningReducer():
	""" Test DMD mapReduceOnPoints() against handmade output with a None returning Reducer """
	mapReduceOnPoints_handmadeNoneReturningReducer(constructor)



##########################
# toRowListData() #
##########################


def test_toRowListData_handmade_defaultFeatureNames():
	""" Test DMD toRowListData with default featureNames """
	toRowListData_handmade_defaultFeatureNames(constructor)

	
def test_toRowListData_handmade_assignedFeatureNames():
	""" Test DMD toRowListData with assigned featureNames """
	toRowListData_handmade_assignedFeatureNames(constructor)



##############################
# toDenseMatrixData() #
##############################


def test_toDenseMatrixData_handmade_defaultFeatureNames():
	""" Test DMD toDenseMatrixData with default featureNames """
	toDenseMatrixData_handmade_defaultFeatureNames(constructor)

	
def test_toDenseMatrixData_handmade_assignedFeatureNames():
	""" Test DMD toDenseMatrixData with assigned featureNames """
	toDenseMatrixData_handmade_assignedFeatureNames(constructor)



############
# writeFile #
############

def test_writeFileCSV_handmade():
	""" Test DMD writeFile() for csv extension with both data and featureNames """
	writeFileCSV_handmade(constructor)

def test_writeFileMTX_handmade():
	""" Test DMD writeFile() for mtx extension with both data and featureNames """
	writeFileMTX_handmade(constructor)


#####################
# copyReferences #
#####################


@raises(ArgumentException)
def test_copyReferences_exceptionWrongType():
	""" Test DMD copyReferences() throws exception when other is not the same type """
	copyReferences_exceptionWrongType(constructor)

def test_copyReferences_sameReference():
	""" Test DMD copyReferences() successfully records the same reference """
	copyReferences_sameReference(constructor)


#############
# duplicate #
#############

def test_duplicate_withZeros():
	""" Test DMD duplicate() produces an equal object and doesn't just copy the references """
	duplicate_withZeros(constructor)


###################
# copyPoints #
###################

@raises(ArgumentException)
def test_copyPoints_exceptionNone():
	""" Test DMD copyPoints() for exception when argument is None """
	copyPoints_exceptionNone(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionNonIndex():
	""" Test DMD copyPoints() for exception when a value in the input is not a valid index """
	copyPoints_exceptionNonIndex(constructor)

def test_copyPoints_handmadeContents():
	""" Test DMD copyPoints() returns the correct data """
	copyPoints_handmadeContents(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionStartInvalid():
	""" Test DMD copyPoints() for ArgumentException when start is not a valid point index """
	copyPoints_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionEndInvalid():
	""" Test DMD copyPoints() for ArgumentException when start is not a valid feature index """
	copyPoints_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_copyPoints_exceptionInversion():
	""" Test DMD copyPoints() for ArgumentException when start comes after end """
	copyPoints_exceptionInversion(constructor)

def test_copyPoints_handmadeRange():
	""" Test DMD copyPoints() against handmade output for range copying """
	copyPoints_handmadeRange(constructor)

def test_copyPoints_handmadeRangeWithFeatureNames():
	""" Test DMD copyPoints() against handmade output for range copying with featureNames """
	copyPoints_handmadeRangeWithFeatureNames(constructor)

def test_copyPoints_handmadeRangeDefaults():
	""" Test DMD copyPoints uses the correct defaults in the case of range based copying """
	copyPoints_handmadeRangeDefaults(constructor)

#####################
# copyFeatures #
#####################

@raises(ArgumentException)
def test_copyFeatures_exceptionNone():
	""" Test DMD copyFeatures() for exception when argument is None """
	copyFeatures_exceptionNone(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionNonIndex():
	""" Test DMD copyFeatures() for exception when a value in the input is not a valid index """
	copyFeatures_exceptionNonIndex(constructor)


def test_copyFeatures_handmadeContents():
	""" Test DMD copyFeatures() returns the correct data """
	copyFeatures_handmadeContents(constructor)


@raises(ArgumentException)
def test_copyFeatures_exceptionStartInvalid():
	""" Test DMD copyFeatures() for ArgumentException when start is not a valid feature index """
	copyFeatures_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionStartInvalidFeatureName():
	""" Test DMD copyFeatures() for ArgumentException when start is not a valid feature FeatureName """
	copyFeatures_exceptionStartInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionEndInvalid():
	""" Test DMD copyFeatures() for ArgumentException when start is not a valid feature index """
	copyFeatures_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionEndInvalidFeatureName():
	""" Test DMD copyFeatures() for ArgumentException when start is not a valid featureName """
	copyFeatures_exceptionEndInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionInversion():
	""" Test DMD copyFeatures() for ArgumentException when start comes after end """
	copyFeatures_exceptionInversion(constructor)

@raises(ArgumentException)
def test_copyFeatures_exceptionInversionFeatureName():
	""" Test DMD copyFeatures() for ArgumentException when start comes after end as FeatureNames"""
	copyFeatures_exceptionInversionFeatureName(constructor)


def test_copyFeatures_handmadeRange():
	""" Test DMD copyFeatures() against handmade output for range copying """
	copyFeatures_handmadeRange(constructor)

def test_copyFeatures_handmadeWithFeatureNames():
	""" Test DMD copyFeatures() against handmade output for range copying with FeatureNames """
	copyFeatures_handmadeWithFeatureNames(constructor)


##############
# __getitem__#
##############

def test_getitem_simpleExampeWithZeroes():
	""" Test DMD __getitem__ returns the correct output for a number of simple queries """
	getitem_simpleExampeWithZeroes(constructor)



################
# getPointView #
################

def test_getPointView_isinstance():
	""" Test DMD getPointView returns an instance of the View in base_data """
	getPointView_isinstance(constructor)


##################
# getFeatureView #
##################

def test_getFeatureView_isinstance():
	""" Test DMD getFeatureView() returns an instance of the View in base_data """
	getFeatureView_isinstance(constructor)
