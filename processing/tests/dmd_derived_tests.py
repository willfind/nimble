"""
Unit tests for the DenseMatrixData object. Calls upon the functions defined
in derived_backend.py using appropriate input


"""

from derived_backend import *
from ..dense_matrix_data import *
from nose.tools import *

def constructor(data=None, featureNames=None, file=None):
	return DenseMatrixData(data, featureNames, file)


##############
# __init__() #
##############

def test_init_allEqual():
	""" Test DMD __init__() so that each possible way to instantiate produces equal objects """
	init_allEqual(constructor)


############
# equals() #
############

def test_equals_False():
	""" Test DMD equals() against some non-equal input """
	equals_False(constructor)

def test_equals_True():
	""" Test DMD equals() against some actually equal input """
	equals_True(constructor)

def test_equals_empty():
	""" Test DMD equals() for empty objects """
	equals_empty(constructor)


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

def test_sortPoints_handmadeNatural():
	""" Test DMD sortPoints() against handmade, naturally ordered output """	
	sortPoints_handmadeNatural(constructor)

def test_sortPoints_handmadeWithFcn():
	""" Test DMD sortPoints() against handmade output when given cmp and key functions """	
	sortPoints_handmadeWithFcn(constructor)

#################
# sortFeatures() #
#################


def test_sortFeatures_handmadeWithFcn():
	""" Test DMD sortFeatures() against handmade output when given cmp and key functions """	
	sortFeatures_handmadeWithFcn(constructor)



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

def test_extractPoints_handmade():
	""" Test DMD extractPoints() against handmade output for range extraction """
	extractPoints_handmade(constructor)

def test_extractPoints_handmadeWithFeatureNames():
	""" Test DMD extractPoints() against handmade output for range extraction with featureNames """
	extractPoints_handmadeWithFeatureNames(constructor)





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
# writeCSV #
############

def test_writeCSV_handmade():
	""" Test DMD writeCSV with both data and featureNames """
	writeCSV_handmade(constructor)


###########
# writeMM #
###########


def test_writeMM_handmade():
	""" Test dmd writeMM with both data and featureNames """
	writeMM_handmade(constructor)


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



###################
# duplicatePoints #
###################

@raises(ArgumentException)
def test_duplicatePoints_exceptionNone():
	""" Test DMD duplicatePoints() for exception when argument is None """
	duplicatePoints_exceptionNone(constructor)

@raises(ArgumentException)
def test_duplicatePoints_exceptionNonIndex():
	""" Test DMD duplicatePoints() for exception when a value in the input is not a valid index """
	duplicatePoints_exceptionNonIndex(constructor)

def test_duplicatePoints_handmadeContents():
	""" Test DMD duplicatePoints() returns the correct data """
	duplicatePoints_handmadeContents(constructor)
