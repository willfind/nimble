"""
Unit tests for the RowListData object. Calls upon the functions defined
in derived_backend.py using appropriate input


"""

from derived_backend import *
from ..row_list_data import *
from nose.tools import *

def constructor(data=None, featureNames=None, file=None):
	return RowListData(data, featureNames, file)


##############
# __init__() #
##############

def test_init_allEqual():
	""" Test RLD __init__() so that each possible way to instantiate produces equal objects """
	init_allEqual(constructor)

############
# equals() #
############

def test_equals_False():
	""" Test RLD equals() against some non-equal input """
	equals_False(constructor)

def test_equals_True():
	""" Test RLD equals() against some actually equal input """
	equals_True(constructor)

def test_equals_empty():
	""" Test RLD equals() for empty objects """
	equals_empty(constructor)


###############
# transpose() #
###############

def test_transpose_handmade():
	""" Test RLD transpose() function against handmade output """
	transpose_handmade(constructor)


#############
# appendPoints() #
#############

@raises(ArgumentException)
def test_appendPoints_exceptionNone():
	""" Test RLD appendPoints() for ArgumentException when toAppend is None"""
	appendPoints_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendPoints_exceptionWrongSize():
	""" Test RLD appendPoints() for ArgumentException when toAppend has too many features """
	appendPoints_exceptionWrongSize(constructor)

def test_appendPoints_handmadeSingle():
	""" Test RLD appendPoints() against handmade output for a single added point """
	appendPoints_handmadeSingle(constructor)

def test_appendPoints_handmadeSequence():
	""" Test RLD appendPoints() against handmade output for a sequence of additions"""
	appendPoints_handmadeSequence(constructor)


################
# appendFeatures() #
################

@raises(ArgumentException)
def test_appendFeatures_exceptionNone():
	""" Test RLD appendFeatures() for ArgumentException when toAppend is None """
	appendFeatures_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendFeatures_exceptionWrongSize():
	""" Test RLD appendFeatures() for ArgumentException when toAppend has too many points """
	appendFeatures_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_appendFeatures_exceptionSameFeatureName():
	""" Test RLD appendFeatures() for ArgumentException when toAppend and self have a featureName in common """
	appendFeatures_exceptionSameFeatureName(constructor)

def test_appendFeatures_handmadeSingle():
	""" Test RLD appendFeatures() against handmade output for a single added feature"""
	appendFeatures_handmadeSingle(constructor)

def test_appendFeatures_handmadeSequence():
	""" Test RLD appendFeatures() against handmade output for a sequence of additions"""
	appendFeatures_handmadeSequence(constructor)



##############
# sortPoints() #
##############

def test_sortPoints_handmadeNatural():
	""" Test RLD sortPoints() against handmade , naturally ordered  output """	
	sortPoints_handmadeNatural(constructor)

def test_sortPoints_handmadeWithFcn():
	""" Test RLD sortPoints() against handmade output when given a key function """	
	sortPoints_handmadeWithFcn(constructor)

#################
# sortFeatures() #
#################


def test_sortFeatures_handmadeWithFcn():
	""" Test RLD sortFeatures() against handmade output when given cmp and key functions """	
	sortFeatures_handmadeWithFcn(constructor)



#################
# extractPoints() #
#################

def test_extractPoints_emptyInput(): 
	""" Test RLD extractPoints() does nothing when not provided with any input """
	extractPoints_emptyInput(constructor)

def test_extractPoints_handmadeSingle():
	""" Test RLD extractPoints() against handmade output when extracting one point """
	extractPoints_handmadeSingle(constructor)

def test_extractPoints_handmadeListSequence():
	""" Test RLD extractPoints() against handmade output for several list extractions """
	extractPoints_handmadeListSequence(constructor)

def test_extractPoints_handmadeFunction():
	""" Test RLD extractPoints() against handmade output for function extraction """
	extractPoints_handmadeFunction(constructor)

def test_extractPoints_handmadeFuncionWithFeatureNames():
	""" Test RLD extractPoints() against handmade output for function extraction with featureNames"""
	extractPoints_handmadeFuncionWithFeatureNames(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionStartInvalid():
	""" Test RLD extractPoints() for ArgumentException when start is not a valid point index """
	extractPoints_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionEndInvalid():
	""" Test RLD extractPoints() for ArgumentException when start is not a valid feature index """
	extractPoints_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionInversion():
	""" Test RLD extractPoints() for ArgumentException when start comes after end """
	extractPoints_exceptionInversion(constructor)

def test_extractPoints_handmade():
	""" Test RLD extractPoints() against handmade output for range extraction """
	extractPoints_handmade(constructor)

def test_extractPoints_handmadeWithFeatureNames():
	""" Test RLD extractPoints() against handmade output for range extraction with featureNames """
	extractPoints_handmadeWithFeatureNames(constructor)


####################
# extractFeatures() #
####################

def test_extractFeatures_handmadeSingle():
	""" Test RLD extractFeatures() against handmade output when extracting one feature """
	extractFeatures_handmadeSingle(constructor)

def test_extractFeatures_handmadeListSequence():
	""" Test RLD extractFeatures() against handmade output for several extractions by list """
	extractFeatures_handmadeListSequence(constructor)

def test_extractFeatures_handmadeListWithFeatureName():
	""" Test RLD extractFeatures() against handmade output for list extraction when specifying featureNames """
	extractFeatures_handmadeListWithFeatureName(constructor)

def test_extractFeatures_handmadeFunction():
	""" Test RLD extractFeatures() against handmade output for function extraction """
	extractFeatures_handmadeFunction(constructor)

def test_extractFeatures_handmadeFunctionWithFeatureName():
	""" Test RLD extractFeatures() against handmade output for function extraction with featureNames """
	extractFeatures_handmadeFunctionWithFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionStartInvalid():
	""" Test RLD extractFeatures() for ArgumentException when start is not a valid feature index """
	extractFeatures_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionStartInvalidFeatureName():
	""" Test RLD extractFeatures() for ArgumentException when start is not a valid featureName """
	extractFeatures_exceptionStartInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionEndInvalid():
	""" Test RLD extractFeatures() for ArgumentException when start is not a valid feature index """
	extractFeatures_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionEndInvalidFeatureName():
	""" Test RLD extractFeatures() for ArgumentException when start is not a valid featureName """
	extractFeatures_exceptionEndInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionInversion():
	""" Test RLD extractFeatures() for ArgumentException when start comes after end """
	extractFeatures_exceptionInversion(constructor)

@raises(ArgumentException)
def test_extractFeatures_exceptionInversionFeatureName():
	""" Test RLD extractFeatures() for ArgumentException when start comes after end as FeatureNames"""
	extractFeatures_exceptionInversionFeatureName(constructor)

def test_extractFeatures_handmadeRange():
	""" Test RLD extractFeatures() against handmade output for range extraction """
	extractFeatures_handmadeRange(constructor)

def test_extractFeatures_handmadeWithFeatureNames():
	""" Test RLD extractFeatures() against handmade output for range extraction with FeatureNames """
	extractFeatures_handmadeWithFeatureNames(constructor)


####################
# applyFunctionToEachPoint() #
####################

@raises(ArgumentException)
def test_applyFunctionToEachPoint_exceptionInputNone():
	""" Test RLD applyFunctionToEachPoint() for ArgumentException when function is None """
	applyFunctionToEachPoint_exceptionInputNone(constructor)

def test_applyFunctionToEachPoint_Handmade():
	""" Test RLD applyFunctionToEachPoint() with handmade output """
	applyFunctionToEachPoint_Handmade(constructor)


#######################
# applyFunctionToEachFeature() #
#######################

@raises(ArgumentException)
def test_applyFunctionToEachFeature_exceptionInputNone():
	""" Test RLD applyFunctionToEachFeature() for ArgumentException when function is None """
	applyFunctionToEachFeature_exceptionInputNone(constructor)

def test_applyFunctionToEachFeature_Handmade():
	""" Test RLD applyFunctionToEachFeature() with handmade output """
	applyFunctionToEachFeature_Handmade(constructor)



#####################
# mapReduceOnPoints() #
#####################

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionNoneMap():
	""" Test RLD mapReduceOnPoints() for ArgumentException when mapper is None """
	mapReduceOnPoints_argumentExceptionNoneMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionNoneReduce():
	""" Test RLD mapReduceOnPoints() for ArgumentException when reducer is None """
	mapReduceOnPoints_argumentExceptionNoneReduce(constructor)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionUncallableMap():
	""" Test RLD mapReduceOnPoints() for ArgumentException when mapper is not callable """
	mapReduceOnPoints_argumentExceptionUncallableMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionUncallableReduce():
	""" Test RLD mapReduceOnPoints() for ArgumentException when reducer is not callable """
	mapReduceOnPoints_argumentExceptionUncallableReduce(constructor)


def test_mapReduceOnPoints_handmade():
	""" Test RLD mapReduceOnPoints() against handmade output """
	mapReduceOnPoints_handmade(constructor)

def test_mapReduceOnPoints_handmadeNoneReturningReducer():
	""" Test RLD mapReduceOnPoints() against handmade output with a None returning Reducer """
	mapReduceOnPoints_handmadeNoneReturningReducer(constructor)


##########################
# toRowListData() #
##########################


def test_toRowListData_handmade_defaultFeatureNames():
	""" Test RLD toRowListData with default featureNames """
	toRowListData_handmade_defaultFeatureNames(constructor)

	
def test_toRowListData_handmade_assignedFeatureNames():
	""" Test RLD toRowListData with assigned featureNames """
	toRowListData_handmade_assignedFeatureNames(constructor)



##############################
# toDenseMatrixData() #
##############################


def test_toDenseMatrixData_handmade_defaultFeatureNames():
	""" Test RLD toDenseMatrixData with default featureNames """
	toDenseMatrixData_handmade_defaultFeatureNames(constructor)

	
def test_toDenseMatrixData_handmade_assignedFeatureNames():
	""" Test RLD toDenseMatrixData with assigned featureNames """
	toDenseMatrixData_handmade_assignedFeatureNames(constructor)




############
# writeCSV #
############

def test_writeCSV_handmade():
	""" Test RLD writeCSV with both data and featureNames """
	writeCSV_handmade(constructor)


###########
# writeMM #
###########


def test_writeMM_handmade():
	""" Test RLD writeMM with both data and featureNames """
	writeMM_handmade(constructor)



#####################
# copyDataReference #
#####################

@raises(ArgumentException)
def test_copyDataReference_exceptionInconsistentFeatures():
	""" Test RLD copyDataReference() throws exception when the number of features doesn't match"""
	copyDataReference_exceptionInconsistentFeatures(constructor)

@raises(ArgumentException)
def test_copyDataReference_exceptionWrongType():
	""" Test RLD copyDataReference() throws exception when other is not the same type """
	copyDataReference_exceptionWrongType(constructor)

def test_copyDataReference_sameReference():
	""" Test RLD copyDataReference() successfully records the same reference """
	copyDataReference_sameReference(constructor)


