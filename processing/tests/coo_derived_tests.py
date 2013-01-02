"""
Unit tests for the DenseMatrixData object. Calls upon the functions defined
in derived_backend.py using appropriate input


"""

import tempfile

from derived_backend import *
from ..coo_sparse_data import *
from nose.tools import *

from numpy import matrix as npm

def constructor(data,featureNames=None):
	return CooSparseData(npm(data),featureNames)

############
# equals() #
############

def test_equals_False():
	""" Test Coo Sparse equals() against some non-equal input """
	equals_False(constructor)

def test_equals_True():
	""" Test Coo Sparse equals() against some actually equal input """
	equals_True(constructor)

def test_equals_empty():
	""" Test CooSparse equals() for empty objects """
	equals_empty(constructor)


###############
# transpose() #
###############

def test_transpose_handmade():
	""" Test CooSparse transpose() function against handmade output """
	transpose_handmade(constructor)


#############
# appendPoints() #
#############

@raises(ArgumentException)
def test_appendPoints_exceptionNone():
	""" Test CooSparse appendPoints() for ArgumentException when toAppend is None"""
	appendPoints_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendPoints_exceptionWrongSize():
	""" Test CooSparse appendPoints() for ArgumentException when toAppend has too many columns """
	appendPoints_exceptionWrongSize(constructor)

def test_appendPoints_handmadeSingle():
	""" Test CooSparse appendPoints() against handmade output for a single added point """
	appendPoints_handmadeSingle(constructor)

def test_appendPoints_handmadeSequence():
	""" Test CooSparse appendPoints() against handmade output for a sequence of additions"""
	appendPoints_handmadeSequence(constructor)


################
# appendColumns() #
################

@raises(ArgumentException)
def test_appendColumns_exceptionNone():
	""" Test CooSparse appendColumns() for ArgumentException when toAppend is None """
	appendColumns_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendColumns_exceptionWrongSize():
	""" Test CooSparse appendColumns() for ArgumentException when toAppend has too many points """
	appendColumns_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_appendColumns_exceptionSameFeatureName():
	""" Test CooSparse appendColumns() for ArgumentException when toAppend and self have a featureName in common """
	appendColumns_exceptionSameFeatureName(constructor)

def test_appendColumns_handmadeSingle():
	""" Test CooSparse appendColumns() against handmade output for a single added column"""
	appendColumns_handmadeSingle(constructor)

def test_appendColumns_handmadeSequence():
	""" Test CooSparse appendColumns() against handmade output for a sequence of additions"""
	appendColumns_handmadeSequence(constructor)



##############
# sortPoints() #
##############

def test_sortPoints_handmadeNatural():
	""" Test CooSparse sortPoints() against handmade, naturally ordered output """	
	sortPoints_handmadeNatural(constructor)

def test_sortPoints_handmadeWithFcn():
	""" Test CooSparse sortPoints() against handmade output when given cmp and key functions """	
	sortPoints_handmadeWithFcn(constructor)

#################
# sortColumns() #
#################


def test_sortColumns_handmadeWithFcn():
	""" Test CooSparse sortColumns() against handmade output when given cmp and key functions """	
	sortColumns_handmadeWithFcn(constructor)



#################
# extractPoints() #
#################

def test_extractPoints_emptyInput(): 
	""" Test CooSparse extractPoints() does nothing when not provided with any input """
	extractPoints_emptyInput(constructor)

def test_extractPoints_handmadeSingle():
	""" Test CooSparse extractPoints() against handmade output when extracting one point """
	extractPoints_handmadeSingle(constructor)

def test_extractPoints_handmadeListSequence():
	""" Test CooSparse extractPoints() against handmade output for several list extractions """
	extractPoints_handmadeListSequence(constructor)

def test_extractPoints_handmadeFunction():
	""" Test CooSparse extractPoints() against handmade output for function extraction """
	extractPoints_handmadeFunction(constructor)

def test_extractPoints_handmadeFuncionWithFeatureNames():
	""" Test CooSparse extractPoints() against handmade output for function extraction with featureNames"""
	extractPoints_handmadeFuncionWithFeatureNames(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionStartInvalid():
	""" Test CooSparse extractPoints() for ArgumentException when start is not a valid point index """
	extractPoints_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionEndInvalid():
	""" Test CooSparse extractPoints() for ArgumentException when start is not a valid column index """
	extractPoints_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractPoints_exceptionInversion():
	""" Test CooSparse extractPoints() for ArgumentException when start comes after end """
	extractPoints_exceptionInversion(constructor)

def test_extractPoints_handmade():
	""" Test CooSparse extractPoints() against handmade output for range extraction """
	extractPoints_handmade(constructor)

def test_extractPoints_handmadeWithFeatureNames():
	""" Test CooSparse extractPoints() against handmade output for range extraction with featureNames """
	extractPoints_handmadeWithFeatureNames(constructor)



####################
# extractColumns() #
####################




def test_extractColumns_handmadeSingle():
	""" Test CooSparse extractColumns() against handmade output when extracting one column """
	extractColumns_handmadeSingle(constructor)

def test_extractColumns_handmadeListSequence():
	""" Test CooSparse extractColumns() against handmade output for several extractions by list """
	extractColumns_handmadeListSequence(constructor)

def test_extractColumns_handmadeListWithFeatureName():
	""" Test CooSparse extractColumns() against handmade output for list extraction when specifying featureNames """
	extractColumns_handmadeListWithFeatureName(constructor)

def test_extractColumns_handmadeFunction():
	""" Test CooSparse extractColumns() against handmade output for function extraction """
	extractColumns_handmadeFunction(constructor)

def test_extractColumns_handmadeFunctionWithFeatureName():
	""" Test CooSparse extractColumns() against handmade output for function extraction with featureNames """
	extractColumns_handmadeFunctionWithFeatureName(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionStartInvalid():
	""" Test CooSparse extractColumns() for ArgumentException when start is not a valid column index """
	extractColumns_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionStartInvalidFeatureName():
	""" Test CooSparse extractColumns() for ArgumentException when start is not a valid featureName """
	extractColumns_exceptionStartInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionEndInvalid():
	""" Test CooSparse extractColumns() for ArgumentException when start is not a valid column index """
	extractColumns_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionEndInvalidFeatureName():
	""" Test CooSparse extractColumns() for ArgumentException when start is not a valid featureName """
	extractColumns_exceptionEndInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionInversion():
	""" Test CooSparse extractColumns() for ArgumentException when start comes after end """
	extractColumns_exceptionInversion(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionInversionFeatureName():
	""" Test CooSparse extractColumns() for ArgumentException when start comes after end as FeatureNames"""
	extractColumns_exceptionInversionFeatureName(constructor)

def test_extractColumns_handmadeRange():
	""" Test CooSparse extractColumns() against handmade output for range extraction """
	extractColumns_handmadeRange(constructor)

def test_extractColumns_handmadeWithFeatureNames():
	""" Test CooSparse extractColumns() against handmade output for range extraction with FeatureNames """
	extractColumns_handmadeWithFeatureNames(constructor)




####################
# applyFunctionToEachPoint() #
####################

@raises(ArgumentException)
def test_applyFunctionToEachPoint_exceptionInputNone():
	""" Test CooSparse applyFunctionToEachPoint() for ArgumentException when function is None """
	applyFunctionToEachPoint_exceptionInputNone(constructor)

def test_applyFunctionToEachPoint_Handmade():
	""" Test CooSparse applyFunctionToEachPoint() with handmade output """
	applyFunctionToEachPoint_Handmade(constructor)


#######################
# applyFunctionToEachColumn() #
#######################

@raises(ArgumentException)
def test_applyFunctionToEachColumn_exceptionInputNone():
	""" Test CooSparse applyFunctionToEachColumn() for ArgumentException when function is None """
	applyFunctionToEachColumn_exceptionInputNone(constructor)

def test_applyFunctionToEachColumn_Handmade():
	""" Test CooSparse applyFunctionToEachColumn() with handmade output """
	applyFunctionToEachColumn_Handmade(constructor)


#####################
# mapReduceOnPoints() #
#####################

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionNoneMap():
	""" Test CooSparse mapReduceOnPoints() for ArgumentException when mapper is None """
	mapReduceOnPoints_argumentExceptionNoneMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionNoneReduce():
	""" Test CooSparse mapReduceOnPoints() for ArgumentException when reducer is None """
	mapReduceOnPoints_argumentExceptionNoneReduce(constructor)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionUncallableMap():
	""" Test CooSparse mapReduceOnPoints() for ArgumentException when mapper is not callable """
	mapReduceOnPoints_argumentExceptionUncallableMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionUncallableReduce():
	""" Test CooSparse mapReduceOnPoints() for ArgumentException when reducer is not callable """
	mapReduceOnPoints_argumentExceptionUncallableReduce(constructor)



def test_mapReduceOnPoints_handmade():
	""" Test CooSparse mapReduceOnPoints() against handmade output """
	mapReduceOnPoints_handmade(constructor)

def test_mapReduceOnPoints_handmadeNoneReturningReducer():
	""" Test CooSparse mapReduceOnPoints() against handmade output with a None returning Reducer """
	mapReduceOnPoints_handmadeNoneReturningReducer(constructor)


##########################
# convertToRowListData() #
##########################


def test_convertToRowListData_handmade_defaultFeatureNames():
	""" Test CooSparse convertToRowListData with default featureNames """
	convertToRowListData_handmade_defaultFeatureNames(constructor)

	
def test_convertToRowListData_handmade_assignedFeatureNames():
	""" Test CooSparse convertToRowListData with assigned featureNames """
	convertToRowListData_handmade_assignedFeatureNames(constructor)



##############################
# convertToDenseMatrixData() #
##############################


def test_convertToDenseMatrixData_incompatible():
	""" Test CooSparse convertToDenseMatrixData with data that cannot be used by dense matrices """
	convertToDenseMatrixData_incompatible(constructor)


def test_convertToDenseMatrixData_handmade_defaultFeatureNames():
	""" Test CooSparse convertToDenseMatrixData with default featureNames """
	convertToDenseMatrixData_handmade_defaultFeatureNames(constructor)

	
def test_convertToDenseMatrixData_handmade_assignedFeatureNames():
	""" Test CooSparse convertToDenseMatrixData with assigned featureNames """
	convertToDenseMatrixData_handmade_assignedFeatureNames(constructor)




###########
# File IO #
###########





