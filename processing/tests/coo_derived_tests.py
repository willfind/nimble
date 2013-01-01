"""
Unit tests for the DenseMatrixData object. Calls upon the functions defined
in derived_backend.py using appropriate input


"""

import tempfile

from derived_backend import *
from ..coo_sparse_data import *
from nose.tools import *

from numpy import matrix as npm

def constructor(data,labels=None):
	return CooSparseData(npm(data),labels)

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
# appendRows() #
#############

@raises(ArgumentException)
def test_appendRows_exceptionNone():
	""" Test CooSparse appendRows() for ArgumentException when toAppend is None"""
	appendRows_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendRows_exceptionWrongSize():
	""" Test CooSparse appendRows() for ArgumentException when toAppend has too many columns """
	appendRows_exceptionWrongSize(constructor)

def test_appendRows_handmadeSingle():
	""" Test CooSparse appendRows() against handmade output for a single added row """
	appendRows_handmadeSingle(constructor)

def test_appendRows_handmadeSequence():
	""" Test CooSparse appendRows() against handmade output for a sequence of additions"""
	appendRows_handmadeSequence(constructor)


################
# appendColumns() #
################

@raises(ArgumentException)
def test_appendColumns_exceptionNone():
	""" Test CooSparse appendColumns() for ArgumentException when toAppend is None """
	appendColumns_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendColumns_exceptionWrongSize():
	""" Test CooSparse appendColumns() for ArgumentException when toAppend has too many rows """
	appendColumns_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_appendColumns_exceptionSameLabel():
	""" Test CooSparse appendColumns() for ArgumentException when toAppend and self have a column label in common """
	appendColumns_exceptionSameLabel(constructor)

def test_appendColumns_handmadeSingle():
	""" Test CooSparse appendColumns() against handmade output for a single added column"""
	appendColumns_handmadeSingle(constructor)

def test_appendColumns_handmadeSequence():
	""" Test CooSparse appendColumns() against handmade output for a sequence of additions"""
	appendColumns_handmadeSequence(constructor)



##############
# sortRows() #
##############

def test_sortRows_handmadeNatural():
	""" Test CooSparse sortRows() against handmade, naturally ordered output """	
	sortRows_handmadeNatural(constructor)

def test_sortRows_handmadeWithFcn():
	""" Test CooSparse sortRows() against handmade output when given cmp and key functions """	
	sortRows_handmadeWithFcn(constructor)

#################
# sortColumns() #
#################


def test_sortColumns_handmadeWithFcn():
	""" Test CooSparse sortColumns() against handmade output when given cmp and key functions """	
	sortColumns_handmadeWithFcn(constructor)



#################
# extractRows() #
#################

def test_extractRows_emptyInput(): 
	""" Test CooSparse extractRows() does nothing when not provided with any input """
	extractRows_emptyInput(constructor)

def test_extractRows_handmadeSingle():
	""" Test CooSparse extractRows() against handmade output when extracting one row """
	extractRows_handmadeSingle(constructor)

def test_extractRows_handmadeListSequence():
	""" Test CooSparse extractRows() against handmade output for several list extractions """
	extractRows_handmadeListSequence(constructor)

def test_extractRows_handmadeFunction():
	""" Test CooSparse extractRows() against handmade output for function extraction """
	extractRows_handmadeFunction(constructor)

def test_extractRows_handmadeFuncionWithLabels():
	""" Test CooSparse extractRows() against handmade output for function extraction with labels"""
	extractRows_handmadeFuncionWithLabels(constructor)

@raises(ArgumentException)
def test_extractRows_exceptionStartInvalid():
	""" Test CooSparse extractRows() for ArgumentException when start is not a valid row index """
	extractRows_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractRows_exceptionEndInvalid():
	""" Test CooSparse extractRows() for ArgumentException when start is not a valid column index """
	extractRows_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractRows_exceptionInversion():
	""" Test CooSparse extractRows() for ArgumentException when start comes after end """
	extractRows_exceptionInversion(constructor)

def test_extractRows_handmade():
	""" Test CooSparse extractRows() against handmade output for range extraction """
	extractRows_handmade(constructor)

def test_extractRows_handmadeWithLabels():
	""" Test CooSparse extractRows() against handmade output for range extraction with labels """
	extractRows_handmadeWithLabels(constructor)



####################
# extractColumns() #
####################




def test_extractColumns_handmadeSingle():
	""" Test CooSparse extractColumns() against handmade output when extracting one column """
	extractColumns_handmadeSingle(constructor)

def test_extractColumns_handmadeListSequence():
	""" Test CooSparse extractColumns() against handmade output for several extractions by list """
	extractColumns_handmadeListSequence(constructor)

def test_extractColumns_handmadeListWithLabel():
	""" Test CooSparse extractColumns() against handmade output for list extraction when specifying labels """
	extractColumns_handmadeListWithLabel(constructor)

def test_extractColumns_handmadeFunction():
	""" Test CooSparse extractColumns() against handmade output for function extraction """
	extractColumns_handmadeFunction(constructor)

def test_extractColumns_handmadeFunctionWithLabel():
	""" Test CooSparse extractColumns() against handmade output for function extraction with labels """
	extractColumns_handmadeFunctionWithLabel(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionStartInvalid():
	""" Test CooSparse extractColumns() for ArgumentException when start is not a valid column index """
	extractColumns_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionStartInvalidLabel():
	""" Test CooSparse extractColumns() for ArgumentException when start is not a valid column Label """
	extractColumns_exceptionStartInvalidLabel(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionEndInvalid():
	""" Test CooSparse extractColumns() for ArgumentException when start is not a valid column index """
	extractColumns_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionEndInvalidLabel():
	""" Test CooSparse extractColumns() for ArgumentException when start is not a valid column label """
	extractColumns_exceptionEndInvalidLabel(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionInversion():
	""" Test CooSparse extractColumns() for ArgumentException when start comes after end """
	extractColumns_exceptionInversion(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionInversionLabel():
	""" Test CooSparse extractColumns() for ArgumentException when start comes after end as Labels"""
	extractColumns_exceptionInversionLabel(constructor)

def test_extractColumns_handmadeRange():
	""" Test CooSparse extractColumns() against handmade output for range extraction """
	extractColumns_handmadeRange(constructor)

def test_extractColumns_handmadeWithLabels():
	""" Test CooSparse extractColumns() against handmade output for range extraction with Labels """
	extractColumns_handmadeWithLabels(constructor)




####################
# applyFunctionToEachRow() #
####################

@raises(ArgumentException)
def test_applyFunctionToEachRow_exceptionInputNone():
	""" Test CooSparse applyFunctionToEachRow() for ArgumentException when function is None """
	applyFunctionToEachRow_exceptionInputNone(constructor)

def test_applyFunctionToEachRow_Handmade():
	""" Test CooSparse applyFunctionToEachRow() with handmade output """
	applyFunctionToEachRow_Handmade(constructor)


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
# mapReduceOnRows() #
#####################

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionNoneMap():
	""" Test CooSparse mapReduceOnRows() for ArgumentException when mapper is None """
	mapReduceOnRows_argumentExceptionNoneMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionNoneReduce():
	""" Test CooSparse mapReduceOnRows() for ArgumentException when reducer is None """
	mapReduceOnRows_argumentExceptionNoneReduce(constructor)

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionUncallableMap():
	""" Test CooSparse mapReduceOnRows() for ArgumentException when mapper is not callable """
	mapReduceOnRows_argumentExceptionUncallableMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionUncallableReduce():
	""" Test CooSparse mapReduceOnRows() for ArgumentException when reducer is not callable """
	mapReduceOnRows_argumentExceptionUncallableReduce(constructor)



def test_mapReduceOnRows_handmade():
	""" Test CooSparse mapReduceOnRows() against handmade output """
	mapReduceOnRows_handmade(constructor)

def test_mapReduceOnRows_handmadeNoneReturningReducer():
	""" Test CooSparse mapReduceOnRows() against handmade output with a None returning Reducer """
	mapReduceOnRows_handmadeNoneReturningReducer(constructor)


##########################
# convertToRowListData() #
##########################


def test_convertToRowListData_handmade_defaultLabels():
	""" Test CooSparse convertToRowListData with default labels """
	convertToRowListData_handmade_defaultLabels(constructor)

	
def test_convertToRowListData_handmade_assignedLabels():
	""" Test CooSparse convertToRowListData with assigned labels """
	convertToRowListData_handmade_assignedLabels(constructor)



##############################
# convertToDenseMatrixData() #
##############################


def test_convertToDenseMatrixData_incompatible():
	""" Test CooSparse convertToDenseMatrixData with data that cannot be used by dense matrices """
	convertToDenseMatrixData_incompatible(constructor)


def test_convertToDenseMatrixData_handmade_defaultLabels():
	""" Test CooSparse convertToDenseMatrixData with default labels """
	convertToDenseMatrixData_handmade_defaultLabels(constructor)

	
def test_convertToDenseMatrixData_handmade_assignedLabels():
	""" Test CooSparse convertToDenseMatrixData with assigned labels """
	convertToDenseMatrixData_handmade_assignedLabels(constructor)




###########
# File IO #
###########





