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
# addColumns() #
################

@raises(ArgumentException)
def test_addColumns_exceptionNone():
	""" Test CooSparse addColumns() for ArgumentException when toAdd is None """
	addColumns_exceptionNone(constructor)

@raises(ArgumentException)
def test_addColumns_exceptionWrongSize():
	""" Test CooSparse addColumns() for ArgumentException when toAdd has too many rows """
	addColumns_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_addColumns_exceptionSameLabel():
	""" Test CooSparse addColumns() for ArgumentException when toAdd and self have a column label in common """
	addColumns_exceptionSameLabel(constructor)

def test_addColumns_handmadeSingle():
	""" Test CooSparse addColumns() against handmade output for a single added column"""
	addColumns_handmadeSingle(constructor)

def test_addColumns_handmadeSequence():
	""" Test CooSparse addColumns() against handmade output for a sequence of additions"""
	addColumns_handmadeSequence(constructor)



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

@raises(ArgumentException)
def test_extractRows_exceptionNone():
	""" Test CooSparse extractRows() for ArgumentException when toExtract is none """
	extractRows_exceptionNone(constructor)

def test_extractRows_handmadeSingle():
	""" Test CooSparse extractRows() against handmade output when extracting one row """
	extractRows_handmadeSingle(constructor)

def test_extractRows_handmadeSequence():
	""" Test CooSparse extractRows() against handmade output for several extractions """
	extractRows_handmadeSequence(constructor)


####################
# extractColumns() #
####################

@raises(ArgumentException)
def test_extractColumns_exceptionNone():
	""" Test CooSparse extractColumns() for ArgumentException when toExtract is none """
	extractColumns_exceptionNone(constructor)

def test_extractColumns_handmadeSingle():
	""" Test CooSparse extractColumns() against handmade output when extracting one row """
	extractColumns_handmadeSingle(constructor)

def test_extractColumns_handmadeSequence():
	""" Test CooSparse extractColumns() against handmade output for several extractions """
	extractColumns_handmadeSequence(constructor)

def test_extractColumns_handmadeByLabel():
	""" Test CooSparse extractColumns() against handmade output when specifying labels """
	extractColumns_handmadeByLabel(constructor)

###########################
# extractSatisfyingRows() #
###########################

@raises(ArgumentException)
def test_extractSatisfyingRows_exceptionNone():
	""" Test CooSparse extractSatisfyingRows() for ArgumentException when toExtract is none """
	extractSatisfyingRows_exceptionNone(constructor)

def test_extractSatisfyingRows_handmade():
	""" Test CooSparse extractSatisfyingRows() against handmade output """
	extractSatisfyingRows_handmade(constructor)

def test_extractSatisfyingRows_handmadeWithLabels():
	""" Test CooSparse extractSatisfyingRows() against handmade output with labels"""
	extractSatisfyingRows_handmadeWithLabels(constructor)

##############################
# extractSatisfyingColumns() #
##############################

@raises(ArgumentException)
def test_extractSatisfyingColumns_exceptionNone():
	""" Test CooSparse extractSatisfyingColumns() for ArgumentException when toExtract is none """
	extractSatisfyingColumns_exceptionNone(constructor)


def test_extractSatisfyingColumns_handmade():
	""" Test CooSparse extractSatisfyingColumns() against handmade output """
	extractSatisfyingColumns_handmade(constructor)

def test_extractSatisfyingColumns_handmadeWithLabel():
	""" Test CooSparse extractSatisfyingColumns() against handmade output with labels """
	extractSatisfyingColumns_handmadeWithLabel(constructor)


######################
# extractRangeRows() #
######################

@raises(ArgumentException)
def test_extractRangeRows_exceptionStartNone():
	""" Test CooSparse extractRangeRows() for ArgumentException when start is None"""
	extractRangeRows_exceptionStartNone(constructor)	

@raises(ArgumentException)
def test_extractRangeRows_exceptionStartInvalid():
	""" Test CooSparse extractRangeRows() for ArgumentException when start is not a valid row index """
	extractRangeRows_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractRangeRows_exceptionEndNone():
	""" Test CooSparse extractRangeRows() for ArgumentException when end is None"""
	extractRangeRows_exceptionEndNone(constructor)

@raises(ArgumentException)
def test_extractRangeRows_exceptionEndInvalid():
	""" Test CooSparse extractRangeRows() for ArgumentException when start is not a valid column index """
	extractRangeRows_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractRangeRows_exceptionInversion():
	""" Test CooSparse extractRangeRows() for ArgumentException when start comes after end """
	extractRangeRows_exceptionInversion(constructor)

def test_extractRangeRows_handmade():
	""" Test CooSparse extractRangeRows() against handmade output """
	extractRangeRows_handmade(constructor)

def test_extractRangeRows_handmadeWithLabels():
	""" Test CooSparse extractRangeRows() against handmade output """
	extractRangeRows_handmadeWithLabels(constructor)


#########################
# extractRangeColumns() #
#########################

@raises(ArgumentException)
def test_extractRangeColumns_exceptionStartNone():
	""" Test CooSparse extractRangeColumns() for ArgumentException when start is None"""
	extractRangeColumns_exceptionStartNone(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionStartInvalid():
	""" Test CooSparse extractRangeColumns() for ArgumentException when start is not a valid column index """
	extractRangeColumns_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionStartInvalidLabel():
	""" Test CooSparse extractRangeColumns() for ArgumentException when start is not a valid column Label """
	extractRangeColumns_exceptionStartInvalidLabel(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionEndNone():
	""" Test CooSparse extractRangeColumns() for ArgumentException when end is None"""
	extractRangeColumns_exceptionEndNone(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionEndInvalid():
	""" Test CooSparse extractRangeColumns() for ArgumentException when start is not a valid column index """
	extractRangeColumns_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionEndInvalidLabel():
	""" Test CooSparse extractRangeColumns() for ArgumentException when start is not a valid column label """
	extractRangeColumns_exceptionEndInvalidLabel(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionInversion():
	""" Test CooSparse extractRangeColumns() for ArgumentException when start comes after end """
	extractRangeColumns_exceptionInversion(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionInversionLabel():
	""" Test CooSparse extractRangeColumns() for ArgumentException when start comes after end as Labels"""
	extractRangeColumns_exceptionInversionLabel(constructor)

def test_extractRangeColumns_handmade():
	""" Test CooSparse extractRangeColumns() against handmade output """
	extractRangeColumns_handmade(constructor)

def test_extractRangeColumns_handmadeWithLabels():
	""" Test CooSparse extractRangeColumns() against handmade output with Labels """
	extractRangeColumns_handmadeWithLabels(constructor)

####################
# applyToEachRow() #
####################

@raises(ArgumentException)
def test_applyToEachRow_exceptionInputNone():
	""" Test CooSparse applyToEachRow() for ArgumentException when function is None """
	applyToEachRow_exceptionInputNone(constructor)

def test_applyToEachRow_Handmade():
	""" Test CooSparse applyToEachRow() with handmade output """
	applyToEachRow_Handmade(constructor)


#######################
# applyToEachColumn() #
#######################

@raises(ArgumentException)
def test_applyToEachColumn_exceptionInputNone():
	""" Test CooSparse applyToEachColumn() for ArgumentException when function is None """
	applyToEachColumn_exceptionInputNone(constructor)

def test_applyToEachColumn_Handmade():
	""" Test CooSparse applyToEachColumn() with handmade output """
	applyToEachColumn_Handmade(constructor)


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





