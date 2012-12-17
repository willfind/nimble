"""
Unit tests for the RowListData object. Calls upon the functions defined
in derived_backend.py using appropriate input


"""

import tempfile

from derived_backend import *
from ..row_list_data import *
from nose.tools import *

def constructor(data,labels=None):
	return RowListData(data,labels)

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
# addRows() #
#############

@raises(ArgumentException)
def test_addRows_exceptionNone():
	""" Test RLD addRows() for ArgumentException when toAdd is None"""
	addRows_exceptionNone(constructor)

@raises(ArgumentException)
def test_addRows_exceptionWrongSize():
	""" Test RLD addRows() for ArgumentException when toAdd has too many columns """
	addRows_exceptionWrongSize(constructor)

def test_addRows_handmadeSingle():
	""" Test RLD addRows() against handmade output for a single added row """
	addRows_handmadeSingle(constructor)

def test_addRows_handmadeSequence():
	""" Test RLD addRows() against handmade output for a sequence of additions"""
	addRows_handmadeSequence(constructor)


################
# addColumns() #
################

@raises(ArgumentException)
def test_addColumns_exceptionNone():
	""" Test RLD addColumns() for ArgumentException when toAdd is None """
	addColumns_exceptionNone(constructor)

@raises(ArgumentException)
def test_addColumns_exceptionWrongSize():
	""" Test RLD addColumns() for ArgumentException when toAdd has too many rows """
	addColumns_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_addColumns_exceptionSameLabel():
	""" Test RLD addColumns() for ArgumentException when toAdd and self have a column label in common """
	addColumns_exceptionSameLabel(constructor)

def test_addColumns_handmadeSingle():
	""" Test RLD addColumns() against handmade output for a single added column"""
	addColumns_handmadeSingle(constructor)

def test_addColumns_handmadeSequence():
	""" Test RLD addColumns() against handmade output for a sequence of additions"""
	addColumns_handmadeSequence(constructor)



##############
# sortRows() #
##############

def test_sortRows_handmadeNatural():
	""" Test RLD sortRows() against handmade , naturally ordered  output """	
	sortRows_handmadeNatural(constructor)

def test_sortRows_handmadeWithFcn():
	""" Test RLD sortRows() against handmade output when given a key function """	
	sortRows_handmadeWithFcn(constructor)

#################
# sortColumns() #
#################


def test_sortColumns_handmadeWithFcn():
	""" Test RLD sortColumns() against handmade output when given cmp and key functions """	
	sortColumns_handmadeWithFcn(constructor)



#################
# extractRows() #
#################

@raises(ArgumentException)
def test_extractRows_exceptionNone():
	""" Test RLD extractRows() for ArgumentException when toExtract is none """
	extractRows_exceptionNone(constructor)

def test_extractRows_handmadeSingle():
	""" Test RLD extractRows() against handmade output when extracting one row """
	extractRows_handmadeSingle(constructor)

def test_extractRows_handmadeSequence():
	""" Test RLD extractRows() against handmade output for several extractions """
	extractRows_handmadeSequence(constructor)


####################
# extractColumns() #
####################

@raises(ArgumentException)
def test_extractColumns_exceptionNone():
	""" Test RLD extractColumns() for ArgumentException when toExtract is none """
	extractColumns_exceptionNone(constructor)

def test_extractColumns_handmadeSingle():
	""" Test RLD extractColumns() against handmade output when extracting one row """
	extractColumns_handmadeSingle(constructor)

def test_extractColumns_handmadeSequence():
	""" Test RLD extractColumns() against handmade output for several extractions """
	extractColumns_handmadeSequence(constructor)

def test_extractColumns_handmadeByLabel():
	""" Test RLD extractColumns() against handmade output when specifying labels """
	extractColumns_handmadeByLabel(constructor)

###########################
# extractSatisfyingRows() #
###########################

@raises(ArgumentException)
def test_extractSatisfyingRows_exceptionNone():
	""" Test RLD extractSatisfyingRows() for ArgumentException when toExtract is none """
	extractSatisfyingRows_exceptionNone(constructor)

def test_extractSatisfyingRows_handmade():
	""" Test RLD extractSatisfyingRows() against handmade output """
	extractSatisfyingRows_handmade(constructor)

def test_extractSatisfyingRows_handmadeWithLabels():
	""" Test RLD extractSatisfyingRows() against handmade output with labels"""
	extractSatisfyingRows_handmadeWithLabels(constructor)


##############################
# extractSatisfyingColumns() #
##############################

@raises(ArgumentException)
def test_extractSatisfyingColumns_exceptionNone():
	""" Test RLD extractSatisfyingColumns() for ArgumentException when toExtract is none """
	extractSatisfyingColumns_exceptionNone(constructor)


def test_extractSatisfyingColumns_handmade():
	""" Test RLD extractSatisfyingColumns() against handmade output """
	extractSatisfyingColumns_handmade(constructor)

def test_extractSatisfyingColumns_handmadeWithLabel():
	""" Test RLD extractSatisfyingColumns() against handmade output with labels """
	extractSatisfyingColumns_handmadeWithLabel(constructor)


######################
# extractRangeRows() #
######################

@raises(ArgumentException)
def test_extractRangeRows_exceptionStartNone():
	""" Test RLD extractRangeRows() for ArgumentException when start is None"""
	extractRangeRows_exceptionStartNone(constructor)	

@raises(ArgumentException)
def test_extractRangeRows_exceptionStartInvalid():
	""" Test RLD extractRangeRows() for ArgumentException when start is not a valid row index """
	extractRangeRows_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractRangeRows_exceptionEndNone():
	""" Test RLD extractRangeRows() for ArgumentException when end is None"""
	extractRangeRows_exceptionEndNone(constructor)

@raises(ArgumentException)
def test_extractRangeRows_exceptionEndInvalid():
	""" Test RLD extractRangeRows() for ArgumentException when start is not a valid column index """
	extractRangeRows_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractRangeRows_exceptionInversion():
	""" Test RLD extractRangeRows() for ArgumentException when start comes after end """
	extractRangeRows_exceptionInversion(constructor)

def test_extractRangeRows_handmade():
	""" Test RLD extractRangeRows() against handmade output """
	extractRangeRows_handmade(constructor)

def test_extractRangeRows_handmadeWithLabels():
	""" Test RLD extractRangeRows() against handmade output with Labels """
	extractRangeRows_handmadeWithLabels(constructor)


#########################
# extractRangeColumns() #
#########################

@raises(ArgumentException)
def test_extractRangeColumns_exceptionStartNone():
	""" Test RLD extractRangeColumns() for ArgumentException when start is None"""
	extractRangeColumns_exceptionStartNone(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionStartInvalid():
	""" Test RLD extractRangeColumns() for ArgumentException when start is not a valid column index """
	extractRangeColumns_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionStartInvalidLabel():
	""" Test RLD extractRangeColumns() for ArgumentException when start is not a valid column Label """
	extractRangeColumns_exceptionStartInvalidLabel(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionEndNone():
	""" Test RLD extractRangeColumns() for ArgumentException when end is None"""
	extractRangeColumns_exceptionEndNone(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionEndInvalid():
	""" Test RLD extractRangeColumns() for ArgumentException when start is not a valid column index """
	extractRangeColumns_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionEndInvalidLabel():
	""" Test RLD extractRangeColumns() for ArgumentException when start is not a valid column label """
	extractRangeColumns_exceptionEndInvalidLabel(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionInversion():
	""" Test RLD extractRangeColumns() for ArgumentException when start comes after end """
	extractRangeColumns_exceptionInversion(constructor)

@raises(ArgumentException)
def test_extractRangeColumns_exceptionInversionLabel():
	""" Test RLD extractRangeColumns() for ArgumentException when start comes after end as Labels"""
	extractRangeColumns_exceptionInversionLabel(constructor)

def test_extractRangeColumns_handmade():
	""" Test RLD extractRangeColumns() against handmade output """
	extractRangeColumns_handmade(constructor)

def test_extractRangeColumns_handmadeWithLabels():
	""" Test RLD extractRangeColumns() against handmade output with Labels """
	extractRangeColumns_handmadeWithLabels(constructor)


####################
# applyToEachRow() #
####################

@raises(ArgumentException)
def test_applyToEachRow_exceptionInputNone():
	""" Test RLD applyToEachRow() for ArgumentException when function is None """
	applyToEachRow_exceptionInputNone(constructor)

def test_applyToEachRow_Handmade():
	""" Test RLD applyToEachRow() with handmade output """
	applyToEachRow_Handmade(constructor)


#######################
# applyToEachColumn() #
#######################

@raises(ArgumentException)
def test_applyToEachColumn_exceptionInputNone():
	""" Test RLD applyToEachColumn() for ArgumentException when function is None """
	applyToEachColumn_exceptionInputNone(constructor)

def test_applyToEachColumn_Handmade():
	""" Test RLD applyToEachColumn() with handmade output """
	applyToEachColumn_Handmade(constructor)



#####################
# mapReduceOnRows() #
#####################

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionNoneMap():
	""" Test RLD mapReduceOnRows() for ArgumentException when mapper is None """
	mapReduceOnRows_argumentExceptionNoneMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionNoneReduce():
	""" Test RLD mapReduceOnRows() for ArgumentException when reducer is None """
	mapReduceOnRows_argumentExceptionNoneReduce(constructor)

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionUncallableMap():
	""" Test RLD mapReduceOnRows() for ArgumentException when mapper is not callable """
	mapReduceOnRows_argumentExceptionUncallableMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionUncallableReduce():
	""" Test RLD mapReduceOnRows() for ArgumentException when reducer is not callable """
	mapReduceOnRows_argumentExceptionUncallableReduce(constructor)




def test_mapReduceOnRows_handmade():
	""" Test RLD mapReduceOnRows() against handmade output """
	mapReduceOnRows_handmade(constructor)

def test_mapReduceOnRows_handmadeNoneReturningReducer():
	""" Test RLD mapReduceOnRows() against handmade output with a None returning Reducer """
	mapReduceOnRows_handmadeNoneReturningReducer(constructor)


###########
# File IO #
###########

def test_LoadData():
	""" Test RLD loadCSV() by writing to, and then reading from, a temporary file """
	tmpFile = tempfile.NamedTemporaryFile()
	tmpFile.write("#number,lower,upper\n")
	origData = [['1','a','A'], ['1','a','B'], ['1','a','C'], ['1','b','B'],
				['2','a','B'], ['2','c','A'], ['2','b','C'], ['2','c','C']]

	for point in origData:
		for value in point:
			if point.index(value) != 0:
				tmpFile.write(',')		
			tmpFile.write(str(value))
		tmpFile.write('\n')
	tmpFile.flush()

	loaded = loadCSV(tmpFile.name)

	assert (loaded.data == origData)
	assert (loaded.labels == {'number':0, 'lower':1, 'upper':2})


def test_LoadDataWithParser():
	""" Test RLD loadCSV() with the optional line parsering argument """
	tmpFile = tempfile.NamedTemporaryFile()
	tmpFile.write("#number,lower,upper\n")
	origData = [[1,'a','A'], [1,'a','B'], [1,'a','C'], [1,'b','B'],
				[2,'a','B'], [2,'c','A'], [2,'b','C'], [2,'c','C']]

	for point in origData:
		for value in point:
			if point.index(value) != 0:
				tmpFile.write(',')		
			tmpFile.write(str(value))
		tmpFile.write('\n')
	tmpFile.flush()

	def parseLine (line):
		currList = line.split(',')
		currList[0] = int(currList[0])
		return currList

	loaded = loadCSV(tmpFile.name,parseLine)

	assert (loaded.data == origData)
	assert (loaded.labels == {'number':0, 'lower':1, 'upper':2})


def test_RoundTrip():
	""" Test RLD loadCSV() and RLD writeToCSV() in a round trip test, including a label line """
	roundTripBackend(True)

def test_RoundTripNoLabels():
	""" Test RLD loadCSV() and RLD writeToCSV() in a round trip test, without a label line """
	roundTripBackend(False)


def roundTripBackend(includeLabels):
	tmpFile = tempfile.NamedTemporaryFile()
	labels = None	
	if includeLabels:
		labels = {'number':0,'upper':2,'lower':1}
	origData = [['1','a','a'], ['1','a','B'], ['1','a','C'], ['1','b','B'],
				['2','a','B'], ['2','c','2'], ['2','b','C'], ['2','c','C']]

	if includeLabels:
		origObj = RowListData(origData,labels)
	else:
		origObj = RowListData(origData)

	writeToCSV(origObj,tmpFile.name,includeLabels)

	loaded = loadCSV(tmpFile.name)

	# test equality of data
	assert (loaded.data == origData)

	# test equality of the label map, if it exists
	if includeLabels:
		assert(loaded.labels == labels)



##########################
# convertToRowListData() #
##########################


def test_convertToRowListData_handmade_defaultLabels():
	""" Test RLD convertToRowListData with default labels """
	convertToRowListData_handmade_defaultLabels(constructor)

	
def test_convertToRowListData_handmade_assignedLabels():
	""" Test RLD convertToRowListData with assigned labels """
	convertToRowListData_handmade_assignedLabels(constructor)



##############################
# convertToDenseMatrixData() #
##############################


def test_convertToDenseMatrixData_handmade_defaultLabels():
	""" Test RLD convertToDenseMatrixData with default labels """
	convertToDenseMatrixData_handmade_defaultLabels(constructor)

	
def test_convertToDenseMatrixData_handmade_assignedLabels():
	""" Test RLD convertToDenseMatrixData with assigned labels """
	convertToDenseMatrixData_handmade_assignedLabels(constructor)









