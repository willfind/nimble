"""
Unit tests for the RowListData object. Calls upon the functions defined
in derived_backend.py using appropriate input


"""

import tempfile

from derived_backend import *
from ..row_list_data import *
from nose.tools import *

def constructor(data,featureNames=None):
	return RowListData(data,featureNames)

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
# appendRows() #
#############

@raises(ArgumentException)
def test_appendRows_exceptionNone():
	""" Test RLD appendRows() for ArgumentException when toAppend is None"""
	appendRows_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendRows_exceptionWrongSize():
	""" Test RLD appendRows() for ArgumentException when toAppend has too many columns """
	appendRows_exceptionWrongSize(constructor)

def test_appendRows_handmadeSingle():
	""" Test RLD appendRows() against handmade output for a single added row """
	appendRows_handmadeSingle(constructor)

def test_appendRows_handmadeSequence():
	""" Test RLD appendRows() against handmade output for a sequence of additions"""
	appendRows_handmadeSequence(constructor)


################
# appendColumns() #
################

@raises(ArgumentException)
def test_appendColumns_exceptionNone():
	""" Test RLD appendColumns() for ArgumentException when toAppend is None """
	appendColumns_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendColumns_exceptionWrongSize():
	""" Test RLD appendColumns() for ArgumentException when toAppend has too many rows """
	appendColumns_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_appendColumns_exceptionSameFeatureName():
	""" Test RLD appendColumns() for ArgumentException when toAppend and self have a featureName in common """
	appendColumns_exceptionSameFeatureName(constructor)

def test_appendColumns_handmadeSingle():
	""" Test RLD appendColumns() against handmade output for a single added column"""
	appendColumns_handmadeSingle(constructor)

def test_appendColumns_handmadeSequence():
	""" Test RLD appendColumns() against handmade output for a sequence of additions"""
	appendColumns_handmadeSequence(constructor)



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

def test_extractRows_emptyInput(): 
	""" Test RLD extractRows() does nothing when not provided with any input """
	extractRows_emptyInput(constructor)

def test_extractRows_handmadeSingle():
	""" Test RLD extractRows() against handmade output when extracting one row """
	extractRows_handmadeSingle(constructor)

def test_extractRows_handmadeListSequence():
	""" Test RLD extractRows() against handmade output for several list extractions """
	extractRows_handmadeListSequence(constructor)

def test_extractRows_handmadeFunction():
	""" Test RLD extractRows() against handmade output for function extraction """
	extractRows_handmadeFunction(constructor)

def test_extractRows_handmadeFuncionWithFeatureNames():
	""" Test RLD extractRows() against handmade output for function extraction with featureNames"""
	extractRows_handmadeFuncionWithFeatureNames(constructor)

@raises(ArgumentException)
def test_extractRows_exceptionStartInvalid():
	""" Test RLD extractRows() for ArgumentException when start is not a valid row index """
	extractRows_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractRows_exceptionEndInvalid():
	""" Test RLD extractRows() for ArgumentException when start is not a valid column index """
	extractRows_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractRows_exceptionInversion():
	""" Test RLD extractRows() for ArgumentException when start comes after end """
	extractRows_exceptionInversion(constructor)

def test_extractRows_handmade():
	""" Test RLD extractRows() against handmade output for range extraction """
	extractRows_handmade(constructor)

def test_extractRows_handmadeWithFeatureNames():
	""" Test RLD extractRows() against handmade output for range extraction with featureNames """
	extractRows_handmadeWithFeatureNames(constructor)


####################
# extractColumns() #
####################

def test_extractColumns_handmadeSingle():
	""" Test RLD extractColumns() against handmade output when extracting one column """
	extractColumns_handmadeSingle(constructor)

def test_extractColumns_handmadeListSequence():
	""" Test RLD extractColumns() against handmade output for several extractions by list """
	extractColumns_handmadeListSequence(constructor)

def test_extractColumns_handmadeListWithFeatureName():
	""" Test RLD extractColumns() against handmade output for list extraction when specifying featureNames """
	extractColumns_handmadeListWithFeatureName(constructor)

def test_extractColumns_handmadeFunction():
	""" Test RLD extractColumns() against handmade output for function extraction """
	extractColumns_handmadeFunction(constructor)

def test_extractColumns_handmadeFunctionWithFeatureName():
	""" Test RLD extractColumns() against handmade output for function extraction with featureNames """
	extractColumns_handmadeFunctionWithFeatureName(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionStartInvalid():
	""" Test RLD extractColumns() for ArgumentException when start is not a valid column index """
	extractColumns_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionStartInvalidFeatureName():
	""" Test RLD extractColumns() for ArgumentException when start is not a valid featureName """
	extractColumns_exceptionStartInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionEndInvalid():
	""" Test RLD extractColumns() for ArgumentException when start is not a valid column index """
	extractColumns_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionEndInvalidFeatureName():
	""" Test RLD extractColumns() for ArgumentException when start is not a valid column featureName """
	extractColumns_exceptionEndInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionInversion():
	""" Test RLD extractColumns() for ArgumentException when start comes after end """
	extractColumns_exceptionInversion(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionInversionFeatureName():
	""" Test RLD extractColumns() for ArgumentException when start comes after end as FeatureNames"""
	extractColumns_exceptionInversionFeatureName(constructor)

def test_extractColumns_handmadeRange():
	""" Test RLD extractColumns() against handmade output for range extraction """
	extractColumns_handmadeRange(constructor)

def test_extractColumns_handmadeWithFeatureNames():
	""" Test RLD extractColumns() against handmade output for range extraction with FeatureNames """
	extractColumns_handmadeWithFeatureNames(constructor)


####################
# applyFunctionToEachRow() #
####################

@raises(ArgumentException)
def test_applyFunctionToEachRow_exceptionInputNone():
	""" Test RLD applyFunctionToEachRow() for ArgumentException when function is None """
	applyFunctionToEachRow_exceptionInputNone(constructor)

def test_applyFunctionToEachRow_Handmade():
	""" Test RLD applyFunctionToEachRow() with handmade output """
	applyFunctionToEachRow_Handmade(constructor)


#######################
# applyFunctionToEachColumn() #
#######################

@raises(ArgumentException)
def test_applyFunctionToEachColumn_exceptionInputNone():
	""" Test RLD applyFunctionToEachColumn() for ArgumentException when function is None """
	applyFunctionToEachColumn_exceptionInputNone(constructor)

def test_applyFunctionToEachColumn_Handmade():
	""" Test RLD applyFunctionToEachColumn() with handmade output """
	applyFunctionToEachColumn_Handmade(constructor)



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
	assert (loaded.featureNames == {'number':0, 'lower':1, 'upper':2})


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
	assert (loaded.featureNames == {'number':0, 'lower':1, 'upper':2})


def test_RoundTrip():
	""" Test RLD loadCSV() and RLD writeToCSV() in a round trip test, including a featureName line """
	roundTripBackend(True)

def test_RoundTripNoFeatureNames():
	""" Test RLD loadCSV() and RLD writeToCSV() in a round trip test, without a featureName line """
	roundTripBackend(False)


def roundTripBackend(includeFeatureNames):
	tmpFile = tempfile.NamedTemporaryFile()
	featureNames = None	
	if includeFeatureNames:
		featureNames = {'number':0,'upper':2,'lower':1}
	origData = [['1','a','a'], ['1','a','B'], ['1','a','C'], ['1','b','B'],
				['2','a','B'], ['2','c','2'], ['2','b','C'], ['2','c','C']]

	if includeFeatureNames:
		origObj = RowListData(origData,featureNames)
	else:
		origObj = RowListData(origData)

	writeToCSV(origObj,tmpFile.name,includeFeatureNames)

	loaded = loadCSV(tmpFile.name)

	# test equality of data
	assert (loaded.data == origData)

	# test equality of the featureName map, if it exists
	if includeFeatureNames:
		assert(loaded.featureNames == featureNames)



##########################
# convertToRowListData() #
##########################


def test_convertToRowListData_handmade_defaultFeatureNames():
	""" Test RLD convertToRowListData with default featureNames """
	convertToRowListData_handmade_defaultFeatureNames(constructor)

	
def test_convertToRowListData_handmade_assignedFeatureNames():
	""" Test RLD convertToRowListData with assigned featureNames """
	convertToRowListData_handmade_assignedFeatureNames(constructor)



##############################
# convertToDenseMatrixData() #
##############################


def test_convertToDenseMatrixData_handmade_defaultFeatureNames():
	""" Test RLD convertToDenseMatrixData with default featureNames """
	convertToDenseMatrixData_handmade_defaultFeatureNames(constructor)

	
def test_convertToDenseMatrixData_handmade_assignedFeatureNames():
	""" Test RLD convertToDenseMatrixData with assigned featureNames """
	convertToDenseMatrixData_handmade_assignedFeatureNames(constructor)









