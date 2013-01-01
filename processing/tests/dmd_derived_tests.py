"""
Unit tests for the DenseMatrixData object. Calls upon the functions defined
in derived_backend.py using appropriate input


"""

import tempfile

from derived_backend import *
from ..dense_matrix_data import *
from nose.tools import *

def constructor(data,featureNames=None):
	return DenseMatrixData(data,featureNames)

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
# appendRows() #
#############

@raises(ArgumentException)
def test_appendRows_exceptionNone():
	""" Test DMD appendRows() for ArgumentException when toAppend is None"""
	appendRows_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendRows_exceptionWrongSize():
	""" Test DMD appendRows() for ArgumentException when toAppend has too many columns """
	appendRows_exceptionWrongSize(constructor)

def test_appendRows_handmadeSingle():
	""" Test DMD appendRows() against handmade output for a single added row """
	appendRows_handmadeSingle(constructor)

def test_appendRows_handmadeSequence():
	""" Test DMD appendRows() against handmade output for a sequence of additions"""
	appendRows_handmadeSequence(constructor)


################
# appendColumns() #
################

@raises(ArgumentException)
def test_appendColumns_exceptionNone():
	""" Test DMD appendColumns() for ArgumentException when toAppend is None """
	appendColumns_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendColumns_exceptionWrongSize():
	""" Test DMD appendColumns() for ArgumentException when toAppend has too many rows """
	appendColumns_exceptionWrongSize(constructor)

@raises(ArgumentException)
def test_appendColumns_exceptionSameFeatureName():
	""" Test DMD appendColumns() for ArgumentException when toAppend and self have a featureName in common """
	appendColumns_exceptionSameFeatureName(constructor)

def test_appendColumns_handmadeSingle():
	""" Test DMD appendColumns() against handmade output for a single added column"""
	appendColumns_handmadeSingle(constructor)

def test_appendColumns_handmadeSequence():
	""" Test DMD appendColumns() against handmade output for a sequence of additions"""
	appendColumns_handmadeSequence(constructor)



##############
# sortRows() #
##############

def test_sortRows_handmadeNatural():
	""" Test DMD sortRows() against handmade, naturally ordered output """	
	sortRows_handmadeNatural(constructor)

def test_sortRows_handmadeWithFcn():
	""" Test DMD sortRows() against handmade output when given cmp and key functions """	
	sortRows_handmadeWithFcn(constructor)

#################
# sortColumns() #
#################


def test_sortColumns_handmadeWithFcn():
	""" Test DMD sortColumns() against handmade output when given cmp and key functions """	
	sortColumns_handmadeWithFcn(constructor)



#################
# extractRows() #
#################

def test_extractRows_emptyInput(): 
	""" Test DMD extractRows() does nothing when not provided with any input """
	extractRows_emptyInput(constructor)

def test_extractRows_handmadeSingle():
	""" Test DMD extractRows() against handmade output when extracting one row """
	extractRows_handmadeSingle(constructor)

def test_extractRows_handmadeListSequence():
	""" Test DMD extractRows() against handmade output for several list extractions """
	extractRows_handmadeListSequence(constructor)

def test_extractRows_handmadeFunction():
	""" Test DMD extractRows() against handmade output for function extraction """
	extractRows_handmadeFunction(constructor)

def test_extractRows_handmadeFuncionWithFeatureNames():
	""" Test DMD extractRows() against handmade output for function extraction with featureNames"""
	extractRows_handmadeFuncionWithFeatureNames(constructor)

@raises(ArgumentException)
def test_extractRows_exceptionStartInvalid():
	""" Test DMD extractRows() for ArgumentException when start is not a valid row index """
	extractRows_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractRows_exceptionEndInvalid():
	""" Test DMD extractRows() for ArgumentException when start is not a valid column index """
	extractRows_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractRows_exceptionInversion():
	""" Test DMD extractRows() for ArgumentException when start comes after end """
	extractRows_exceptionInversion(constructor)

def test_extractRows_handmade():
	""" Test DMD extractRows() against handmade output for range extraction """
	extractRows_handmade(constructor)

def test_extractRows_handmadeWithFeatureNames():
	""" Test DMD extractRows() against handmade output for range extraction with featureNames """
	extractRows_handmadeWithFeatureNames(constructor)





####################
# extractColumns() #
####################


def test_extractColumns_handmadeSingle():
	""" Test DMD extractColumns() against handmade output when extracting one column """
	extractColumns_handmadeSingle(constructor)

def test_extractColumns_handmadeListSequence():
	""" Test DMD extractColumns() against handmade output for several extractions by list """
	extractColumns_handmadeListSequence(constructor)

def test_extractColumns_handmadeListWithFeatureName():
	""" Test DMD extractColumns() against handmade output for list extraction when specifying featureNames """
	extractColumns_handmadeListWithFeatureName(constructor)

def test_extractColumns_handmadeFunction():
	""" Test DMD extractColumns() against handmade output for function extraction """
	extractColumns_handmadeFunction(constructor)

def test_extractColumns_handmadeFunctionWithFeatureName():
	""" Test DMD extractColumns() against handmade output for function extraction with featureNames """
	extractColumns_handmadeFunctionWithFeatureName(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionStartInvalid():
	""" Test DMD extractColumns() for ArgumentException when start is not a valid column index """
	extractColumns_exceptionStartInvalid(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionStartInvalidFeatureName():
	""" Test DMD extractColumns() for ArgumentException when start is not a valid featureName """
	extractColumns_exceptionStartInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionEndInvalid():
	""" Test DMD extractColumns() for ArgumentException when start is not a valid column index """
	extractColumns_exceptionEndInvalid(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionEndInvalidFeatureName():
	""" Test DMD extractColumns() for ArgumentException when start is not a valid featureName """
	extractColumns_exceptionEndInvalidFeatureName(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionInversion():
	""" Test DMD extractColumns() for ArgumentException when start comes after end """
	extractColumns_exceptionInversion(constructor)

@raises(ArgumentException)
def test_extractColumns_exceptionInversionFeatureName():
	""" Test DMD extractColumns() for ArgumentException when start comes after end as FeatureNames"""
	extractColumns_exceptionInversionFeatureName(constructor)

def test_extractColumns_handmadeRange():
	""" Test DMD extractColumns() against handmade output for range extraction """
	extractColumns_handmadeRange(constructor)

def test_extractColumns_handmadeWithFeatureNames():
	""" Test DMD extractColumns() against handmade output for range extraction with FeatureNames """
	extractColumns_handmadeWithFeatureNames(constructor)



####################
# applyFunctionToEachRow() #
####################

@raises(ArgumentException)
def test_applyFunctionToEachRow_exceptionInputNone():
	""" Test DMD applyFunctionToEachRow() for ArgumentException when function is None """
	applyFunctionToEachRow_exceptionInputNone(constructor)

def test_applyFunctionToEachRow_Handmade():
	""" Test DMD applyFunctionToEachRow() with handmade output """
	applyFunctionToEachRow_Handmade(constructor)


#######################
# applyFunctionToEachColumn() #
#######################

@raises(ArgumentException)
def test_applyFunctionToEachColumn_exceptionInputNone():
	""" Test DMD applyFunctionToEachColumn() for ArgumentException when function is None """
	applyFunctionToEachColumn_exceptionInputNone(constructor)

def test_applyFunctionToEachColumn_Handmade():
	""" Test DMD applyFunctionToEachColumn() with handmade output """
	applyFunctionToEachColumn_Handmade(constructor)


#####################
# mapReduceOnRows() #
#####################

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionNoneMap():
	""" Test DMD mapReduceOnRows() for ArgumentException when mapper is None """
	mapReduceOnRows_argumentExceptionNoneMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionNoneReduce():
	""" Test DMD mapReduceOnRows() for ArgumentException when reducer is None """
	mapReduceOnRows_argumentExceptionNoneReduce(constructor)

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionUncallableMap():
	""" Test DMD mapReduceOnRows() for ArgumentException when mapper is not callable """
	mapReduceOnRows_argumentExceptionUncallableMap(constructor)

@raises(ArgumentException)
def test_mapReduceOnRows_argumentExceptionUncallableReduce():
	""" Test DMD mapReduceOnRows() for ArgumentException when reducer is not callable """
	mapReduceOnRows_argumentExceptionUncallableReduce(constructor)



def test_mapReduceOnRows_handmade():
	""" Test DMD mapReduceOnRows() against handmade output """
	mapReduceOnRows_handmade(constructor)

def test_mapReduceOnRows_handmadeNoneReturningReducer():
	""" Test DMD mapReduceOnRows() against handmade output with a None returning Reducer """
	mapReduceOnRows_handmadeNoneReturningReducer(constructor)

###########
# File IO #
###########

def test_LoadData():
	""" Test DMD loadCSV() by writing to, and then reading from, a temporary file without featureNames """
	tmpFile = tempfile.NamedTemporaryFile()
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02],
				[2,0.1,0.02], [2,0.3,0.01], [2,0.2,0.03], [2,0.3,0.03]]
	test = constructor(origData)

	for point in origData:
		for value in point:
			if point.index(value) != 0:
				tmpFile.write(',')		
			tmpFile.write(str(value))
		tmpFile.write('\n')
	tmpFile.flush()

	loaded = loadCSV(tmpFile.name)
	assert loaded.equals(test)

def test_LoadDataFeatureNames():
	""" Test DMD loadCSV() by writing to, and then reading from, a temporary file with featureNames """
	tmpFile = tempfile.NamedTemporaryFile()
	tmpFile.write("#number,deci,centi\n")
	featureNames = (['number','deci','centi'])
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02],
				[2,0.1,0.02], [2,0.3,0.01], [2,0.2,0.03], [2,0.3,0.03]]
	test = constructor(origData,featureNames)

	for point in origData:
		for value in point:
			if point.index(value) != 0:
				tmpFile.write(',')		
			tmpFile.write(str(value))
		tmpFile.write('\n')
	tmpFile.flush()

	loaded = loadCSV(tmpFile.name)
	
	assert loaded.equals(test)


def test_RoundTrip():
	""" Test DMD loadCSV() and DMD writeToCSV() in a round trip test, including a featureName line """
	roundTripBackend(True)

def test_RoundTripNoFeatureNames():
	""" Test DMD loadCSV() and DMD writeToCSV() in a round trip test, without a featureName line """
	roundTripBackend(False)


def roundTripBackend(includeFeatureNames):
	tmpFile = tempfile.NamedTemporaryFile()
	featureNames = None	
	if includeFeatureNames:
		featureNames = (['number','deci','centi'])
	origData = [[1,0.1,0.01], [1,0.1,0.02], [1,0.1,0.03], [1,0.2,0.02],
				[2,0.1,0.02], [2,0.3,0.01], [2,0.2,0.03], [2,0.3,0.03]]

	origObj = constructor(origData,featureNames)

	writeToCSV(origObj,tmpFile.name,includeFeatureNames)

	loaded = loadCSV(tmpFile.name)

	assert origObj.equals(loaded)
	assert loaded.equals(origObj)




##########################
# convertToRowListData() #
##########################


def test_convertToRowListData_handmade_defaultFeatureNames():
	""" Test DMD convertToRowListData with default featureNames """
	convertToRowListData_handmade_defaultFeatureNames(constructor)

	
def test_convertToRowListData_handmade_assignedFeatureNames():
	""" Test DMD convertToRowListData with assigned featureNames """
	convertToRowListData_handmade_assignedFeatureNames(constructor)



##############################
# convertToDenseMatrixData() #
##############################


def test_convertToDenseMatrixData_handmade_defaultFeatureNames():
	""" Test DMD convertToDenseMatrixData with default featureNames """
	convertToDenseMatrixData_handmade_defaultFeatureNames(constructor)

	
def test_convertToDenseMatrixData_handmade_assignedFeatureNames():
	""" Test DMD convertToDenseMatrixData with assigned featureNames """
	convertToDenseMatrixData_handmade_assignedFeatureNames(constructor)








