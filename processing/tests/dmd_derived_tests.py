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
# appendPoints() #
#############

@raises(ArgumentException)
def test_appendPoints_exceptionNone():
	""" Test DMD appendPoints() for ArgumentException when toAppend is None"""
	appendPoints_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendPoints_exceptionWrongSize():
	""" Test DMD appendPoints() for ArgumentException when toAppend has too many columns """
	appendPoints_exceptionWrongSize(constructor)

def test_appendPoints_handmadeSingle():
	""" Test DMD appendPoints() against handmade output for a single added point """
	appendPoints_handmadeSingle(constructor)

def test_appendPoints_handmadeSequence():
	""" Test DMD appendPoints() against handmade output for a sequence of additions"""
	appendPoints_handmadeSequence(constructor)


################
# appendColumns() #
################

@raises(ArgumentException)
def test_appendColumns_exceptionNone():
	""" Test DMD appendColumns() for ArgumentException when toAppend is None """
	appendColumns_exceptionNone(constructor)

@raises(ArgumentException)
def test_appendColumns_exceptionWrongSize():
	""" Test DMD appendColumns() for ArgumentException when toAppend has too many points """
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
# sortPoints() #
##############

def test_sortPoints_handmadeNatural():
	""" Test DMD sortPoints() against handmade, naturally ordered output """	
	sortPoints_handmadeNatural(constructor)

def test_sortPoints_handmadeWithFcn():
	""" Test DMD sortPoints() against handmade output when given cmp and key functions """	
	sortPoints_handmadeWithFcn(constructor)

#################
# sortColumns() #
#################


def test_sortColumns_handmadeWithFcn():
	""" Test DMD sortColumns() against handmade output when given cmp and key functions """	
	sortColumns_handmadeWithFcn(constructor)



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
	""" Test DMD extractPoints() for ArgumentException when start is not a valid column index """
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








