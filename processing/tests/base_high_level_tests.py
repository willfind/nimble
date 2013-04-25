"""
Unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, each
test will call the backend test for each possible representation

"""

from ..base_data import *
from ..row_list_data import RowListData
from ..dense_matrix_data import DenseMatrixData
from ..coo_sparse_data import CooSparseData
from high_level_backend import *

from copy import deepcopy
from nose.tools import *




def rld(data,featureNames=None):
	return RowListData(data,featureNames)

def dmd(data,featureNames=None):
	return DenseMatrixData(data,featureNames)

def coo(data,featureNames=None):
	return CooSparseData(data, featureNames)

def callAll(func):
	func(rld)
	func(dmd)
#	func(coo)



###########################
# dropStringValuedFeatures #
###########################


#hmmm but this only applies to representations that can have strings.



#################################
# featureToBinaryCategoryFeatures #
#################################


def test_featureToBinaryCategoryFeatures_handmade():
	""" Test featureToBinaryCategoryFeaturess() against handmade output """
	callAll(featureToBinaryCategoryFeatures_handmade)
	


#############################
# featureToIntegerCategories #
#############################

def test_featureToIntegerCategories_handmade():
	""" Test featureToIntegerCategories() against handmade output """
	callAll(featureToIntegerCategories_handmade)


###############################
# selectConstantOfPointsByValue #
###############################

@raises(ArgumentException)
def _selectConstantOfPointsByValue_exceptionNumToSelectNone():
	""" Test selectConstantOfPointsByValue() for Argument exception when numToSelect is None """
	callAll(selectConstantOfPointsByValue_exceptionNumToSelectNone)

@raises(ArgumentException)
def _selectConstantOfPointsByValue_exceptionNumToSelectLEzero():
	""" Test selectConstantOfPointsByValue() for Argument exception when numToSelect <= 0 """
	callAll(selectConstantOfPointsByValue_exceptionNumToSelectLEzero)

@raises(ArgumentException)
def _selectConstantOfPointsByValue_handmade():
	""" Test selectConstantOfPointsByValue() against handmade output """
	callAll(selectConstantOfPointsByValue_handmade)

def _selectConstantOfPointsByValue_handmadeLimit():
	""" Test selectConstantOfPointsByValue() against handmade output when the constant exceeds the available points """
	callAll(selectConstantOfPointsByValue_handmadeLimit)


##############################
# selectPercentOfPointsByValue #
##############################

@raises(ArgumentException)
def _selectPercentOfPointsByValue_exceptionPercentNone():
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is None """
	callAll(selectPercentOfPointsByValue_exceptionPercentNone)

@raises(ArgumentException)
def _selectPercentOfPointsByValue_exceptionPercentZero():
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is <= 0 """
	callAll(selectPercentOfPointsByValue_exceptionPercentZero)

@raises(ArgumentException)
def _selectPercentOfPointsByValue_exceptionPercentOneHundrend():
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is >= 100 """
	callAll(selectPercentOfPointsByValue_exceptionPercentOneHundrend)

def _selectPercentOfPointsByValue_handmade():
	""" Test selectPercentOfPointsByValue() against handmade output """
	callAll(selectPercentOfPointsByValue_handmade)


#########################
# extractPointsByCoinToss #
#########################

@raises(ArgumentException)
def test_extractPointsByCoinToss_exceptionNoneProbability():
	""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is None """
	callAll(extractPointsByCoinToss_exceptionNoneProbability)

@raises(ArgumentException)
def test_extractPointsByCoinToss_exceptionLEzero():
	""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is <= 0 """
	callAll(extractPointsByCoinToss_exceptionLEzero)

@raises(ArgumentException)
def test_extractPointsByCoinToss_exceptionGEone():
	""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is >= 1 """
	callAll(extractPointsByCoinToss_exceptionGEone)


def test_extractPointsByCoinToss_handmade():
	""" Test extractPointsByCoinToss() against handmade output with the test seed """
	callAll(extractPointsByCoinToss_handmade)



################
# foldIterator #
################

@raises(ArgumentException)
def test_foldIterator_exceptionTooManyFolds():
	""" Test foldIterator() for exception when given too many folds """
	callAll(foldIterator_exceptionTooManyFolds)


def test_foldIterator_verifyPartitions():
	""" Test foldIterator() yields the correct number and size of folds partitioning the data """
	callAll(foldIterator_verifyPartitions)



####################
# applyFunctionToEachPoint() #
####################

@raises(ArgumentException)
def test_applyFunctionToEachPoint_exceptionInputNone():
	""" Test applyFunctionToEachPoint() for ArgumentException when function is None """
	callAll(applyFunctionToEachPoint_exceptionInputNone)

def test_applyFunctionToEachPoint_Handmade():
	""" Test applyFunctionToEachPoint() with handmade output """
	callAll(applyFunctionToEachPoint_Handmade)


def test_applyFunctionToEachPoint_nonZeroItAndLen():
	""" Test applyFunctionToEachPoint() for the correct usage of the nonzero iterator """
	callAll(applyFunctionToEachPoint_nonZeroItAndLen)



#######################
# applyFunctionToEachFeature() #
#######################

@raises(ArgumentException)
def test_applyFunctionToEachFeature_exceptionInputNone():
	""" Test applyFunctionToEachFeature() for ArgumentException when function is None """
	callAll(applyFunctionToEachFeature_exceptionInputNone)

def test_applyFunctionToEachFeature_Handmade():
	""" Test applyFunctionToEachFeature() with handmade output """
	callAll(applyFunctionToEachFeature_Handmade)


def test_applyFunctionToEachFeature_nonZeroItAndLen():
	""" Test applyFunctionToEachFeature() for the correct usage of the nonzero iterator """
	callAll(applyFunctionToEachFeature_nonZeroItAndLen)


#####################
# mapReduceOnPoints() #
#####################

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionNoneMap():
	""" Test mapReduceOnPoints() for ArgumentException when mapper is None """
	callAll(mapReduceOnPoints_argumentExceptionNoneMap)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionNoneReduce():
	""" Test mapReduceOnPoints() for ArgumentException when reducer is None """
	callAll(mapReduceOnPoints_argumentExceptionNoneReduce)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionUncallableMap():
	""" Test mapReduceOnPoints() for ArgumentException when mapper is not callable """
	callAll(mapReduceOnPoints_argumentExceptionUncallableMap)

@raises(ArgumentException)
def test_mapReduceOnPoints_argumentExceptionUncallableReduce():
	""" Test mapReduceOnPoints() for ArgumentException when reducer is not callable """
	callAll(mapReduceOnPoints_argumentExceptionUncallableReduce)

def test_mapReduceOnPoints_handmade():
	""" Test mapReduceOnPoints() against handmade output """
	callAll(mapReduceOnPoints_handmade)


def test_mapReduceOnPoints_handmadeNoneReturningReducer():
	""" Test mapReduceOnPoints() against handmade output with a None returning Reducer """
	callAll(mapReduceOnPoints_handmadeNoneReturningReducer)



####################
# transformPoint() #
####################




######################
# transformFeature() #
######################



#####################################
# computeListOfValuesFromElements() #
#####################################


def test_computeList_passthrough():
	""" test computeListOfValuesFromElements() can construct a list by just passing values through  """
	callAll(computeList_passthrough)

def test_computeList_passthroughSkip():
	""" test computeListOfValuesFromElements() skipZeros flag """
	callAll(computeList_passthroughSkip)


def test_computeList_passthroughExclude():
	""" test computeListOfValuesFromElements() excludeNoneResultValues flag  """
	callAll(computeList_passthroughSkip)




########################
# isApproxEquivalent() #
########################


def test_isApproxEquivalent_randomTest():
	""" Test isApproxEquivalent() using randomly generated data """
	callAll(isApproxEquivalent_randomTest)
