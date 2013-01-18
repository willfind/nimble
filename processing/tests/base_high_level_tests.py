"""
Unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, each
test will call the backend test for each possible representation

"""

from ..base_data import *
from ..row_list_data import RowListData
from ..dense_matrix_data import DenseMatrixData
from high_level_backend import *

from copy import deepcopy
from nose.tools import *




def rld(data,featureNames=None):
	return RowListData(data,featureNames)

def dmd(data,featureNames=None):
	return DenseMatrixData(data,featureNames)

def callAll(func):
	func(rld)
	func(dmd)




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
def test_selectConstantOfPointsByValue_exceptionNumToSelectNone():
	""" Test selectConstantOfPointsByValue() for Argument exception when numToSelect is None """
	callAll(selectConstantOfPointsByValue_exceptionNumToSelectNone)

@raises(ArgumentException)
def test_selectConstantOfPointsByValue_exceptionNumToSelectLEzero():
	""" Test selectConstantOfPointsByValue() for Argument exception when numToSelect <= 0 """
	callAll(selectConstantOfPointsByValue_exceptionNumToSelectLEzero)

@raises(ArgumentException)
def test_selectConstantOfPointsByValue_handmade():
	""" Test selectConstantOfPointsByValue() against handmade output """
	callAll(selectConstantOfPointsByValue_handmade)

def selectConstantOfPointsByValue_handmadeLimit():
	""" Test selectConstantOfPointsByValue() against handmade output when the constant exceeds the available points """
	callAll(selectConstantOfPointsByValue_handmadeLimit)


##############################
# selectPercentOfPointsByValue #
##############################

@raises(ArgumentException)
def test_selectPercentOfPointsByValue_exceptionPercentNone():
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is None """
	callAll(selectPercentOfPointsByValue_exceptionPercentNone)

@raises(ArgumentException)
def test_selectPercentOfPointsByValue_exceptionPercentZero():
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is <= 0 """
	callAll(selectPercentOfPointsByValue_exceptionPercentZero)

@raises(ArgumentException)
def test_selectPercentOfPointsByValue_exceptionPercentOneHundrend():
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is >= 100 """
	callAll(selectPercentOfPointsByValue_exceptionPercentOneHundrend)

def test_selectPercentOfPointsByValue_handmade():
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
