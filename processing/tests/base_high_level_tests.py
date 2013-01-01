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


#############
# duplicate #
#############




#################
# duplicateRows #
#################




####################
# duplicateColumns #
####################





###########################
# dropStringValuedColumns #
###########################


#hmmm but this only applies to representations that can have strings.



#################################
# columnToBinaryCategoryColumns #
#################################


def test_columnToBinaryCategoryColumns_handmade():
	""" Test columnToBinaryCategoryColumnss() against handmade output """
	callAll(columnToBinaryCategoryColumns_handmade)
	


#############################
# columnToIntegerCategories #
#############################

def test_columnToIntegerCategories_handmade():
	""" Test columnToIntegerCategories() against handmade output """
	callAll(columnToIntegerCategories_handmade)


###############################
# selectConstantOfRowsByValue #
###############################

@raises(ArgumentException)
def test_selectConstantOfRowsByValue_exceptionNumToSelectNone():
	""" Test selectConstantOfRowsByValue() for Argument exception when numToSelect is None """
	callAll(selectConstantOfRowsByValue_exceptionNumToSelectNone)

@raises(ArgumentException)
def test_selectConstantOfRowsByValue_exceptionNumToSelectLEzero():
	""" Test selectConstantOfRowsByValue() for Argument exception when numToSelect <= 0 """
	callAll(selectConstantOfRowsByValue_exceptionNumToSelectLEzero)

@raises(ArgumentException)
def test_selectConstantOfRowsByValue_handmade():
	""" Test selectConstantOfRowsByValue() against handmade output """
	callAll(selectConstantOfRowsByValue_handmade)

def selectConstantOfRowsByValue_handmadeLimit():
	""" Test selectConstantOfRowsByValue() against handmade output when the constant exceeds the available rows """
	callAll(selectConstantOfRowsByValue_handmadeLimit)


##############################
# selectPercentOfRowsByValue #
##############################

@raises(ArgumentException)
def test_selectPercentOfRowsByValue_exceptionPercentNone():
	""" Test selectPercentOfRowsByValue() for ArgumentException when percent to select is None """
	callAll(selectPercentOfRowsByValue_exceptionPercentNone)

@raises(ArgumentException)
def test_selectPercentOfRowsByValue_exceptionPercentZero():
	""" Test selectPercentOfRowsByValue() for ArgumentException when percent to select is <= 0 """
	callAll(selectPercentOfRowsByValue_exceptionPercentZero)

@raises(ArgumentException)
def test_selectPercentOfRowsByValue_exceptionPercentOneHundrend():
	""" Test selectPercentOfRowsByValue() for ArgumentException when percent to select is >= 100 """
	callAll(selectPercentOfRowsByValue_exceptionPercentOneHundrend)

def test_selectPercentOfRowsByValue_handmade():
	""" Test selectPercentOfRowsByValue() against handmade output """
	callAll(selectPercentOfRowsByValue_handmade)


##########################
# selectPercentOfAllRows #
##########################

@raises(ArgumentException)
def test_selectPercentOfAllRows_exceptionPercentNone():
	""" Test selectPercentOfAllRows() for ArgumentException when percent to select is None """
	callAll(selectPercentOfAllRows_exceptionPercentNone)

@raises(ArgumentException)
def test_selectPercentOfAllRows_exceptionPercentZero():
	""" Test selectPercentOfAllRows() for ArgumentException when percent to select is <= 0 """
	callAll(selectPercentOfAllRows_exceptionPercentZero)

@raises(ArgumentException)
def test_selectPercentOfAllRows_exceptionPercentOneHundrend():
	""" Test selectPercentOfAllRows() for ArgumentException when percent to select is >= 100 """
	callAll(selectPercentOfAllRows_exceptionPercentOneHundrend)

def test_selectPercentOfAllRows_handmade():
	""" Test selectPercentOfAllRows() against handmade output with the test seed """
	callAll(selectPercentOfAllRows_handmade)

#########################
# extractRowsByCoinToss #
#########################

@raises(ArgumentException)
def test_extractRowsByCoinToss_exceptionNoneProbability():
	""" Test extractRowsByCoinToss() for ArgumentException when extractionProbability is None """
	callAll(extractRowsByCoinToss_exceptionNoneProbability)

@raises(ArgumentException)
def test_extractRowsByCoinToss_exceptionLEzero():
	""" Test extractRowsByCoinToss() for ArgumentException when extractionProbability is <= 0 """
	callAll(extractRowsByCoinToss_exceptionLEzero)

@raises(ArgumentException)
def test_extractRowsByCoinToss_exceptionGEone():
	""" Test extractRowsByCoinToss() for ArgumentException when extractionProbability is >= 1 """
	callAll(extractRowsByCoinToss_exceptionGEone)


def test_extractRowsByCoinToss_handmade():
	""" Test extractRowsByCoinToss() against handmade output with the test seed """
	callAll(extractRowsByCoinToss_handmade)



