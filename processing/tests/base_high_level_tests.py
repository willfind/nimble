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




def rld(data,labels=None):
	return RowListData(data,labels)

def dmd(data,labels=None):
	return DenseMatrixData(data,labels)

def callAll(func):
	func(rld)
	func(dmd)


###################
# duplicateObject #
###################




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



##################################
# convertColumnToCategoryColumns #
##################################


def test_convertColumnToCategoryColumns_handmade():
	""" Test convertColumnToCategoryColumns() against handmade output """
	callAll(convertColumnToCategoryColumns_handmade)
	


####################################
# convertColumnToIntegerCategories #
####################################

def test_convertColumnToIntegerCategories_handmade():
	""" Test convertColumnToIntegerCategories() against handmade output """
	callAll(convertColumnToIntegerCategories_handmade)


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

##############################
# selectEachRowWithGivenBias #
##############################

@raises(ArgumentException)
def test_selectEachRowWithGivenBias_exceptionNoneBias():
	""" Test selectEachRowWithGivenBias() for ArgumentException when bias is None """
	callAll(selectEachRowWithGivenBias_exceptionNoneBias)

@raises(ArgumentException)
def test_selectEachRowWithGivenBias_exceptionNoneLEzero():
	""" Test selectEachRowWithGivenBias() for ArgumentException when bias is <= 0 """
	callAll(selectEachRowWithGivenBias_exceptionNoneLEzero)

@raises(ArgumentException)
def test_selectEachRowWithGivenBias_exceptionNoneGEone():
	""" Test selectEachRowWithGivenBias() for ArgumentException when bias is >= 1 """
	callAll(selectEachRowWithGivenBias_exceptionNoneGEone)


def test_selectEachRowWithGivenBias_handmade():
	""" Test selectEachRowWithGivenBias() against handmade output with the test seed """
	callAll(selectEachRowWithGivenBias_handmade)



