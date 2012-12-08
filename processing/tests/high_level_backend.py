"""
Backend for unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, these
functions are to be called by unit tests of the derived class, with the appropiate
objects provided.

"""

from ..base_data import *
from copy import deepcopy


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


def convertColumnToCategoryColumns_handmade(constructor):
	""" Test convertColumnToCategoryColumns() against handmade output """
	data = [[1],[2],[3]]
	labels = ['col']
	toTest = constructor(data,labels)
	toTest.convertColumnToCategoryColumns(0)

	expData = [[1,0,0], [0,1,0], [0,0,1]]
	expLabels = ['col=1','col=2','col=3']
	exp = constructor(expData, expLabels)

	print toTest.labels

	assert toTest.equals(exp)
	


####################################
# convertColumnToIntegerCategories #
####################################

def convertColumnToIntegerCategories_handmade(constructor):
	""" Test convertColumnToIntegerCategories() against handmade output """
	data = [[10],[20],[30.5],[20],[10]]
	labels = ['col']
	toTest = constructor(data,labels)
	toTest.convertColumnToIntegerCategories(0)

	expLabels = {'col:categories':0}

	assert toTest.labels == expLabels
	assert toTest.data[0] == toTest.data[4]
	assert toTest.data[1] == toTest.data[3]
	assert toTest.data[0] != toTest.data[1]
	assert toTest.data[0] != toTest.data[2]




###############################
# selectConstantOfRowsByValue #
###############################

def selectConstantOfRowsByValue_exceptionNumToSelectNone(constructor):
	""" Test selectConstantOfRowsByValue() for Argument exception when numToSelect is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectConstantOfRowsByValue(None,'1')

def selectConstantOfRowsByValue_exceptionNumToSelectLEzero(constructor):
	""" Test selectConstantOfRowsByValue() for Argument exception when numToSelect <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectConstantOfRowsByValue(0,'1')

def selectConstantOfRowsByValue_handmade(constructor):
	""" Test selectConstantOfRowsByValue() against handmade output """
	data = [[1,2,3],[1,5,6],[1,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectConstantOfRowsByValue(2,'1')

	expRet = constructor([[1,2,3],[1,8,9]],labels)
	expTest = constructor([[1,5,6],],labels)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)

def selectConstantOfRowsByValue_handmadeLimit(constructor):
	""" Test selectConstantOfRowsByValue() against handmade output when the constant exceeds the available rows """
	data = [[1,2,3],[1,5,6],[1,8,9],[2,11,12]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectConstantOfRowsByValue(2,'1')

	expRet = constructor([[1,2,3],[1,8,9],[2,11,12]],labels)
	expTest = constructor([[1,5,6],],labels)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)



##############################
# selectPercentOfRowsByValue #
##############################

def selectPercentOfRowsByValue_exceptionPercentNone(constructor):
	""" Test selectPercentOfRowsByValue() for ArgumentException when percent to select is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectPercentOfRowsByValue(None,'1')

def selectPercentOfRowsByValue_exceptionPercentZero(constructor):
	""" Test selectPercentOfRowsByValue() for ArgumentException when percent to select is <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectPercentOfRowsByValue(0,'1')

def selectPercentOfRowsByValue_exceptionPercentOneHundrend(constructor):
	""" Test selectPercentOfRowsByValue() for ArgumentException when percent to select is >= 100 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectPercentOfRowsByValue(100,'1')

def selectPercentOfRowsByValue_handmade(constructor):
	""" Test selectPercentOfRowsByValue() against handmade output """
	data = [[1,2,3],[1,5,6],[1,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectPercentOfRowsByValue(50,'1')

	expRet = constructor([[1,2,3]],labels)
	expTest = constructor([[1,5,6],[1,8,9]],labels)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)


##########################
# selectPercentOfAllRows #
##########################

def selectPercentOfAllRows_exceptionPercentNone(constructor):
	""" Test selectPercentOfAllRows() for ArgumentException when percent to select is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectPercentOfAllRows(None)

def selectPercentOfAllRows_exceptionPercentZero(constructor):
	""" Test selectPercentOfAllRows() for ArgumentException when percent to select is <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectPercentOfAllRows(0)

def selectPercentOfAllRows_exceptionPercentOneHundrend(constructor):
	""" Test selectPercentOfAllRows() for ArgumentException when percent to select is >= 100 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectPercentOfAllRows(100)

def selectPercentOfAllRows_handmade(constructor):
	""" Test selectPercentOfAllRows() against handmade output with the test seed """
	data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	ret = toTest.selectPercentOfAllRows(50)

	expRet = constructor([[1,2,3],[10,11,12]],labels)
	expTest = constructor([[4,5,6],[7,8,9]],labels)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)

##############################
# selectEachRowWithGivenBias #
##############################

def selectEachRowWithGivenBias_exceptionNoneBias(constructor):
	""" Test selectEachRowWithGivenBias() for ArgumentException when bias is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectEachRowWithGivenBias(None)

def selectEachRowWithGivenBias_exceptionNoneLEzero(constructor):
	""" Test selectEachRowWithGivenBias() for ArgumentException when bias is <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectEachRowWithGivenBias(0)

def selectEachRowWithGivenBias_exceptionNoneGEone(constructor):
	""" Test selectEachRowWithGivenBias() for ArgumentException when bias is >= 1 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	toTest.selectEachRowWithGivenBias(1)

def selectEachRowWithGivenBias_handmade(constructor):
	""" Test selectEachRowWithGivenBias() against handmade output with the test seed """
	data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	labels = ['1','2','3']
	toTest = constructor(data,labels)
	ret = toTest.selectEachRowWithGivenBias(0.5)

	expRet = constructor([[1,2,3],[4,5,6],[7,8,9]],labels)
	expTest = constructor([[10,11,12]],labels)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)





