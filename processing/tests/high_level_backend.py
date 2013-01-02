"""
Backend for unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, these
functions are to be called by unit tests of the derived class, with the appropiate
objects provided.

"""

from ..base_data import *
from copy import deepcopy


###################
# duplicate #
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



#################################
# columnToBinaryCategoryColumns #
#################################


def columnToBinaryCategoryColumns_handmade(constructor):
	""" Test convertColumnToCategoryColumns() against handmade output """
	data = [[1],[2],[3]]
	featureNames = ['col']
	toTest = constructor(data,featureNames)
	toTest.columnToBinaryCategoryColumns(0)

	expData = [[1,0,0], [0,1,0], [0,0,1]]
	expFeatureNames = ['col=1','col=2','col=3']
	exp = constructor(expData, expFeatureNames)

	print toTest.featureNames

	assert toTest.equals(exp)
	


#############################
# columnToIntegerCategories #
#############################

def columnToIntegerCategories_handmade(constructor):
	""" Test convertColumnToIntegerCategories() against handmade output """
	data = [[10],[20],[30.5],[20],[10]]
	featureNames = ['col']
	toTest = constructor(data,featureNames)
	toTest.columnToIntegerCategories(0)

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
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfRowsByValue(None,'1')

def selectConstantOfRowsByValue_exceptionNumToSelectLEzero(constructor):
	""" Test selectConstantOfRowsByValue() for Argument exception when numToSelect <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfRowsByValue(0,'1')

def selectConstantOfRowsByValue_handmade(constructor):
	""" Test selectConstantOfRowsByValue() against handmade output """
	data = [[1,2,3],[1,5,6],[1,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfRowsByValue(2,'1')

	expRet = constructor([[1,2,3],[1,8,9]],featureNames)
	expTest = constructor([[1,5,6],],featureNames)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)

def selectConstantOfRowsByValue_handmadeLimit(constructor):
	""" Test selectConstantOfRowsByValue() against handmade output when the constant exceeds the available rows """
	data = [[1,2,3],[1,5,6],[1,8,9],[2,11,12]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfRowsByValue(2,'1')

	expRet = constructor([[1,2,3],[1,8,9],[2,11,12]],featureNames)
	expTest = constructor([[1,5,6],],featureNames)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)



##############################
# selectPercentOfRowsByValue #
##############################

def selectPercentOfRowsByValue_exceptionPercentNone(constructor):
	""" Test selectPercentOfRowsByValue() for ArgumentException when percent to select is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfRowsByValue(None,'1')

def selectPercentOfRowsByValue_exceptionPercentZero(constructor):
	""" Test selectPercentOfRowsByValue() for ArgumentException when percent to select is <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfRowsByValue(0,'1')

def selectPercentOfRowsByValue_exceptionPercentOneHundrend(constructor):
	""" Test selectPercentOfRowsByValue() for ArgumentException when percent to select is >= 100 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfRowsByValue(100,'1')

def selectPercentOfRowsByValue_handmade(constructor):
	""" Test selectPercentOfRowsByValue() against handmade output """
	data = [[1,2,3],[1,5,6],[1,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfRowsByValue(50,'1')

	expRet = constructor([[1,2,3]],featureNames)
	expTest = constructor([[1,5,6],[1,8,9]],featureNames)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)


#########################
# extractRowsByCoinToss #
#########################

def extractRowsByCoinToss_exceptionNoneProbability(constructor):
	""" Test extractRowsByCoinToss() for ArgumentException when extractionProbability is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.extractRowsByCoinToss(None)

def extractRowsByCoinToss_exceptionLEzero(constructor):
	""" Test extractRowsByCoinToss() for ArgumentException when extractionProbability is <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.extractRowsByCoinToss(0)

def extractRowsByCoinToss_exceptionGEone(constructor):
	""" Test extractRowsByCoinToss() for ArgumentException when extractionProbability is >= 1 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.extractRowsByCoinToss(1)

def extractRowsByCoinToss_handmade(constructor):
	""" Test extractRowsByCoinToss() against handmade output with the test seed """
	data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	ret = toTest.extractRowsByCoinToss(0.5)

	expRet = constructor([[4,5,6],[7,8,9]],featureNames)
	expTest = constructor([[1,2,3],[10,11,12]],featureNames)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)





