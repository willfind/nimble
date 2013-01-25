"""
Backend for unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, these
functions are to be called by unit tests of the derived class, with the appropiate
objects provided.

"""

from ..base_data import *
from copy import deepcopy




###########################
# dropStringValuedFeatures #
###########################


#hmmm but this only applies to representations that can have strings.



#################################
# featureToBinaryCategoryFeatures #
#################################


def featureToBinaryCategoryFeatures_handmade(constructor):
	""" Test featureToBinaryCategoryFeatures() against handmade output """
	data = [[1],[2],[3]]
	featureNames = ['col']
	toTest = constructor(data,featureNames)
	toTest.featureToBinaryCategoryFeatures(0)

	expData = [[1,0,0], [0,1,0], [0,0,1]]
	expFeatureNames = ['col=1','col=2','col=3']
	exp = constructor(expData, expFeatureNames)

	print toTest.featureNames

	assert toTest.equals(exp)
	


#############################
# featureToIntegerCategories #
#############################

def featureToIntegerCategories_handmade(constructor):
	""" Test featureToIntegerCategories() against handmade output """
	data = [[10],[20],[30.5],[20],[10]]
	featureNames = ['col']
	toTest = constructor(data,featureNames)
	toTest.featureToIntegerCategories(0)

	assert toTest.data[0] == toTest.data[4]
	assert toTest.data[1] == toTest.data[3]
	assert toTest.data[0] != toTest.data[1]
	assert toTest.data[0] != toTest.data[2]




###############################
# selectConstantOfPointsByValue #
###############################

def selectConstantOfPointsByValue_exceptionNumToSelectNone(constructor):
	""" Test selectConstantOfPointsByValue() for Argument exception when numToSelect is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfPointsByValue(None,'1')

def selectConstantOfPointsByValue_exceptionNumToSelectLEzero(constructor):
	""" Test selectConstantOfPointsByValue() for Argument exception when numToSelect <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfPointsByValue(0,'1')

def selectConstantOfPointsByValue_handmade(constructor):
	""" Test selectConstantOfPointsByValue() against handmade output """
	data = [[1,2,3],[1,5,6],[1,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfPointsByValue(2,'1')

	expRet = constructor([[1,2,3],[1,8,9]],featureNames)
	expTest = constructor([[1,5,6],],featureNames)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)

def selectConstantOfPointsByValue_handmadeLimit(constructor):
	""" Test selectConstantOfPointsByValue() against handmade output when the constant exceeds the available points """
	data = [[1,2,3],[1,5,6],[1,8,9],[2,11,12]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectConstantOfPointsByValue(2,'1')

	expRet = constructor([[1,2,3],[1,8,9],[2,11,12]],featureNames)
	expTest = constructor([[1,5,6],],featureNames)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)



##############################
# selectPercentOfPointsByValue #
##############################

def selectPercentOfPointsByValue_exceptionPercentNone(constructor):
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfPointsByValue(None,'1')

def selectPercentOfPointsByValue_exceptionPercentZero(constructor):
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfPointsByValue(0,'1')

def selectPercentOfPointsByValue_exceptionPercentOneHundrend(constructor):
	""" Test selectPercentOfPointsByValue() for ArgumentException when percent to select is >= 100 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfPointsByValue(100,'1')

def selectPercentOfPointsByValue_handmade(constructor):
	""" Test selectPercentOfPointsByValue() against handmade output """
	data = [[1,2,3],[1,5,6],[1,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.selectPercentOfPointsByValue(50,'1')

	expRet = constructor([[1,2,3]],featureNames)
	expTest = constructor([[1,5,6],[1,8,9]],featureNames)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)


#########################
# extractPointsByCoinToss #
#########################

def extractPointsByCoinToss_exceptionNoneProbability(constructor):
	""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is None """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.extractPointsByCoinToss(None)

def extractPointsByCoinToss_exceptionLEzero(constructor):
	""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is <= 0 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.extractPointsByCoinToss(0)

def extractPointsByCoinToss_exceptionGEone(constructor):
	""" Test extractPointsByCoinToss() for ArgumentException when extractionProbability is >= 1 """
	data = [[1,2,3],[4,5,6],[7,8,9]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	toTest.extractPointsByCoinToss(1)

def extractPointsByCoinToss_handmade(constructor):
	""" Test extractPointsByCoinToss() against handmade output with the test seed """
	data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
	featureNames = ['1','2','3']
	toTest = constructor(data,featureNames)
	ret = toTest.extractPointsByCoinToss(0.5)

	expRet = constructor([[4,5,6],[7,8,9]],featureNames)
	expTest = constructor([[1,2,3],[10,11,12]],featureNames)

	assert ret.equals(expRet)
	assert expTest.equals(toTest)



################
# foldIterator #
################

def foldIterator_exceptionTooManyFolds(constructor):
	""" Test foldIterator() for exception when given too many folds """
	data = [[1],[2],[3],[4],[5]]
	names = ['col']
	toTest = constructor(data,names)
	toTest.foldIterator(6)


def foldIterator_verifyPartitions(constructor):
	""" Test foldIterator() yields the correct number folds and partitions the data """
	data = [[1],[2],[3],[4],[5]]
	names = ['col']
	toTest = constructor(data,names)
	folds = toTest.foldIterator(2)

	(fold1Train, fold1Test) = folds.next()
	(fold2Train, fold2Test) = folds.next()

	try:
		folds.next()
		assert False
	except StopIteration:
		pass

	assert fold1Train.points() + fold1Test.points() == 5
	assert fold2Train.points() + fold2Test.points() == 5

	fold1Train.appendPoints(fold1Test)
	fold2Train.appendPoints(fold2Test)

	#TODO some kind of rigourous partition check
