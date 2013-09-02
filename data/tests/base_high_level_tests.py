"""
Unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, each
test will call the backend test for each possible representation

"""

from nose.tools import *


from UML.data import List
from UML.data import Matrix
from UML.data import Sparse
from UML.data.tests.high_level_backend import *
from UML import createData

from UML.exceptions import ImproperActionException
from UML.exceptions import ArgumentException






def listInit(data,featureNames=None):
	return List(data,featureNames)

def matrixInit(data,featureNames=None):
	return Matrix(data,featureNames)

def sparseInit(data,featureNames=None):
	return Sparse(data, featureNames)

def callAll(func):
	func(listInit)
	func(matrixInit)
	func(sparseInit)



###########################
# dropFeaturesContainingType #
###########################

def test_dropFeaturesContainingType_emptyTest():
	""" Test dropFeaturesContainingType() when the data is empty """
	callAll(dropFeaturesContainingType_emptyTest)


def test_dropFeaturesContainingType_ListOnlyTest():
	""" Test dropFeaturesContainingType() only on List data """
	callAll(dropFeaturesContainingType_ListOnlyTest)

#################################
# replaceFeatureWithBinaryFeatures #
#################################

@raises(ArgumentException)
def test_replaceFeatureWithBinaryFeatures_PemptyException():
	""" Test replaceFeatureWithBinaryFeatures() with a point empty object """
	callAll(replaceFeatureWithBinaryFeatures_PemptyException)

def test_replaceFeatureWithBinaryFeatures_handmade():
	""" Test replaceFeatureWithBinaryFeaturess() against handmade output """
	callAll(replaceFeatureWithBinaryFeatures_handmade)
	


#############################
# transformFeartureToIntegerFeature #
#############################

@raises(ImproperActionException)
def test_transformFeartureToIntegerFeature_emptyException():
	""" Test transformFeartureToIntegerFeature() with an empty object """
	callAll(transformFeartureToIntegerFeature_emptyException)

def test_transformFeartureToIntegerFeature_handmade():
	""" Test transformFeartureToIntegerFeature() against handmade output """
	callAll(transformFeartureToIntegerFeature_handmade)


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

@raises(ImproperActionException)
def test_extractPointsByCoinToss_exceptionEmpty():
	""" Test extractPointsByCoinToss() for ImproperActionException when object is empty """
	callAll(extractPointsByCoinToss_exceptionEmpty)

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

@raises(ImproperActionException)
def test_foldIterator_exceptionEmpty():
	""" Test foldIterator() for exception when object is empty """
	callAll(foldIterator_exceptionEmpty)

@raises(ArgumentException)
def test_foldIterator_exceptionTooManyFolds():
	""" Test foldIterator() for exception when given too many folds """
	callAll(foldIterator_exceptionTooManyFolds)


def test_foldIterator_verifyPartitions():
	""" Test foldIterator() yields the correct number and size of folds partitioning the data """
	callAll(foldIterator_verifyPartitions)

def test_foldIterator_ordering():
	""" Test that foldIterator() yields folds in the proper order: X and Y folds should be in the same order"""
	twoColumnData = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]]
	matrixObj = createData('Matrix', twoColumnData)
	Ydata = matrixObj.extractFeatures([1])
	Xdata = matrixObj
	XIterator = Xdata.foldIterator(numFolds=2)
	YIterator = Ydata.foldIterator(numFolds=2)
	
	while True: #need to add a test here for when iterator .next() is done
		try:
			curTrainX, curTestX = XIterator.next()
			curTrainY, curTestY = YIterator.next()
		except StopIteration:	#once we've gone through all the folds, this exception gets thrown and we're done!
			break
		curTrainXList = curTrainX.copy(asType="python list")
		curTestXList = curTestX.copy(asType="python list")
		curTrainYList = curTrainY.copy(asType="python list")
		curTestYList = curTestY.copy(asType="python list")

#		import pdb
#		pdb.set_trace()

		for i in range(len(curTrainXList)):
			assert curTrainXList[i][0] == curTrainYList[i][0]

		for i in range(len(curTestXList)):
			assert curTestXList[i][0] == curTestYList[i][0]


####################
# applyToPoints() #
####################

@raises(ImproperActionException)
def test_applyToPoints_exceptionEmpty():
	""" Test applyToPoints() for ImproperActionException when object is empty """
	callAll(applyToPoints_exceptionEmpty)

@raises(ArgumentException)
def test_applyToPoints_exceptionInputNone():
	""" Test applyToPoints() for ArgumentException when function is None """
	callAll(applyToPoints_exceptionInputNone)

def test_applyToPoints_Handmade():
	""" Test applyToPoints() with handmade output """
	callAll(applyToPoints_Handmade)

def test_applyToPoints_HandmadeLimited():
	""" Test applyToPoints() with handmade output on a limited portion of points """
	callAll(applyToPoints_HandmadeLimited)

def test_applyToPoints_nonZeroItAndLen():
	""" Test applyToPoints() for the correct usage of the nonzero iterator """
	callAll(applyToPoints_nonZeroItAndLen)

def test_applyToPoints_HandmadeInPlace():
	""" Test applyToPoints() with handmade output. InPlace """
	callAll(applyToPoints_HandmadeInPlace)

def test_applyToPoints_HandmadeLimitedInPlace():
	""" Test applyToPoints() with handmade output on a limited portion of points. InPlace"""
	callAll(applyToPoints_HandmadeLimitedInPlace)

def test_applyToPoints_nonZeroItAndLenInPlace():
	""" Test applyToPoints() for the correct usage of the nonzero iterator. InPlace """
	callAll(applyToPoints_nonZeroItAndLenInPlace)

#######################
# applyToFeatures() #
#######################

@raises(ImproperActionException)
def test_applyToFeatures_exceptionEmpty():
	""" Test applyToFeatures() for ImproperActionException when object is empty """
	callAll(applyToFeatures_exceptionEmpty)

@raises(ArgumentException)
def test_applyToFeatures_exceptionInputNone():
	""" Test applyToFeatures() for ArgumentException when function is None """
	callAll(applyToFeatures_exceptionInputNone)

def test_applyToFeatures_Handmade():
	""" Test applyToFeatures() with handmade output """
	callAll(applyToFeatures_Handmade)

def test_applyToFeatures_HandmadeLimited():
	""" Test applyToFeatures() with handmade output on a limited portion of features """
	callAll(applyToFeatures_HandmadeLimited)

def test_applyToFeatures_nonZeroItAndLen():
	""" Test applyToFeatures() for the correct usage of the nonzero iterator """
	callAll(applyToFeatures_nonZeroItAndLen)


def test_applyToFeatures_HandmadeInPlace():
	""" Test applyToFeatures() with handmade output. InPlace """
	callAll(applyToFeatures_HandmadeInPlace)


def test_applyToFeatures_HandmadeLimitedInPlace():
	""" Test applyToFeatures() with handmade output on a limited portion of features. InPlace """
	callAll(applyToFeatures_HandmadeLimitedInPlace)


def test_applyToFeatures_nonZeroItAndLenInPlace():
	""" Test applyToFeatures() for the correct usage of the nonzero iterator. InPlace """
	callAll(applyToFeatures_nonZeroItAndLenInPlace)


#####################
# mapReducePoints() #
#####################

@raises(ArgumentException)
def test_mapReducePoints_argumentExceptionNoneMap():
	""" Test mapReducePoints() for ArgumentException when mapper is None """
	callAll(mapReducePoints_argumentExceptionNoneMap)

@raises(ArgumentException)
def test_mapReducePoints_argumentExceptionNoneReduce():
	""" Test mapReducePoints() for ArgumentException when reducer is None """
	callAll(mapReducePoints_argumentExceptionNoneReduce)

@raises(ArgumentException)
def test_mapReducePoints_argumentExceptionUncallableMap():
	""" Test mapReducePoints() for ArgumentException when mapper is not callable """
	callAll(mapReducePoints_argumentExceptionUncallableMap)

@raises(ArgumentException)
def test_mapReducePoints_argumentExceptionUncallableReduce():
	""" Test mapReducePoints() for ArgumentException when reducer is not callable """
	callAll(mapReducePoints_argumentExceptionUncallableReduce)

def test_mapReducePoints_handmade():
	""" Test mapReducePoints() against handmade output """
	callAll(mapReducePoints_handmade)


def test_mapReducePoints_handmadeNoneReturningReducer():
	""" Test mapReducePoints() against handmade output with a None returning Reducer """
	callAll(mapReducePoints_handmadeNoneReturningReducer)



#######################
# pointIterator() #
#######################


def test_pointIterator_exactValueViaFor():
	""" Test pointIterator() gives views that contain exactly the correct data """
	callAll(pointIterator_exactValueViaFor)

#########################
# featureIterator() #
#########################


def test_featureIterator_exactValueViaFor():
	""" Test featureIterator() gives views that contain exactly the correct data """
	callAll(featureIterator_exactValueViaFor)


####################
# transformPoint() #
####################




######################
# transformFeature() #
######################



#####################################
# applyToElements() #
#####################################


def test_applyToElements_passthrough():
	""" test applyToElements() can construct a list by just passing values through  """
	callAll(applyToElements_passthrough)

def test_applyToElements_plusOnePreserve():
	""" test applyToElements can modify elements other than zero """
	callAll(applyToElements_plusOnePreserve)


def test_applyToElements_plusOneExclude():
	""" test applyToElements() skipNoneReturnValues flag  """
	callAll(applyToElements_plusOneExclude)

def test_applyToElements_plusOneLimited():
	""" test applyToElements() on limited portions of the points and features """
	callAll(applyToElements_plusOneLimited)


########################
# isApproximatelyEqual() #
########################


def test_isApproximatelyEqual_randomTest():
	""" Test isApproximatelyEqual() using randomly generated data """
	callAll(isApproximatelyEqual_randomTest)



###################
# shufflePoints() #
###################


def test_shufflePoints_noLongerEqual():
	""" Tests shufflePoints() results in a changed object """
	callAll(shufflePoints_noLongerEqual)



#####################
# shuffleFeatures() #
#####################


def test_shuffleFeatures_noLongerEqual():
	""" Tests shuffleFeatures() results in a changed object """
	callAll(shuffleFeatures_noLongerEqual)

