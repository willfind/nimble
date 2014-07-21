#test scripts for
# crossValidate



#so you can run as main:
import sys
sys.path.append('../..')

import nose

from UML import crossValidate
from UML import createData
from UML.metrics import *
from UML.randomness import pythonRandom

from pdb import set_trace as ttt


def _randomLabeledDataSet(dataType='matrix', numPoints=50, numFeatures=5, numLabels=3):
	"""returns a tuple of two data objects of type dataType
	the first object in the tuple contains the feature information ('X' in UML language)
	the second object in the tuple contains the labels for each feature ('Y' in UML language)
	"""
	if numLabels is None:
		labelsRaw = [[pythonRandom.random()] for _x in xrange(numPoints)]
	else: #labels data set
		labelsRaw = [[int(pythonRandom.random()*numLabels)] for _x in xrange(numPoints)]

	rawFeatures = [[pythonRandom.random() for _x in xrange(numFeatures)] for _y in xrange(numPoints)]

	return (createData(dataType, rawFeatures), createData(dataType, labelsRaw))

def test_crossValidate_XY_unchanged():
	"""assert that after running cross validate on datasets passed to 
	X and Y, the original data is unchanged"""
	classifierAlgo = 'Custom.KNNClassifier'
	X, Y = _randomLabeledDataSet(numLabels=5)
	copyX = X.copy()
	copyY = Y.copy()
	result = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
	assert X.hashCode() == copyX.hashCode()
	assert Y.hashCode() == copyY.hashCode()



def test_crossValidate_runs():
	"""tests that crossValidate gives results (in form of float) 
	for different algorithms
	and different data types (children of base)
	"""
	#just scrap data to make sure it doesn't crash
	numLabelsInSet = 3
	numPointsInSet = 50
	#todo add other data types - currently crashes in sklearn interface for list and sparse
	for dType in ['matrix',]:
		X, Y = _randomLabeledDataSet(numPoints=numPointsInSet, numLabels=numLabelsInSet, dataType=dType)	
		classifierAlgos = ['Custom.KNNClassifier']
		for curAlgo in classifierAlgos:
			result = crossValidate(curAlgo, X, Y, fractionIncorrect, {}, numFolds=3)
			assert isinstance(result, float)


		#With regression dataset (no repeated labels)
		X, Y = _randomLabeledDataSet(numLabels=None, dataType=dType)	
		classifierAlgos = ['Custom.RidgeRegression']
		for curAlgo in classifierAlgos:
			result = crossValidate(curAlgo, X, Y, meanAbsoluteError, {}, numFolds=3)
			assert isinstance(result, float)
		

def _assertClassifierErrorOnRandomDataPlausible(actualError, numLabels, tolerance=.1):
	"""assert the actual error on a labeled data set (for a classifier)
	is plausible, given the number of (evenly distributed) labels in hte data set
	"""
	idealFractionIncorrect = 1.0 - 1.0/numLabels
	error = abs(actualError - idealFractionIncorrect)
	assert error <= tolerance


def test_crossValidate_reasonable_results():
	"""Assert that crossValidate returns reasonable errors for known algorithms
	on cooked data sets:
	crossValidate should do the following:
	classifiers:
		have no error when there is only one label in the data set
		classify random data at roughly the accuracy of 1/numLabels
	regressors:
		LinearRegression - have no error when the dataset all lies on one plane
	"""
	classifierAlgo = 'Custom.KNNClassifier'
	#assert that when whole dataset has the same label, crossValidated score 
	#reflects 100% accruacy (with a classifier)
	X, Y = _randomLabeledDataSet(numLabels=1)
	result = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
	assert result < 0.000001 #0 incorrect ever

	#assert that a random dataset will have accuracy roughly equal to 1/numLabels
	numLabelsList = [2,3,5]
	for curNumLabels in numLabelsList:
		X, Y = _randomLabeledDataSet(numPoints=50, numLabels=curNumLabels)
		result = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
		_assertClassifierErrorOnRandomDataPlausible(result, curNumLabels, tolerance=(1.0/curNumLabels))

	#assert that for an easy dataset (no noise, overdetermined linear hyperplane!), 
	#crossValidated error is perfect 
	#for all folds, with simple LinearRegression
	regressionAlgo = 'Custom.RidgeRegression'
	def linearFunc(points):
		return sum(points)
	#make random data set where all points lie on a linear hyperplane
	numFeats = 3
	numPoints = 50
	points = [[pythonRandom.gauss(0,1) for _x in xrange(numFeats)] for _y in xrange(numPoints)]
	labels = [[sum(featVector)] for featVector in points]
	X = createData('matrix', points)
	Y = createData('matrix', labels)
	
	#run in crossValidate
	result = crossValidate(regressionAlgo, X, Y, meanAbsoluteError, {}, numFolds=5)
	#assert error essentially zero since there's no noise
	assert result < .001 

@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
def test_crossValidateShuffleSeed():
	"""Assert that for a dataset, the same algorithm will generate the same model 
	(and have the same accuracy) when presented with identical random state (and
	therefore identical folds).
	Assert that the model is different when the random state is different
	"""
	numTrials = 5
	for _ in xrange(numTrials):
		X, Y = _randomLabeledDataSet(numPoints=50, numFeatures=10, numLabels=5)
		seed = UML.randomness.pythonRandom.randint(0, sys.maxint)
		UML.setRandomSeed(seed)
		resultOne = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, {}, numFolds=3)
		UML.setRandomSeed(seed)
		resultTwo = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, {}, numFolds=3)
		assert resultOne == resultTwo

	resultThree = crossValidate('Custom.KNNClassifier', X, Y, fractionIncorrect, {}, numFolds=3)
	#assert that models have diffeerent errors when different random state is available.
	#the idea being that different seeds create different folds
	#which create different models, which create different accuracies
	#for sufficiently large datasets.
	assert resultOne != resultThree


def main():
	test_crossValidate_XY_unchanged()
	test_crossValidate_runs()
	test_crossValidateShuffleSeed()
	test_crossValidate_reasonable_results()


if __name__ == '__main__':
	main()



