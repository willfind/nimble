#test scripts for
# crossValidate



#so you can run as main:
import sys
sys.path.append('../..')

from UML import crossValidate
from UML import createData
from UML.metrics import *
import random
from pdb import set_trace as ttt


def _randomLabeledDataSet(dataType='matrix', numPoints=100, numFeatures=5, numLabels=3, seed=None):
	"""returns a tuple of two data objects of type dataType
	the first object in the tuple contains the feature information ('X' in UML language)
	the second object in the tuple contains the labels for each feature ('Y' in UML language)
	"""
	random.seed(seed)
	if numLabels is None:
		labelsRaw = [[random.random()] for _x in xrange(numPoints)]
	else: #labels data set
		labelsRaw = [[int(random.random()*numLabels)] for _x in xrange(numPoints)]

	rawFeatures = [[random.random() for _x in xrange(numFeatures)] for _y in xrange(numPoints)]

	return (createData(dataType, rawFeatures), createData(dataType, labelsRaw))

def test_crossValidate_XY_unchanged():
	"""assert that after running cross validate on datasets passed to 
	X and Y, the original data is unchanged"""
	classifierAlgo = 'sciKitLearn.KNeighborsClassifier'
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
	numPointsInSet = 100
	#todo add other data types - currently crashes in sklearn interface for list and sparse
	for dType in ['matrix',]:
		X, Y = _randomLabeledDataSet(numPoints=numPointsInSet, numLabels=numLabelsInSet, dataType=dType)	
		classifierAlgos = ['sciKitLearn.KNeighborsClassifier', 'sciKitLearn.PassiveAggressiveClassifier']
		for curAlgo in classifierAlgos:
			result = crossValidate(curAlgo, X, Y, fractionIncorrect, {}, numFolds=3, foldSeed=random.random())
			assert isinstance(result, float)


		#With regression dataset (no repeated labels)
		X, Y = _randomLabeledDataSet(numLabels=None, dataType=dType)	
		classifierAlgos = ['sciKitLearn.LinearRegression', ]
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
	"""Assert that crossValidate returns reasonable erros for known algorithms
	on cooked data sets:
	crossValidate should do the following:
	classifiers:
		have no error when there is only one label in the data set
		classify random data at roughly the accuracy of 1/numLabels
	regressors:
		LinearRegression - have no error when the dataset all lies on one plane
	"""
	classifierAlgo = 'sciKitLearn.KNeighborsClassifier'
	#assert that when whole dataset has the same label, crossValidated score 
	#reflects 100% accruacy (with a classifier)
	X, Y = _randomLabeledDataSet(numLabels=1)
	result = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
	assert result < 0.000001 #0 incorrect ever

	#assert that a radom dataset will have accuracy roughly equal to 1/numLabels
	numLabelsList = [2,3,5]
	for curNumLabels in numLabelsList:
		X, Y = _randomLabeledDataSet(numPoints=1000, numLabels=curNumLabels)
		result = crossValidate(classifierAlgo, X, Y, fractionIncorrect, {}, numFolds=5)
		_assertClassifierErrorOnRandomDataPlausible(result, curNumLabels, tolerance=.1)

	#assert that for an easy dataset (no noise, overdetermined linear hyperplane!), 
	#crossValidated error is perfect 
	#for all folds, with simple LinearRegression
	regressionAlgo = 'sciKitLearn.LinearRegression'
	def linearFunc(points):
		return sum(points)
	#make random data set where all points lie on a linear hyperplane
	numFeats = 3
	numPoints = 1000
	points = [[random.gauss(0,1) for _x in xrange(numFeats)] for _y in xrange(numPoints)]
	labels = [[sum(featVector)] for featVector in points]
	X = createData('matrix', points)
	Y = createData('matrix', labels)
	
	#run in crossValidate
	result = crossValidate(regressionAlgo, X, Y, meanAbsoluteError, {}, numFolds=5)
	#assert error essentially zero since there's no noise
	assert result < .001 


def test_crossValidateShuffleSeed():
	"""Assert that for a dataset, the same algorithm will generate the same model 
	(and have the same accuracy) when the foldSeed is identical.
	Assert that the model is different when the foldSeed is different
	"""
	numTrials = 5
	for _ in xrange(numTrials):
		X, Y = _randomLabeledDataSet(numPoints=1000, numFeatures=10, numLabels=5)
		theSeed = 'theseed'
		resultOne = crossValidate('sciKitLearn.KNeighborsClassifier', X, Y, fractionIncorrect, {}, numFolds=3, foldSeed=theSeed)
		resultTwo = crossValidate('sciKitLearn.KNeighborsClassifier', X, Y, fractionIncorrect, {}, numFolds=3, foldSeed=theSeed)
		assert resultOne == resultTwo
	newSeed = 'newseed'
	resultThree = crossValidate('sciKitLearn.KNeighborsClassifier', X, Y, fractionIncorrect, {}, numFolds=3, foldSeed=newSeed)
	#assert that models have diffeerent errors when different seeds are used.
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



