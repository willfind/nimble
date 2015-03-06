
import numpy
import math
from math import fabs

from nose.tools import *
from nose.plugins.attrib import attr

import UML

from UML import learnerType
from UML import createData

from UML.exceptions import ArgumentException, ImproperActionException

from UML.helpers import extractWinningPredictionLabel
from UML.helpers import generateAllPairs
from UML.helpers import findBestInterface
from UML.helpers import makeFoldIterator
from UML.helpers import sumAbsoluteDifference
from UML.helpers import generateClusteredPoints
from UML.helpers import trainAndTestOneVsOne
from UML.helpers import trainAndApplyOneVsOne
from UML.helpers import trainAndApplyOneVsAll
from UML.helpers import _mergeArguments
from UML.helpers import computeMetrics
from UML.calculate import rootMeanSquareError
from UML.calculate import meanAbsoluteError
from UML.calculate import fractionIncorrect
from UML.randomness import pythonRandom



##########
# TESTER #
##########

class FoldIteratorTester(object):
	def __init__(self, constructor):
		self.constructor = constructor

	@raises(ArgumentException)
	def test_makeFoldIterator_exceptionPEmpty(self):
		""" Test makeFoldIterator() for ArgumentException when object is point empty """
		data = [[],[]]
		data = numpy.array(data).T
		toTest = self.constructor(data)
		makeFoldIterator([toTest],2)

#	@raises(ImproperActionException)
#	def test_makeFoldIterator_exceptionFEmpty(self):
#		""" Test makeFoldIterator() for ImproperActionException when object is feature empty """
#		data = [[],[]]
#		data = numpy.array(data)
#		toTest = self.constructor(data)
#		makeFoldIterator([toTest],2)


	@raises(ArgumentException)
	def test_makeFoldIterator_exceptionTooManyFolds(self):
		""" Test makeFoldIterator() for exception when given too many folds """
		data = [[1],[2],[3],[4],[5]]
		names = ['col']
		toTest = self.constructor(data,names)
		makeFoldIterator([toTest, toTest],6)



	def test_makeFoldIterator_verifyPartitions(self):
		""" Test makeFoldIterator() yields the correct number folds and partitions the data """
		data = [[1],[2],[3],[4],[5]]
		names = ['col']
		toTest = self.constructor(data,names)
		folds = makeFoldIterator([toTest],2)

		[(fold1Train, fold1Test)] = folds.next()
		[(fold2Train, fold2Test)] = folds.next()

		try:
			folds.next()
			assert False
		except StopIteration:
			pass

		assert fold1Train.pointCount + fold1Test.pointCount == 5
		assert fold2Train.pointCount + fold2Test.pointCount == 5

		fold1Train.appendPoints(fold1Test)
		fold2Train.appendPoints(fold2Test)



	def test_makeFoldIterator_verifyMatchups(self):
		""" Test makeFoldIterator() maintains the correct pairings when given multiple data objects """
		data0 = [[1],[2],[3],[4],[5],[6],[7]]
		toTest0 = self.constructor(data0)

		data1 = [[1,1],[2,2],[3,3],[4,4],[5,5], [6,6], [7,7]]
		toTest1 = self.constructor(data1)
		
		data2 = [[-1],[-2],[-3],[-4],[-5],[-6],[-7]]
		toTest2 = self.constructor(data2)


		folds = makeFoldIterator([toTest0, toTest1, toTest2], 2)

		fold0 = folds.next()
		fold1 = folds.next()
		[(fold0Train0, fold0Test0), (fold0Train1, fold0Test1), (fold0Train2, fold0Test2)] = fold0
		[(fold1Train0, fold1Test0), (fold1Train1, fold1Test1), (fold1Train2, fold1Test2)] = fold1

		try:
			folds.next()
			assert False
		except StopIteration:
			pass

		# check that the partitions are the right size (ie, no overlap in training and testing)
		assert fold0Train0.pointCount + fold0Test0.pointCount == 7
		assert fold1Train0.pointCount + fold1Test0.pointCount == 7

		assert fold0Train1.pointCount + fold0Test1.pointCount == 7
		assert fold1Train1.pointCount + fold1Test1.pointCount == 7

		assert fold0Train2.pointCount + fold0Test2.pointCount == 7
		assert fold1Train2.pointCount + fold1Test2.pointCount == 7

		# check that the data is in the same order accross objects, within
		# the training or testing sets of a single fold
		for fold in [fold0, fold1]:
			trainList = []
			testList = []
			for (train, test) in fold:
				trainList.append(train)
				testList.append(test)

			for train in trainList:
				assert train.pointCount == trainList[0].pointCount
				for index in xrange(train.pointCount):
					assert fabs(train[index,0]) == fabs(trainList[0][index,0])

			for test in testList:
				assert test.pointCount == testList[0].pointCount
				for index in xrange(test.pointCount):
					assert fabs(test[index,0]) == fabs(testList[0][index,0])



class TestList(FoldIteratorTester):
	def __init__(self):
		def maker(data=None, featureNames=None):
			return UML.createData("List", data=data, featureNames=featureNames)

		super(TestList, self).__init__(maker)


class TestMatrix(FoldIteratorTester):
	def __init__(self):
		def maker(data, featureNames=None):
			return UML.createData("Matrix", data=data, featureNames=featureNames)

		super(TestMatrix, self).__init__(maker)


class TestSparse(FoldIteratorTester):
	def __init__(self):	
		def maker(data, featureNames=None):
			return UML.createData("Sparse", data=data, featureNames=featureNames)

		super(TestSparse, self).__init__(maker)

class TestRand(FoldIteratorTester):
	def __init__(self):	
		def maker(data, featureNames=None):
			possible = ['List', 'Matrix', 'Sparse']
			retType = possible[pythonRandom.randint(0, 2)]
			return UML.createData(retType=retType, data=data, featureNames=featureNames)

		super(TestRand, self).__init__(maker)


@attr('slow')
def testClassifyAlgorithms(printResultsDontThrow=False):
	"""tries the algorithm names (which are keys in knownAlgorithmToTypeHash) with learnerType().
	Next, compares the result to the algorithm's assocaited value in knownAlgorithmToTypeHash.
	If the algorithm types don't match, an AssertionError is thrown."""

	knownAlgorithmToTypeHash = {    'Custom.KNNClassifier':'classification', 
									'Custom.RidgeRegression':'regression',
								}
	try:
		findBestInterface('sciKitLearn')
		knownAlgorithmToTypeHash['sciKitLearn.RadiusNeighborsClassifier'] = 'classification'
		knownAlgorithmToTypeHash['sciKitLearn.RadiusNeighborsRegressor'] = 'regression'
	except ArgumentException:
		pass
	try:
		findBestInterface('mlpy')
		knownAlgorithmToTypeHash['mlpy.LDAC'] = 'classification'
		knownAlgorithmToTypeHash['mlpy.Ridge'] = 'regression'
	except ArgumentException:
		pass
	try:
		findBestInterface('shogun')
		knownAlgorithmToTypeHash['shogun.MulticlassOCAS'] = 'classification'
		knownAlgorithmToTypeHash['shogun.LibSVR'] = 'regression'
	except ArgumentException:
		pass

	for curAlgorithm in knownAlgorithmToTypeHash.keys():
		actualType = knownAlgorithmToTypeHash[curAlgorithm]
		predictedType = UML.learnerType(curAlgorithm)
		try:
			assert(actualType in predictedType)
		except AssertionError:
			errorString = 'Classification failure. Classified ' + curAlgorithm + ' as ' + predictedType + ', when it really is a ' + actualType
			if printResultsDontThrow:
				print errorString
			else:
				raise AssertionError(errorString)
		else:
			if printResultsDontThrow:
				print 'Passed test for ' + curAlgorithm

def testGenerateClusteredPoints():
	"""tests that the shape of data produced by generateClusteredPoints() is predictable and that the noisiness of the data
	matches that requested via the addFeatureNoise and addLabelNoise flags"""
	clusterCount = 3
	pointsPer = 10
	featuresPer = 5

	dataset, labelsObj, noiselessLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer, addFeatureNoise=True, addLabelNoise=True, addLabelColumn=True)
	pts, feats = noiselessLabels.pointCount, noiselessLabels.featureCount
	for i in xrange(pts):
		for j in xrange(feats):
			#assert that the labels don't have noise in noiselessLabels
			assert(noiselessLabels[i,j] % 1 == 0.0)

	pts, feats = dataset.pointCount, dataset.featureCount
	for i in xrange(pts):
		for j in xrange(feats):
			#assert dataset has noise for all entries
			assert(dataset[i,j] % 1 != 0.0)

	dataset, labelsObj, noiselessLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer, addFeatureNoise=False, addLabelNoise=False, addLabelColumn=True)
	pts, feats = noiselessLabels.pointCount, noiselessLabels.featureCount
	for i in xrange(pts):
		for j in xrange(feats):
			#assert that the labels don't have noise in noiselessLabels
			assert(noiselessLabels[i,j] % 1 == 0.0)

	pts, feats = dataset.pointCount, dataset.featureCount
	for i in xrange(pts):
		for j in xrange(feats):
			#assert dataset has no noise for all entries
			assert(dataset[i,j] % 1 == 0.0)

	#test that addLabelColumn flag works
	dataset, labelsObj, noiselessLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer, addFeatureNoise=False, addLabelNoise=False, addLabelColumn=False)
	labelColumnlessRows, labelColumnlessCols = dataset.pointCount, dataset.featureCount
	#columnLess should have one less column in the DATASET, rows should be the same
	assert(labelColumnlessCols - feats == -1)
	assert(labelColumnlessRows - pts == 0)


	#test that generated points have plausible values expected from the nature of generateClusteredPoints
	allNoiseDataset, labsObj, noiselessLabels = generateClusteredPoints(clusterCount, pointsPer, featuresPer, addFeatureNoise=True, addLabelNoise=True, addLabelColumn=True)
	pts, feats = allNoiseDataset.pointCount, allNoiseDataset.featureCount
	for curRow in xrange(pts):
		for curCol in xrange(feats):
			#assert dataset has no noise for all entries
			assert(allNoiseDataset[curRow,curCol] % 1 > 0.0000000001)
			
			currentClusterNumber = math.floor(curRow / pointsPer)
			expectedNoiselessValue = currentClusterNumber
			absoluteDifference = abs(allNoiseDataset[curRow,curCol] - expectedNoiselessValue)
			#assert that the noise is reasonable:
			assert(absoluteDifference < 0.01)

def testSumDifferenceFunction():
	""" Function verifies that for different shaped matricies, generated via createData, sumAbsoluteDifference() throws an ArgumentException."""

	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	data2 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2], [0,0,0,0]]
	matrix1 = createData('Matrix', data1)
	matrix2 = createData('Matrix', data2)
	failedShape = False
	try:
		result = sumAbsoluteDifference(matrix1, matrix2)
	except ArgumentException:
		failedShape = True
	assert(failedShape)

	data1 = [[0,0,1], [1,0,2], [0,1,3], [0,0,1], [1,0,2], [0,1,3], [0,0,1], [1,0,2], [0,1,3], [0,0,1], [1,0,2], [0,1,3], [0,0,1], [1,0,2], [0,1,3], [0,0,3], [1,0,1], [0,1,2]]
	data2 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	matrix1 = createData('Matrix', data1)
	matrix2 = createData('Matrix', data2)
	failedShape = False
	try:
		result = sumAbsoluteDifference(matrix1, matrix2)
	except ArgumentException:
		failedShape = True
	assert(failedShape)

	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	data2 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	matrix1 = createData('Matrix', data1)
	matrix2 = createData('Matrix', data2)
	failedShape = False
	try:
		result = sumAbsoluteDifference(matrix1, matrix2)
	except ArgumentException:
		failedShape = True
	assert(failedShape is False)

	#asserts differece function gets absolute difference correct (18 * 0.1 * 2)
	#18 rows
	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	data2 = [[1-0.1,0,0,1.1], [0-0.1,1,0,2.1], [0-0.1,0,1,3.1], [1-0.1,0,0,1.1], [0-0.1,1,0,2.1], [0-0.1,0,1,3.1], [1-0.1,0,0,1.1], [0-0.1,1,0,2.1], [0-0.1,0,1,3.1], [1-0.1,0,0,1.1], [0-0.1,1,0,2.1], [0-0.1,0,1,3.1], [1-0.1,0,0,1.1], [0-0.1,1,0,2.1], [0-0.1,0,1,3.1], [1-0.1,0,0,3.1], [0-0.1,1,0,1.1], [0-0.1,0,1,2.1]]
	matrix1 = createData('Matrix', data1)
	matrix2 = createData('Matrix', data2)
	diffResult = sumAbsoluteDifference(matrix1, matrix2)
	shouldBe = 18 * 0.1 * 2
	# 18 entries, discrepencies of 0.1 in the first column, and the last column
	discrepencyEffectivelyZero = (diffResult - shouldBe) < .000000001 and (diffResult - shouldBe) > -.0000000001
	if not discrepencyEffectivelyZero:
		raise AssertionError("difference result should be " + str(18 * 0.1 * 2) + ' but it is ' + str(diffResult))
	# assert(diffResult == 18 * 0.1 * 2)


def testtrainAndTestOneVsOne():
	variables = ["x1", "x2", "x3", "label"]
	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	data2 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [0,1,1,4], [0,1,1,4], [0,1,1,4], [0,1,1,4], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	trainObj1 = createData('Matrix', data=data1, featureNames=variables)
	trainObj2 = createData('Matrix', data=data2, featureNames=variables)

	testData1 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3]]
	testData2 = [[1, 0, 0, 1],[0, 1, 0, 2],[0, 0, 1, 3], [0, 1, 1, 2]]
	testObj1 = createData('Matrix', data=testData1)
	testObj2 = createData('Matrix', data=testData2)

	metricFunc = fractionIncorrect

	results1 = trainAndTestOneVsOne('Custom.KNNClassifier', trainObj1, trainY=3, testX=testObj1, testY=3, performanceFunction=metricFunc)
	results2 = trainAndTestOneVsOne('Custom.KNNClassifier', trainObj2, trainY=3, testX=testObj2, testY=3, performanceFunction=metricFunc)

	assert results1[0] == 0.0
	assert results2[0] == 0.25

def testtrainAndApplyOneVsAll():
	variables = ["x1", "x2", "x3", "label"]
	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	data2 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [0,1,1,4], [0,1,1,4], [0,1,1,4], [0,1,1,4], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	trainObj1 = createData('Sparse', data=data1, featureNames=variables)
	trainObj2 = createData('Sparse', data=data2, featureNames=variables)

	testData1 = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
	testData2 = [[1, 0, 0],[0, 1, 0],[0, 0, 1], [0, 1, 1]]
	testObj1 = createData('Sparse', data=testData1)
	testObj2 = createData('Sparse', data=testData2)

	results1 = trainAndApplyOneVsAll('Custom.KNNClassifier', trainObj1, trainY=3, testX=testObj1, scoreMode='label')
	results2 = trainAndApplyOneVsAll('Custom.KNNClassifier', trainObj1, trainY=3, testX=testObj1, scoreMode='bestScore')
	results3 = trainAndApplyOneVsAll('Custom.KNNClassifier', trainObj1, trainY=3, testX=testObj1, scoreMode='allScores')

	print "Results 1 output: " + str(results1.data)
	print "Results 2 output: " + str(results2.data)
	print "Results 3 output: " + str(results3.data)

	assert results1.copyAs(format="python list")[0][0] >= 0.0
	assert results1.copyAs(format="python list")[0][0] <= 3.0

	assert results2.copyAs(format="python list")[0][0] 

def testtrainAndApplyOneVsOne():
	variables = ["x1", "x2", "x3", "label"]
	data1 = [[1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1], [0,1,0,2], [0,0,1,3], [1,0,0,1],[0,1,0,2], [0,0,1,3], [1,0,0,3], [0,1,0,1], [0,0,1,2]]
	trainObj1 = createData('Matrix', data=data1, featureNames=variables)

	testData1 = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
	testObj1 = createData('Matrix', data=testData1)

#	metricFuncs = []
#	metricFuncs.append(fractionIncorrect)

	results1 = trainAndApplyOneVsOne('Custom.KNNClassifier', trainObj1, trainY=3, testX=testObj1, scoreMode='label')
	results2 = trainAndApplyOneVsOne('Custom.KNNClassifier', trainObj1, trainY=3, testX=testObj1, scoreMode='bestScore')
	results3 = trainAndApplyOneVsOne('Custom.KNNClassifier', trainObj1, trainY=3, testX=testObj1, scoreMode='allScores')

	assert results1.data[0][0] == 1.0
	assert results1.data[1][0] == 2.0
	assert results1.data[2][0] == 3.0
	assert len(results1.data) == 3

	assert results2.data[0][0] == 1.0
	assert results2.data[0][1] == 2
	assert results2.data[1][0] == 2.0
	assert results2.data[1][1] == 2
	assert results2.data[2][0] == 3.0
	assert results2.data[2][1] == 2

	results3FeatureMap = results3.featureNamesInverse
	for i in range(len(results3.data)):
		row = results3.data[i]
		for j in range(len(row)):
			score = row[j]
			# because our input data was matrix, we have to check feature names
			# as they would have been generated from float data
			if i == 0:
				if score == 2:
					assert results3FeatureMap[j] == str(float(1))
			elif i == 1:
				if score == 2:
					assert results3FeatureMap[j] == str(float(2))
			else:
				if score == 2:
					assert results3FeatureMap[j] == str(float(3))


@raises(ArgumentException)
def testMergeArgumentsException():
	""" Test helpers._mergeArguments will throw the exception it should """
	args = {1:'a', 2:'b', 3:'d'}
	kwargs = {1:1, 2:'b'}

	_mergeArguments(args, kwargs)


def testMergeArgumentsHand():
	""" Test helpers._mergeArguments is correct on hand construsted data """
	args = {1:'a', 2:'b', 3:'d'}
	kwargs = {1:'a', 4:'b'}

	ret = _mergeArguments(args, kwargs)

	assert ret == {1:'a', 2:'b', 3:'d', 4:'b'}



def test_computeMetrics_1d_2arg():
	knownLabels = numpy.array([[1.0],[2.0],[3.0]])
	predictedLabels = numpy.array([[1.0],[2.0],[3.0]])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	metricFunctions = rootMeanSquareError
	result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
	assert result == 0.0

	knownLabels = numpy.array([[1.5],[2.5],[3.5]])
	predictedLabels = numpy.array([[1.0],[2.0],[3.0]])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	metricFunctions = meanAbsoluteError
	result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
	assert result > 0.49
	assert result < 0.51

def test_computeMetrics_1d_labelsInData():
	training = numpy.array([[1.0, 5],[2.0, 27],[3.0, 42]])
	predictedLabels = numpy.array([[1.0],[2.0],[3.0]])

	trainingObj = createData('Matrix', data=training)
	predictedObj = createData('Matrix', data=predictedLabels)

	metricFunctions = rootMeanSquareError
	result = computeMetrics(0, trainingObj, predictedObj, metricFunctions)
	assert result == 0.0

	result = computeMetrics([0], trainingObj, predictedObj, metricFunctions)
	assert result == 0.0


# single val labels, three arg metric
def test_computeMetrics_1d_3arg():
	knownsData = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
	correctData = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
	wrongData = [[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]]

	knowns = createData('List', data=knownsData)
	correct = createData('List', data=correctData, featureNames=['0','1'])
	wrong = createData('List', data=wrongData, featureNames=['0','1'])

	metric = UML.calculate.fractionIncorrectBottom10

	correctScore = computeMetrics(knowns, None, correct, metric, negativeLabel=0)
	wrongScore = computeMetrics(knowns, None, wrong, metric, negativeLabel=0)

	assert correctScore == 0
	assert wrongScore == 1


# multi val, two arg metric
def test_computeMetrics_2d_2arg():
	knownLabels = numpy.array([[1.0, 1.0], [2.0, 2.0],[3.0, 3.0]])
	predictedLabels = numpy.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

	knownLabelsMatrix = createData('Matrix', data=knownLabels)
	predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

	metricFunctions = UML.calculate.meanFeaturewiseRootMeanSquareError
	result = computeMetrics(knownLabelsMatrix, None, predictedLabelsMatrix, metricFunctions)
	assert isinstance(result, float)
	assert result == 0.0

def test_computeMetrics_2d_labelsInData():
	training = numpy.array([[1.0, 5, 1.0],[2.0, 27, 2.0],[3.0, 42, 3.0]])
	predictedLabels = numpy.array([[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]])

	trainingObj = createData('Matrix', data=training)
	predictedObj = createData('Matrix', data=predictedLabels)

	metricFunctions = UML.calculate.meanFeaturewiseRootMeanSquareError
	result = computeMetrics([0,2], trainingObj, predictedObj, metricFunctions)
	assert result == 0.0


# TODO multi val, three arg metric. do we have one of those?

# single val, symetric two arg metric (similarity)
def test_computeMetrics_1d_2d_symmetric():	
	origData = numpy.array([[1.0],[2.0],[3.0]])
	outputData = numpy.array([[1.0],[2.0],[3.0]])

	origObj = createData('Matrix', data=origData)
	outObj = createData('Matrix', data=outputData)

	metricFunctions = UML.calculate.cosineSimilarity
	result = computeMetrics(origObj, None, outObj, metricFunctions)
	assert result == 1.0

# multi metrics should not be allowed
@raises(ArgumentException)
def test_cocmputeMetrics_multiple_metrics_disallowed():
	origData = numpy.array([[1.0],[2.0],[3.0]])
	outputData = numpy.array([[1.0],[2.0],[3.0]])

	origObj = createData('Matrix', data=origData)
	outObj = createData('Matrix', data=outputData)

	metricFunctions = [UML.calculate.cosineSimilarity, rootMeanSquareError]
	computeMetrics(origObj, None, outObj, metricFunctions)



def testExtractWinningPredictionLabel():
	"""
	Unit test for extractWinningPrediction function in runner.py
	"""
	predictionData = [[1, 3, 3, 2, 3, 2], [2, 3, 3, 2, 2, 2], [1, 1, 1, 1, 1, 1], [4, 4, 4, 3, 3, 3]]
	BaseObj = createData('Matrix', predictionData)
	BaseObj.transpose()
	predictions = BaseObj.applyToFeatures(extractWinningPredictionLabel, inPlace=False)
	listPredictions = predictions.copyAs(format="python list")
	
	assert listPredictions[0][0] - 3 == 0.0
	assert listPredictions[0][1] - 2 == 0.0
	assert listPredictions[0][2] - 1 == 0.0
	assert (listPredictions[0][3] - 4 == 0.0) or (listPredictions[0][3] - 3 == 0.0)



def testGenerateAllPairs():
	"""
	Unit test function for testGenerateAllPairs
	"""
	testList1 = [1, 2, 3, 4]
	testPairs = generateAllPairs(testList1)
	print testPairs

	assert len(testPairs) == 6
	assert ((1, 2) in testPairs) or ((2, 1) in testPairs)
	assert not (((1, 2) in testPairs) and ((2, 1) in testPairs))
	assert ((1, 3) in testPairs) or ((3, 1) in testPairs)
	assert not (((1, 3) in testPairs) and ((3, 1) in testPairs))
	assert ((1, 4) in testPairs) or ((4, 1) in testPairs)
	assert not (((1, 4) in testPairs) and ((4, 1) in testPairs))
	assert ((2, 3) in testPairs) or ((3, 2) in testPairs)
	assert not (((2, 3) in testPairs) and ((3, 2) in testPairs))
	assert ((2, 4) in testPairs) or ((4, 2) in testPairs)
	assert not (((2, 4) in testPairs) and ((4, 2) in testPairs))
	assert ((3, 4) in testPairs) or ((4, 3) in testPairs)
	assert not (((3, 4) in testPairs) and ((4, 3) in testPairs))

	testList2 = []
	testPairs2 = generateAllPairs(testList2)
	assert testPairs2 is None
