"""
Select the best questions to keep in a survey from a larger set of questions
"""
import numpy
import sys
import random
import copy
import math
import pylab
import bisect

from allowImports import boilerplate
boilerplate()

import os.path
import UML
from UML import trainAndTest
from UML import trainAndApply
from UML import train
from UML import createData
from UML.calculate import fractionIncorrect
from UML import calculate

def rSquared(knownValues, predictedValues):
	return 1.0 - varianceFractionRemaining(knownValues, predictedValues)

def varianceFractionRemaining(knownValues, predictedValues):
	if knownValues.pointCount != predictedValues.pointCount: raise Exception("Objects had different numbers of points")
	if knownValues.featureCount != predictedValues.featureCount: raise Exception("Objects had different numbers of features. Known values had " + str(knownValues.featureCount) + " and predicted values had " + str(predictedValues.featureCount))
	diffObject = predictedValues - knownValues
	rawDiff = diffObject.copyAs("numpy array")
	rawKnowns = knownValues.copyAs("numpy array")
	return numpy.var(rawDiff)/float(numpy.var(rawKnowns))


def buildTrainingAndTestingSetsForPredictions(data, fractionOfDataForTesting, featuresToPredict, functionsToExcludePoints):
	"""creates  the training and testing sets for each label we're going to predict"""
	trainXs = []
	trainYs = []
	testXs = []
	testYs = []

	#make our training and testing sets
	#for each of the different labels we want to predict
	for labelNumber in xrange(len(featuresToPredict)):
		#create the features for the current labelNum, and exclude irrelevant poins
		currentFeatures = data.copy()
		currentFeatures.extractPoints(functionsToExcludePoints[labelNumber]) #get rid of points that aren't relevant to this label
		
		#remove all the different labels from the features
		labelsMatrix = currentFeatures.extractFeatures(featuresToPredict)

		#get the labels we'll be predicting 
		featureToPredict = featuresToPredict[labelNumber]
		currentLabels = labelsMatrix.copyFeatures(featureToPredict)

		#add just those labels we'll be predicting to the features to form a combined set
		featuresWithLabels = currentFeatures.copy()
		featuresWithLabels.appendFeatures(currentLabels)
		labelFeatureNum = featuresWithLabels.featureCount-1

		#get the training and testing data for this label
		trainX, trainY, testX, testY = featuresWithLabels.trainAndTestSets(testFraction=fractionOfDataForTesting, labels=labelFeatureNum)

		assert not(featureToPredict in trainX.getFeatureNames())
		trainXs.append(trainX)
		trainYs.append(trainY)
		testXs.append(testX)
		testYs.append(testY)

	#confirm the training X sets all have the same number of features (even though they may not have the same number of points)
	for trainX, trainY in zip(trainXs, trainYs):
		assert trainX.featureCount == trainXs[0].featureCount
		assert trainY.pointCount == trainX.pointCount
		assert trainY.featureCount == 1

	return trainXs, trainYs, testXs, testYs


def testBuildTrainingAndTestingSetsForPredictions():
	data = [["x1", "x2", "x3", "y1", "x4", "y2"],[1,5,2,3,7,1],[2,2,3.2,5,9.1,-7],[3,5,2,1,3,9],[4,9.2,3,5,5,1], [5,-4,2,1,1,0], [6,-2,-3,-1,-2,-3]]
	data = createData("Matrix", data, featureNames=True)
	fractionOfDataForTesting = 1.0/3.0
	featuresToPredict = ["y1", "y2"]
	functionsToExcludePoints = [lambda r: r["x2"]<3, lambda r: False]
	trainXs, trainYs, testXs, testYs = buildTrainingAndTestingSetsForPredictions(data, fractionOfDataForTesting, featuresToPredict, functionsToExcludePoints)
	assert(len(trainXs)) == 2
	assert trainXs[0].featureCount == 4
	assert trainXs[1].featureCount == 4
	assert trainXs[0].getFeatureNames() == ["x1","x2","x3","x4"]
	assert trainXs[1].getFeatureNames() == ["x1","x2","x3","x4"]
	assert trainXs[0].pointCount == 2
	assert testXs[0].pointCount == 1
	assert trainXs[1].pointCount == 4
	assert testXs[1].pointCount == 2
	jointXs0 = trainXs[0].copy()
	jointXs0.appendPoints(testXs[0])
	jointXs0.sortPoints("x1")
	print "jointXs0\n", jointXs0
	assert jointXs0.isApproximatelyEqual(createData("Matrix", [[1,5,2,7],[3,5,2,3],[4,9.2,3,5]]))

	jointYs0 = trainYs[0].copy()
	jointYs0.appendPoints(testYs[0])
	jointYs0.sortPoints(0)

	print "jointYs0\n", jointYs0
	assert jointYs0.isApproximatelyEqual(createData("Matrix", [[1],[3],[5]]))

	jointXs1 = trainXs[1].copy()
	jointXs1.appendPoints(testXs[1])
	jointXs1.sortPoints("x1")
	assert jointXs1.isApproximatelyEqual(createData("Matrix", [[1,5,2,7],[2,2,3.2,9.1],[3,5,2,3],[4,9.2,3,5], [5,-4,2,1], [6,-2,-3,-2]]))

	jointYs1 = trainYs[1].copy()
	jointYs1.appendPoints(testYs[1])
	jointYs1.sortPoints(0)

	print "jointYs1\n", jointYs1
	assert jointYs1.isApproximatelyEqual(createData("Matrix", [[-7],[-3],[0],[1],[1],[9]]))


def reduceDataToBestFeatures(trainXs, trainYs, testXs, testYs, numFeaturesToKeep, predictionAlgorithms, featuresToPredict, plot=False):
	"tries dropping one feature at a time from all datasets, and kill off the feature that is most useless until we have the right number"""
	assert isinstance(trainXs, list)
	assert isinstance(trainYs, list)
	assert isinstance(testXs, list)
	assert isinstance(testYs, list)
	assert len(trainXs) == len(trainYs) and len(trainXs) == len(testXs) and len(trainXs) == len(testYs)

	#prevent us from modifying the original objects
	for i in xrange(len(trainXs)):
		trainXs[i] = trainXs[i].copy()
		trainYs[i] = trainYs[i].copy()
		testXs[i] = testXs[i].copy()
		testYs[i] = testYs[i].copy()

	if numFeaturesToKeep > trainXs[0].featureCount: raise Exception("Cannot keep " + str(numFeaturesToKeep) + " features since the data has only " + str(trainXs[0].featureCount) + " features.")

	droppedFeatureErrorsListOutSample = []
	droppedFeatureErrorsListInSample = []

	while trainXs[0].featureCount > numFeaturesToKeep:
		print str(trainXs[0].featureCount) + " features left"

		errorForEachFeatureDropped = []

		#try dropping each feature one by one 
		for featureNumToDrop in xrange(trainXs[0].featureCount):
			#sys.stdout.write(" " + str(trainXs[0].getFeatureNames()[featureNumToDrop]))
			errorsForThisFeatureDrop = []
			#for each label we're predicting
			for labelNum, trainX, trainY in zip(range(len(trainXs)), trainXs, trainYs):
				#print "got here: " + str(featureNumToDrop) + ", " + str(labelNum)
				#print "trainX", trainX
				#build a feature set to train on that doesn't include the feature we're dropping
				trainXWithoutFeature = trainX.copy()
				trainXWithoutFeature.extractFeatures(featureNumToDrop)
				
				algorithmName = predictionAlgorithms[labelNum]
				if "Logistic" in algorithmName: 
					#C = tuple([10.0**k for k in xrange(-6, 6)])
					C = 10**9 #large means no regularization #a large value for C results in less regularization
					error = trainAndTest(algorithmName, trainXWithoutFeature, trainY, testX=trainXWithoutFeature, testY=trainY, performanceFunction=fractionIncorrect, C=C)
				elif "Ridge" in algorithmName:
					#alpha = tuple([10.0**k for k in xrange(-6, 6)])
					alpha = 0 #0 means no regularization, large means heavy regularization
					error = trainAndTest(algorithmName, trainXWithoutFeature, trainY, testX=trainXWithoutFeature, testY=trainY, performanceFunction=varianceFractionRemaining, alpha=alpha)
				else:
					raise Exception("Don't know how to set parameters for algorithm: " + str(algorithmName))

				errorsForThisFeatureDrop.append(error)

			combinedErrorForFeatureDrop = numpy.mean(errorsForThisFeatureDrop)
			errorForEachFeatureDropped.append((combinedErrorForFeatureDrop, featureNumToDrop))

		errorForEachFeatureDropped.sort(lambda x,y: cmp(y[0],x[0])) #sort descending by error so that the last element corresponds to the most useless feature (i.e. the feature where we have the lowest error without it)
		mostUselessFeatureErrorInSample = errorForEachFeatureDropped[-1][0]
		droppedFeatureErrorsListInSample.append(mostUselessFeatureErrorInSample)
		mostUselessFeatureNum = errorForEachFeatureDropped[-1][1]
		#print "\nRemoving feature " + str(trainX.getFeatureNames()[mostUselessFeatureNum]) + " with combined error " + str(round(errorForEachFeatureDropped[-1][0],3))
		for trainX, testX in zip(trainXs, testXs):
			trainX.extractFeatures(mostUselessFeatureNum)
			testX.extractFeatures(mostUselessFeatureNum)
		outSampleErrorsHash, outSampleParamshash = getPredictionErrors(trainXs, trainYs, testXs, testYs, predictionAlgorithms, featuresToPredict)
		mostUselessFeatureErrorOutSample = outSampleErrorsHash["Combined Error"]
		droppedFeatureErrorsListOutSample.append(mostUselessFeatureErrorOutSample)
		#print "viableFeaturesLeft", trainXs[0].featureCount

	if plot:
		pylab.plot(droppedFeatureErrorsListOutSample, ".", color="green")
		pylab.plot(droppedFeatureErrorsListInSample, ".", color="blue")
		pylab.legend(["Out Sample", "In Sample"], loc="best")
		pylab.title("Combined Error With Different Numbers of Features Removed")
		pylab.xlabel("Features removed")
		pylab.ylabel("Combined Error")
		#pylab.show()

	return trainXs, trainYs, testXs, testYs


def getPredictionErrors(trainXs, trainYs, testXs, testYs, predictionAlgorithms, featuresToPredict):
	errorsHash = {}
	parametersHash = {}
	#firstTrainX = trainXs[0]
	#now test the models out of sample on our final feature sets!
	for labelNum, trainX, trainY, testX, testY in zip(range(len(trainXs)), trainXs, trainYs, testXs, testYs):
		algorithmName = predictionAlgorithms[labelNum]
		featureToPredict = featuresToPredict[labelNum]
		if "Logistic" in algorithmName: 
			#C = tuple([10.0**k for k in xrange(-6, 6)])
			C = 10**9
			#error = trainAndTest(algorithmName, trainX, trainY, testX=testX, testY=testY, performanceFunction=fractionIncorrect, C=C)
			learner = UML.train(algorithmName, trainX, trainY, C=C)
			error = learner.test(testX=testX, testY=testY, performanceFunction=fractionIncorrect)
			backend = learner.backend
			parametersHash[featureToPredict] = {"intercept":backend.intercept_, "coefs":backend.coef_}
		elif "Ridge" in algorithmName:
			#alpha = tuple([10.0**k for k in xrange(-6, 6)])
			alpha = 0
			#accuracy = trainAndTest(algorithmName, trainX, trainY, testX=testX, testY=testY, performanceFunction=varianceFractionRemaining, alpha=alpha)
			learner = UML.train(algorithmName, trainX, trainY, alpha=alpha)
			error = learner.test(testX=testX, testY=testY, performanceFunction=varianceFractionRemaining)
			backend = learner.backend
			parametersHash[featureToPredict] = {"intercept":backend.intercept_, "coefs":backend.coef_}
		else:
			raise Exception("Don't know how to set parameters for algorithm: " + str(algorithmName))
		errorsHash[featureToPredict] = error #record the accuracy

	combinedError = numpy.mean(errorsHash.values())
	errorsHash["Combined Error"] = combinedError


	return errorsHash, parametersHash



def getBestFeaturesAndErrors(trainXs, trainYs, testXs, testYs, numFeaturesToKeep, predictionAlgorithms, featuresToPredict, plot=False):
	#prevent us from modifying the original objects
	#for i in xrange(len(trainXs)):
	#	trainXs[i] = trainXs[i].copy()
	#	trainYs[i] = trainYs[i].copy()
	#	testXs[i] = testXs[i].copy()
	#	testYs[i] = testYs[i].copy()
	trainXs = copy.copy(trainXs)
	trainYs = copy.copy(trainYs)
	testXs = copy.copy(testXs)
	testYs = copy.copy(testYs)

	trainXs, trainYs, testXs, testYs = reduceDataToBestFeatures(trainXs, trainYs, testXs, testYs, numFeaturesToKeep=numFeaturesToKeep, predictionAlgorithms=predictionAlgorithms, featuresToPredict=featuresToPredict, plot=plot)
	errorsHash, parametersHash = getPredictionErrors(trainXs, trainYs, testXs, testYs, predictionAlgorithms=predictionAlgorithms, featuresToPredict=featuresToPredict)

	bestFeatures = trainXs[0].getFeatureNames()
	return bestFeatures, errorsHash, parametersHash


def getBestFeaturesAndErrorsPurelyRandomlyManyTimes(trials, trainXs, trainYs, testXs, testYs, numFeaturesToKeep, predictionAlgorithms, featuresToPredict):
	random.seed(5)
	combinedErrors = {}
	for trialNum in xrange(trials):
		trainXsTemp = copy.copy(trainXs)
		trainYsTemp = copy.copy(trainYs)
		testXsTemp = copy.copy(testXs)
		testYsTemp = copy.copy(testYs)
		numFeatures = trainXs[0].featureCount
		featuresToKeep = random.sample(range(numFeatures), numFeaturesToKeep)
		#print "featuresToKeep", featuresToKeep
		for datasetNum in xrange(len(trainXsTemp)):
			trainXsTemp[datasetNum] = trainXsTemp[datasetNum].copyFeatures(featuresToKeep)
			testXsTemp[datasetNum] = testXsTemp[datasetNum].copyFeatures(featuresToKeep)

		errorsHash, parametersHash = getPredictionErrors(trainXs=trainXsTemp, trainYs=trainYsTemp, testXs=testXsTemp, testYs=testYsTemp, predictionAlgorithms=predictionAlgorithms, featuresToPredict=featuresToPredict)
		for key, error in errorsHash.iteritems():
			if not key in combinedErrors:
				combinedErrors[key] = []
			combinedErrors[key].append(error)

	return combinedErrors


def toListOfLists(matrix):
	output = []
	for row in xrange(len(matrix)):
		rowVals = []
		for col in xrange(len(matrix.T)):
			rowVals.append(matrix[row,col])
		output.append(rowVals)
	return output

def testToListOfLists():
	print "toListOfLists(numpy.matrix([[1,2],[4,2]]))", toListOfLists(numpy.matrix([[1,2],[4,2]]))
	assert toListOfLists(numpy.matrix([[1,2],[4,2]])) == [[1,2],[4,2]]

def testgetBestFeaturesAndErrors():
	#data = [["x0", "x1", "x2", "x3", "p1", "p2", "x6", "x7"],[1,5,2,3,7,1,3,9],[2,2,3.2,5,9.1,-7,2,7],[3,5,2,1,3,9,4,8],[4,9.2,3,5,5,1,8,1],[5,-4,2,1,1,0,6,3], [6,-2,-3,-1,-2,-3,-3,-2],[7,1,4,18,5,6,1,4],[8,2,3,4,5,6,3,2],[9,1,12,10,5,2,8,5],[10,8,3,2,1,5,4,1], [11,5,2,10,4,2,8,1], [12,6,2,14,4,5.3,2,-1], [13,2,2,12,4,5.4,3,-1], [14,6,2,15,4,5.3,3,-1], [15,2,1,13,2,0.3,-3,4], [16,6,3,4,4,-1,3,0], [17,1,4,20,3,-6,-2,1], [18,4,3,2,5,8,9,3]]
	seed = 4
	numpy.random.seed(seed)
	random.seed(seed)
	points = 200
	featuresToPredict = ["p1", "p2"]
	featureNames = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
	featureNames.extend(featuresToPredict)
	features = len(featureNames)
	data = toListOfLists(numpy.random.normal(0,1,(points, features)))
	#setup values for the label
	noiseLevel = 0.01
	feature0Mean = numpy.mean(numpy.matrix(data)[:, 0])
	for row in data:
		#set p1
		row[-2] = 10*row[0] + row[1] + 100*row[3] + noiseLevel*numpy.random.normal(0,1) 
		#set p2
		#if row[0] > row[3] + noiseLevel*numpy.random.uniform(0,1): row[5] = 1
		if row[3] + noiseLevel*numpy.random.normal(0,1) > feature0Mean: row[-1] = 1
		else: row[-1] = 0

	data.insert(0, featureNames) #put the column headers at the top of the data
	data = createData("Matrix", data, featureNames=True)
	fractionOfDataForTesting = 1.0/3.0
	
	print "data\n", data

	functionsToExcludePoints = [lambda r: False, lambda r: False]
	predictionAlgorithms = ["SciKitLearn.Ridge",  "SciKitLearn.LogisticRegression"]
	
	numFeaturesToKeep = 3

	trainXs, trainYs, testXs, testYs = buildTrainingAndTestingSetsForPredictions(data.copy(), fractionOfDataForTesting=fractionOfDataForTesting, featuresToPredict=featuresToPredict, functionsToExcludePoints=functionsToExcludePoints)
	bestFeatures, errorsHash, parametersHash = getBestFeaturesAndErrors(trainXs, trainYs, testXs, testYs, numFeaturesToKeep, predictionAlgorithms, featuresToPredict)
	
	print "errorsHash:", errorsHash
	print "parametersHash", parametersHash
	print "bestFeatures", bestFeatures
	if not ("x3" in bestFeatures): raise Exception("x3 was the best feature by far by was removed!")
	assert bestFeatures == ["x0", "x1", "x3"]

	assert errorsHash["p1"] < 0.02
	assert errorsHash["p2"] < 0.02

	numFeaturesToKeep = 2
	bestFeatures, errorsHash, parametersHash = getBestFeaturesAndErrors(trainXs, trainYs, testXs, testYs, numFeaturesToKeep, predictionAlgorithms, featuresToPredict)
	assert bestFeatures == ["x0", "x3"]

	numFeaturesToKeep = 1
	bestFeatures, errorsHash, parametersHash = getBestFeaturesAndErrors(trainXs, trainYs, testXs, testYs, numFeaturesToKeep, predictionAlgorithms, featuresToPredict)
	assert bestFeatures == ["x3"]

def printParameters(name, parametersHash):
	print "\n" + name
	for key, value in parametersHash.iteritems():
		if isinstance(value, dict):
			print "\t" + str(key).rjust(12)
			for lowerKey, lowerValue in value.iteritems():
				print "\t\t" + str(lowerKey).rjust(12) + ": " + str(lowerValue)
		else:
			print "\t" + str(key).rjust(12) + ": " + str(value)



def range95String(values, digits=2):
	sortedValues = list(sorted(values)) #sort asscending
	lowerPosition = int(round(0.05*len(sortedValues)))
	upperPosition = int(round(0.95*len(sortedValues)))
	lowerValue = sortedValues[lowerPosition]
	upperValue = sortedValues[upperPosition]
	strOut = str(round(lowerValue, digits)) + " to " + str(round(upperValue, digits)) + " 95% Range"
	return strOut

def confidenceInterval95String(values, digits=2):
	mean = numpy.mean(values)
	stddev = numpy.std(values)
	stderr = stddev/math.sqrt(len(values))
	lowerValue = mean - 1.96*stderr
	upperValue = mean + 1.96*stderr
	strOut = str(round(lowerValue, digits)) + " to " + str(round(upperValue, digits)) + " CI 95%"
	return strOut

def percentileOf(value, inValues):
	sortedValues = list(sorted(inValues)) #sort asscending
	pos = bisect.bisect_left(sortedValues, value)
	percentile = pos / float(len(sortedValues))
	return percentile

def testPercentileOf():
	assert percentileOf(-1, [0,2]) == 0
	assert percentileOf(3, [0,2]) == 1
	assert percentileOf(1, [0,2]) == 0.50


"""
def plotFeatureDistributionForDifferentFilters(X, featureName, filterFunc):
	pointsFiltered = X.extractPoints(filterFunc)
	pointsUnfiltered = X
	featureFiltered = pointsFiltered.copyFeatures(featureName)
	featureUnfiltered = pointsUnfiltered.copyFeatures(featureName)
	featureFiltered.show() WAS WORKING HERE
	pylab.hist(featureFiltered.data, color="blue")
	pylab.hist(featureUnfiltered.data, color="green")
	pylab.legend(["Filtered", "Unfiltered"], loc="best")
"""

def applyPointFilterThenPlotDistribution(X, featureName, filterFunc):
	Z = X.copy()
	pointsToUse = Z.extractPoints(filterFunc)
	pointsToUse.plotFeatureDistribution(featureName)


def trainAndTestClassificationErrorUsingOneVariable(data, variableToUse, labelID, fractionOfDataForTesting):
	trainX, trainY, testX, testY = data.trainAndTestSets(testFraction=fractionOfDataForTesting, labels=labelID)
	
	#just use this one variable in our predictions
	trainX = trainX.copyFeatures(variableToUse)
	testX = testX.copyFeatures(variableToUse)

	### use logistic regression using this one variable
	algorithmName = "SciKitLearn.LogisticRegression"
	C = 10**9
	learner = UML.train(algorithmName, trainX, trainY, C=C)
	trainErr = learner.test(testX=trainX, testY=trainY, performanceFunction=fractionIncorrect)
	testErr = learner.test(testX=testX, testY=testY, performanceFunction=fractionIncorrect)
	#backend = learner.backend
	#parameters = {"intercept":backend.intercept_, "coefs":backend.coef_}
	return trainErr, testErr


def plotStandardDeviationDistributionsForLiarsAndHonest(allFeatures):
	featuresForStats = allFeatures.copy()
	liarsForStats = featuresForStats.copy().extractPoints(lambda x: x["inLyingGroup"] == 1)
	honestForStats = featuresForStats.copy().extractPoints(lambda x: x["inLyingGroup"] == 0)

	#featuresForStats.show()
	featuresForStats.extractFeatures(featuresToRemoveCompletely)
	featuresForStats.extractFeatures(featuresToPredict)
	liarsForStats.extractFeatures(featuresToRemoveCompletely)
	liarsForStats.extractFeatures(featuresToPredict)
	honestForStats.extractFeatures(featuresToRemoveCompletely)
	honestForStats.extractFeatures(featuresToPredict)

	honestForStats.pointStatistics("standarddeviation").plotFeatureDistribution(0)
	liarsForStats.pointStatistics("standarddeviation").plotFeatureDistribution(0)

	featuresForStats = liarsForStats

	#print "point means"
	pointMeans = featuresForStats.pointStatistics("mean")
	#print "lyers"
	#featuresForStats.show()
	#print "means"
	print "minMean", pointMeans.featureStatistics("min")
	print "maxMean", pointMeans.featureStatistics("max")
	#pointMeans.show()
	pointStdDevs = featuresForStats.pointStatistics("standarddeviation")
	print "meanStdDev", pointStdDevs.featureStatistics("mean")

	#pointStdDevs.show()
	#stats = pointMeans.copy()
	#stats.appendFeatures(pointStdDevs)
	#stats.plotFeatureCross(0,1)

#def pointsToZScores(data):
#	means = data.pointStatistics("mean")
#	stddevs = data.pointStatistics("standarddeviation")
#	return (data - means) / stddevs #this doesn't work because of size mismatch

def pointsToZScores(data):
	means = data.pointStatistics("mean")
	stddevs = data.pointStatistics("standarddeviation")
	#identity = UML.createData("Matrix", numpy.ones(data.featureCount))
	rawIdentity = numpy.atleast_2d(numpy.ones(data.featureCount))
	onesPoint = UML.createData("Matrix", rawIdentity)
	means *= onesPoint
	stddevs *= onesPoint
	data.show("data")
	return (data - means) / stddevs 


if __name__ == "__main__":
	import time
	fileName = "/Users/spencer2/Dropbox/Spencer/Work/Personality Matrix/public domain datasets/Selected personality data from SAPA/sapaTempData696items08dec2013thru26jul2014.csv"
	
	print "Loading data..."
	startTime = time.time()
	data = createData("Matrix", fileName, featureNames=True)
	print "Loaded data in " + str(round(time.time()-startTime)) + " seconds."



	"""

	allFeatures.extractFeatures(featuresToRemoveCompletely)

	#debugging only
	pointsToZScores(allFeatures)

	trainXs, trainYs, testXs, testYs = buildTrainingAndTestingSetsForPredictions(allFeatures, fractionOfDataForTesting=fractionOfDataForTesting, featuresToPredict=featuresToPredict, functionsToExcludePoints=functionsToExcludePoints)
	
	trainErr, testErr = trainAndTestClassificationErrorUsingOneVariable(allFeatures, variableToUse="TotalAcademicScore", labelID="inLyingGroup", fractionOfDataForTesting=fractionOfDataForTesting)
	print "Using just total academic score to predict lying"
	print "\ttrain error:", trainErr
	print "\ttest error:", testErr

	#find the best features
	doPlot = False
	bestFeatures, errorsHash, parametersHash = getBestFeaturesAndErrors(trainXs=trainXs, trainYs=trainYs, testXs=testXs, testYs=testYs, numFeaturesToKeep=numFeaturesToKeep, predictionAlgorithms=predictionAlgorithms, featuresToPredict=featuresToPredict, plot=doPlot)

	#for comparison purposes, pick the same number of features at random many times to see what performance is like
	print "\nDoing trials of random sets of " + str(numFeaturesToKeep) + " fts"
	trials = 500
	combinedErrorsHash = getBestFeaturesAndErrorsPurelyRandomlyManyTimes(trials, trainXs, trainYs, testXs, testYs, numFeaturesToKeep=numFeaturesToKeep, predictionAlgorithms=predictionAlgorithms, featuresToPredict=featuresToPredict)

	print "\n\n"
	print "Best features: " + str(bestFeatures)
	printParameters("Error", errorsHash)
	#printParameters("Random Errors", combinedErrorsHash)

	print "\n\nRandom Errors using " + str(numFeaturesToKeep) + " fts"
	for key, errors in combinedErrorsHash.iteritems():
		print "\t" + str(key).rjust(12)
		#print "\t\tmean: " + str(round(numpy.mean(errors),2))
		print "\t\tpercentile: " + str(int(round(100*percentileOf(errorsHash[key], errors),0))) + "%"
		#print "\t\trange: " + range95String(errors)
		print ""
		#print "\tmean error" + str(numpy.mean(errors))

	printParameters("Parameters", parametersHash)

	print "\n\n"
	if doPlot:
		pylab.show()

	"""
	
				

