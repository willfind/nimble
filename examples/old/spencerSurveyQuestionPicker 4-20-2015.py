"""
Select the best questions to keep in a survey from a larger set of questions
"""
import numpy
import sys

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
	diffObject = predictedValues - knownValues
	rawDiff = diffObject.copyAs("numpy array")
	rawKnowns = knownValues.copyAs("numpy array")
	return 1.0 - numpy.var(rawDiff)/float(numpy.var(rawKnowns))


def getBestFeaturesAndAccuracies(data, numFeaturesToKeep, fractionOfDataForTesting, featuresToPredict, predictionAlgorithms, functionsToExcludePoints):
	#these will store the training and testing data for each label we're going to predict
	trainXs = []
	trainYs = []
	testXs = []
	testYs = []

	print "AllFeatures:\n", allFeatures

	#make our training and testing sets
	#for each of the different labels we want to predict
	for labelNumber in xrange(len(featuresToPredict)):
		#create the features for the current labelNum, and exclude irrelevant poins
		currentFeatures = allFeatures.copy()
		currentFeatures.extractPoints(functionsToExcludePoints[labelNumber]) #get rid of points that aren't relevant to this label
		
		#remove the labels from the features
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
		trainXs.append(trainX)
		trainYs.append(trainY)
		testXs.append(testX)
		testYs.append(testY)

	#confirm the training X sets all have the same number of features (even though they may not have the same number of points)
	for trainX in trainXs:
		assert trainX.featureCount == trainXs[0].featureCount

	#create a set of all features we might want to keep
	#viableFeaturesLeft = range(trainX.featureCount)

	#try dropping one feature at a time, and kill off the feature that is most useless until we have the right number

	while trainXs[0].featureCount > numFeaturesToKeep:
		print str(trainXs[0].featureCount) + " features left"

		accuraciesForEachFeatureDropped = []

		#try dropping each feature one by one 
		for featureNumToDrop in xrange(trainXs[0].featureCount):
			sys.stdout.write(" " + str(featureNumToDrop))
			accuraciesForThisFeatureDrop = []
			#for each label we're predicting
			for labelNum, trainX, trainY in zip(range(len(trainXs)), trainXs, trainYs):
				#print "trainX", trainX
				#build a feature set to train on that doesn't include the feature we're dropping
				trainXWithoutFeature = trainX.copy()
				trainXWithoutFeature.extractFeatures(featureNumToDrop)
				
				algorithmName = predictionAlgorithms[labelNum]
				if "Logistic" in algorithmName: 
					#C = tuple([10.0**k for k in xrange(-6, 6)])
					C = 1000000
					error = trainAndTest(algorithmName, trainXWithoutFeature, trainY, testX=trainXWithoutFeature, testY=trainY, performanceFunction=fractionIncorrect, C=C)
					accuracy = 1.0 - error
				elif "Ridge" in algorithmName:
					#alpha = tuple([10.0**k for k in xrange(-6, 6)])
					alpha = 0
					accuracy = trainAndTest(algorithmName, trainXWithoutFeature, trainY, testX=trainXWithoutFeature, testY=trainY, performanceFunction=rSquared, alpha=alpha)
				else:
					raise Exception("Don't know how to set parameters for algorithm: " + str(algorithmName))

				accuraciesForThisFeatureDrop.append(accuracy)

			combinedAccuracyForFeatureDrop = numpy.mean(accuraciesForThisFeatureDrop)
			accuraciesForEachFeatureDropped.append((combinedAccuracyForFeatureDrop, featureNumToDrop))

		accuraciesForEachFeatureDropped.sort(lambda x,y: cmp(x[0],y[0])) #sort asscending by accuracy so that the last element corresponds to the most useless feature
		mostUselessFeatureNum = accuraciesForEachFeatureDropped[-1][1]
		print "\nRemoving feature " + str(mostUselessFeatureNum) + " with combined accuracy " + str(round(accuraciesForEachFeatureDropped[-1][0],3))
		for trainX, testX in zip(trainXs, testXs):
			trainX.extractFeatures(mostUselessFeatureNum)
			testX.extractFeatures(mostUselessFeatureNum)
		print "viableFeaturesLeft", trainXs[0].featureCount

	accuraciesHash = {}
	parametersHash = {}
	#now test the models out of sample on our final feature sets!
	for labelNum, trainX, trainY, testX, testY in zip(range(len(trainXs)), trainXs, trainYs, testXs, testYs):
		algorithmName = predictionAlgorithms[labelNum]
		featureToPredict = featuresToPredict[labelNum]
		if "Logistic" in algorithmName: 
			#C = tuple([10.0**k for k in xrange(-6, 6)])
			C = 1000000
			#error = trainAndTest(algorithmName, trainX, trainY, testX=testX, testY=testY, performanceFunction=fractionIncorrect, C=C)
			learner = UML.train(algorithmName, trainX, trainY, C=C)
			error = learner.test(testX=testX, testY=testY, performanceFunction=fractionIncorrect)
			accuracy = 1.0 - error
			backend = learner.backend
			parametersHash[featureToPredict] = {"intercept":backend.intercept_, "coefs":backend.coef_}
		elif "Ridge" in algorithmName:
			#alpha = tuple([10.0**k for k in xrange(-6, 6)])
			alpha = 0
			#accuracy = trainAndTest(algorithmName, trainX, trainY, testX=testX, testY=testY, performanceFunction=rSquared, alpha=alpha)
			learner = UML.train(algorithmName, trainX, trainY, alpha=0)
			accuracy = learner.test(testX=testX, testY=testY, performanceFunction=rSquared)
			backend = learner.backend
			parametersHash[featureToPredict] = {"intercept":backend.intercept_, "coefs":backend.coef_}
		else:
			raise Exception("Don't know how to set parameters for algorithm: " + str(algorithmName))
		accuraciesHash[featureToPredict] = accuracy #record the accuracy

	bestFeatures = trainXs[0].getFeatureNames()
	return bestFeatures, accuraciesHash, parametersHash


if __name__ == "__main__":
	#fileName = "spencerTest.csv" 
	fileName = "Academic_Cons_Survey_445_points.csv"
	numFeaturesToKeep = 10
	fractionOfDataForTesting = 0.25
	featuresToPredict = ["TotalAcademicScore", "inLyingGroup"] #the features we'll be predicting
	#these functions determine what to exclude for each feature prediction
	functionsToExcludePoints = [lambda x: x["inLyingGroup"] == 1, lambda x: False] 
	predictionAlgorithms = ["SciKitLearn.Ridge", "SciKitLearn.LogisticRegression"]
	featuresToRemoveCompletely = ["User Number", "attention1", "attention2", "AttentionChecksPassed", "AcademicAtt1", "AcademicAtt2", "AcademicAttChksPassed", "AcademicScore"]

	#load the data
	pathIn = os.path.join(UML.UMLPath, "datasets/", fileName)
	allFeatures = createData("Matrix", pathIn, featureNames=True)
	allFeatures.extractFeatures(featuresToRemoveCompletely)

	bestFeatures, accuraciesHash, parametersHash = getBestFeaturesAndAccuracies(allFeatures, numFeaturesToKeep=numFeaturesToKeep, fractionOfDataForTesting=fractionOfDataForTesting, featuresToPredict=featuresToPredict, predictionAlgorithms=predictionAlgorithms, functionsToExcludePoints=functionsToExcludePoints)

	print "Best features: " + str(bestFeatures)
	print ""
	print "Accuracies: " + str(accuraciesHash)
	print ""
	print "Paramaters: " + str(parametersHash)


	
				

