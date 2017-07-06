

import os
import sys
import numpy
import math

from allowImports import boilerplate
boilerplate()

import UML
from UML.customLearners import CustomLearner
from UML.calculate import fractionIncorrect


class LogisticRegressionNoTraining(CustomLearner):
	learnerType = "classification"

	def train(self, trainX, trainY, coefs, intercept):
		if coefs.featureCount != 1:
			coefs.transpose()
		self.coefs = coefs
		self.intercept = float(intercept)

	def apply(self, testX):
		exponents = -(self.intercept + (testX * self.coefs))
		
		def getProb(val):
			return 1 / (1 + math.e**(val))

		probabilities = exponents.calculateForEachElement(getProb)

		def roundProb(val):
			return 1 if val > .5 else 0

		ret = probabilities.calculateForEachElement(roundProb)
		
		for i,name in enumerate(testX.getFeatureNames()):
			print name + " : " + str(self.coefs[i])

		return ret


# assumptions: have data in the form of point per person,
# with features per question and correct scale scores. - what is correct? - we've decided RAW
# Also have metatdata showing which cat each question is in, and the scales
#
def predict_gender_sanityCheck(responses, catTestFraction, splitSeed):
	
	UML.setRandomSeed(splitSeed)
	catTrain, catTest = responses.trainAndTestSets(testFraction=catTestFraction)
	
	for i in xrange(0, 5):
		currData = catTest.copy()
		s = i*100
		print s
		testX = currData.extractPoints(start=s, end=s+100)
		testY = testX.extractFeatures('male0female1')

		trainY = currData.extractFeatures('male0female1')
		trainX = currData

		C = tuple([100. / (10**n) for n in range(7)])
		ret = UML.trainAndTest('scikitlearn.LogisticRegression', trainX, trainY, testX, testY, fractionIncorrect, C=C)
		print ret


def predict_gender_fullTrain(responses, catTestFraction, splitSeed):
	UML.setRandomSeed(splitSeed)
	catTrain, catTest = responses.trainAndTestSets(testFraction=catTestFraction)

	trainY = catTest.extractFeatures('male0female1')
	trainX = catTest

	C = tuple([1000. / (10**n) for n in range(9)])
	sklModel = UML.train('scikitlearn.LogisticRegression', trainX, trainY, fractionIncorrect, C=C)

#	print sklModel.getAttributes()['C']

	coefs = sklModel.getAttributes()['coef_']
	coefsObj = UML.createData("Matrix", coefs)
	intercept = sklModel.getAttributes()['intercept_'][0]

	fromModelPredictions = sklModel.apply(trainX)
	independentModel = UML.train("custom.LogisticRegressionNoTraining", trainX, trainY, coefs=coefsObj, intercept=intercept)
	independentPredictions = independentModel.apply(trainX)
	assert fromModelPredictions.isIdentical(independentPredictions)
	assert sklModel.test(trainX, trainY, fractionIncorrect) == independentModel.test(trainX, trainY, fractionIncorrect)

	return coefs.flatten(), intercept

def outputFile_PredictionMetadata(outPath, responses, questionMetadata, coefs, categoryMetadata):
	allCats = questionMetadata.featureView(1).copyAs("numpyarray")
	allCats = numpy.unique(allCats).tolist()
	raw = []
#	print coefs
	for i,point in enumerate(questionMetadata.pointIterator()):
		question = point[0]
		category = point[1]

		newPoint = [question, 1, coefs[i]]
		
		# do that inclusion/scale scoring of each possible category
		for possibleCat in allCats:
			if category == possibleCat:
				catAgreement = categoryMetadata[category,0]
				val = 1 if point[2] == catAgreement else -1
			else:
				val = 0
			newPoint.append(val)

		raw.append(newPoint)

	coefCats = map(lambda x: "coefs|" + x, allCats)

	fnames = ["questions", "coding", "coefs|gender_prediction"] + coefCats
	toOutput = UML.createData("List", raw, featureNames=fnames)
#	toOutput.show("", maxWidth=120, maxColumnWidth=44)

	toOutput.writeFile(outPath)


if __name__ == '__main__':
	CAT_TRAIN_NUMBER = 300
	SPLITSEED = 13

	UML.registerCustomLearner("custom", LogisticRegressionNoTraining)

	# Source Data
	sourceDir = sys.argv[1]
	path_responses = os.path.join(sourceDir, "inData", "questions_selected_and_formated.csv")
	path_categories_metadata = os.path.join(sourceDir, "inData", "categories_selectedCatMetadata.csv")
	path_question_metatdata = os.path.join(sourceDir, "inData", "questions_selectedMetadata.csv")

	outPath_predictionMetadata = os.path.join(sourceDir, "inData", "prediction_metadata.csv")

	# load question responses and gender data
	responses = UML.createData("Matrix", path_responses)

	# load the associated data for categories
	categoryMetadata = UML.createData("List", path_categories_metadata)
	cm_pnames = categoryMetadata.extractFeatures(0).copyAs("pythonList", outputAs1D=True)
	categoryMetadata.setPointNames(cm_pnames)

	# load the associated data for questions
	questionMetadata = UML.createData("List", path_question_metatdata)


	catTestFraction = float(responses.pointCount - CAT_TRAIN_NUMBER) / responses.pointCount
#	predict_gender_sanityCheck(responses, catTestFraction, SPLITSEED)	
	coefs, intercept = predict_gender_fullTrain(responses, catTestFraction, SPLITSEED)

	outputFile_PredictionMetadata(outPath_predictionMetadata, responses, questionMetadata, coefs, categoryMetadata)

#	path_inter = os.path.join(sourceDir, "inData", "prediction_intercept.csv")
#	inter = UML.createData("Matrix", path_inter)

#	print inter[0,0]
