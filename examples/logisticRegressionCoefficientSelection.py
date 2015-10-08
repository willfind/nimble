"""
An example script defining a custom learner and running it on the
data given via a path on the command line.

"""

from allowImports import boilerplate
boilerplate()

import sys
import numpy

import UML
from UML.customLearners import CustomLearner
from UML.helpers import generateClassificationData
from UML.calculate import fractionIncorrect

class LogisticRegressionWithSelection(CustomLearner):
	learnerType = "classification"

	def train(
			self, trainX, trainY, desiredNonZero, verboseStandardOut=False,
			allowSubLogging=False):
		if desiredNonZero > trainX.featureCount:
			desiredNonZero = trainX.featureCount

		def countNonZero(tl):
			coefs = tl.getAttributes()['coef_']
			return numpy.count_nonzero(coefs)

		# Define those arguments that will be fixed over all calls
		arguments = {}
		arguments['penalty'] = "l1"

		inC = 1
		sklLogReg = "scikitlearn.LogisticRegression"
		trained = UML.train(sklLogReg, trainX, trainY, C=inC, useLog=allowSubLogging, **arguments)

		# the first entry will be our downward multiplicative adjustment,
		# the second entry will be our upward adjustment
		adjustment = [3./4, 5./4]
		iteration = 1
		numNZ = countNonZero(trained)
		
		# we stop when the number of non-zero coefficients is precisely
		# what we want it to be
		while numNZ != desiredNonZero:
			# determine which adjustment is appropriate
			if numNZ < desiredNonZero:
				multiplier = adjustment[1]
			else:  # numNZ > desiredNonZero
				multiplier = adjustment[0]
			inC = inC * multiplier
			
			trained = UML.train(sklLogReg, trainX, trainY, C=inC,  useLog=allowSubLogging, **arguments)
			numNZ = countNonZero(trained)
			iteration += 1

			if verboseStandardOut:
				print '\n# iteration #: ' + str(iteration)
				print 'multiplier: ' + str(multiplier)
				print 'inC: ' + str(inC)
				print 'nz: '+str(numpy.count_nonzero(trained.getAttributes()['coef_']))

		self._trained = trained
		# expose everything from the underlying learner
		toExpose = trained.getAttributes()
		for key in toExpose:
			setattr(self, key, toExpose[key])

	def apply(self, testX):
		return self._trained.apply(testX)


def sanityCheck(trainX, totalScores):
	assert trainX.featureCount == 84

	for name in trainX.getFeatureNames():
		assert name[-3:] == "(M)" or name[-3:] == "(F)"

	# gotta fix data's __getitem__ so we can just pass python's sum function
	def summer(point):
		total = 0
		for value in point:
			total += value
		return total

	summed = trainX.applyToPoints(summer, inPlace=False)
	summed.setFeatureName(0, "totalScorePosOrNeg")
	assert summed == totalScores

def seperateData(dataAll, omitCultureQuestions):
	# grab the features we want to be in the training data; including raw
	# data that we may later choose to omit
	nameOfFirst = "I do not enjoy watching dance performances. (M)"
	indexOfFirst = dataAll.getFeatureIndex(nameOfFirst)

	usedData = dataAll.extractFeatures(start=indexOfFirst, end=None)
	usedData.appendFeatures(dataAll.copyFeatures("isMale"))
	usedData.appendFeatures(dataAll.copyFeatures("InTestSet"))

	if omitCultureQuestions:
		toOmit = []
		toOmit.append("I do not enjoy watching dance performances. (M)")
		toOmit.append(" I would be good at rescuing someone from a burning building. (M)")
		toOmit.append(" I am interested in science. (M)")
		toOmit.append(" I find political discussions interesting. (M)")
		toOmit.append(" I face danger confidently. (M)")
		toOmit.append(" I do not like concerts. (M)")
		toOmit.append(" I do not enjoy going to art museums. (M)")
		toOmit.append(" I would fear walking in a high-crime part of a city. (F)")
		toOmit.append(" I begin to panic when there is danger. (F)")
		usedData.extractFeatures(toOmit)

	print "Splitting data into training and testing..."

	def selectInTestSet(point):
		if point["InTestSet"] > 0:
			return True
		return False

	testingSet = usedData.extractPoints(selectInTestSet)
	testY = testingSet.extractFeatures("isMale")
	testingSet.extractFeatures("InTestSet")
	testX = testingSet

	trainY = usedData.extractFeatures("isMale")
	usedData.extractFeatures("InTestSet")
	trainX = usedData

	return trainX, trainY, testX, testY

def printCoefficients(trainedLearner):
	coefs = trainedLearner.getAttributes()["coef_"]
#	intercept = trainedLearner.getAttributes()["intercept_"]
	coefs = coefs.flatten()
	chosen = []
	chosenCoefs = []
	for i in xrange(len(coefs)):
		value = coefs[i]
		if value != 0:
			chosen.append(trainX.getFeatureName(i))
			chosenCoefs.append(coefs[i])

	# display those questions which were the most useful
	print "\n"
	i = 1
	for question, coef in zip(chosen, chosenCoefs):
		print str(i).ljust(3) + "    " + str(round(coef,2)).ljust(8) + question.strip()
		i = i + 1

if __name__ == "__main__":

	# Some variables to control the flow of the program
	defaultFile = "/Users/spencer2/Dropbox/Spencer/Work/ClearerThinking.org/Programs and modules/Gender continuum test/gender continuum train and test ready to predict.csv"
	omitCultureQuestions = True
	desiredNonZeroCoefficients = 75  # 50
	performSanityCheck = False

	UML.registerCustomLearner("custom", LogisticRegressionWithSelection)

	if len(sys.argv) <= 1:
		origFileName = defaultFile
	else:
		origFileName = sys.argv[1]
	
	dataAll = UML.createData("Matrix", origFileName, featureNames=True, fileType='csv',
		ignoreNonNumericalFeatures=True)

	# call helper to remove extraneous features, omit undesired
	# ones, and to seperate into training and testing sets according
	# to the 'InTestSet' feature.
	trainX, trainY, testX, testY = seperateData(dataAll, omitCultureQuestions)

	trainX.name = "Training Data"
	trainY.name = "Training Labels"
	testX.name = "Testing Data"
	testY.name = "Testing Labels"

	# Check some of the expectations we had on the data to make sure
	# we've extracted the right stuff
	if performSanityCheck:
		totalScores = dataAll.extractFeatures("totalScorePosOrNeg")
		sanityCheck(trainX, totalScores)  # defined above __main__

	print ""
	print "Train points: " + str(trainX.pointCount) 
	print "Test points: " + str(testX.pointCount) 
	print ""


	predictionPossibilities = []
	predictionPossibilities.append("SVC full data out-sample only")  # 0
	predictionPossibilities.append("Reduced number of coefficients")  # 1
	predictionPossibilities.append("cross validation")  # 2
	predictionPossibilities.append("Prediction: LogReg with L1")  # 3
	predictionPossibilities.append("Prediction: LogReg with L2")  # 4
	predictionPossibilities.append("Prediction: SVM")  # 5
	predictionPossibilities.append("Prediction: SVM with poly kernel deg 2")  # 6
	predictionPossibilities.append("Prediction: SVM with poly kernel deg 3")  # 7
	predictionMode = predictionPossibilities[5]

	print "Learning..."
	print predictionMode
	print ""

	# 0
	if predictionMode == "SVC full data out-sample only":
		Cs = tuple([4**k for k in xrange(-8,8)])
#		bestError = UML.trainAndTest("scikitlearn.SVC", trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, C=0.3) #19.7% out of sample error
#		bestError = UML.trainAndTest("scikitlearn.SVC", trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, kernel="poly", degree=2, coef0=1, C=0.01) #19.2%
		bestError = UML.trainAndTest("scikitlearn.SVC", trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, kernel="poly", degree=3, coef0=1, C=0.1) 
		print "bestError out of sample: ", str(round(bestError*100,1)) + "%"
		sys.exit(0)
	# 1
	elif predictionMode == "Reduced number of coefficients":
		name = "custom.LogisticRegressionWithSelection"
		print "Finding exactly " + str(desiredNonZeroCoefficients) + " coefficients..."
		trainedLearner = UML.train(name, trainX, trainY, desiredNonZero=desiredNonZeroCoefficients)
	# 2
	elif predictionMode == "cross validation":
		#Cs = [4**k for k in xrange(-8,8)]
		#print "Cross validating..."
		#trainedLearner = UML.train("scikitlearn.LogisticRegression", trainX, trainY, C=10, penalty="l1")
		#trainedLearner = UML.train("scikitlearn.LogisticRegression", trainX, trainY, C=5, penalty="l2")
		#trainedLearner = UML.train("scikitlearn.LogisticRegression", trainX, trainY)
		trainedLearner = UML.train("scikitlearn.SVC", trainX, trainY, C=0.3)
	# 3
	elif predictionMode == "Prediction: LogReg with L1":
		name = "scikitlearn.LogisticRegression"
		cVals = tuple([100. / (10**n) for n in range(7)])
		print "Cross validated over C with values of: " + str(cVals)
		trainedLearner = UML.train(
			name, trainX, trainY, C=cVals, penalty='l1',
			performanceFunction=fractionIncorrect)
	# 4
	elif predictionMode == "Prediction: LogReg with L2":
		name = "scikitlearn.LogisticRegression"
		cVals = tuple([100. / (10**n) for n in range(7)])
		print "Cross validated over C with values of: " + str(cVals)
		trainedLearner = UML.train(
			name, trainX, trainY, C=cVals, penalty='l2',
			performanceFunction=fractionIncorrect)
	# 5
	elif predictionMode == "Prediction: SVM":
		name = "scikitlearn.SVC"
		cVals = tuple([100. / (10**n) for n in range(7)])
		print "Cross validated over C with values of: " + str(cVals)
		trainedLearner = UML.train(
			name, trainX, trainY, C=cVals, kernel='linear', performanceFunction=fractionIncorrect)
	# 6
	elif predictionMode == "Prediction: SVM with poly kernel deg 2":
		name = "scikitlearn.SVC"
		cVals = tuple([100. / (10**n) for n in range(7)])
		print "Cross validated over C with values of: " + str(cVals)
		trainedLearner = UML.train(
			name, trainX, trainY, C=cVals, kernel='poly', degree=2,
			performanceFunction=fractionIncorrect)
	# 7
	elif predictionMode == "Prediction: SVM with poly kernel deg 3":
		name = "scikitlearn.SVC"
		cVals = tuple([100. / (10**n) for n in range(7)])
		print "Cross validated over C with values of: " + str(cVals)
		trainedLearner = UML.train(
			name, trainX, trainY, C=cVals, kernel='poly', degree=3,
			performanceFunction=fractionIncorrect)
	else:
		raise Exception("Bad prediction mode!")
	
	# grab the feature names associated with the non-zero coefficients
#	printCoefficients(trainedLearner)

	#Now measure the accuracy of the model
	print "\n\n"
	errorOutSample = trainedLearner.test(testX, testY, performanceFunction=fractionIncorrect)
	print "Out of sample error rate: " + str(round(errorOutSample*100,1)) + "%"
	errorInSample = trainedLearner.test(trainX, trainY, performanceFunction=fractionIncorrect)
	print "In sample error rate: " + str(round(errorInSample*100,1)) + "%"
	print ""

	exit(0)
