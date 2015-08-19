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

	def train(self, trainX, trainY, desiredNonZero, verboseStandardOut=False):
		def countNonZero(tl):
			coefs = tl.getAttributes()['coef_']
			return numpy.count_nonzero(coefs)

		# Define those arguments that will be fixed over all calls
		arguments = {}
		arguments['penalty'] = "l1"

		inC = 1
		sklLogReg = "scikitlearn.LogisticRegression"
		trained = UML.train(sklLogReg, trainX, trainY, C=inC, **arguments)

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
			
			trained = UML.train(sklLogReg, trainX, trainY, C=inC, **arguments)	
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



def cleanForCommas(origFileName):
	origFile = open(origFileName, 'r')

	cleanedFileName = origFileName[:-len(".csv")] + " - clean.csv"
	cleanedFile = open(cleanedFileName, 'w')

	for line in origFile:
		splitQuote = line.split('"')
		for i in xrange(len(splitQuote)):
			curr = splitQuote[i]
			# If we are at an odd index, we have a string that was previously
			# contained in quotes. Assumes we are operating on something in
			# which quotes come in pairs.
			if i % 2 == 1:
				splitComma = curr.split(',')
				# write it out without the contained commas, while adding
				# back in the quotes
				toWrite = '"' + ''.join(splitComma) + '"'
				cleanedFile.write(toWrite)
			else:
				cleanedFile.write(curr)
		# The newline will still be on the end of the last value in splitQuote

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



if __name__ == "__main__":

	defaultFile = "/Users/spencer2/Dropbox/Spencer/Work/ClearerThinking.org/Programs and modules/Gender continuum test/gender continuum train and test ready to predict.csv"
	
	if len(sys.argv) <= 1:
		origFileName = defaultFile
	
	else:
		rigFileName = sys.argv[1]
	

	desiredNonZeroCoefficients = 84 #50
		
	# check to see if we've cleaned the data for extra commas
	cleanedSuffix = " - clean.csv"
	if origFileName[-len(cleanedSuffix):] != cleanedSuffix:
		cleanForCommas(origFileName)
		toOpen = origFileName[:-len('.csv')] + cleanedSuffix
	else:
		toOpen = origFileName

	openedFile = open(toOpen, 'r')
	dataAll = UML.createData("List", openedFile, featureNames=0, fileType='csv',
			ignoreNonNumericalFeatures=True)

	# grab the features we want to be in the training data
	nameOfFirst = "I do not enjoy watching dance performances. (M)"
	indexOfFirst = dataAll.getFeatureIndex(nameOfFirst)

	usedData = dataAll.extractFeatures(start=indexOfFirst, end=None)
	usedData.appendFeatures(dataAll.copyFeatures("isMale"))
	usedData = usedData.copyAs("Matrix")

	#usedData.show("usedData")

	print "Splitting data into training and testing..."
	trainX, trainY, testX, testY = usedData.trainAndTestSets(testFraction=0.30, labels="isMale")

	#trainX = dataAll.extractFeatures(start=indexOfFirst, end=None)
	
	# Check some of the expectations we had on the data to make sure
	# we've extracted the right stuff
	#totalScores = dataAll.extractFeatures("totalScorePosOrNeg")
	#sanityCheck(trainX, totalScores)  # defined above __main__

	# convert for processing
	#trainX = trainX.copyAs("Matrix")

	# grab the prediction variable
	#trainY = dataAll.extractFeatures("isMale")

	UML.registerCustomLearner("custom", LogisticRegressionWithSelection)

	print ""
	print "Train points: " + str(trainX.pointCount) 
	print "Test points: " + str(testX.pointCount) 
	print""

	#Cs = tuple([4**k for k in xrange(-8,8)])
	#print "Learning..."
	#print ""
	#bestError = UML.trainAndTest("scikitlearn.SVC", trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, C=0.3) #19.7% out of sample error
	#bestError = UML.trainAndTest("scikitlearn.SVC", trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, kernel="poly", degree=2, coef0=1, C=0.01) #19.2%
	#bestError = UML.trainAndTest("scikitlearn.SVC", trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, kernel="poly", degree=3, coef0=1, C=0.1) 
	#print "bestError out of sample: ", str(round(bestError*100,1)) + "%"

	
	predictionMode = ["set number of coefficients", "cross validation"][0]
	if predictionMode == "set number of coefficients":
		name = "custom.LogisticRegressionWithSelection"
		print "Finding exactly " + str(desiredNonZeroCoefficients) + " coefficients..."
		trainedLearner = UML.train(name, trainX, trainY, desiredNonZero=desiredNonZeroCoefficients)
	elif predictionMode == "cross validation":
		#Cs = [4**k for k in xrange(-8,8)]
		#print "Cross validating..."
		#trainedLearner = UML.train("scikitlearn.LogisticRegression", trainX, trainY, C=10, penalty="l1")
		#trainedLearner = UML.train("scikitlearn.LogisticRegression", trainX, trainY, C=5, penalty="l2")
		#trainedLearner = UML.train("scikitlearn.LogisticRegression", trainX, trainY)
		trainedLearner = UML.train("scikitlearn.SVC", trainX, trainY, C=0.3)
	else: raise Exception("Bad prediction mode!")
	

	# grab the feature names associated with the non-zero coefficients
	
	#print "trainedLearner.getAttributes()", trainedLearner.getAttributes()
	coefs = trainedLearner.getAttributes()["coef_"]
	intercept = trainedLearner.getAttributes()["intercept_"]
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

	

	#Now measure the accuracy of the model
	print "\n\n"
	errorOutSample = trainedLearner.test(testX, testY, performanceFunction=fractionIncorrect)
	print "Out of sample error rate: " + str(round(errorOutSample*100,1)) + "%"
	errorInSample = trainedLearner.test(trainX, trainY, performanceFunction=fractionIncorrect)
	print "In sample error rate: " + str(round(errorInSample*100,1)) + "%"
	print ""


	exit(0)




