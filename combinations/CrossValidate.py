import Combinations
#import DenseMatrix

def runAlgorithm(train, test, algorithm, parameters):
	pass #this would already be implemented in other UML code


def crossValidate(X, Y, functionsToApply, numFolds=10):
	"""applies crossValidation using numFolds folds, applying each function in the list functionsToApply (which are text of python functions)
	one by one. Assumes that thefunctionsToApply is a list of the text functions, that use the variables trainX, trainY, testX, testY within
	their text."""
	aggregatedResults = {}
	for function in functionsToApply:
		curResults = []
		XIterator = X.foldIterator(numFolds=numFolds)	#assumes that the dataMatrix class has a function foldIterator that returns an iterator
														#each time you call .next() on this iterator, it gives you the next (trainData, testData)
														#pair, one for each fold. So for 10 fold cross validation, next works 10 times before failing.
		YIterator = Y.foldIterator(numFolds=numFolds)
		while True: #need to add a test here for when iterator .next() is done
			try:
				curTrainX, curTestX = XIterator.next()
				curTrainY, curTestY = YIterator.next()
			except StopIteration:	#once we've gone through all the folds, this exception gets thrown and we're done!
					break
			dataHash = {}
			dataHash["trainX"] = curTrainX; dataHash["testX"] = curTestX	#assumes that the function text in functionsToApply uses these variables
			dataHash["trainY"] = curTrainY; dataHash["testY"] = curTestY
			curResults.append(Combinations.executeCode(function, dataHash))
		aggregatedResults[function] = sum(curResults)/float(len(curResults)) #NOTE: this could be bad if the sets have different size!!
	return aggregatedResults


def crossValidateReturnBest(X, Y, functionsToApply, minimize, numFolds=10):
	"""runs cross validation on the functions whose text is in the list functionsToApply, and returns the text of the best performer together with
	its performance"""
	if not isinstance(minimize, bool): raise Exception("minimize must be True or False!")
	resultsHash = crossValidate(X,Y, functionsToApply=functionsToApply, numFolds=numFolds)
	if minimize: bestPerformance = float('inf')
	else: bestPerformance = float('-inf')
	bestFuncText = None
	for functionText, performance in resultsHash:
		if (minimize and performance < bestPerformance) or (not minimize and performance > bestPerformance): #if it's the best we've seen so far
				bestPerformance = performance
				bestFuncText = functionText
	return bestFuncText, bestPerformance




def normalize(train, test, algorithm, parameters=None):
	"""use this command to normalize training and testing data using an algorithm. For instance:
	normalize(trainX, testX, algorithm="mean") would run mean normalization on trainX, and apply those learned column means to both trainX
	and testX, modifying trainX and testX to be the new normalized dataMatrix objects."""
	test.copyOf(runAlgorithm(train, test, algorithm=algorithm, parameters=parameters))	#copyOf() would set train so that it is the same as what's passed to it
	train.copyOf(runAlgorithm(train, train, algorithm=algorithm, parameters=parameters))	#copyOf() would set test so that it is the same as what's passed to it

def loadTrainingAndTesting(fileName, labelID, fractionForTestSet):
	"""this is a helpful function that makes it easy to do the common task of loading a dataset and splitting it into training and testing sets.
	It returns training X, training Y, testing X and testing Y"""
	trainX = DenseMatrix.DenseMatrix("myFile.txt")	#load all our data (both X & Y) I'm not sure actually how we do this in the current code
													#we'll have to deal with different data types (dense, sparse, etc.) but I'm ignoring that here
	testX = trainX.extractPoints(number=int(round(fractionForTestSet*len(trainX))), randomize=True)	#pull out a testing set
	trainY = trainX.extractFeatures(labelID)	#construct the column vector of training labels
	testY = testX.extractFeatures(labelID)	#construct the column vector of testing labels
	return trainX, trainY, testX, testY



if __name__ == "__main__":

	#### Some sample use cases ####

	###############################################
	#mean normalize your training and testing data#

	trainX, trainY, testX, testY = loadTrainingAndTesting("myFile.txt", "predictionLabels", fractionForTestSet=.15) #load and split up the data
	normalize(trainX, testX, algorithm="mean") #perform mean normalization

	##################################################
	#cross validate over some different possibilities#

	#these are the runs we're going to try
	run1 = 'normalize(trainX, testX, algorithm="dropFeatures", parameters={"start":1,"end":3}); runAlgorithmWithPeformance(trainX, trainY, testX, testY, algorithm="mlpy.svm")'
	run2 = 'normalize(trainX, testX, algorithm="dropFeatures", parameters={"start":1,"end":5}); runAlgorithmWithPeformance(trainX, trainY, testX, testY, algorithm="mlpy.svm")'

	#this will return the text of whichever function performed better, as well as that best performance value
	bestFunction, performance = crossValidateReturnBest(trainX, trainY, [run1, run2], numFolds=10)

	#####################################################
	#cross validate over a large number of possibilities#

	#we'll be trying all combinations of C in [0.1, 1, 10, 100] and iterations in [100, 1000]
	runs = Combinations.functionCombinations('runAlgorithmWithPeformance(trainX, trainY, testX, testY, algorithm="mlpy.svm", parameters={"C":<0.1|1|10|100>, "iterations":<100|1000>})')

	#this will return the text of whichever function performed better, as well as that best performance value
	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, numFolds=10)




