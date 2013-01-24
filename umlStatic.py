"""
Wrapper module, allowing for easy access to the major functionality of the package that
is static.

"""

from UML import run


# run() with a return type of the predicted labels added back into the object?


def normalize(package, algorithm, trainData, testData, dependentVar=None, arguments={}, mode=True):
	"""
	Calls on the functionality of a package to train on some data and then modify both
	the training data and a set of test data accroding to the produced model.


	"""
	# single call normalize, combined data
	if mode:
		testLength = testData.points()
		# glue training data at the end of test data
		testData.appendPoints(trainData)
		normalizedAll = run(package, algorithm, trainData, testData, dependentVar=dependentVar, arguments=arguments)
		# resplit normalized
		normalizedTrain = normalizedAll.extractPoints(start=testLength, end=normalizedAll.points())
		normalizedTest = normalizedAll
	# two call normalize, no data combination
	else:
		normalizedTrain = run(package, algorithm, trainData, trainData, dependentVar=dependentVar, arguments=arguments)
		normalizedTest = run(package, algorithm, trainData, testData, dependentVar=dependentVar, arguments=arguments)
		
	# modify references for trainData and testData
	trainData.copyReferences(normalizedTrain)
	testData.copyReferences(normalizedTest)




# def runWithPerformance()  # same as run, with extra parameter?


def crossValidate(X, Y, functionsToApply, numFolds=10):
	return cvImplementation(X, Y, functionsToApply, numFolds)


#combinations() -- maybe

#listAllAlgorithms()

