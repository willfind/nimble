"""
Module containing most of the user facing functions for the top level uml import.

"""

import numpy
import scipy.sparse
import inspect
import operator
import re 
import datetime
import os

import UML
from UML.exceptions import ArgumentException

from UML.logger import UmlLogger
from UML.logger import Stopwatch

from UML.helpers import findBestInterface
from UML.helpers import _learnerQuery
from UML.helpers import _validScoreMode
from UML.helpers import _validMultiClassStrategy
from UML.helpers import _unpackLearnerName
from UML.helpers import _validArguments
from UML.helpers import _validData
from UML.helpers import LearnerInspector
from UML.helpers import copyLabels
from UML.helpers import ArgumentIterator
from UML.helpers import trainAndApplyOneVsAll
from UML.helpers import trainAndApplyOneVsOne
from UML.helpers import _mergeArguments
from UML.helpers import crossValidateBackend
from UML.helpers import isAllowedRaw
from UML.helpers import initDataObject
from UML.helpers import createDataFromFile

from UML.randomness import numpyRandom

from UML.interfaces.interface_helpers import checkClassificationStrategy





UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def createRandomData(retType, numPoints, numFeatures, sparsity, numericType="float", featureNames=None, name=None):
	"""
	Generates a data object with random contents and numPoints points and numFeatures features. 

	If numericType is 'float' (default) then the value of (point, feature) pairs are sampled from a normal
	distribution (location 0, scale 1).

	If numericType is 'int' then value of (point, feature) pairs are sampled from uniform integer distribution [1 100].

	The sparsity is the likelihood that the value of a (point,feature) pair is zero.

	Zeros are not counted in/do not affect the aforementioned sampling distribution.
	"""

	if numPoints < 1:
		raise ArgumentException("must specify a positive nonzero number of points")
	if numFeatures < 1:
		raise ArgumentException("must specify a positive nonzero number of features")
	if sparsity < 0 or sparsity >=1:
		raise ArgumentException("sparsity must be greater than zero and less than one")
	if numericType != "int" and numericType != "float":
		raise ArgumentException("numericType may only be 'int' or 'float'")


	#note: sparse is not stochastic sparsity, it uses rigid density measures
	if retType.lower() == 'sparse':

		density = 1.0 - float(sparsity)
		numNonZeroValues = int(numPoints * numFeatures * density)

		pointIndices = numpyRandom.randint(low=0, high=numPoints, size=numNonZeroValues)
		featureIndices = numpyRandom.randint(low=0, high=numFeatures, size=numNonZeroValues)

		if numericType == 'int':
			dataVector = numpyRandom.randint(low=1, high=100, size=numNonZeroValues)
		#numeric type is float; distribution is normal
		else: 
			dataVector = numpyRandom.normal(0, 1, size=numNonZeroValues) 

		#pointIndices and featureIndices are 
		randData = scipy.sparse.coo.coo_matrix((dataVector, (pointIndices, featureIndices)), (numPoints, numFeatures))
			
	#for non-sparse matrices, use numpy to generate matrices with sparsity characterics
	else:
		if numericType == 'int':
			filledIntMatrix = numpyRandom.randint(1, 100, (numPoints, numFeatures))
		else:
			filledFloatMatrix = numpyRandom.normal(loc=0.0, scale=1.0, size=(numPoints,numFeatures))

		#if sparsity is zero
		if abs(float(sparsity) - 0.0) < 0.0000000001:
			if numericType == 'int':
				randData = filledIntMatrix
			else:
				randData = filledFloatMatrix
		else:
			binarySparsityMatrix = numpyRandom.binomial(1, 1.0-sparsity, (numPoints, numFeatures))

			if numericType == 'int':
				randData = binarySparsityMatrix * filledIntMatrix
			else:
				randData = binarySparsityMatrix * filledFloatMatrix

	return createData(retType, data=randData, featureNames=featureNames, name=name)



def splitData(toSplit, fractionForTestSet, labelID=None):
	"""this is a helpful function that makes it easy to do the common task of loading a dataset and splitting it into training and testing sets.
	It returns training X, training Y, testing X and testing Y"""
	testXSize = int(round(fractionForTestSet*toSplit.pointCount))
	#shuffle data before pulling anything out
	toSplit.shufflePoints()
	#pull out a testing set
	testX = toSplit.extractPoints(start=0, end=testXSize)	
	trainY = None
	testY = None
	if labelID is not None:
		trainY = toSplit.extractFeatures(labelID)	#construct the column vector of training labels
		testY = testX.extractFeatures(labelID)	#construct the column vector of testing labels
	return toSplit, trainY, testX, testY



def normalizeData(learnerName, trainX, trainY=None, testX=None, arguments={}, **kwarguments):
	"""
	Calls on the functionality of a package to train on some data and then modify both
	the training data and a set of test data according to the produced model.

	Parameters:

	learnerName : String name of the learner to be called, in the form 'package.learner'

	trainX: data to be used for training (as some form of UML data Base object)

	trainY: used to retrieve the known class labels of the training data. Either
	contains the labels themselves (as a Base object) or an identifier (numerical
	index or string name) that defines their placement in the trainX object as a
	feature ID.

	testX: data set to be used for testing (as some form of Base object)

	arguments : dictionary mapping argument names (strings) to their values,
	to be used during training and application. example: {'dimensions':5, 'k':5}

	**kwarguments : kwargs specified variables that are passed to the learner. Same
	format as the arguments parameter.

	"""
	(packName, trueLearnerName) = _unpackLearnerName(learnerName)

	tl = UML.train(learnerName, trainX, trainY, arguments, **kwarguments)
	normalizedTrain = tl.apply(trainX, arguments=arguments, **kwarguments)
	if normalizedTrain.getTypeString() != trainX.getTypeString():
		normalizedTrain = normalizedTrain.copyAs(trainX.getTypeString())

	if testX is not None:
		normalizedTest = tl.apply(testX, arguments=arguments, **kwarguments)
		if normalizedTest.getTypeString() != testX.getTypeString():
			normalizedTest = normalizedTest.copyAs(testX.getTypeString())

	# modify references and names for trainX and testX
	trainX.referenceDataFrom(normalizedTrain)
	trainX.name = trainX.name + " " + trueLearnerName

	if testX is not None:
		testX.referenceDataFrom(normalizedTest)
		testX.name = testX.name + " " + trueLearnerName

def registerCustomLearner(customPackageName, learnerClassObject):
	"""
	Register the given customLearner class so that it is callable by the top level UML
	functions through the interface of the specified custom package.

	customPackageName : The string name of the package preface you want to use when calling
	the learner. If there is already an interface for a custom package with this name, the
	learner will be accessible through that interface. If there is no interface to a custom
	package of that name, then one will be created. You cannot register a custom learner to
	be callable through the interface for a non-custom package (such as ScikitLearn or MLPY).
	Therefore, customPackageName cannot be a value which is the accepted alias of another
	package's interface.

	learnerClassObject : The class object implementing the learner you want registered. It
	will be checked using UML.interfaces.CustomLearner.validateSubclass to ensure that all
	details of the provided implementation are acceptable.

	"""
	# detect name collision
	for currInterface in UML.interfaces.available:
		if not isinstance(currInterface, UML.interfaces.CustomLearnerInterface):
			if currInterface.isAlias(customPackageName):
				raise ArgumentException("The customPackageName '" + customPackageName + "' cannot be used: it is an accepted alias of a non-custom package")

	# do validation before we potentially construct an interface to a custom package
	UML.customLearners.CustomLearner.validateSubclass(learnerClassObject)

	try:
		currInterface = findBestInterface(customPackageName)
	except ArgumentException:
		currInterface = UML.interfaces.CustomLearnerInterface(customPackageName)
		UML.interfaces.available.append(currInterface)

	currInterface.registerLearnerClass(learnerClassObject)

	opName = customPackageName + "." + learnerClassObject.__name__
	opValue = learnerClassObject.__module__ + '.' + learnerClassObject.__name__
	UML.settings.set('RegisteredLearners', opName, opValue)
	UML.settings.saveChanges('RegisteredLearners', opName)
	
	# check if new option names introduced, call sync if needed
	if learnerClassObject.options() != []:
		UML.configuration.syncWithInterfaces(UML.settings)

def deregisterCustomLearner(customPackageName, learnerName):
	"""
	Remove accessibility of the learner with the given name from the interface of the package
	with the given name.

	customPackageName : the name of the interface / custom package from which the learner
	named 'learnerName' is to be removed from. If that learner was the last one grouped in
	that custom package, then the interface is removed from the UML.interfaces.available list.

	learnerName : the name of the learner to be removed from the interface / custom package with
	the name 'customPackageName'

	"""
	currInterface = findBestInterface(customPackageName)
	if not isinstance(currInterface, UML.interfaces.CustomLearnerInterface):
		raise ArgumentException("May only attempt to deregister learners from the interfaces of custom packages. '" + customPackageName + "' is not a custom package")
	origOptions = currInterface.optionNames
	empty = currInterface.deregisterLearner(learnerName)
	newOptions = currInterface.optionNames

	# remove options
	for optName in origOptions:
		if optName not in newOptions:
			UML.settings.delete(customPackageName, optName)

	if empty:
		UML.interfaces.available.remove(currInterface)
		#remove section
		UML.settings.delete(customPackageName, None)

	regOptName = customPackageName + '.' + learnerName
	# delete from registered learner list
	UML.settings.delete('RegisteredLearners', regOptName)
	UML.settings.saveChanges('RegisteredLearners')


def learnerParameters(name):
	"""
	Takes a string of the form 'package.learnerName' and returns a list of
	strings which are the names of the parameters when calling package.learnerName

	If the name cannot be found within the package, then an exception will be thrown.
	If the name is found, be for some reason we cannot determine what the parameters
	are, then we return None. Note that if we have determined that there are no
	parameters, we return an empty list. 

	"""
	return _learnerQuery(name, 'parameters')

def learnerDefaultValues(name):
	"""
	Takes a string of the form 'package.learnerName' and returns a returns a
	dict mapping of parameter names to their default values when calling
	package.learnerName

	If the name cannot be found within the package, then an exception will be thrown.
	If the name is found, be for some reason we cannot determine what the parameters
	are, then we return None. Note that if we have determined that there are no
	parameters, we return an empty dict. 

	"""
	return _learnerQuery(name, 'defaults')


def listLearners(package=None):
	"""
	Takes the name of a package, and returns a list of learners that are callable through that
	package's trainAndApply() interface.

	"""
	results = []
	if package is None:
		for interface in UML.interfaces.available:
			packageName = interface.getCanonicalName()
			currResults = interface.listLearners()
			for learnerName in currResults:
				results.append(packageName + "." + learnerName)
	else:
		interface = findBestInterface(package)
		currResults = interface.listLearners()
		for learnerName in currResults:
			results.append(learnerName)

	return results


def listDataFunctions():
	methodList = dir(UML.data.Base)
	visibleMethodList = []
	for methodName in methodList:
		if not methodName.startswith('_'):
			visibleMethodList.append(methodName)

	ret = []
	for methodName in visibleMethodList:
		currMethod = getattr(UML.data.Base, methodName)
		try:
			(args, varargs, keywords, defaults) = inspect.getargspec(currMethod)
		except TypeError:
			continue

		retString = methodName + "("
		for i in xrange(0, len(args)):
			if i != 0:
				retString += ", "
			retString += args[i]
			if defaults is not None and i >= (len(args) - len(defaults)):
				retString += "=" + str(defaults[i - (len(args) - len(defaults))])
			
		# obliterate the last comma
		retString += ")"
		ret.append(retString)

	return ret


def listUMLFunctions():
	methodList = dir(UML)

	visibleMethodList = []
	for methodName in methodList:
		if not methodName.startswith('_'):
			visibleMethodList.append(methodName)

	ret = []
	for methodName in visibleMethodList:
		currMethod = eval("UML." + methodName)
		if "__call__" not in dir(currMethod):
			continue
		(args, varargs, keywords, defaults) = inspect.getargspec(currMethod)

		retString = methodName + "("
		for i in xrange(0, len(args)):
			if i != 0:
				retString += ", "
			retString += args[i]
			if defaults is not None and i >= (len(args) - len(defaults)):
				retString += "=" + str(defaults[i - (len(args) - len(defaults))])
			
		# obliterate the last comma
		retString += ")"
		ret.append(retString)

	return ret


def createData(retType, data, pointNames=None, featureNames=None, fileType=None, name=None, useLog=None):
	retAllowed = ['List', 'Matrix', 'Sparse', None]
	if retType not in retAllowed:
		raise ArgumentException("retType must be a value in " + str(retAllowed))

	if isAllowedRaw(data):
		return initDataObject(retType, data, pointNames, featureNames, name)
	elif isinstance(data, basestring):
		inPN = pointNames if isinstance(pointNames, int) else None
		inFN = featureNames if isinstance(featureNames, int) else None
		(tempData, tempPNames, tempFNames) = createDataFromFile(retType, data, fileType, inPN, inFN)
		if pointNames is None or isinstance(pointNames, int):
			pointNames = tempPNames
		if featureNames is None or isinstance(featureNames, int):
			featureNames = tempFNames
		return initDataObject(retType, tempData, pointNames, featureNames, name, data)
	else:
		raise ArgumentException("data must contain either raw data or the path to a file to be loaded")


def crossValidate(learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label', negativeLabel=None, useLog=None, **kwarguments):
	"""
	K-fold cross validation.
	Returns mean performance (float) across numFolds folds on a X Y.

	Parameters:

	learnerName (string) - UML compliant algorithm name in the form 
	'package.algorithm' e.g. 'sciKitLearn.KNeighborsClassifier'

	X (UML.Base subclass) - points/features data

	Y (UML.Base subclass or int index for X) - labels/data about points in X

	performanceFunction (function) - Look in UML.calculate for premade options.
	Function used by computeMetrics to generate a performance score for the run.
	function is of the form:
	def func(knownValues, predictedValues, negativeLabel).

	arguments (dict) - dictionary mapping argument names (strings)
	to their values. The parameter is sent to trainAndApply() through its arguments
	parameter. example: {'dimensions':5, 'k':5}

	numFolds (int) - the number of folds used in the cross validation. Can't
	exceed the number of points in X, Y

	scoreMode - used by computeMetrics

	negativeLabel - used by computeMetrics

	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments - kwargs specified variables that are passed to the learner.
	To make use of multiple permutations, specify different values for a
	parameter as a tuple. eg. a=(1,2,3) will generate an error score for 
	the learner when the learner was passed all three values of a, separately.

	"""
	bestResult = crossValidateReturnBest(learnerName, X, Y, performanceFunction, arguments, numFolds, scoreMode, negativeLabel, useLog, **kwarguments)
	return bestResult[1]
	#return crossValidateBackend(learnerName, X, Y, performanceFunction, arguments, numFolds, scoreMode, negativeLabel, useLog, **kwarguments)

def crossValidateReturnAll(learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label', negativeLabel=None, useLog=None, **kwarguments):
	"""
	Calculates the cross validated error for each argument permutation that can 
	be generated by the merge of arguments and kwarguments.

	example **kwarguments: {'a':(1,2,3), 'b':(4,5)}
	generates permutations of dict in the format:
	{'a':1, 'b':4}, {'a':2, 'b':4}, {'a':3, 'b':4}, {'a':1, 'b':5}, 
	{'a':2, 'b':5}, {'a':3, 'b':5}

	For each permutation of 'arguments', crossValidateReturnAll uses cross 
	validation to generate a performance score for the algorithm, given the 
	particular argument permutation.

	Returns a list of tuples, where every tuple contains a dict representing 
	the argument sent to trainAndApply, and a float represennting the cross 
	validated error associated with that argument dict.
	example list element: ({'arg1':2, 'arg2':'max'}, 89.0000123)

	Arguments:

	learnerName (string) - UML compliant algorithm name in the form 
	'package.algorithm' e.g. 'sciKitLearn.KNeighborsClassifier'

	X (UML.Base subclass) - points/features data

	Y (UML.Base subclass or int index for X) - labels/data about points in X

	performanceFunction (function) - Look in UML.calculate for premade options.
	Function used by computeMetrics to generate a performance score for the run.
	function is of the form:
	def func(knownValues, predictedValues, negativeLabel).

	arguments (dict) - dictionary mapping argument names (strings)
	to their values, to be merged with kwargs. To make use of multiple
	permutations, specify different values for a parameter as a tuple. eg.
	a=(1,2,3) will generate an error score for  the learner when the learner
	was passed all three values of a, separately.

	numFolds (int) - the number of folds used in the cross validation. Can't
	exceed the number of points in X, Y

	scoreMode - used by computeMetrics

	negativeLabel - used by computeMetrics

	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments - kwargs specified variables that are passed to the learner,
	after being merged with arguments. To make use of multiple permutations,
	specify different values for a parameter as a tuple. eg. a=(1,2,3) will
	generate an error score for the learner when the learner was passed all
	three values of a, separately.

	"""	
	return crossValidateBackend(learnerName, X, Y, performanceFunction, arguments, numFolds, scoreMode, negativeLabel, useLog, **kwarguments)


def crossValidateReturnBest(learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label', negativeLabel=None, useLog=None, maximize=False, **kwarguments):
	"""
	For each possible argument permutation generated by arguments, 
	crossValidateReturnBest runs crossValidate to compute a mean error for the 
	argument combination. 

	crossValidateReturnBest then RETURNS the best argument and error as a tuple:
	(argument_as_dict, cross_validated_performance_float)

	Arguments:
	learnerName (string) - UML compliant algorithm name in the form 
	'package.algorithm' e.g. 'sciKitLearn.KNeighborsClassifier'

	X (UML.Base subclass) - points/features data

	Y (UML.Base subclass or int index for X) - labels/data about points in X

	performanceFunction (function) - Look in UML.calculate for premade options.
	Function used by computeMetrics to generate a performance score for the run.
	function is of the form:
	def func(knownValues, predictedValues, negativeLabel).

	arguments (dict) - dictionary mapping argument names (strings)
	to their values, to be merged with kwargs. To make use of multiple
	permutations, specify different values for a parameter as a tuple. eg.
	a=(1,2,3) will generate an error score for  the learner when the learner
	was passed all three values of a, separately.

	numFolds (int) - the number of folds used in the cross validation. Can't
	exceed the number of points in X, Y

	scoreMode - used by computeMetrics

	negativeLabel - used by computeMetrics

	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments - kwargs specified variables that are passed to the learner,
	after being merged with arguments. To make use of multiple permutations,
	specify different values for a parameter as a tuple. eg. a=(1,2,3) will
	generate an error score for the learner when the learner was passed all
	three values of a, separately.

	"""
	resultsAll = crossValidateReturnAll(learnerName, X, Y, performanceFunction, arguments, numFolds, scoreMode, negativeLabel, useLog, **kwarguments)

	bestArgumentAndScoreTuple = None

	for curResultTuple in resultsAll:
		curArgument, curScore = curResultTuple
		#if curArgument is the first or best we've seen: 
		#store its details in bestArgumentAndScoreTuple
		if bestArgumentAndScoreTuple is None:
			bestArgumentAndScoreTuple = curResultTuple
		else:
			if (maximize and curScore > bestArgumentAndScoreTuple[1]) or ((not maximize) and curScore < bestArgumentAndScoreTuple[1]):
				bestArgumentAndScoreTuple = curResultTuple

	return bestArgumentAndScoreTuple


def learnerType(learnerNames):
	"""
	Returns the string or list of strings representation of a best guess for 
	the type of learner(s) specified by the learner name(s) in learnerNames.

	If learnerNames is a single string (not a list of strings), then only a single 
	result is returned, instead of a list.
	
	LearnerType first queries the appropriate interface object for a definitive return
	value. If the interface doesn't provide a satisfactory answer, then this method
	calls a backend which generates a series of artificial data sets with particular
	traits to look for heuristic evidence of a classifier, regressor, etc.
	"""
	#argument checking
	if not isinstance(learnerNames, list):
		learnerNames = [learnerNames]

	resultsList = []
	secondPassLearnerNames = []
	for name in learnerNames:
		if not isinstance(name, str):
			raise ArgumentException("learnerNames must be a string or a list of strings.")

		splitTuple = _unpackLearnerName(name)
		currInterface = findBestInterface(splitTuple[0])
		allValidLearnerNames = currInterface.listLearners()
		if not splitTuple[1] in allValidLearnerNames:
			raise ArgumentException(name + " is not a valid learner on your machine.")
		result = currInterface.learnerType(splitTuple[1])
		if result == 'UNKNOWN' or result == 'other' or result is None:
			resultsList.append(None)
			secondPassLearnerNames.append(name)
		else:
			resultsList.append(result)
			secondPassLearnerNames.append(None)
		
	#have valid arguments - a list of learner names
	learnerInspectorObj = LearnerInspector()

	for index in range(len(secondPassLearnerNames)):
		curLearnerName = secondPassLearnerNames[index]
		if curLearnerName is None:
			continue
		resultsList[index] = learnerInspectorObj.learnerType(curLearnerName)

	#if only one algo was requested, remove type from list an return as single string
	if len(resultsList) == 1:
		resultsList = resultsList[0]

	return resultsList


def train(learnerName, trainX, trainY=None, arguments={},  multiClassStrategy='default', useLog=None, **kwarguments):
	"""
	Trains and returns the specified learner using the provided data. The return value is a
	UniversalInterface.trainedLearner object.

	ARGUMENTS:
	
	learnerName: algorithm to be called, in the form 'package.learnerName'.

	trainX: data set to be used for training (as some form of Base object)

	trainY: used to retrieve the known class labels of the traing data. Either
	contains the labels themselves (as a Base object) or an index (numerical or string) 
	that defines their locale in the trainX object

	arguments: dict containing the parameters to be passed to the learner, in the
	form of a mapping between (string) parameter names, and values. Will be merged
	with the contents of **kwarguments before being passed on.

	multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'
	
	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments: The collection of extra key:value argument pairs included in this call to
	train(). They will be merged with the arguments dict, and passed on through to the
	learner.

	"""
	(package, learnerName) = _unpackLearnerName(learnerName)
	_validData(trainX, trainY, None, None, [False, False])
	_validArguments(arguments)
	_validArguments(kwarguments)
	merged = _mergeArguments(arguments, kwarguments)

	if useLog is None:
		useLog = UML.settings.get("logger", "enabledByDefault")
		useLog = True if useLog.lower() == 'true' else False

	if useLog:
		timer = Stopwatch()
	else:
		timer = None

	interface = findBestInterface(package)

	# TODO how do we do multiclassStrategy?

	trainedLearner = interface.train(learnerName, trainX, trainY, merged, timer)

	if useLog:
		funcString = interface.getCanonicalName() + '.' + learnerName
		UML.logger.active.logRun(trainX, trainY, None, None, funcString, None, None, None, timer, extraInfo=merged)

	return trainedLearner

def trainAndApply(learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', useLog=None, **kwarguments):
	"""
	Trains and returns the results of applying the learner to the test data (i.e.
	performing prediction, transformation, etc. as appropriate to the learner).

	ARGUMENTS:
	
	learnerName: algorithm to be called, in the form 'package.learnerName'.

	trainX: data set to be used for training (as some form of Base object)

	trainY: used to retrieve the known class labels of the training data. Either
	contains the labels themselves (as a Base object) or an index (numerical or string) 
	that defines their locale in the trainX object

	testX: data set on which the trained learner will be applied (i.e. performing
	prediction, transformation, etc. as appropriate to the learner). Must be
	some form of UML data Base object. 

	arguments: dict containing the parameters to be passed to the learner, in the
	form of a mapping between (string) parameter names, and values. Will be merged
	with the contents of **kwarguments before being passed on.

	output: The kind of UML data object that the output of this function should be
	in. Any of the normal string inputs to the createData 'retType' parameter are
	accepted here. Alternatively, the value 'match' will indicate to use the type
	of the 'trainX' parameter.

	scoreMode: In the case of a classifying learner, this specifies the type of output
	wanted: 'label' if we class labels are desired, 'bestScore' if both the class
	label and the score associated with that class are desired, or 'allScores' if
	a matrix containing the scores for every class label are desired.

	multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'
	
	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments: The collection of extra key:value argument pairs included in this call to
	train(). They will be merged with the arguments dict, and passed on through to the
	learner.

	"""
	(package, learnerName) = _unpackLearnerName(learnerName)
	fullName = package + '.' + learnerName
	_validData(trainX, trainY, testX, None, [False, False])
	_validScoreMode(scoreMode)
	_validMultiClassStrategy(multiClassStrategy)
	_validArguments(arguments)
	_validArguments(kwarguments)
	merged = _mergeArguments(arguments, kwarguments)

	if testX is None:
		testX = trainX

	if useLog is None:
		useLog = UML.settings.get("logger", "enabledByDefault")
		useLog = True if useLog.lower() == 'true' else False

	if useLog:
		timer = Stopwatch()
	else:
		timer = None

	interface = findBestInterface(package)

	results = None
	if multiClassStrategy != 'default':
		trialResult = checkClassificationStrategy(interface, learnerName, merged)
		# We only use our own version of the strategy if the internal method is different than
		# what we want.
		if multiClassStrategy == 'OneVsAll' and trialResult != 'OneVsAll':
			results = trainAndApplyOneVsAll(fullName, trainX, trainY, testX, arguments=merged, scoreMode=scoreMode, useLog=useLog, timer=timer)
		if multiClassStrategy == 'OneVsOne' and trialResult != 'OneVsOne':
			results = trainAndApplyOneVsOne(fullName, trainX, trainY, testX, arguments=merged, scoreMode=scoreMode, useLog=useLog, timer=timer)

	if results is None:
		results = interface.trainAndApply(learnerName, trainX, trainY, testX, merged, output, scoreMode, timer)

	if useLog:
		funcString = interface.getCanonicalName() + '.' + learnerName
		UML.logger.active.logRun(trainX, trainY, testX, None, funcString, None, results, None, timer, extraInfo=merged)

	return results


def trainAndTest(learnerName, trainX, trainY, testX, testY, performanceFunction, arguments={}, output=None, scoreMode='label', negativeLabel=None, useLog=None, **kwarguments):
	"""
	For each permutation of the merge of 'arguments' and 'kwarguments' (more below),
	trainAndTest uses cross validation to generate a performance score for the algorithm,
	given the particular argument permutation. The argument permutation that performed
	best cross validating over the training data is then used as the lone argument for
	training on the whole training data set. Finally, the learned model generates
	predictions for the testing set, an the performance of those predictions is
	calculated and returned.

	If no additional arguments are supplied via arguments or **kwarguments, then
	trainAndTest just returns the performance of the algorithm with default arguments
	on the testing data.

	ARGUMENTS:

	learnerName: training algorithm to be called, in the form 'package.algorithmName'.

	trainX: data set to be used for training (as some form of Base object)

	trainY: used to retrieve the known class labels of the training data. Either
	contains the labels themselves (as a Base object) or an index (numerical or string) 
	that defines their locale in the trainX object

	testX: data set to be used for testing (as some form of Base object)

	testY: used to retrieve the known class labels of the test data. Either
	contains the labels themselves (as a Base object) or an index (numerical or string) 
	that defines their location in the testX object.

	performanceFunction: Function used by computeMetrics to generate a performance score
	for the run. function is of the form: def func(knownValues, predictedValues, negativeLabel).
	Look in UML.calculate for pre-made options.

	arguments: dict containing the parameters to be passed to the learner, in the
	form of a mapping between (string) parameter names, and values. Will be merged
	with the contents of **kwarguments before being passed on. The syntax for prescribing
	different arguments for algorithm: arguments of the form {arg1=(1,2,3), arg2=(4,5,6)}
	correspond to permutations/argument states with one element from arg1 and one element 
	from arg2, such that an example generated permutation/argument state would be "arg1=2, arg2=4"

	negativeLabel: Argument required if performanceFunction contains proportionPercentPositive90
	or proportionPercentPositive50.  Identifies the 'negative' label in the data set.  Only
	applies to data sets with 2 class labels.

	multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'

	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments: optional arguments to be passed to the specified learner. Will be merged
	with the arguments parameter before being passed on to the learner.
	The syntax for prescribing different arguments for algorithm:
	**kwarguments of the form arg1=(1,2,3), arg2=(4,5,6)
	correspond to permutations/argument states with one element from arg1 and one element 
	from arg2, such that an example generated permutation/argument state would be "arg1=2, arg2=4"

	"""
	(package, trueLearnerName) = _unpackLearnerName(learnerName)
	_validData(trainX, trainY, testX, testY, [True, True])
	_validArguments(arguments)
	_validArguments(kwarguments)
	merged = _mergeArguments(arguments, kwarguments)

	trainY = copyLabels(trainX, trainY)
	testY = copyLabels(testX, testY)

	interface = findBestInterface(package)
	
	if useLog is None:
		useLog = UML.settings.get("logger", "enabledByDefault")
		useLog = True if useLog.lower() == 'true' else False

	#if we are logging this run, we need to start the timer
	if useLog:
		timer = Stopwatch()
		timer.start('crossValidateReturnBest')
	else:
		timer = None
	#sig (learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label', negativeLabel=None, useLog=None, maximize=False, **kwarguments):
	# we explicitly set useLog=False here, 
	bestArgument, bestScore = UML.crossValidateReturnBest(learnerName, trainX, trainY, performanceFunction, merged, scoreMode=scoreMode, useLog=useLog)

	if useLog:
		timer.stop('crossValidateReturnBest')

	predictions = interface.trainAndApply(trueLearnerName, trainX, trainY, testX, arguments=bestArgument, output=output, scoreMode=scoreMode, timer=timer)
	performance = UML.helpers.computeMetrics(testY, None, predictions, performanceFunction, negativeLabel)

	if useLog:
		funcString = interface.getCanonicalName() + '.' + learnerName
		UML.logger.active.logRun(trainX, trainY, testX, testY, funcString, [performanceFunction], predictions, [performance], timer, bestArgument)

	return performance
