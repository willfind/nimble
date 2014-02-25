"""
Module containing most of the user facing functions for the top level uml import.

"""

import random
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
from UML.logger import LogManager
from UML.logger import Stopwatch

from UML.runners import run

from UML.umlHelpers import _loadSparse
from UML.umlHelpers import _loadMatrix
from UML.umlHelpers import _loadList
from UML.umlHelpers import executeCode
from UML.umlHelpers import _incrementTrialWindows
from UML.umlHelpers import _learningAlgorithmQuery


UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def createRandomizedData(retType, numPoints, numFeatures, sparcity, numericType="int", featureNames=None, name=None):
	if numPoints < 1:
		raise ArgumentException("must specify a positive nonzero number of points")
	if numFeatures < 1:
		raise ArgumentException("must specify a positive nonzero number of features")
	if sparcity < 0 or sparcity >=1:
		raise ArgumentException("sparcity must be greater than zero and less than one")
	if numericType != "int" and numericType != "float":
		raise ArgumentException("numericType may only be 'int' or 'float'")

	randData = numpy.zeros((numPoints,numFeatures))
	for i in xrange(numPoints):
		for j in xrange(numFeatures):
			if random.random() > sparcity:
				if numericType == 'int':
					randData[i,j] = numpy.random.randint(1,100)
				else:
					randData[i,j] = numpy.random.rand()

	return createData(retType, data=randData, featureNames=featureNames, name=name)


def splitData(toSplit, fractionForTestSet, labelID=None):
	"""this is a helpful function that makes it easy to do the common task of loading a dataset and splitting it into training and testing sets.
	It returns training X, training Y, testing X and testing Y"""
	testXSize = int(round(fractionForTestSet*toSplit.pointCount))
	#pull out a testing set
	testX = toSplit.extractPoints(start=0, end=toSplit.pointCount, number=testXSize, randomize=True)	
	trainY = None
	testY = None
	if labelID is not None:
		trainY = toSplit.extractFeatures(labelID)	#construct the column vector of training labels
		testY = testX.extractFeatures(labelID)	#construct the column vector of testing labels
	return toSplit, trainY, testX, testY



def normalizeData(learningAlgorithm, trainX, trainY=None, testX=None, arguments={}, mode=True):
	"""
	Calls on the functionality of a package to train on some data and then modify both
	the training data and a set of test data accroding to the produced model.

	"""
	# single call normalize, combined data
	if mode and testX is not None:
		testLength = testX.pointCount
		# glue training data at the end of test data
		testX.appendPoints(trainX)
		try:
			normalizedAll = run(learningAlgorithm, trainX, trainY, testX, arguments=arguments)
		except ArgumentException:
			testX.extractPoints(start=testLength, end=normalizedAll.pointCount)
		# resplit normalized
		normalizedTrain = normalizedAll.extractPoints(start=testLength, end=normalizedAll.pointCount)
		normalizedTest = normalizedAll
	# two call normalize, no data combination
	else:
		normalizedTrain = run(learningAlgorithm, trainX, trainY, trainX, arguments=arguments)
		if testX is not None:
			normalizedTest = run(learningAlgorithm, trainX, trainY, testX, arguments=arguments)
		
	# modify references for trainX and testX
	trainX.referenceDataFrom(normalizedTrain)
	if testX is not None:
		testX.referenceDataFrom(normalizedTest)


def learningAlgorithmParameters(name):
	"""
	Takes a string of the form 'package.learningAlgorithm' and returns a list of
	strings which are the names of the parameters when calling package.learningAlgorithm

	If the name cannot be found within the package, then an exception will be thrown.
	If the name is found, be for some reason we cannot determine what the parameters
	are, then we return None. Note that if we have determined that there are no
	parameters, we return an empty list. 

	"""
	return _learningAlgorithmQuery(name, 'parameters')

def learningAlgorithmDefaultValues(name):
	"""
	Takes a string of the form 'package.learningAlgorithm' and returns a returns a
	dict mapping of parameter names to their default values when calling
	package.learningAlgorithm

	If the name cannot be found within the package, then an exception will be thrown.
	If the name is found, be for some reason we cannot determine what the parameters
	are, then we return None. Note that if we have determined that there are no
	parameters, we return an empty dict. 

	"""
	return _learningAlgorithmQuery(name, 'defaults')


def listLearningAlgorithms(package=None):
	listAll = False
	if package is not None:
		if not isinstance(package, basestring):
			raise ArgumentException("package may only be None (to list all learning functions), or the string name of a package")
		package = package.lower()
		available = ['mahout', 'regressor', 'scikitlearn', 'mlpy', 'shogun']
		if not package in available:
			raise ArgumentException("unrecognized package, only allowed are: " + str(available))
	else:
		listAll = True
	results = None
	allResults = []
	def addToAll(packageName, toAdd, toAppendTo):
		for funcName in toAdd:
			toAppendTo.append(packageName + '.' + funcName)
	if package == 'mahout' or listAll:
		import UML.interfaces.mahout_interface
		results = UML.interfaces.mahout_interface.listMahoutLearningAlgorithms()
		if listAll:
			addToAll('mahout', results, allResults)
	if package == 'regressor' or listAll:
		import UML.interfaces.regressors_interface
		results = UML.interfaces.regressors_interface.listRegressorLearningAlgorithms()
		if listAll:
			addToAll('regressor', results, allResults)
	if package == 'scikitlearn' or listAll:
		import UML.interfaces.scikit_learn_interface
		results = UML.interfaces.scikit_learn_interface.listSciKitLearnLearningAlgorithms()
		if listAll:
			addToAll('sciKitLearn', results, allResults)
	if package == 'mlpy' or listAll:
		import UML.interfaces.mlpy_interface
		results = UML.interfaces.mlpy_interface.listMlpyLearningAlgorithms()
		if listAll:
			addToAll('mlpy', results, allResults)
	if package == 'shogun' or listAll:
		import UML.interfaces.shogun_interface
		results = UML.interfaces.shogun_interface.listShogunLearningAlgorithms()
		if listAll:
			addToAll('shogun', results, allResults)

	if listAll:
		return allResults
	else:
		return results

def listDataFunctions():
	methodList = dir(UML.data.Base)
	visibleMethodList = []
	for methodName in methodList:
		if not methodName.startswith('_'):
			visibleMethodList.append(methodName)

	ret = []
	for methodName in visibleMethodList:
		currMethod = eval("UML.data.Base." + methodName)
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


def createData(retType, data=None, featureNames=None, fileType=None, name=None, sendToLog=True):
	automatedRetType = False
	# determine if its a file we have to read; we assume if its a string its a path
	if isinstance(data, basestring):
		#we may log this event
		if sendToLog is not None and sendToLog is not False:
			if isinstance(sendToLog, UmlLogger):
				logger = sendToLog
			else:
				logger = LogManager()
			#logger.logLoad(retType, data, name)
		# determine the extension, call the appropriate helper
		split = data.rsplit('.', 1)
		extension = None
		if len(split) > 1:
			extension = split[1].lower()
		# if fileType is specified, it takes priority, otherwise, use the extension
		if fileType is None:
			fileType = extension
		if fileType is None:
			raise ArgumentException("The file must be recognizable by extension, or a type must be specified")
		
		if retType is None:
			automatedRetType = True
			if fileType == 'csv':
				retType = 'Matrix'
			if fileType == 'mtx':
				retType = 'Sparse'

	# if data is not a path to a file, then we don't care about the value of this flag;
	# instead we use it as an indicator flag to directly instantiate
	else:
		fileType = None
		if retType is None:
			if isinstance(data, list):
				retType = 'List'
			if isinstance(data, numpy.matrix):
				retType = 'Matrix'
			if isinstance(data, numpy.array):
				retType = 'Matrix'
			if scipy.sparse.issparse(data):
				retType = 'Sparse'


	# these should be lowercase to avoid ambiguity
	retType = retType.lower()
	sparseAlias = ["sparse"]
	matrixAlias = ['matrix']
	listAlias = ["list"]
	if retType in sparseAlias:
		ret = _loadSparse(data, featureNames, fileType, automatedRetType)
	elif retType in matrixAlias:
		ret = _loadMatrix(data, featureNames, fileType, automatedRetType)
	elif retType in listAlias:
		ret = _loadList(data, featureNames, fileType)
	else:
		msg = "Unknown data type, cannot instantiate. Only allowable inputs: "
		msg += "'List' for data in python lists, 'Matrix' for a numpy matrix, "
		msg += "'Sparse' for a scipy sparse coo_matrix, and None for automated choice."
		raise ArgumentException(msg)

	if name is not None:
		ret.nameData(name)
	return ret






