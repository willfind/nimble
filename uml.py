"""
Wrapper module, allowing for easy access to the major functions of the package from the
root.

"""

import numpy
import scipy.io

from .interfaces import mahout
from .interfaces import regressor
from .interfaces import sciKitLearn
from .interfaces import mlpy
from .processing import CooSparseData
from .processing import DenseMatrixData
from .processing import RowListData
from .utility import ArgumentException


def run(package, algorithm, trainData, testData, output=None, dependentVar=None, arguments={}):
	if package == 'mahout':
		return mahout(algorithm, trainData, testData, output, dependentVar, arguments)
	if package == 'regressor':
		return regressors(algorithm, trainData, testData, output, dependentVar, arguments)
	if package == 'sciKitLearn':
		return sciKitLearn(algorithm, trainData, testData, output, dependentVar, arguments)
	if package == 'mlpy':
		return mlpy(algorithm, trainData, testData, output, dependentVar, arguments)


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


def data(retType, data=None, featureNames=None, fileType=None):
	# determine if its a file we have to read; we assume if its a string its a path
	if isinstance(data, basestring):
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
	# if data is not a path to a file, then we don't care about the value of this flag;
	# instead we use it as an indicator flag to directly instantiate
	else:
		fileType = None

	# these should be lowercase to avoid ambiguity
	retType = retType.lower()
	cooAlias = ["coosparsedata", "coo", "coosparse"]
	dmdAlias = ["densematrixdata", 'dmd', 'dense']
	rldAlias = ["rowlistdata", 'rld']
	if retType in cooAlias:
		return _loadCoo(data, featureNames, fileType)
	elif retType in dmdAlias:
		return _loadDMD(data, featureNames, fileType)
	elif retType in rldAlias:
		return _loadRLD(data, featureNames, fileType)
	else:
		raise ArgumentException("Unknown data type, cannot instantiate")


# def runWithPerformance()  # same as run, with extra parameter?


def crossValidate(X, Y, functionsToApply, numFolds=10):
	return cvImplementation(X, Y, functionsToApply, numFolds)


#combinations() -- maybe

#listAllAlgorithms()




def _loadCoo(data, featureNames, fileType):
	tempFeatureNames = None
	if fileType == 'csv':
		(data, tempFeatureNames) = _loadCSVtoMatrix(data)
	elif fileType == 'mtx':
		(data, tempFeatureNames) = _loadMTXtoAuto(data)
	elif fileType is not None:
		raise ArgumentException("Unrecognized file type")

	if tempFeatureNames is not None:
			featureNames = tempFeatureNames
	return CooSparseData(data, featureNames)


def _loadDMD(data, featureNames, fileType):
	tempFeatureNames = None
	if fileType == 'csv':
		(data, tempFeatureNames) = _loadCSVtoMatrix(data)
	elif fileType == 'mtx':
		(data, tempFeatureNames) = _loadMTXtoAuto(data)
	elif fileType is not None:
		raise ArgumentException("Unrecognized file type")

	if tempFeatureNames is not None:
			featureNames = tempFeatureNames
	return DenseMatrixData(data, featureNames)


def _loadRLD(data, featureNames, fileType):
	tempFeatureNames = None
	if fileType == 'csv':
		(data, tempFeatureNames) =_loadCSVtoList(data)
	elif fileType == 'mtx':
		(data, tempFeatureNames) = _loadMTXtoAuto(data)
	elif fileType is not None:
		raise ArgumentException("Unrecognized file type")

	# if we load from file, we assume that data will be read; feature names may not
	# be, thus we check
	if tempFeatureNames is not None:
			featureNames = tempFeatureNames

	return RowListData(data, featureNames)

def _loadCSVtoMatrix(path):
	inFile = open(path, 'r')
	firstLine = inFile.readline()
	featureNames = None
	skip_header = 0

	# test if this is a line defining featureNames
	if firstLine[0] == "#":
		# strip '#' from the begining of the line
		scrubbedLine = firstLine[1:]
		# strip newline from end of line
		scrubbedLine = scrubbedLine.rstrip()
		featureNames = scrubbedLine.split(',')
		skip_header = 1

	inFile.close()

	data = numpy.genfromtxt(path, delimiter=',', skip_header=skip_header)
	return (data, featureNames)

def _loadMTXtoAuto(path):
	"""
	Uses scipy helpers to read a matrix market file; returning whatever is most
	appropriate for the file. If it is a matrix market array type, a numpy
	dense matrix is returned as data, if it is a matrix market coordinate type, a
	sparse scipy coo_matrix is returned as data. If featureNames are present,
	they are also read.

	"""
	inFile = open(path, 'r')
	featureNames = None

	# read through the comment lines
	while True:
		currLine = inFile.readline()
		if currLine[0] != '%':
			break
		if len(currLine) > 1 and currLine[1] == "#":
			# strip '%#' from the begining of the line
			scrubbedLine = currLine[2:]
			# strip newline from end of line
			scrubbedLine = scrubbedLine.rstrip()
			featureNames = scrubbedLine.split(',')

	inFile.close()

	data = scipy.io.mmread(path)
	return (data, featureNames)

def _intFloatOrString(str):
	ret = str
	try:
		ret = int(str)
	except exceptions.ValueError:
		ret = float(str)
	# this will return an int or float if either of the above two are successful
	finally:
		return ret

def _defaultParser(line):
	"""
	When given a comma separated value line, it will attempt to convert
	the values first to int, then to float, and if all else fails will
	keep values as strings. Returns list of values.

	"""
	ret = []
	lineList = line.split(',')
	for entry in lineList:
		ret.append(_intFloatOrString(entry))
	return ret


def _loadCSVtoList(path):
	inFile = open(path, 'r')
	firstLine = inFile.readline()
	featureNameList = None

	# test if this is a line defining featureNames
	if firstLine[0] == "#":
		# strip '#' from the begining of the line
		scrubbedLine = firstLine[1:]
		# strip newline from end of line
		scrubbedLine = scrubbedLine.rstrip()
		featureNameList = scrubbedLine.split(',')
		featureNameMap = {}
		for name in featureNameList:
			featureNameMap[name] = featureNameList.index(name)
	#if not, get the iterator pointed back at the first line again	
	else:
		inFile.close()
		inFile = open(path, 'r')

	#list of datapoints in the file, where each data point is a list
	data = []
	for currLine in inFile:
		currLine = currLine.rstrip()
		#ignore empty lines
		if len(currLine) == 0:
			continue

		data.append(_defaultParser(currLine))

	if featureNameList == None:
		return (data, None)

	inFile.close()

	return (data, featureNameMap)
