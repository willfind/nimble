"""
Wrapper module, allowing for easy access to the major functionality of the package that might
later be extended with more features, packages, or datatypes.

"""

import numpy
import scipy.io
import time
import os.path

from .interfaces import mahout
from .interfaces import regressor
from .interfaces import sciKitLearn
from .interfaces import mlpy
from .processing import CooSparseData
from .processing import DenseMatrixData
from .processing import RowListData
from .logging.log_manager import LogManager
from .logging.stopwatch import Stopwatch
from .utility import ArgumentException


def run(algorithm, trainData, testData, output=None, dependentVar=None, arguments={}, sendToLog=True):
	splitList = algorithm.split('.',1)
	if len(splitList) < 2:
		raise ArgumentException("The algorithm must be prefeced with the package name and a dot. Example:'mlpy.KNN'")
	package = splitList[0]
	algorithm = splitList[1]

	if sendToLog:
		timer = Stopwatch()

	if package == 'mahout':
		results = mahout(algorithm, trainData, testData, output, dependentVar, arguments, timer)
	elif package == 'regressor':
		results = regressor(algorithm, trainData, testData, output, dependentVar, arguments, timer)
	elif package == 'sciKitLearn':
		results = sciKitLearn(algorithm, trainData, testData, output, dependentVar, arguments, timer)
	elif package == 'mlpy':
		results = mlpy(algorithm, trainData, testData, output, dependentVar, arguments, timer)
	elif package == 'self':
		raise ArgumentException("self modification not yet implemented")
	else:
		raise ArgumentException("package not recognized")

	if sendToLog:
			logManager = LogManager()
			if package == 'regressor':
				funcString = 'regressors.' + algorithm
			else:
				funcString = package + '.' + algorithm
			logManager.logRun(trainData, testData, funcString, None, timer, extraInfo=arguments)

	return results


def data(retType, data=None, featureNames=None, fileType=None, name=None):
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
		ret = _loadCoo(data, featureNames, fileType)
	elif retType in dmdAlias:
		ret = _loadDMD(data, featureNames, fileType)
	elif retType in rldAlias:
		ret = _loadRLD(data, featureNames, fileType)
	else:
		raise ArgumentException("Unknown data type, cannot instantiate")

	if name is not None:
		ret.setName(name)
	return ret


def _loadCoo(data, featureNames, fileType):
	if fileType is None:
		return CooSparseData(data, featureNames)

	# since file type is not None, that is an indicator that we must read from a file
	path = data
	tempFeatureNames = None
	if fileType == 'csv':
		(data, tempFeatureNames) = _loadCSVtoMatrix(path)
	elif fileType == 'mtx':
		(data, tempFeatureNames) = _loadMTXtoAuto(path)
	else:
		raise ArgumentException("Unrecognized file type")

	if tempFeatureNames is not None:
			featureNames = tempFeatureNames
	return CooSparseData(data, featureNames, os.path.basename(path), path)


def _loadDMD(data, featureNames, fileType):
	if fileType is None:
		return DenseMatrixData(data, featureNames)

	# since file type is not None, that is an indicator that we must read from a file
	path = data
	tempFeatureNames = None
	if fileType == 'csv':
		(data, tempFeatureNames) = _loadCSVtoMatrix(path)
	elif fileType == 'mtx':
		(data, tempFeatureNames) = _loadMTXtoAuto(path)
	else:
		raise ArgumentException("Unrecognized file type")

	if tempFeatureNames is not None:
			featureNames = tempFeatureNames
	return DenseMatrixData(data, featureNames, os.path.basename(path), path)


def _loadRLD(data, featureNames, fileType):
	if fileType is None:
		return RowListData(data, featureNames)

	# since file type is not None, that is an indicator that we must read from a file
	path = data
	tempFeatureNames = None
	if fileType == 'csv':
		(data, tempFeatureNames) =_loadCSVtoList(data)
	elif fileType == 'mtx':
		(data, tempFeatureNames) = _loadMTXtoAuto(data)
	else:
		raise ArgumentException("Unrecognized file type")

	# if we load from file, we assume that data will be read; feature names may not
	# be, thus we check
	if tempFeatureNames is not None:
			featureNames = tempFeatureNames

	return RowListData(data, featureNames, os.path.basename(path), path)

def _loadCSVtoMatrix(path):
	inFile = open(path, 'rU')
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

	# check the types in the first data containing line.
	line = firstLine
	while (line == "") or (line[0] == '#'):
		line = inFile.readline()
	lineList = line.split(',')
	for datum in lineList:
		try:
			num = numpy.float(datum)
		except ValueError:
			raise ValueError("Cannot load a file with non numerical typed columns")

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
	inFile = open(path, 'rU')
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

def _intFloatOrString(inString):
	ret = inString
	try:
		ret = int(inString)
	except exceptions.ValueError:
		ret = float(inString)
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
	inFile = open(path, 'rU')
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
		inFile = open(path, 'rU')

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
