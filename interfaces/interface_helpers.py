"""
Utility functions that could be useful in multiple interfaces

"""

import sys
import UML
from UML.processing import BaseData


def makeArgString(wanted, argDict, prefix, infix, postfix):
	"""
	Construct and return a long string containing argument and value pairs,
	each separated and surrounded by the given strings. If wanted is None,
	then no args are put in the string

	"""
	argString = ""
	if wanted is None:
		return argString
	for arg in wanted:
		if arg in argDict:
			value = argDict[arg]
			if isinstance(value,basestring):
				value = "\"" + value + "\""
			else:
				value = str(value)
			argString += prefix + arg + infix + value + postfix
	return argString


#TODO what about multiple levels???
def findModule(algorithmName, packageName, packageLocation):
	"""
	Import the desired python package, and search for the module containing
	the wanted algorithm. For use by interfaces to python packages.

	"""

	putOnSearchPath(packageLocation)
	exec("import " + packageName)

	contents = eval("dir(" + packageName + ")")

	# if all is defined, then we defer to the package to provide
	# a sensible list of what it contains
	if "__all__" in contents:
		contents = eval(packageName + ".__all__")

	for moduleName in contents:
		if moduleName.startswith("__"):
			continue
		cmd = "import " + packageName + "." + moduleName
		try:		
			exec(cmd)
		except ImportError as e:
			continue
		subContents = eval("dir(" + packageName + "." + moduleName + ")")
		if ".__all__" in subContents:
			contents = eval(packageName+ "." + moduleName + ".__all__")
		if algorithmName in subContents:
			return moduleName

	return None



def putOnSearchPath(wantedPath):
	if wantedPath is None:
		return
	elif wantedPath in sys.path:
		return
	else:
		sys.path.append(wantedPath)


def pythonIOWrapper(algorithm, trainData, testData, output, dependentVar, arguments, kernel, config):

	inType = config['inType']
	inTypeLabels = config['inTypeLabels']
	
	toCall = config['toCall']
	checkPackage = config['checkPackage']
	pythonOutType = config['pythonOutType']
	fileOutType = config['fileOutType']


	if not isinstance(trainData, BaseData):
		trainObj = UML.data(inType, data=trainData)
	else: # input is an object
		trainObj = convertTo(trainObj, inType)
	if not isinstance(testData, BaseData):
		testObj = UML.data(inType, data=testData)
	else: # input is an object
		testObj = convertTo(testObj, inType)
	
	trainObjY = None
	# directly assign target values, if present
	if isinstance(dependentVar, BaseData):
		trainObjY = dependentVar
	# otherwise, isolate the target values from training examples
	elif dependentVar is not None:
		trainCopy = trainObj.duplicate()
		trainObjY = trainCopy.extractFeatures([dependentVar])		
	# could be None for unsupervised learning, in which case it remains none

	# get the correct type for the labels
	if trainObjY is not None:	
		trainObjY = convertTo(trainObjY, inTypeLabels)
	
	# pull out data from obj
	trainObj.transpose()
	trainRawData = trainObj.data
	if trainObjY is not None:
		# corrects the dimensions of the matrix data to be just an array
		trainRawDataY = numpy.array(trainObjY.data).flatten()
	else:
		trainRawDataY = None
	testObj.transpose()
	testRawData = testObj.data

	# call backend
	try:
		retData = toCall(algorithm,  trainRawData, trainRawDataY, testRawData, arguments, kernel)
	except ImportError as e:
		if not checkPackage():
			print "Package was not importable."
			print "It must be either on the search path, or have its location set in UML"
		raise e

	if retData is None:
		return

	outputObj = UML.data(pythonOutType, data=retData)

	if output is None:
		# we want to return a column vector
		outputObj.transpose()
		return outputObj

	outputObj.writeFile(fileOutType, output, False)



def convertTo(data, retType):
	return eval("data.to" + retType + "()")

def generateAllPairs(items):
	"""
		Given a list of items, generate a list of all possible pairs 
		(2-combinations) of items from the list, and return as a list
		of tuples.  Assumes that no two items in the list refer to the same
		object or number.  If there are duplicates in the input list, there
		will be duplicates in the output list.
	"""
	if items is None or len(items) == 0:
		return None

	pairs = []
	for i in range(len(items)):
		firstItem = items[i]
		for j in range(i+1, len(items)):
			secondItem = items[j]
			pair = (firstItem, secondItem)
			pairs.append(pair)

	return pairs

def testGenerateAllPairs():
	"""
	Unit test function for testGenerateAllPairs
	"""
	testList1 = [1, 2, 3, 4]
	testPairs = generateAllPairs(testList1)
	print testPairs

	assert len(testPairs) == 6
	assert ((1, 2) in testPairs) or ((2, 1) in testPairs)
	assert not (((1, 2) in testPairs) and ((2, 1) in testPairs))
	assert ((1, 3) in testPairs) or ((3, 1) in testPairs)
	assert not (((1, 3) in testPairs) and ((3, 1) in testPairs))
	assert ((1, 4) in testPairs) or ((4, 1) in testPairs)
	assert not (((1, 4) in testPairs) and ((4, 1) in testPairs))
	assert ((2, 3) in testPairs) or ((3, 2) in testPairs)
	assert not (((2, 3) in testPairs) and ((3, 2) in testPairs))
	assert ((2, 4) in testPairs) or ((4, 2) in testPairs)
	assert not (((2, 4) in testPairs) and ((4, 2) in testPairs))
	assert ((3, 4) in testPairs) or ((4, 3) in testPairs)
	assert not (((3, 4) in testPairs) and ((4, 3) in testPairs))

	testList2 = []
	testPairs2 = generateAllPairs(testList2)
	assert testPairs2 is None

