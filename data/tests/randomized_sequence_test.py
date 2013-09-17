"""
Implementation and support functions for a single randomized
test which performs a sequence of operations on each
implemented type of data representation, comparing both the
results, and an approximately equal representation of their
contained values.

"""

import random
#import numpy.random
import inspect

import UML
from UML.data import Base


numberOperations = 10
numPoints = 50
numFeatures = 10
unTestedMethods = ['nameData', 'writeFile', 'getTypeString', 'report',

	'extractPointsByCoinToss',
#	'replaceFeatureWithBinaryFeatures',
	'transformFeartureToIntegerFeature',  'appendFeatures', 'appendPoints', 
	'featureView',
	'pointView',
	
	'extractFeatures', 'extractPoints', # overloaded params, mutally exclusive params
	'referenceDataFrom',  # actually broken??
	'sortFeatures', 'sortPoints',  # takes a funciton
	'transformPoint', 'transformFeature',  # takes a funciton
	'applyToPoints', 'applyToElements', # takes a funciton
	'dropFeaturesContainingType', # coosparse cannot return emptry from extract
	'foldIterator', 'featureIterator', 'pointIterator', #iterator equality
	'featureReport', # floating point equality errors?
	'applyToFeatures', 'mapReducePoints',  # takes a funciton
	'copyFeatures', 'copyPoints', # cannot specify both points and a range -- our code provides both
	]

nameToType = {}
nameToType['assignments'] = 'blist_string'
nameToType['randomize'] = 'boolean'
nameToType['start'] = 'bint'
nameToType['end'] = 'bint'
nameToType['number'] = 'int'
nameToType['point'] = 'bint'
nameToType['numFolds'] = 'int'
nameToType['displayDigits'] = 'int'
nameToType['toExtract'] = 'blist_bint'
nameToType['points'] = 'blist_bint'
nameToType['features'] = 'blist_fname'
nameToType['sortBy'] = 'ID'
nameToType['oldIdentifier'] = 'ID'
nameToType['feature'] = 'ID'
nameToType['ID'] = 'ID'
nameToType['featureToConvert'] = 'fname'
nameToType['newFeatureName'] = 'string'
nameToType['extractionProbability'] = 'float'
nameToType['toCopy'] = 'bobject'
nameToType['other'] = 'object'
nameToType['toAppend'] = 'shaped_bobject'
nameToType['otherDataMatrix'] = 'object'
nameToType['seed'] = 'default'

#FUNCTION:
#sortHelper(view -> value)
#function(value -> value)
#function(view -> value)



def testRandomSequenceOfMethods():
	# always use this number of points and features
	points = numPoints
	features = numFeatures

	return

	# dense int trial
	sparcity = 0.05
	objectList = []
	first = UML.createRandomizedData('List', points, features, sparcity, 'int')
	objectList.append(first)
	objectList.append(UML.createData('Matrix', first.data))
	objectList.append(UML.createData('Sparse', first.data))
	runSequence(objectList)

	return

	## dense float trial
	sparcity = 0.05
	objectList = []
	first = UML.createRandomizedData('List', points, features, sparcity, 'float')
	objectList.append(first)
	objectList.append(UML.createData('Matrix', first.data))
	objectList.append(UML.createData('Sparse', first.data))
	runSequence(objectList)

	# sparse int trial
	sparcity = 0.9
	objectList = []
	first = UML.createRandomizedData('List', points, features, sparcity, 'int')
	objectList.append(first)
	objectList.append(UML.createData('Matrix', first.data))
	objectList.append(UML.createData('Sparse', first.data))
	runSequence(objectList)

	# sparse float trial
	sparcity = 0.9
	objectList = []
	first = UML.createRandomizedData('List', points, features, sparcity, 'float')
	objectList.append(first)
	objectList.append(UML.createData('Matrix', first.data))
	objectList.append(UML.createData('Sparse', first.data))
	runSequence(objectList)


def runSequence(objectList):
	# setup method list, from Base dir
	availableMethods = setupMethodList()

	# loop over specified number of operations
#	for i in xrange(numberOperations):
	for i in xrange(len(availableMethods)):
		# random number as index into available operations list
#		index = random.randint(0,len(availableMethods)-1)
		index = i
		currName = availableMethods[index]
#		if currName != 'pointView':
#			continue
		print currName

		# call method on each object, collect results
		results = []
		for dataObj in objectList:
			# get param names
			funcToCheck = eval('Base.' + currName)
			(args, varargs, keywords, defaults) = inspect.getargspec(funcToCheck)
			params = []
			for paramName in args:
				if paramName != 'self':
					paramValue = generateRandomParameter(i, dataObj, paramName, currName)
					if paramValue != 'default':
						params.append(paramValue)
					else:
						pass
						# need to query defaults

			toEval = 'dataObj.' + currName + '('
			for i in xrange(len(params)):
				toEval += 'params[' + str(i) + '],'
			toEval += ')'
			currResult = eval(toEval)
			results.append(currResult)	

		# need to check equality of results
		for i in xrange(len(results)):
			ithResult = results[i]
			for j in xrange(i+1, len(results)):
				jthResult = results[j]
				equalityWrapper(ithResult, jthResult)

		# call hash equal on each of the objects
		for i in xrange(len(objectList)):
			ithDataObject = objectList[i]
			for j in xrange(i+1, len(objectList)):
				jthDataObject = objectList[j]
				assert ithDataObject.isApproximatelyEqual(jthDataObject)



def equalityWrapper(left, right):
	""" takes two parameters and uses the most probable equality function to
	assert that they are equal.
	It is important to note that the preference of equality checks is
	strictly ordered: we prefer using isApproximatelyEqual() more than
	isIdentical(), more than equality of contained, iterable objects,
	more than == equality. """

	if hasattr(left, 'isApproximatelyEqual'): 
		assert left.isApproximatelyEqual(right)
	elif hasattr(left, 'isIdentical'):
		assert left.isIdentical(right)
	elif hasattr(left, '__iter__'):
		leftIter = left.__iter__()
		rightIter = right.__iter__()
		leftList = []
		rightList = []
		try:
			while(True): leftList.append(leftIter.next())
		except StopIteration: pass
		try:
			while(True): rightList.append(rightIter.next())
		except StopIteration: pass

		assert len(leftList) == len(rightList)
		for i in xrange(len(leftList)):
			currLeft = leftList[i]
			currRight = rightList[i]
			equalityWrapper(currLeft, currRight)
	else:
		assert left == right




def setupMethodList():
	"""
	Uses the outputs of dir and subtracts methods we know we don't want to
	test from the available callable methods. Returns a list of names of
	methods, not the methods themselves.

	"""
	allMethods = dir(Base)
	ret = []
	for name in allMethods:
		if name.startswith('_'):
			continue
		if name in unTestedMethods:
			continue
		ret.append(name)

	return ret




def generateRandomParameter(numCall, dataObject, paramName, funcName):
	""" uses the parameter name and context from the given object to produce
	an appropriate random value for the parameter
	"""
	paramType = nameToType[paramName]
	ret = None

	if paramType == 'boolean':
		ret = random.randint(0,1) == 1
	elif paramType == 'int':
		ret = random.randint(0,49) 
	elif paramType == 'bint':
		ret = randBInt(dataObject, funcName)
	elif paramType == 'blist_bint':
		ret = 'list_bint'
	elif paramType == 'blist_fname':
		ret = 'list_fname'
	elif paramType == 'blist_string':
		ret = randBList_String(numCall, dataObject, funcName)
	elif paramType =='string':
		ret = randString(numCall)
	elif paramType == 'ID':
		if 'point' in funcName.lower():
			ret = randBInt(dataObject, funcName)
		else:
			ret = randFName(dataObject)
	elif paramType == 'fname':
		ret = randFName(dataObject)
	elif paramType == 'float':
		ret = random.random()
	elif paramType == 'object':
		ret = randObject(dataObject, False)
	elif paramType == 'bobject':
		ret = randObject(dataObject, True)
	elif paramType == 'shaped_object':		
		ret = 'shaped_object'

	return ret


def randString(numCall):
	return str(numCall) + '_' + str(random.randint(0,50))

def randBList_String(numCall, dataObject, funcName):
	ret = []
	if 'feature' in funcName.lower():
		length = dataObject.featureCount
	else:
		length = dataObject.featureCount

	for i in xrange(length):
		randString = str(numCall) + '_' + str(i)
		ret.append(randString)

	return ret

def randBList_FName(numCall, dataObject, funcName):
	ret = []
	if 'feature' in funcName.lower():
		length = dataObject.featureCount
	else:
		length = dataObject.featureCount

	for i in xrange(length):
		randString = str(numCall) + '_' + str(i)
		ret.append(randString)

	return ret

def randBoolean():
	return random.randint(0,1) == 1

def randBInt(dataObject, funcName):
	if 'point' in funcName.lower():
		cap = dataObject.pointCount - 1
	else:
		cap = dataObject.featureCount - 1
	return random.randint(0,cap)

def randFName(dataObject):
	numF = dataObject.featureCount - 1
	return dataObject.featureNamesInverse[random.randint(0,numF)]

def randObject(dataObject, sameType):
	if sameType:
		dataType = dataObject.getTypeString()
	else:
		trial = random.randint(0,2)
		if trial == 0:
			dataType = 'List'
		elif trial == 1:
			dataType = 'Matrix'
		else:
			dataType = 'Sparse'

	return UML.createRandomizedData(dataType, 50, 10, .5, 'int')


def randShapedObject(dataObject):
	trial = random.randint(0,2)
	dataType = dataObject.getTypeString()
	return UML.createRandomizedData('List', 50, 10, .5, 'int')









