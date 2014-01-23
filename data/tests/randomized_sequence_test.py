"""
Implementation and support functions for a single randomized
test which performs a sequence of operations on each
implemented type of data representation, comparing both the
results, and an approximately equal representation of their
contained values.

"""

import random
import inspect
import pdb

import UML
from UML.data import Base
from UML.data import View
from UML.exceptions import ArgumentException


numberOperations = 100
numPoints = 4
numFeatures = 4
unTestedMethods = [
	# Core passing
	# -- but broken in some cases
#	'transformFeatureToIntegerFeature', 
#	'replaceFeatureWithBinaryFeatures',
#	'appendPoints',
#	'appendFeatures',
#	'extractPointsByCoinToss',
	'shufflePoints', 'shuffleFeatures',
	'applyToPoints','applyToFeatures',
	'pointView', 'featureView',
#	'mapReducePoints', 
#	'sortPoints', 'sortFeatures',   # takes a function
	'extractFeatures', 'extractPoints', # overloaded params, mutually exclusive params
	'copyFeatures', 'copyPoints', # cannot specify both points and a range -- our code provides both
#	'transpose', # Sparse is broken?
	'copy', #something about numpy equality?
	'referenceDataFrom',  # often causes contradictory shapes to be seen later

	'applyToElements', # sometimes hangs????
	
	# Final exclusion List
	'dropFeaturesContainingType', # how can you choose the type?
	'featureIterator', 'pointIterator', #iterator equality isn't a sensible thing to check
	'writeFile', # lets not test this yet
	'getTypeString', # won't actually be equal
	'summaryReport', # do we really care about testing this?
	'featureReport', # floating point equality errors? / do we care?
	'foldIterator', # deprecated.
	'pointCount', 'featureCount' # not runnable
	]

unavailableNoPoints = [
	'replaceFeatureWithBinaryFeatures',
	'transformFeatureToIntegerFeature',
	'applyToPoints',
	'applyToFeatures',
	'featureIterator',
	'applyToEachElement',
	'shufflePoints',
	'shuffleFeatures',
	'writeFile',
	'copyPoints',
	'pointView',
	]

unavailableNoFeatures = [
	'setFeatureName',
	'setFeatureNamesFromList',
	'setFeatureNamesFromDict',
	'replaceFeatureWithBinaryFeatures',
	'transformFeatureToIntegerFeature',
	'applyToPoints',
	'applyToFeatures',
	'mapReducePoints',
	'pointIterator',
	'applyToEachElement',
	'shufflePoints',
	'shuffleFeatures',
	'writeFile',
	'copyFeatures',
	'featureView',
	]


mutuallyDependentParams = {}
mutuallyDependentParams['start'] = ('le', 'end')
mutuallyDependentParams['end'] = ('ge', 'start')

mutuallyExclusiveParams = {}
mutuallyExclusiveParams['sortPoints'] = ('sortBy', 'sortHelper')
mutuallyExclusiveParams['sortFeatures'] = ('sortBy', 'sortHelper')
mutuallyExclusiveParams['extractFeatures'] = ('toExtract', ('start', 'end'))
mutuallyExclusiveParams['extractPoints'] = ('toExtract', ('start', 'end'))
mutuallyExclusiveParams['copyFeatures'] = ('features', ('start', 'end'))
mutuallyExclusiveParams['copyPoints'] = ('points', ('start', 'end'))

def inExclusive(funcName, toCheck):
	exclusive = mutuallyExclusiveParams[funcName]
	for value in exclusive:
		if value == toCheck:
			return True
		if isinstance(value, tuple):
			for tupValue in value:
				if tupValue == toCheck:
					return True
	return False

nameToType = {}
nameToType['assignments'] = 'assignments' 
nameToType['asType'] = 'asType'
nameToType['end'] = 'param_bounded_int'
nameToType['extractionProbability'] = 'float'
nameToType['features'] = ('sub_bounded_list_ID', 'ID')
nameToType['featureToReplace'] = 'ID'
nameToType['featureToConvert'] = 'ID'
nameToType['function'] = 'apply_function'
nameToType['ID'] = 'ID'
nameToType['indices'] = 'permutation_list'
nameToType['inPlace'] = 'boolean'
nameToType['level'] = 'default'
nameToType['mapper'] = 'mapper'
nameToType['name'] = 'string'
nameToType['newFeatureName'] = 'string'
nameToType['number'] = 'bounded_int'
nameToType['oldIdentifier'] = 'ID'
nameToType['other'] = 'typed_object' # should express it should be sometimes equal
nameToType['points'] = ('sub_bounded_list_ID', 'ID')
nameToType['preserveZeros'] = 'boolean'
nameToType['randomize'] = 'boolean'
nameToType['reducer'] = 'reducer'
nameToType['seed'] = 'default'
nameToType['skipNoneReturnValues'] = 'boolean'
nameToType['sortBy'] = 'ID'
nameToType['sortHelper'] = ('scorer', 'comparator')
nameToType['start'] = 'param_bounded_int'
nameToType['toAppend'] = 'shaped_typed_object'
nameToType['toExtract'] = 'sub_bounded_list_ID'




def testRandomSequenceOfMethods():
	# always use this number of points and features
	points = numPoints
	features = numFeatures

	testSeed = random.random()
#	testSeed = 0.935074333142
	random.seed(testSeed)
	print testSeed

	# dense int trial
	sparcity = 0.05
	objectList = []
	first = UML.createRandomizedData('List', points, features, sparcity, 'int')
	objectList.append(first)
#	objectList.append(first.copyAs(format='Matrix'))
#	objectList.append(first.copyAs(format='Sparse'))
	runSequence(objectList)

	## dense float trial
	sparcity = 0.05
	objectList = []
	first = UML.createRandomizedData('List', points, features, sparcity, 'float')
	objectList.append(first)
	objectList.append(first.copyAs(format='Matrix'))
	objectList.append(first.copyAs(format='Sparse'))
#	runSequence(objectList)

	# sparse int trial
	sparcity = 0.9
	objectList = []
	first = UML.createRandomizedData('List', points, features, sparcity, 'int')
	objectList.append(first)
	objectList.append(first.copyAs(format='Matrix'))
	objectList.append(first.copyAs(format='Sparse'))
#	runSequence(objectList)

	# sparse float trial
	sparcity = 0.9
	objectList = []
	first = UML.createRandomizedData('List', points, features, sparcity, 'float')
	objectList.append(first)
	objectList.append(first.copyAs(format='Matrix'))
	objectList.append(first.copyAs(format='Sparse'))
#	runSequence(objectList)



def runSequence(objectList):
	# setup method list, from Base dir
	availableMethods = setupMethodList()

	# loop over specified number of operations
#	for trial in xrange(numberOperations):
	for trial in xrange(len(availableMethods)):
		# random number as index into available operations list
#		index = random.randint(0,len(availableMethods)-1)
		index = trial
		currFunc = availableMethods[index]
		# exclude operation we know are not runnabel given certain configurations
		if objectList[0].pointCount == 0:
			if currFunc in unavailableNoPoints:
				continue
		if objectList[0].featureCount == 0:
			if currFunc in unavailableNoFeatures:
				continue

#		if currFunc != 'copyPoints':
#			continue
		print currFunc

#		pdb.set_trace()

		# set up parameters
		funcToCheck = eval('Base.' + currFunc)
		(args, varargs, keywords, defaults) = inspect.getargspec(funcToCheck)
		paramsPerObj = []
		for i in xrange(len(objectList)):
			paramsPerObj.append({})
		currShape = (objectList[0].pointCount, objectList[0].featureCount)
		for paramName in args:
			if paramName != 'self':
				paramValues = generateRandomParameter(trial, currShape, objectList, paramName, currFunc, paramsPerObj[i])
				if paramValues[0] != 'default':
					for i in xrange(len(paramsPerObj)):
						paramsPerObj[i][paramName] = paramValues[i]

#		print paramsPerObj

		if currFunc in mutuallyExclusiveParams:
			exclusive = mutuallyExclusiveParams[currFunc]
			nonDefaultIndex = random.randint(0, len(exclusive)-1)
			nonDefault = exclusive[nonDefaultIndex]
			if not isinstance(nonDefault, tuple):
				nonDefault = (nonDefault)
			for paramHash in paramsPerObj:
				for param in paramHash.keys():
					if inExclusive(currFunc, param):
						if not param in nonDefault:
							argIndex = args.index(param)
							defaultsIndex = argIndex - len(args)
							paramHash[param] = defaults[defaultsIndex]

#		print paramsPerObj

#		pdb.set_trace()

		# call method on each object, collect results
		results = []
		for i in xrange(len(objectList)):
			funcToCall = eval('objectList[i].' + currFunc)
			currResult = funcToCall(**paramsPerObj[i])
			results.append(currResult)	

		# need to check equality of results
		for i in xrange(len(results)):
			ithResult = results[i]
			for j in xrange(i+1, len(results)):
				jthResult = results[j]
				equalityWrapper(ithResult, jthResult)

		# check approximate equality on each of the objects
		for i in xrange(len(objectList)):
			ithDataObject = objectList[i]
			for j in xrange(i+1, len(objectList)):
				jthDataObject = objectList[j]
				assert ithDataObject.isApproximatelyEqual(jthDataObject)


def generateRandomParameter(numCall, shape, objectList, paramName, funcName, currParams):
	""" uses the parameter name and context from the given object to produce
	an appropriate random value for the parameter
	"""
	if paramName not in nameToType:
		raise ArgumentException("Unrecognized paramName " + paramName)
	paramType = nameToType[paramName]
	if isinstance(paramType, tuple):
		choosenIndex = random.randint(0, len(paramType)-1)
		paramType = paramType[choosenIndex]

	retValue = None
	retFull = None

	# 'default'
	if paramType == 'default':
		retValue = 'default'
	# 'boolean'
	elif paramType == 'boolean':
		retValue = random.randint(0,1) == 1
	# 'int'
	elif paramType == 'int':
		retValue = random.randint(0,49) 
	# 'float'
	elif paramType == 'float':
		retValue = random.random()
	# 'string'
	elif paramType =='string':
		retValue = randString(numCall)
	# 'ID' -- either a bounded int or a feature name
	elif paramType == 'ID':
		retValue = randID(shape, objectList, funcName)
	# 'asType'
	elif paramType == 'asType':
		possible = ['default', 'List', 'Matrix', 'Sparse', 'pythonlist', 'numpyarray', 'numpymatrix']
		retValue = random.choice(possible)
	# 'bounded_int' -- an int between 0 and length of the appropriate axis in the data
	elif paramType == 'bounded_int':
		retValue = randBInt(shape, funcName)
	elif paramType == 'param_bounded_int':
		retValue = randParamBInt(shape, paramName, funcName, currParams)
	# 'typed_object' -- random data matching the types of the object in objectList
	elif paramType == 'typed_object':
		retFull = []
		orig = randObject(shape, objectList[0], True)
		retFull.append(orig)
		for i in xrange(1,len(objectList)):
			retFull.append(orig.copyAs(format=objectList[i].getTypeString()))
	# 'shaped_typed_object' -- a typed object matching the shapes of the data in objectList
	elif paramType == 'shaped_typed_object':
		retFull = []
		if 'feature' in funcName.lower():
			match = 'feature'
		else:
			match = 'point'
		orig = randObject(shape, objectList[0], True, match)
		retFull.append(orig)
		for i in xrange(1,len(objectList)):
			retFull.append(orig.copyAs(format=objectList[i].getTypeString()))
	# 'assignments' either a bounded list of string or a bounded map of strings,
	# depending on the function name
	elif paramType == 'assignments':
		retFull = []
		for i in xrange(len(objectList)):
			retFull.append(randAssignments(numCall, shape, funcName))
	# 'sub_bounded_list_ID' -- a bounded list of ID's, whose length may be less than the bound
	# of the length of the axis 
	elif paramType == 'sub_bounded_list_ID':
		retValue = randSub_bounded_list_ID(shape, objectList, funcName)
	# 'permutation_list' -- permuted list of all indices in the appropriate axis
	elif paramType == 'permutation_list':
		retValue = randPermutation_list(shape, funcName)
	# 'apply_function'
	elif paramType == 'apply_function':
		retValue = randApply_function(funcName)
	# 'mapper'
	elif paramType == 'mapper':
		retValue = simpleMapper
	# 'reducer'
	elif paramType == 'reducer':
		retValue = oddOnlyReducer
	elif paramType == 'scorer':
		def sumScorer(view):
			total = 0
			for value in view:
				total += value
			return value
		retValue = sumScorer
	elif paramType == 'comparator':
		def firstValComparator(view):
			if len(view) == 0:
				return None
			else:
				return view[0]
		retValue = firstValComparator
	else:
		raise ArgumentException("Unrecognized paramType " + paramType)

	if retValue is not None:
		return [retValue,retValue,retValue]
	else:
		return retFull


def randApply_function(funcName):
	toAdd = random.randint(0,9)
	if 'elements' in funcName.lower():
		def addToElements(element):
			return element + toAdd
		return addToElements
	else:
		def addToView(view):
			ret = []
			for value in view:
				ret.append(value + toAdd)
			return ret
		return addToView

def randPermutation_list(shape, funcName):
	ret = []
	if 'feature' in funcName.lower():
		length = shape[1]
	else:
		length = shape[0]

	for i in xrange(length):
		ret.append(i)
	random.shuffle(ret)
	return ret


def randID(shape, objectList, funcName):
	if 'feature' in funcName.lower() and random.randint(0,1) == 1:
		return randFName(objectList[0])
	else:
		return randBInt(shape, funcName)

def randString(numCall):
	return str(numCall) + '_' + str(random.randint(0,50))

def randBList_String(numCall, shape, funcName):
	ret = []
	if 'feature' in funcName.lower():
		length = shape[1]
	else:
		length = shape[0]

	for i in xrange(length):
		randString = str(numCall) + '_' + str(i)
		ret.append(randString)

	return ret

def randBMap_String(numCall, shape, funcName):
	ret = {}
	length = shape[1]

	for i in xrange(length):
		randString = str(numCall) + '_' + str(i)
		ret[randString] = i

	return ret

def randAssignments(numCall, shape, funcName):
	if 'list' in funcName.lower():
		return randBList_String(numCall, shape, funcName)
	else:
		return randBMap_String(numCall, shape, funcName)

def randSub_bounded_list_ID(shape, objectList, funcName):
	ret = []
	if 'feature' in funcName.lower():
		maxlength = shape[1]
	else:
		maxlength = shape[0]
	length = random.randint(1,maxlength)

	ret = []
	for i in xrange(length):
		choice = None
		while choice is None or choice in ret:
			choice = randID(shape, objectList, funcName)
		ret.append(choice)

	return ret

def randBInt(shape, funcName):
	if 'point' in funcName.lower():
		cap = shape[0] - 1
	else:
		cap = shape[1] - 1
	return random.randint(0,cap)


def randParamBInt(shape, paramName, funcName, currParams):
	lowerBound = 0
	if 'point' in funcName.lower():
		upperBound = shape[0] - 1
	else:
		upperBound = shape[1] - 1

	if paramName in mutuallyDependentParams:
		(boundType, boundParam) = mutuallyDependentParams[paramName]
		if boundParam in currParams:
			boundValue = currParams[boundParam]
			if boundType == 'le' and boundValue < upperBound:
				upperBound = boundValue
			if boundType == 'ge' and boundValue > lowerBound:
				lowerBound = boundValue


	return random.randint(lowerBound, upperBound)


def randFName(dataObject):
	numF = dataObject.featureCount - 1
	return dataObject.featureNamesInverse[random.randint(0,numF)]

def randObject(shape, dataObject, matchType, matchShape=None):
	if matchType:
		dataType = dataObject.getTypeString()
	else:
		trial = random.randint(0,2)
		if trial == 0:
			dataType = 'List'
		elif trial == 1:
			dataType = 'Matrix'
		else:
			dataType = 'Sparse'

	points = random.randint(0,numPoints)
	features = random.randint(0,numFeatures)
	if matchShape == 'feature':
		points = shape[0]	
	elif matchShape == 'point':	
		features = shape[1]

	return UML.createRandomizedData(dataType, points, features, .5, 'int')

def simpleMapper(point):
	idInt = point[0]
	intList = []
	for i in xrange(1, len(point)):
		intList.append(point[i])
	ret = []
	for value in intList:
		ret.append((idInt,value))
	return ret

def oddOnlyReducer(identifier, valuesList):
	if identifier % 2 == 0:
		return None
	total = 0
	for value in valuesList:
		total += value
	return (identifier,total)

def equalityWrapper(left, right):
	""" takes two parameters and uses the most probable equality function to
	assert that they are equal.
	It is important to note that the preference of equality checks is
	strictly ordered: we prefer using isApproximatelyEqual() more than
	isIdentical(), more than equality of contained, iterable objects,
	more than == equality. """

	if isinstance(left, Base):
		assert isinstance(right, Base)
		if left.getTypeString() == right.getTypeString():
			assert left.isIdentical(right)
		assert left.isApproximatelyEqual(right)
	elif isinstance(left, View):
		assert isinstance(right, View)
		assert left.equals(right)
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

