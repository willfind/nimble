"""
Utility functions that could be useful in multiple interfaces

"""

import numpy
import random
import sys

import UML
from UML.exceptions import ArgumentException


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
def findModule(learnerName, packageName, packageLocation):
	"""
	Import the desired python package, and search for the module containing
	the wanted learner. For use by interfaces to python packages.

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
		if learnerName in subContents:
			return moduleName

	return None



def putOnSearchPath(wantedPath):
	if wantedPath is None:
		return
	elif wantedPath in sys.path:
		return
	else:
		sys.path.append(wantedPath)


def checkClassificationStrategy(interface, learnerName, algArgs):
	"""
	Helper to determine the classification strategy used for a given learner called
	using the given interface with the given args. Runs a trial on data with 4 classes
	so that we can use structural 
	
	"""
	dataX = [[-100,3], [-122,1], [118,1], [117,5], [1,-191], [-2,-118], [-1,200],[3,222]]
	xObj = UML.createData("Matrix", dataX)
	# we need classes > 2 to test the multiclass strategy, and we should be able
	# to tell structurally when classes != 3
	dataY = [[0],[0],[1],[1],[2],[2],[3],[3]]
	yObj = UML.createData("Matrix", dataY)
	dataTest = [[0,0],[-100,0],[100,0],[0,-100],[0,100]]
	testObj = UML.createData("Matrix", dataTest)

	tlObj = interface.train(learnerName, xObj, yObj, arguments=algArgs)
	applyResults = tlObj.apply(testObj, arguments=algArgs)
	(a, b, testTrans, c) = interface._inputTransformation(learnerName, None, None, testObj, algArgs, tlObj.customDict)
	rawScores = interface._getScores(tlObj.backend, testTrans, algArgs, tlObj.customDict)

	return ovaNotOvOFormatted(rawScores, applyResults, 4)

def ovaNotOvOFormatted(scoresPerPoint, predictedLabels, numLabels, useSize=True):
	"""
	return True if the scoresPerPoint list of list has scores formatted for a
	one vs all strategy, False if it is for a one vs one strategy. None if there
	are no definitive cases. May throw an ArgumentException if there are conflicting
	definitive votes for different strategies.
	"""
	if not isinstance(scoresPerPoint, UML.data.Base):
		scoresPerPoint = UML.data.Matrix(scoresPerPoint, reuseData=True)
	if not isinstance(predictedLabels, UML.data.Base):
		predictedLabels = UML.data.Matrix(predictedLabels, reuseData=True)
	length = scoresPerPoint.pointCount
	scoreLength = scoresPerPoint.featureCount

	# let n = number of classes
	# ova : number scores = n
	# ovo : number scores = (n * (n-1) ) / 2
	# only at n = 3 are they equal
	if useSize and numLabels != 3:
		if scoreLength == numLabels:
			return True
		else:
			return False

	# we want to check random points out of all the possible data
	check = 20
	if length < check:
		check = length
	checkList = random.sample(xrange(length), check)
	results = []
	for i in checkList:
		strategy = verifyOvANotOvOSingleList(scoresPerPoint.pointView(i), predictedLabels[i,0], numLabels)
		results.append(strategy)

	ovaVote = results.count(True)
	ovoVote = results.count(False)

	# different points were unambigously in different scoring strategies. Can't make sense of that
	if ovoVote > 0 and ovaVote > 0:
		raise ArgumentException("We found conflicting scoring strategies for multiclass classification, cannot verify one way or the other")
	# only definitive votes were ova
	elif ovaVote > 0:
		return True
	# only definitive votes were ovo
	elif ovoVote > 0:
		return False
	# no unambiguous cases: return None as a sentinal for unsure
	else:
		return None



def verifyOvANotOvOSingleList(scoreList, predictedLabelIndex, numLabels):
	""" We cannot determine from length
	whether scores are produced using a one vs all strategy or a one vs one
	strategy. This checks a particular set of scores by simulating OvA and
	OvO prediction strategies, and checking the results.
	
	Returns True if it is OvA consistent and not OvO consistent.
	Returns False if it is not OvA consistent but is OvO consistent.
	Returns None otherwise
	"""
	# simulate OvA prediction strategy
	maxScoreIndex = -1
	maxScore = -sys.maxint - 1
	for i in xrange(len(scoreList)):
		if scoreList[i] > maxScore:
			maxScore = scoreList[i]
			maxScoreIndex = i

	ovaConsistent = maxScoreIndex == predictedLabelIndex

	# simulate OvO prediction strategy
	combinedScores = calculateSingleLabelScoresFromOneVsOneScores(scoreList, numLabels)
	maxScoreIndex = -1
	maxScore = -sys.maxint - 1
	for i in xrange(len(combinedScores)):
		if combinedScores[i] > maxScore:
			maxScore = combinedScores[i]
			maxScoreIndex = i
	ovoConsistent = maxScoreIndex == predictedLabelIndex

	if ovaConsistent and not ovoConsistent:
		return True
	elif not ovaConsistent and ovoConsistent:
		return False
	elif ovaConsistent and ovoConsistent:
		return None
	else:
		raise ArgumentException("The given scoreList does not produce the predicted label with either of our combination strategies. We therefore cannot verify the format of the scores")

def calculateSingleLabelScoresFromOneVsOneScores(oneVOneData, numLabels):
	""" oneVOneData is the flat list of scores of each least ordered pair of
	labels, ordered (score label0 vs label1... score label0 vs labeln-1, score label1
	vs label2 ... score labeln-2 vs labeln-1). We return a length n list where
	the ith value is the ratio of wins for label i in the label vs label tournament.
	"""
	ret = []
	for i in xrange(numLabels):
		wins = 0
		for j in xrange(numLabels):
			score = valueFromOneVOneData(oneVOneData, i, j, numLabels)
			if score is not None and score > 0:
				wins += 1
		ret.append( float(wins) / (numLabels - 1) )

	return ret
				


def valueFromOneVOneData(oneVOneData, posLabel, negLabel, numLabels):
	flagNegative = False
	if posLabel == negLabel:
		return None
	if posLabel > negLabel:
		flagNegative = True
		tempLabel = negLabel
		negLabel = posLabel
		posLabel = tempLabel

	start = (posLabel * numLabels) - ((posLabel * (posLabel + 1))/2)
	offset = negLabel - (posLabel + 1) 
	value = oneVOneData[start + offset]
	if flagNegative:
		return 0 - value
	else:
		return value


def scoreModeOutputAdjustment(predLabels, scores, scoreMode, labelOrder):
	"""
	Helper to set up the correct output data for different scoreModes in the multiclass case.
	predLabels is a 2d array, where the data is a column vector, and each row contains a
	single predicted label. scores is a 2d array, where each row corresponds to the confidence
	scores used to predict the corresponding label in predLabels. scoreMode is the string
	valued flag determining the output format. labelOrder is a 1d array where the ith
	entry is the label name corresponding to the ith confidence value in each row of scores.

	"""
	# if 'labels' we just want the predicted labels
	if scoreMode == 'label':
		outData = predLabels
	# in this case we want the first column to be the predicted labels, and the second
	# column to be that label's score
	elif scoreMode == 'bestScore':
		labelToIndexMap = {}
		for i in xrange(len(labelOrder)):
			ithLabel = labelOrder[i]
			labelToIndexMap[ithLabel] = i
		outData = predLabels
		bestScorePerPrediction = numpy.empty((len(scores),1))
		for i in xrange(len(scores)):
			label = predLabels[i,0]
			index = labelToIndexMap[label]
			matchingScore = scores[i][index]
			bestScorePerPrediction[i] = matchingScore
		outData = numpy.concatenate((outData,bestScorePerPrediction), axis=1)
	else:
		outData = scores

	return outData

def generateBinaryScoresFromHigherSortedLabelScores(scoresPerPoint):
	""" Given an indexable containing the score for the label with a higher
	natural ordering corresponding to the ith test point of an n point binary
	classification problem set, construct and return an array with two columns
	and n rows, where the ith row corresponds to the ith test point, the first
	column contains the score for the label with the lower natural sort order,
	and the second column contains the score for the label with the higher natural
	sort order.

	""" 
	newScoresPerPoint = []
	for i in xrange(scoresPerPoint.pointCount):
		pointScoreList = []
		currScore = scoresPerPoint[i,0]
		pointScoreList.append((-1) * currScore)
		pointScoreList.append(currScore)
		newScoresPerPoint.append(pointScoreList)
	return newScoresPerPoint


def pythonIOWrapper(learnerName, trainX, trainY, testX, output, arguments, kernel, config):

	inType = config['inType']
	inTypeLabels = config['inTypeLabels']
	
	toCall = config['toCall']
	checkPackage = config['checkPackage']
	pythonOutType = config['pythonOutType']
	fileOutType = config['fileOutType']


	if not isinstance(trainX, UML.data.Base):
		trainObj = UML.createData(inType, data=trainX)
	else: # input is an object
		trainObj = convertTo(trainObj, inType)
	if not isinstance(testX, UML.data.Base):
		testObj = UML.createData(inType, data=testX)
	else: # input is an object
		testObj = convertTo(testObj, inType)
	
	trainObjY = None
	# directly assign target values, if present
	if isinstance(trainY, UML.data.Base):
		trainObjY = trainY
	# otherwise, isolate the target values from training examples
	elif trainY is not None:
		trainCopy = trainObj.copy()
		trainObjY = trainCopy.extractFeatures([trainY])		
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
		retData = toCall(learnerName,  trainRawData, trainRawDataY, testRawData, arguments, kernel)
	except ImportError as e:
		if not checkPackage():
			print "Package was not importable."
			print "It must be either on the search path, or have its location set in UML"
		raise e

	if retData is None:
		return

	outputObj = UML.createData(pythonOutType, data=retData)

	if output is None:
		# we want to return a column vector
		outputObj.transpose()
		return outputObj

	outputObj.writeFile(output, format=fileOutType, includeNames=False)



def convertTo(data, retType):
	return eval("data.to" + retType + "()")



def cacheWrapper(toWrap):
	cache = {}
	def wrapped(*args):
		if args in cache:
			return cache[args]
		else:
			ret = toWrap(*args)
			cache[args] = ret
		return ret
	return wrapped



