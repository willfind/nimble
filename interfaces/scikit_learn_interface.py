"""
Contains the functions to be used for in-script calls to the scikit-learn
package

Relies on being scikit-learn 0.9 or above

"""

import inspect
import numpy

from interface_helpers import makeArgString
from interface_helpers import checkClassificationStrategy
from interface_helpers import findModule
from interface_helpers import putOnSearchPath
from interface_helpers import generateBinaryScoresFromHigherSortedLabelScores
from interface_helpers import ovaNotOvOFormatted
from interface_helpers import calculateSingleLabelScoresFromOneVsOneScores
from interface_helpers import scoreModeOutputAdjustment
import UML

from UML.exceptions import ArgumentException

# Contains path to sciKitLearn root directory
sciKitLearnDir = None
locationCache = {}


def setSciKitLearnLocation(path):
	""" Sets the location of the root directory of the scikit-Learn installation to be used """
	global sciKitLearnDir
	sciKitLearnDir = path


def getSciKitLearnLocation():
	""" Returns the currently set path to the scikit-Learn root directory """
	return sciKitLearnDir
	

def sciKitLearnPresent():
	"""
	Return true if sciKitLearn is importable. If true, then the interface should
	be accessible.
	
	"""
	putOnSearchPath(sciKitLearnDir)
	try:
		import sklearn
	except ImportError:	
		return False

	return True


def sciKitLearn(algorithm, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', timer=None):
	"""
	Function to call on the estimator objects of the scikit-learn package.
	It will instantiate the estimator case-sensitively matching the given algorithm,
	using the trainX to in the .fit call, and testX in the .predict call,
	with all matching arguments from algArgs supplied at each step. If output
	is non-None, the output from the predict call is written to the supplied file,
	otherwise it is returned as a Matrix object.

	"""
	if scoreMode != 'label' and scoreMode != 'bestScore' and scoreMode != 'allScores':
		raise ArgumentException("scoreMode may only be 'label' 'bestScore' or 'allScores'")
#	multiClassStrategy = multiClassStrategy.lower()	
	if multiClassStrategy != 'default' and multiClassStrategy != 'OneVsAll' and multiClassStrategy != 'OneVsOne':
		raise ArgumentException("multiClassStrategy may only be 'default' 'OneVsAll' or 'OneVsOne'")

	# if we have to enfore a classification strategy, we test the algorithm in question,
	# and call our own strategies if necessary
	if multiClassStrategy != 'default':
		trialResult = checkClassificationStrategy(_sciKitLearnBackend, algorithm, arguments)
		if multiClassStrategy == 'OneVsAll' and trialResult != 'OneVsAll':
			UML.runners.runOneVsAll(algorithm, trainX, trainY, testX, arguments, output, scoreMode, timer)
		if multiClassStrategy == 'OneVsOne' and trialResult != 'OneVsOne':
			UML.runners.runOneVsOne(algorithm, trainX, trainY, testX, arguments, output, scoreMode, timer)


	if not isinstance(trainX, UML.data.Base):
		trainObj = UML.createData('Matrix', trainX)
	else: # input is an object
		trainObj = trainX
	if not isinstance(testX, UML.data.Base):
		if testX is None:
			raise ArgumentException("testX must be an object derived from Base")
		testObj = UML.createData('Matrix', testX)
	else: # input is an object
		testObj = testX
	
	trainObjY = None
	# directly assign target values, if present
	if isinstance(trainY, UML.data.Base):
		trainObjY = trainY
	# otherwise, isolate the target values from training examples
	elif trainY is not None:
		trainObj = trainObj.copy()
		trainObjY = trainObj.extractFeatures([trainY])		
	# could be None for unsupervised learning	

	# necessary format for skl, also makes the following ops easier
	if trainObjY is not None:	
		trainObjY = trainObjY.copy(asType="Matrix")
	
	# pull out data from obj
	trainRawData = trainObj.data
	if trainObjY is not None:
		# corrects the dimensions of the matrix data to be just an array
		trainRawDataY = numpy.array(trainObjY.data).flatten()
	else:
		trainRawDataY = None
	testRawData = testObj.data

	# call backend
	try:
		retData = _sciKitLearnBackend(algorithm, trainRawData, trainRawDataY, testRawData, arguments, scoreMode, timer)
	except ImportError as e:
		print "ImportError: " + str(e)
		if not sciKitLearnPresent():
			print "Scikit-learn not importable."
			print "It must be either on the search path, or have its path set by setSciKitLearnLocation()"
		return

	if retData is None:
		return

	outputObj = UML.createData('Matrix', retData)

	if output is None:
		if scoreMode == 'bestScore':
			outputObj.renameMultipleFeatureNames(['PredictedClassLabel', 'LabelScore'])
		elif scoreMode == 'allScores':
			names = sorted(list(str(i) for i in numpy.unique(trainRawDataY)))
			outputObj.renameMultipleFeatureNames(names)

		return outputObj

	outputObj.writeFile('csv', output, False)


def _sciKitLearnBackend(algorithm, trainX, trainY, testX, algArgs, scoreMode, timer=None):
	"""
	Function to find, construct, and execute the wanted calls to scikit-learn

	"""	
	global locationCache
	if algorithm in locationCache:
		moduleName = locationCache[algorithm]
	else: 
		moduleName = findModule(algorithm, "sklearn", sciKitLearnDir)		
		locationCache[algorithm] = moduleName

	if moduleName is None:
		print ("Could not find the algorithm")
		return

	putOnSearchPath(sciKitLearnDir)
	exec ("from sklearn import " + moduleName)

	# make object
	objectCall = moduleName + "." + algorithm
	(objArgs,v,k,d) = eval("inspect.getargspec(" + objectCall + ".__init__)")
	argString = makeArgString(objArgs, algArgs, "", "=", ", ")
	sklObj = eval(objectCall + "(" + argString + ")")

	#start timer of training, if timer is present
	if timer is not None:
		timer.start('train')

	# fit object
	(fitArgs,v,k,d) = inspect.getargspec(sklObj.fit)
	argString = makeArgString(fitArgs, algArgs, "", "=", ", ")
	sklObj = eval("sklObj.fit(trainX,trainY " + argString + ")")

	#stop timing training and start timing testing, if timer is present
	if timer is not None:
		timer.stop('train')
		timer.start('test')

	#case on scoreMode
	predLabels = None
	scores = None
	numLabels = -1
	if scoreMode != 'label':
		labelOrder = numpy.unique(trainY)
		numLabels = len(labelOrder)
	else:
		labelOrder = None
	# the only case we don't want to actually predict are if we're getting allScores,
	# and the number of labels is not three (in which case we couldn't tell the strategy)
	# just from the number of confidence scores
	if scoreMode != 'allScores' or numLabels == 3:
		# estimate from object
		(preArgs,v,k,d) = inspect.getargspec(sklObj.predict)
		argString = makeArgString(preArgs, algArgs, "", "=", ", ")
		predLabels = eval("sklObj.predict(testX, " + argString + ")")
		predLabels = numpy.atleast_2d(predLabels)
		predLabels = predLabels.T
		#stop timing of testing, if timer is present
		if timer is not None:
			timer.stop('test')
	# the only case where we don't want to get scores is if we're returning labels only
	if scoreMode != 'label':
		try:
			scoresPerPoint = sklObj.decision_function(testX)
		except AttributeError:
			raise ArgumentException("Invalid score mode for this algorithm, does not have the api necessary to report scores")
		# If it was a binary problem, then only one column of scores is returned, so we must
		# generate the other column ourselves
		if numLabels == 2:
			scoresPerPoint = generateBinaryScoresFromHigherSortedLabelScores(scoresPerPoint)
			# this is exactly what we need returned in this case, so we just do it immediately
			if scoreMode.lower() == 'allScores'.lower():
				return scoresPerPoint

		scores = scoresPerPoint
		strategy = ovaNotOvOFormatted(scoresPerPoint, predLabels, numLabels,useSize=(scoreMode!='test'))
		if scoreMode == 'test':
			if strategy: return 'OneVsAll'
			elif not strategy: return 'OneVsOne'
			elif strategy is None: return 'ambiguous'
		# we want the scores to be per label, regardless of the original format, so we
		# check the strategy, and modify it if necessary
		if not strategy:
			scores = []
			for i in xrange(len(scoresPerPoint)):
				combinedScores = calculateSingleLabelScoresFromOneVsOneScores(scoresPerPoint[i], numLabels)
				scores.append(combinedScores)
			scores = numpy.array(scores)

	outData = scoreModeOutputAdjustment(predLabels, scores, scoreMode, labelOrder)

	return outData



def listSciKitLearnAlgorithms():
	"""
	Function to return a list of all algorithms callable through our interface, if scikit learn is present
	
	"""
	if not sciKitLearnPresent():
		return []

	import sklearn

	ret = []
	subpackages = sklearn.__all__

	for sub in subpackages:
		curr = 'sklearn.' + sub
		try:
			exec('import ' + curr)
		except ImportError:
			# no guarantee __all__ is accurate, if something doesn't import, just skip ahead
			continue

		contents = eval('dir(' + curr + ')')
		
		for member in contents:
			memberContents = eval('dir(' + curr + "." + member + ')')
			if 'fit' in memberContents and 'predict' in memberContents:
				ret.append(member)

	return ret


