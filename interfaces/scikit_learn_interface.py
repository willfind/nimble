"""
Contains the functions to be used for in-script calls to the scikit-learn
package

Relies on being scikit-learn 0.9 or above

"""

import inspect
import numpy

from interface_helpers import *
from ..processing.dense_matrix_data import DenseMatrixData as DMData
from ..processing.base_data import BaseData

from ..utility.custom_exceptions import ArgumentException

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


def sciKitLearn(algorithm, trainData, testData, dependentVar=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', timer=None):
	"""
	Function to call on the estimator objects of the scikit-learn package.
	It will instantiate the estimator case-sensitively matching the given algorithm,
	using the trainData to in the .fit call, and testData in the .predict call,
	with all matching arguments from algArgs supplied at each step. If output
	is non-None, the output from the predict call is written to the supplied file,
	otherwise it is returned as a DenseMatrixData object.

	"""
	if scoreMode != 'label' and scoreMode != 'bestScore' and scoreMode != 'allScores':
		raise ArgumentException("scoreMode may only be 'label' 'bestScore' or 'allScores'")
	if multiClassStrategy != 'default' and multiClassStrategy != 'ova' and multiClassStrategy != 'ovo':
		raise ArgumentException("multiClassStrategy may only be 'default' 'ova' or 'ovo'")

	if not isinstance(trainData, BaseData):
		trainObj = DMData(file=trainData)
	else: # input is an object
		trainObj = trainData
	if not isinstance(testData, BaseData):
		testObj = DMData(file=testData)
	else: # input is an object
		testObj = testData
	
	trainObjY = None
	# directly assign target values, if present
	if isinstance(dependentVar, BaseData):
		trainObjY = dependentVar
	# otherwise, isolate the target values from training examples
	elif dependentVar is not None:
		trainObj = trainObj.duplicate()
		trainObjY = trainObj.extractFeatures([dependentVar])		
	# could be None for unsupervised learning	

	# necessary format for skl, also makes the following ops easier
	if trainObjY is not None:	
		trainObjY = trainObjY.toDenseMatrixData()
	
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

	outputObj = DMData(retData)

	if output is None:
		if scoreMode == 'bestScore':
			outputObj.renameMultipleFeatureNames(['PredictedClassLabel', 'LabelScore'])
		elif scoreMode == 'allScores':
			names = sorted(list(str(i) for i in numpy.unique(trainRawDataY)))
			outputObj.renameMultipleFeatureNames(names)

		return outputObj

	outputObj.writeFile('csv', output, False)


def _sciKitLearnBackend(algorithm, trainDataX, trainDataY, testData, algArgs, scoreMode, timer=None):
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
	sklObj = eval("sklObj.fit(trainDataX,trainDataY " + argString + ")")

	#stop timing training and start timing testing, if timer is present
	if timer is not None:
		timer.stop('train')
		timer.start('test')

	#case on scoreMode
	predLabels = None
	scores = None
	if scoreMode != 'label':
		labelOrder = numpy.unique(trainDataY)
		numLabels = len(labelOrder)
	else:
		labelOrder = None
	if scoreMode == 'label' or scoreMode == 'bestScore' or numLabels == 3:
		# estimate from object
		(preArgs,v,k,d) = inspect.getargspec(sklObj.predict)
		argString = makeArgString(preArgs, algArgs, "", "=", ", ")
		predLabels = eval("sklObj.predict(testData, " + argString + ")")
		predLabels = numpy.atleast_2d(predLabels)
		predLabels = predLabels.T
		#stop timing of testing, if timer is present
		if timer is not None:
			timer.stop('test')

	if scoreMode != 'label':
		try:
			scoresPerPoint = sklObj.decision_function(testData)
		except AttributeError:
			raise ArgumentException("Invalid score mode for this algorithm, does not have the api necessary to report scores")
		scores = scoresPerPoint
		if not ovaNotOvOFormatted(scoresPerPoint, predLabels, numLabels):
			scores = []
			for i in xrange(len(scoresPerPoint)):
				combinedScores = calculateSingleLabelScoresFromOneVsOneScores(scoresPerPoint[i], numLabels)
				scores.append(combinedScores)
			scores = numpy.array(scores)

	outData = scoreModeOutputAdjustment(predLabels, scores, scoreMode, labelOrder)

	return outData



def listAlgorithms():
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


