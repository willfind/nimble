"""
Contains the functions to be used for in-script calls to the mlpy
package

"""

import inspect
import numpy

from interface_helpers import makeArgString
from interface_helpers import checkClassificationStrategy
from interface_helpers import putOnSearchPath
from interface_helpers import generateBinaryScoresFromHigherSortedLabelScores
from interface_helpers import ovaNotOvOFormatted
from interface_helpers import calculateSingleLabelScoresFromOneVsOneScores
from interface_helpers import scoreModeOutputAdjustment
from UML.exceptions import ArgumentException
import UML

# Contains path to mlpy root directory
mlpyDir = None
locationCache = {}


def setMlpyLocation(path):
	""" Sets the location of the root directory of the Regressor installation to be used """
	global mlpyDir
	mlpyDir = path


def getMlpyLocation():
	""" Returns the currently set path to the  root directory """
	return mlpyDir
	

def mlpyPresent():
	"""
	Return true if mlpy is importable. If true, then the interface should
	be accessible.
	
	"""
	putOnSearchPath(mlpyDir)
	try:
		import mlpy
	except ImportError:	
		return False

	return True


def mlpy(learningAlgorithm, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', timer=None):
	"""


	"""
	# argument checking
	if scoreMode != 'label' and scoreMode != 'bestScore' and scoreMode != 'allScores':
		raise ArgumentException("scoreMode may only be 'label' 'bestScore' or 'allScores'")
#	multiClassStrategy = multiClassStrategy.lower()	
	if multiClassStrategy != 'default' and multiClassStrategy != 'OneVsAll' and multiClassStrategy != 'OneVsOne':
		raise ArgumentException("multiClassStrategy may only be 'default' 'OneVsAll' or 'OneVsOne'")

	# if we have to enfore a classification strategy, we test the learning algorithm in question,
	# and call our own strategies if necessary
	if multiClassStrategy != 'default':
		trialResult = checkClassificationStrategy(_mlpyBackend, learningAlgorithm, arguments)
		if multiClassStrategy == 'OneVsAll' and trialResult != 'OneVsAll':
			UML.runners.runOneVsAll(learningAlgorithm, trainX, trainY, testX, arguments, output, scoreMode, timer)
		if multiClassStrategy == 'OneVsOne' and trialResult != 'OneVsOne':
			UML.runners.runOneVsOne(learningAlgorithm, trainX, trainY, testX, arguments, output, scoreMode, timer)

	if isinstance(trainX, UML.data.Sparse):
		raise ArgumentException("MLPY does not accept sparse input")
	if isinstance(testX, UML.data.Sparse):
		raise ArgumentException("MLPY does not accept sparse input")

	if not isinstance(trainX, UML.data.Base):
		trainObj = UML.createData('Matrix', trainX)
	else: # input is an object
		trainObj = trainX
	if not isinstance(testX, UML.data.Base):
		if testX is None:
			raise ArgumentException("testX may only be an object derived from Base")
		testObj = UML.createData('Matrix', testX)
	else: # input is an object
		testObj = testX
		
	# used to determine output type
	outputTypeString = None
	trainObjY = None
	# directly assign target values, if present
	if isinstance(trainY, UML.data.Base):
		trainObjY = trainY
		outputTypeString = trainY.getTypeString()
	# otherwise, isolate the target values from training examples
	elif trainY is not None:
		trainObj = trainObj.copy()
		trainObjY = trainObj.extractFeatures([trainY])
		outputTypeString = trainX.getTypeString()		
	# trainY could be None for unsupervised learning	
	else:
		outputTypeString = trainX.getTypeString()	

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
		retData = _mlpyBackend(learningAlgorithm, trainRawData, trainRawDataY, testRawData, arguments, scoreMode, timer)
	except ImportError as e:
		print "ImportError: " + str(e)
		if not mlpyPresent():
			print "mlpy not importable."
			print "It must be either on the search path, or have its path set by setMlpyLocation()"
		return

	if retData is None:
		return

	outputObj = UML.createData(outputTypeString, retData)

	if output is None:
		if scoreMode == 'bestScore':
			outputObj.setFeatureNamesFromList(['PredictedClassLabel', 'LabelScore'])
		elif scoreMode == 'allScores':
			names = sorted(list(str(i) for i in numpy.unique(trainRawDataY)))
			outputObj.setFeatureNamesFromList(names)
		
		return outputObj

	outputObj.writeFile(output, format='csv', includeFeatureNames=False)


def _mlpyBackend(learningAlgorithm, trainX, trainY, testX, algArgs, scoreMode, timer=None):
	"""
	Function to find, construct, and execute the wanted calls to mlpy

	"""	
	putOnSearchPath(mlpyDir)
	import mlpy 

	# Deal with kernel transformations
	kernObj = None
	if 'kernel' in algArgs and algArgs['kernel'] is not None:
		kernelName = algArgs['kernel']
		objectCall = "mlpy." + kernelName
		try:
			(kernArgs,v,k,d) = eval("inspect.getargspec(" + objectCall + ".__init__)")
		except TypeError:
			#default to nothing
			kernArgs = {}

		#start timer of training, if timer is present
		if timer is not None:
			timer.start('train')
		argString = makeArgString(kernArgs, algArgs, "", "=", ", ")
		kernObj = eval(objectCall + "(" + argString + ")")
		del algArgs['kernel']
		#stop timer of training, if timer is present. We don't want to include object instantiation
		if timer is not None:
			timer.stop('train')

	# make object
	objectCall = "mlpy." + learningAlgorithm
	try:
		(objArgs,v,k,d) = eval("inspect.getargspec(" + objectCall + ".__init__)")
	except TypeError:
		# this can occur if we are inspecting a C backed function. We default
		# to adding everything available.
		objArgs = algArgs
	argString = makeArgString(objArgs, algArgs, "", "=", ", ")
	if kernObj is not None:
		argString += "kernel=kernObj" 
	obj = eval(objectCall + "(" + argString + ")")

	#start timer of training, if timer is present
	if timer is not None:
		timer.start('train')

	# call .learn for the object
	try:
		(learnArgs,v,k,d) = inspect.getargspec(obj.learn)
	except TypeError:
		# in this case, we default to adding nothing
		learnArgs = None
	argString = makeArgString(learnArgs, algArgs, "", "=", ", ")
	if hasattr(obj, 'pred'):
		eval("obj.learn(trainX,trainY " + argString + ")")
	# else, we're in the transform paradigm
	else:
		eval("obj.learn(trainX, " + argString + ")")

	#stop timing training and start timing testing, if timer is present
	if timer is not None:
		timer.stop('train')
		timer.start('test')

	# run code for the pred paradigm
	if hasattr(obj, 'pred'):
		# case on score mode:
		predLabels = None
		scores = None
		labelOrder = sorted(obj.labels())
		numLabels = len(labelOrder)
		# the only case we don't want to actually predict are if we're getting allScores,
		# and the number of labels is not three (in which case we couldn't tell the strategy)
		# just from the number of confidence scores
		if scoreMode != 'allScores' or numLabels == 3:
			# call .pred for the object
			try:
				(predArgs,v,k,d) = inspect.getargspec(obj.pred)
			except TypeError:
				# in this case, we default to adding nothing
				predArgs = None
			argString = makeArgString(predArgs, algArgs, "", "=", ", ")
			predLabels = eval("obj.pred(testX, " + argString + ")")
			# .pred() always returns a row vector, we want a column vector
			predLabels.resize(predLabels.size,1)

			#stop timing of testing, if timer is present
			if timer is not None:
				timer.stop('test')
		# the only case where we don't want to get scores is if we're returning labels only
		if scoreMode != 'label':
			try:
				scoresPerPoint = obj.pred_values(testX)
			except AttributeError:
				raise ArgumentException("Invalid score mode for this learningAlgorithm, does not have the api necessary to report scores")
			if numLabels == 2:
				scoresPerPoint = generateBinaryScoresFromHigherSortedLabelScores(scoresPerPoint)
				# this is exactly what we need returned in this case, so we just do it immediately
				if scoreMode.lower() == 'allScores'.lower():
					return scoresPerPoint

			scores = scoresPerPoint
			strategy = ovaNotOvOFormatted(scoresPerPoint, predLabels, numLabels, useSize=(scoreMode!='test'))
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

	# run code for the transform paradigm
	if hasattr(obj, 'transform'):
		# call .transform for the object
		try:
			(transArgs,v,k,d) = inspect.getargspec(obj.transform)
		except TypeError:
			# in this case, we default to adding nothing
			transArgs = None
		argString = makeArgString(transArgs, algArgs, "", "=", ", ")
		outData = eval("obj.transform(testX, " + argString + ")")

		#stop timing of testing, if timer is present
		if timer is not None:
			timer.stop('test')

	return outData



def listMlpyLearningAlgorithms(includeParams=False):
	"""
	Function to return a list of all learning algorithms callable through our interface, if mlpy is present
	
	"""
	if not mlpyPresent():
		return []

	import mlpy

	ret = []
	contents = dir(mlpy)

	for member in contents:
		subContents = eval('dir(mlpy.' + member + ')')
		if 'learn' in subContents and ('pred' in subContents or 'transform' in subContents):
			if not includeParams:
				ret.append(member)
			else:
				try:
					(objArgs,v,k,d) = eval("inspect.getargspec(" + objectCall + ".__init__)")
					(learnArgs,v,k,d) = inspect.getargspec(obj.learn)
					if 'pred' in subContents:
						(lastArgs,v,k,d) = inspect.getargspec(obj.pred)
					else:		
						(lastArgs,v,k,d) = inspect.getargspec(obj.transform)
				except TypeError:
					pass
	return ret


def getParameters(name):
	"""
	Takes the name of some mlpy object or function, returns a list
	of parameters used to instantiate that object or run that function

	"""
	(objArgs,v,k,d) = _paramQuery(name)
	return objArgs

def getDefaultValues(name):
	"""
	Takes the name of some mlpy object or function, returns a dict mapping
	parameter names to their default values 

	"""

	(objArgs,v,k,d) = _paramQuery(name)
	ret = {}
	if d is not None:
		for i in xrange(len(d)):
			ret[objArgs[-(i+1)]] = d[-(i+1)]

	return ret


def _paramQuery(name):
	import mlpy
	if not hasattr(mlpy, name):
		raise ArgumentException("Cannot find " + name + " in mlpy")

	namedModule = getattr(mlpy, name)

	try:
		(objArgs,v,k,d) = inspect.getargspec(namedModule)
		if len(objArgs) > 0 and objArgs[0] == 'self':
			objArgs = objArgs[1:]
		return (objArgs,v,k,d)
	except TypeError as te:
		try:
			(objArgs,v,k,d) = inspect.getargspec(namedModule.__init__)
			if len(objArgs) > 0 and objArgs[0] == 'self':
				objArgs = objArgs[1:]
			return (objArgs,v,k,d)
		except TypeError as te:
			print te
			return None
