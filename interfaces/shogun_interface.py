"""
Contains the functions to be used for in-script calls to the shogun module python
interface

"""

import numpy
import scipy.sparse
import copy

from interface_helpers import findModule
from interface_helpers import putOnSearchPath
from interface_helpers import calculateSingleLabelScoresFromOneVsOneScores
from interface_helpers import ovaNotOvOFormatted
from interface_helpers import scoreModeOutputAdjustment
from ..processing.dense_matrix_data import DenseMatrixData as DMData
from ..processing.base_data import BaseData
from ..processing.sparse_data import SparseData

from ..utility.custom_exceptions import ArgumentException

# Contains path to shogun root directory
shogunDir = None


def setShogunLocation(path):
	""" Sets the location of the root directory of the shogun installation to be used """
	global shogunDir
	shogunDir = path


def getShogunLocation():
	""" Returns the currently set path to the shogun root directory """
	return shogunDir
	

def shogunPresent():
	"""
	Return true if shogun is importable. If true, then the interface should
	be accessible.
	
	"""
	putOnSearchPath(shogunDir)
	try:
		import shogun
	except ImportError:	
		return False

	return True


def shogun(algorithm, trainData, testData, dependentVar=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', timer=None):
	"""


	"""
	if scoreMode != 'label' and scoreMode != 'bestScore' and scoreMode != 'allScores':
		raise ArgumentException("scoreMode may only be 'label' 'bestScore' or 'allScores'")
	if multiClassStrategy != 'default' and multiClassStrategy != 'ova' and multiClassStrategy != 'ovo':
		raise ArgumentException("multiClassStrategy may only be 'default' 'ova' or 'ovo'")

	# if we have to enfore a classification strategy, we test the algorithm in question,
	# and call our own strategies if necessary
	if multiClassStrategy != 'default':
		trialResult = checkClassificationStrategy(_shogunBackend, algorithm, arguments)
		if multiClassStrategy == 'ova' and trialResult != 'ova':
			from ..performance.runner import runOneVsAll
			runOneVsAll(algorithm, trainData, testData, dependentVar, arguments, output, scoreMode, timer)
		if multiClassStrategy == 'ovo' and trialResult != 'ovo':
			from ..performance.runner import runOneVsOne
			runOneVsOne(algorithm, trainData, testData, dependentVar, arguments, output, scoreMode, timer)

	args = copy.copy(arguments)
	if not isinstance(trainData, BaseData):
		trainObj = DMData(file=trainData)
	else: # input is an object
		trainObj = trainData.duplicate()
	if not isinstance(testData, BaseData):
		testObj = DMData(file=testData)
	else: # input is an object
		testObj = testData.duplicate()
	
	trainObjY = None
	# directly assign target values, if present
	if isinstance(dependentVar, BaseData):
		trainObjY = dependentVar.duplicate()
	# otherwise, isolate the target values from training examples
	elif dependentVar is not None:
		trainObjY = trainObj.extractFeatures([dependentVar])		
	# could be None for unsupervised learning	

	# necessary format for shogun, also makes the following ops easier
	if trainObjY is not None:
		trainObjY = trainObjY.toDenseMatrixData()
	
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


	# check some stuff that we know won't work, but shogun will not report intelligently
	if trainRawData.shape[0] != testRawData.shape[0]:
		raise ArgumentException("Points in the training data and testing data must be the same size")

	# call backend
	try:
		retData = _shogunBackend(algorithm,  trainRawData, trainRawDataY, testRawData, args, scoreMode, timer)
	except ImportError as e:
		print "ImportError: " + str(e)
		if not shogunPresent():
			print "Shogun not importable."
			print "It must be either on the search path, or have its path set by setshogunLocation()"
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


def _shogunBackend(algorithm, trainDataX, trainDataY, testData, algArgs, scoreMode, timer=None):
	"""
	Function to find, construct, and execute the wanted calls to shogun

	"""	
	moduleName = findModule(algorithm, "shogun", shogunDir)		

	if moduleName is None:
		raise ArgumentException("Could not find the algorithm")

	putOnSearchPath(shogunDir)
	exec "from shogun import " + moduleName in locals()

	# make object
	objectCall = moduleName + '.' + algorithm
	SGObj = eval(objectCall + "()")

	# convert data to shogun friendly format
	from shogun.Features import RealFeatures
	from shogun.Features import SparseRealFeatures

	if scipy.sparse.issparse(trainDataX):
		trainFeat = SparseRealFeatures()
		trainFeat.set_sparse_feature_matrix(trainDataX.tocsc().astype(numpy.float))
	else:
		trainFeat = RealFeatures()
		trainFeat.set_feature_matrix(numpy.array(trainDataX, dtype=numpy.float))

	if scipy.sparse.issparse(testData):
		testFeat = SparseRealFeatures()
		testFeat.set_sparse_feature_matrix(testData.tocsc().astype(numpy.float))
	else:
		testFeat = RealFeatures()
		testFeat.set_feature_matrix(numpy.array(testData, dtype=numpy.float))

	# set up the correct type of label
	try:
		import shogun.Classifier
		inverseMapping = None
		if isinstance(SGObj, shogun.Classifier.BaseMulticlassMachine):
			tempObj = DMData(trainDataY)
			inverseMapping = remapLabels(tempObj)
			if len(inverseMapping) == 1:
				raise ArgumentException("Cannot train a multiclass classifier with data containing only one label")

			from shogun.Features import MulticlassLabels
			flattened = numpy.array(tempObj.data).flatten()
			trainLabels = MulticlassLabels(flattened.astype(float))
		else:
			regression = False
			for value in trainDataY:
				if value != -1 and value != 1:
					regression = True
			if regression:
				from shogun.Features import RegressionLabels
				trainLabels = RegressionLabels(trainDataY.astype(float))
				if scoreMode != 'label':
					raise ArgumentException("Invalid scoreMode for a regression problem; the default parameter must be used")
			else:
				if scoreMode == 'test':
					return 'binary'
				from shogun.Features import BinaryLabels
				trainLabels = BinaryLabels(trainDataY.astype(float))
	except ImportError:
		from shogun.Features import Labels
		trainLabels = Labels(trainDataY.astype(float))

	#set parameters from input arguments
	SGObj.set_labels(trainLabels)
	blankDataParam = False

	# special parameters 'kernel' and 'distance' -- affects arguments to .train()
	keywords = ['kernel', 'distance']
	for word in keywords:
		if word not in algArgs:
			continue

		wordValue = algArgs[word]
		# both kernels and distances are in shogun.Kernel
		import shogun.Kernel
		if not isinstance(wordValue, basestring):
			raise ArgumentException(word + " parameter must define the name of a Kernel to instantiate")
		
		try:
			constructedObj = eval("shogun.Kernel." + wordValue + "()")
		except AttributeError:
			raise ArgumentException("Failed to instantiate" + wordValue)

		constructedObj.init(trainFeat,trainFeat)

		if word == 'kernel':
			SGObj.set_kernel(constructedObj)
		else:
			SGObj.set_distance(constructedObj)

		blankDataParam = True
		# remove those arguments that were used, so we don't try to readd them further down
		del(algArgs[word])

	# special case parameter 'C' -- we want to fudge the default call
	if 'C' in algArgs:
		Cvalue = algArgs['C']
		# allow a list input
		if not (isinstance(Cvalue, int) or isinstance(Cvalue, float)):
			try:
				SGObj.set_C(Cvalue[0], Cvalue[1])
			except TypeError:
				raise ArgumentException("Error setting C value. " + SGObj.set_C.__doc__)
		# assume the same value for both params
		else:
			try:
				SGObj.set_C(Cvalue, Cvalue)
			except TypeError:
				SGObj.set_C(Cvalue)
		del(algArgs['C'])

	for k in algArgs:
		v = algArgs[k]
		testString = "set_" + k
		if testString not in dir(SGObj):
#			raise ArgumentException("Cannot set argument: " + k)
			continue
		exec("SGObj." + testString + "(v)")

	#start timing training, if timer is present
	if timer is not None:
		timer.start('train')

	# Training Call
	argString = ""
	if not blankDataParam:
		argString = "trainFeat"
	exec("SGObj.train(" + argString + ")")

	#stop timing training and start timing testing, if timer is present
	if timer is not None:
		timer.stop('train')
		timer.start('test')

	predLabels = None
	scores = None
	if scoreMode != 'label':
		labelOrder = numpy.unique(trainDataY)
		numLabels = len(labelOrder)
	else:
		labelOrder = None

	# Prediction / Test Call
	predObject = SGObj.apply(testFeat)
	predLabels = predObject.get_labels()
	predLabels = numpy.atleast_2d(predLabels)
	predLabels = predLabels.T
	# this is going to be our default output in the case of scoreMode == 'label'
	outData = predLabels

	#stop timing of testing, if timer is present
	if timer is not None:
		timer.stop('test')

	if scoreMode != 'label':
		# the multiclass case
		if hasattr(predObject, 'get_multiclass_confidences'):
			# setup an array in the right shape, number of predicted labels by number of possible labels
			scoresPerPoint = numpy.empty((len(predLabels),numLabels))
			for i in xrange(len(predLabels)):
				currConfidences = predObject.get_multiclass_confidences(i)
				for j in xrange(currConfidences.size):
					scoresPerPoint[i,j] = currConfidences[j]
			scores = scoresPerPoint
			strategy = ovaNotOvOFormatted(scoresPerPoint, predLabels, numLabels,useSize=(scoreMode!='test'))
			if scoreMode == 'test':
				if strategy: return 'ova'
				elif not strategy: return 'ovo'
				elif strategy is None: return 'ambiguous'
			# we want the scores to be per label, regardless of the original format, so we
			# check the strategy, and modify it if necessary
			if not strategy:
				scores = []
				for i in xrange(len(scoresPerPoint)):
					combinedScores = calculateSingleLabelScoresFromOneVsOneScores(scoresPerPoint[i], numLabels)
					scores.append(combinedScores)
				scores = numpy.array(scores)
			# helper function will setup the right outputs for the different scoreMode flags
			outData = scoreModeOutputAdjustment(predLabels, scores, scoreMode, labelOrder)
		# otherwise we must be dealing with binary classification
		else:
			# we get a 1d array containing the winning label's confidence value
			scoresPerPoint = predObject.get_values()
			scoresPerPoint.resize(scoresPerPoint.size,1)
			if scoreMode == 'bestScore':
				outData = numpy.concatenate((outData,scoresPerPoint), axis=1)
			else:
				outData = numpy.empty((predObject.get_num_labels(), 2))
				for i in xrange(len(outData)):
					# the confidence value of one label is negative the confidence value of the other
					if predLabels[i] == labelOrder[0]:
						outData[i,0] = scoresPerPoint[i]
						outData[i,1] = -scoresPerPoint[i]
					else:
						outData[i,0] = -scoresPerPoint[i]
						outData[i,1] = scoresPerPoint[i]

	# have to undo the label name packing we performed earlier
	if inverseMapping is not None and scoreMode != 'allScores':
		outputObj = DMData(outData)
		outputObj.transformFeature(0, makeInverseMapper(inverseMapping))
		outData = outputObj.data

	return outData

def makeInverseMapper(inverseMappingParam):
	def inverseMapper(value):
		return inverseMappingParam[int(value)]
	return inverseMapper

def listAlgorithms():
	"""
	Function to return a list of all algorithms callable through our interface, if shogun is present
	
	"""
	if not shogunPresent():
		return []

	import shogun

	ret = []
	subpackages = shogun.__all__

	for sub in subpackages:
		curr = 'shogun.' + sub
		try:
			exec('import ' + curr)
		except ImportError:
			# no guarantee __all__ is accurate, if something doesn't import, just skip ahead
			continue

		contents = eval('dir(' + curr + ')')
		
		for member in contents:
			memberContents = eval('dir(' + curr + "." + member + ')')
			if 'train' in memberContents and 'apply' in memberContents:
				ret.append(member)

	return ret


def _argsFromDoc(objectReference, methodName, algArgs):
	"""
	Helper function to determine the possible parameters for a method from object's docstring.

	"""
	import pydoc
	objDocString = pydoc.render_doc(objectReference)
	retValues = None
	usedArgs = []
	# find and loop through those lines definining possible parameter list
	position = objDocString.find(methodName + "(")
	while position != -1:
		currValues = []
		currUsed = []
		objDocString = objDocString[position:]
		endParams = objDocString.find(")")
		paramList = objDocString[len(methodName)+1:endParams].split(',')

		# at this point we have passed the possible parameters, now we need to check
		# if we have them in our provided args
		for name in paramList:
			if name == 'self':
				continue

			clean = name.strip()
			defaultValue = None
			if '=' in clean:
				splitList = clean.split('=')
				clean = splitList[0].strip()
				defaultValue = splitList[1].strip()
			
			if clean in algArgs:
				currValues.append(algArgs[clean])
				currUsed.append(clean)
			elif defaultValue is not None:
				currValues.append(int(defaultValue))
			# cannot match this parameter set, break out
			else:
				currValues = None
				break

		# we prefer those sets of parameters that use the most of our input arguments
		# as possible (they are there for a reason)
		if currValues is not None:
			if retValues is None or len(currValues) > len(retValues):
				retValues = currValues
				usedArgs = currUsed

		# find the next instance of a string that could define parameter names
		position = objDocString[endParams:].find(methodName + "(")
		if position != -1:
			position = position + endParams

	return (retValues, usedArgs)



def remapLabels(toRemap):
	"""
	Takes the object toRemap, which must be a data representation with a single feature,
	and maps the values in that feature into the range between 0 and n-1, where n is
	the number of distinct values in that feature. The object is modified, and the
	inverse mapping is returned a length n list

	"""

	mapping = {}
	inverse = []
	invIndex = 0

	view = toRemap.getPointView(0)

	for x in xrange(toRemap.features()):
		value = view[x]
		if value not in mapping:
			mapping[value] = invIndex
			inverse.append(value)
			invIndex += 1
		view[x] = mapping[value]

	return inverse

