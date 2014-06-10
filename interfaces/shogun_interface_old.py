"""
Contains the functions to be used for in-script calls to the shogun module python
interface

"""

import numpy
import scipy.sparse
import copy
import json
import os.path

from interface_helpers import findModule
from interface_helpers import putOnSearchPath
from interface_helpers import calculateSingleLabelScoresFromOneVsOneScores
from interface_helpers import ovaNotOvOFormatted
from interface_helpers import scoreModeOutputAdjustment
from interface_helpers import checkClassificationStrategy
from interface_helpers import makeArgString
import UML

from UML.exceptions import ArgumentException

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
	"""`
	Return true if shogun is importable. If true, then the interface should
	be accessible.
	
	"""
	putOnSearchPath(shogunDir)
	try:
		import shogun
		dir(shogun)
	except ImportError:	
		return False

	return True


def shogun(learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', timer=None):
	"""


	"""
	if scoreMode != 'label' and scoreMode != 'bestScore' and scoreMode != 'allScores':
		raise ArgumentException("scoreMode may only be 'label' 'bestScore' or 'allScores'")
#	multiClassStrategy = multiClassStrategy.lower()	
	if multiClassStrategy != 'default' and multiClassStrategy != 'OneVsAll' and multiClassStrategy != 'OneVsOne':
		raise ArgumentException("multiClassStrategy may only be 'default' 'OneVsAll' or 'OneVsOne'")

	# if we have to enfore a classification strategy, we test the learner in question,
	# and call our own strategies if necessary
	if multiClassStrategy != 'default':
		trialResult = checkClassificationStrategy(_shogunBackend, learnerName, arguments)
		# note: these conditionals include a binary return
		if multiClassStrategy == 'OneVsAll' and trialResult != 'OneVsAll':
			UML.umlHelpers.trainAndApplyOneVsAll("shogun." + learnerName, trainX, trainY, testX, arguments=arguments, scoreMode=scoreMode, timer=timer)
		if multiClassStrategy == 'OneVsOne' and trialResult != 'OneVsOne':
			UML.umlHelpers.trainAndApplyOneVsOne("shogun." + learnerName, trainX, trainY, testX, arguments=arguments, scoreMode=scoreMode, timer=timer)

	args = copy.copy(arguments)
	if not isinstance(trainX, UML.data.Base):
		trainObj = UML.createData('Matrix', trainX)
	else: # input is an object
		trainObj = trainX.copy()
	if not isinstance(testX, UML.data.Base):
		if testX is None:
			raise ArgumentException("testX may only be an object derived from UML.data.Base")
		testObj = UML.createData('Matrix', testX)
	else: # input is an object
		testObj = testX.copy()
	

	# used to determine output type
	outputTypeString = None
	trainObjY = None
	# directly assign target values, if present
	if isinstance(trainY, UML.data.Base):
		trainObjY = trainY.copy()
		outputTypeString = trainY.getTypeString()
	# otherwise, isolate the target values from training examples
	elif trainY is not None:
		trainObjY = trainObj.extractFeatures([trainY])	
		outputTypeString = trainX.getTypeString()		
	# trainY could be None for unsupervised learning	
	else:
		outputTypeString = trainX.getTypeString()	

	# necessary format for shogun, also makes the following ops easier
	if trainObjY is not None:
		trainObjY = trainObjY.copyAs(format="Matrix")
	
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
		raise ArgumentException("Length of points in the training data and testing data must be the same")

	# call backend
	try:
		retData = _shogunBackend(learnerName,  trainRawData, trainRawDataY, testRawData, args, scoreMode, timer)
	except ImportError as e:
		print "ImportError: " + str(e)
		if not shogunPresent():
			print "Shogun not importable."
			print "It must be either on the search path, or have its path set by setshogunLocation()"
		raise e

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

	outputObj.writeFile(output, format='csv', includeNames=False)


def _shogunBackend(learnerName, trainX, trainY, testX, algArgs, scoreMode, timer=None):
	"""
	Function to find, construct, and execute the wanted calls to shogun

	"""	
	moduleName = findModule(learnerName, "shogun", shogunDir)		

	if moduleName is None:
		raise ArgumentException("Could not find the learner")

	putOnSearchPath(shogunDir)
	exec "from shogun import " + moduleName in locals()

	# make object
	objectCall = moduleName + '.' + learnerName
	SGObj = eval(objectCall + "()")

	# convert data to shogun friendly format
	from shogun.Features import RealFeatures
	from shogun.Features import SparseRealFeatures

	if scipy.sparse.issparse(trainX):
		trainFeat = SparseRealFeatures()
		trainFeat.set_sparse_feature_matrix(trainX.tocsc().astype(numpy.float))
	else:
		trainFeat = RealFeatures()
		trainFeat.set_feature_matrix(numpy.array(trainX, dtype=numpy.float))

	if scipy.sparse.issparse(testX):
		testFeat = SparseRealFeatures()
		testFeat.set_sparse_feature_matrix(testX.tocsc().astype(numpy.float))
	else:
		testFeat = RealFeatures()
		testFeat.set_feature_matrix(numpy.array(testX, dtype=numpy.float))

	# set up the correct type of label
	try:
		import shogun.Classifier
		inverseMapping = None
		tempObj = UML.createData('Matrix', trainY)
		problemType = SGObj.get_machine_problem_type()
		if problemType == shogun.Classifier.PT_MULTICLASS:
			inverseMapping = remapLabelsRange(tempObj)
			if len(inverseMapping) == 1:
				raise ArgumentException("Cannot train a classifier with data containing only one label")
			from shogun.Features import MulticlassLabels
			flattened = numpy.array(tempObj.data).flatten()
			trainLabels = MulticlassLabels(flattened.astype(float))
		elif problemType == shogun.Classifier.PT_BINARY:
			if scoreMode == 'test':
				return 'binary'
			inverseMapping = remapLabelsSpecific(tempObj, [-1,1])
			if len(inverseMapping) == 1:
				raise ArgumentException("Cannot train a classifier with data containing only one label")
			from shogun.Features import BinaryLabels
			flattened = numpy.array(tempObj.data).flatten()
			trainLabels = BinaryLabels(flattened.astype(float))
		elif problemType == shogun.Classifier.PT_REGRESSION:
			from shogun.Features import RegressionLabels
			trainLabels = RegressionLabels(trainY.astype(float))
			if scoreMode != 'label':
				raise ArgumentException("Invalid scoreMode for a regression problem; the default parameter must be used")
		else:
			raise ArgumentException("Learner problem type (" + str(problemType) + ") not supported")

	except ImportError:
		from shogun.Features import Labels
		trainLabels = Labels(trainY.astype(float))

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
		labelOrder = numpy.unique(trainY)
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
		outputObj = UML.createData('Matrix', outData)
		outputObj.applyToElements(makeInverseMapper(inverseMapping), features=0)
		outData = outputObj.data

	return outData

def makeInverseMapper(inverseMappingParam):
	def inverseMapper(value):
		return inverseMappingParam[int(value)]
	return inverseMapper

def listShogunLearners():
	"""
	Function to return a list of all learners callable through our interface, if shogun is present
	
	"""
	if not shogunPresent():
		return []

	import shogun

	ret = []
	seen = {}
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
			if (not member in seen) and (not member in excludedLearners):
				if 'train' in memberContents and 'apply' in memberContents:
					ret.append(member)
					seen[member] = True

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



def remapLabelsRange(toRemap):
	"""
	Takes the object toRemap, which must be a data representation with a single point,
	and maps the values in that point into the range between 0 and n-1, where n is
	the number of distinct values in that feature. The object is modified, and the
	inverse mapping is returned a length n list

	"""

	mapping = {}
	inverse = []
	invIndex = 0

	view = toRemap.pointView(0)

	for x in xrange(toRemap.featureCount):
		value = view[x]
		if value not in mapping:
			mapping[value] = invIndex
			inverse.append(value)
			invIndex += 1
		view[x] = mapping[value]

	return inverse

def remapLabelsSpecific(toRemap, space):
	"""
	Takes the object toRemap, which must be a data representation with a single point
	containing as many unique values as the length of parameter space, and maps those
	values into the values specified in space, on a first come first served basis. The
	object is modified, and the inverse mapping is returned as a list with the same
	length as space.

	If there are more than unique values than values in space, an ArgumentException is raised

	"""

	mapping = {}
	inverse = []
	invIndex = 0
	maxLength = len(space)

	view = toRemap.pointView(0)

	for x in xrange(toRemap.featureCount):
		value = view[x]
		if value not in mapping:
			mapping[value] = invIndex
			inverse.append(value)
			invIndex += 1
			if invIndex > maxLength:
				if space == [-1,1]:
					raise ArgumentException("Multiclass training data cannot be used by a binary-only classifier")
				else:
					raise ArgumentException("toRemap contains more values than can be mapped into the provided space.")

	for x in xrange(toRemap.featureCount):
		value = view[x]
		view[x] = space[mapping[value]]

	return inverse


def constructObject(objectName, allArgs):
	"""
	Construct the in-shogun object with the given name, taking parameters
	out of the given arguments. Returns the constructed object.

	"""
	ret = None
	moduleName = findModule(objectName, "shogun", shogunDir)		

	if moduleName is None:
		raise ArgumentException("Could not find a constructor for the object " + objectName)

	putOnSearchPath(shogunDir)
	exec "from shogun import " + moduleName in locals()	

	# if we have shogun source, parse from in there
	if shogunDir is not None:
		possibleFileNames = ['C' + objectName, objectName, objectName[1:]]

		pass
	# else: use manifest
	else:
		# load manifest
		manifestPath = os.path.join(UML.UMLPath, 'interfaces', 'metadata', 'shogunConstructors')
		manifestFile = open(manifestPath, 'r')
		allConstructors = json.load(manifestFile)
		if not objectName in allConstructors:
			return ret

		# get parameter orderings, see how they match with the args we have
		paramOrderings = allConstructors[objectName]
		argStrings = []
		for order in paramOrderings:
			argStrings.append(makeArgString(order,allArgs, "", "", "", False))

		#we define the best one to be the longest non-None argument string
		best = []
		for args in argStrings:
			if args is not None and len(args) > len(best):
				best = args

		# make object using the best args
		objectCall = moduleName + '.' + objectName
		ret = eval(objectCall + "(" + best + ")")

	return ret


def getParameters(name):
	"""
	Takes the name of some mlpy object or function, returns a list
	of parameters used to instantiate that object or run that function

	"""
	if name == 'SVMOcas':
		return [[]]
	(objArgs,v,k,d) = _paramQuery(name)
	return [objArgs]

def getDefaultValues(name):
	"""
	Takes the name of some mlpy object or function, returns a dict mapping
	parameter names to their default values 

	"""
	if getParameters(name) == None:
		return None
	return [{}]



excludedLearners = [# parent classes, not actually runable
			'BaseMulticlassMachine', 
			'CDistanceMachine',
			'CSVM', 
			'KernelMachine', 
			'KernelMulticlassMachine',
			'LinearLatentMachine', 
			'LinearMachine',
			'MKL',
			'Machine', 
			'MulticlassMachine', 
			'MulticlassSVM',
			'MultitaskLinearMachineBase',
			'NativeMulticlassMachine',
			'OnlineLinearMachine', 
			'ScatterSVM', # unstable method
			'TreeMachineWithConditionalProbabilityTreeNodeData',
			'TreeMachineWithRelaxedTreeNodeData', 

			# Should be implemented, but don't work
			'BalancedConditionalProbabilityTree', # streaming dense features input
			'ConditionalProbabilityTree', 	
			'DomainAdaptationSVM',
			'DualLibQPBMSOSVM',  # problem type 3
			'FeatureBlockLogisticRegression',  # remapping
			'KernelStructuredOutputMachine', # problem type 3
			'LatentSVM', # problem type 4
			'LibSVMOneClass',
			'LinearMulticlassMachine', # mixes machines. is this even possible to run?
			'LinearStructuredOutputMachine', # problem type 3
			'MKLMulticlass', # needs combined kernel type?
			'MKLClassification', # compute by subkernel not implemented
			'MKLOneClass', # Interleaved MKL optimization is currently only supported with SVMlight
			'MKLRegression',  # kernel stuff?
			'MultitaskClusteredLogisticRegression',   # assertion error
			'MultitaskCompositeMachine',  # takes machine as input?
			'MultitaskL12LogisticRegression',  # assertion error
			'MultitaskLeastSquaresRegression',  #core dump
			'MultitaskLogisticRegression',  #core dump
			'MultitaskTraceLogisticRegression',  # assertion error
			'OnlineLibLinear', # needs streaming dot features
			'OnlineSVMSGD', # needs streaming dot features
			'PluginEstimate', # takes string inputs?
			'RandomConditionalProbabilityTree',  # takes streaming dense features
			'RelaxedTree', # [ERROR] Call set_machine_for_confusion_matrix before training
			'ShareBoost', # non standard input
			'StructuredOutputMachine', # problem type 3
			'SubGradientSVM', #doesn't terminate
			'VowpalWabbit',  # segfault
			'WDSVMOcas', # string input

			# functioning learners
			#'AveragedPerceptron'
			#'GaussianNaiveBayes',
			#'GMNPSVM', 
			#'GNPPSVM', 
			#'GPBTSVM',
			#'Hierarchical',
			#'KMeans', 
			#'KNN', 
			#'LaRank',
			#'LibLinear',
			#'LibSVM',
			#'LibSVR',
			#'MPDSVM',
			#'MulticlassLibLinear',
			#'MulticlassLibSVM',
			#'Perceptron',
			#'SGDQN',
			#'SVMLight',
			#'SVMLightOneClass',
			#'SVMLin',
			#'SVMOcas', 
			#'SVMSGD',
			#'SVRLight',
			]

