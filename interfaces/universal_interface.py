"""




"""

import inspect
import copy
import abc
import functools
import numpy

import UML
from UML.exceptions import ArgumentException
from UML.interfaces.interface_helpers import generateBinaryScoresFromHigherSortedLabelScores
from UML.interfaces.interface_helpers import calculateSingleLabelScoresFromOneVsOneScores
from UML.interfaces.interface_helpers import ovaNotOvOFormatted
from UML.interfaces.interface_helpers import checkClassificationStrategy
from UML.interfaces.interface_helpers import cacheWrapper

from UML.umlHelpers import _mergeArguments

class UniversalInterface(object):
	"""

	"""

	__metaclass__ = abc.ABCMeta

	_listLearnersCached = None

	def __init__(self):
		"""

		"""
		### Validate all the information from abstract functions ###
		# enforce a check that the underlying package is accessible at instantiation,
		# aborting the construction of the interface for this session of UML if
		# it is not.
		if not self.accessible():
			raise ImportError("The underlying package for " + self.getCanonicalName() + " was not accessible, aborting instantiation.")

		# getCanonicalName
		if not isinstance(self.getCanonicalName(), str):
			raise TypeError("Improper implementation of getCanonicalName(), must return a string")
		
		# _configurableOptionNames and _optionDefaults
		optionNames = self._configurableOptionNames()
		if not isinstance(optionNames, list):
			raise TypeError("Improper implementation of _configurableOptionNames(), must return a list of strings") 
		for optionName in optionNames:
			if not isinstance(optionName, str):
				raise TypeError("Improper implementation of _configurableOptionNames(), must return a list of strings")
			# make a call to _optionDefaults just to make sure it doesn't throw an exception
			self._optionDefaults(optionName)
		
		# _exposedFunctions
		exposedFunctions = self._exposedFunctions()
		if exposedFunctions is None or not isinstance(exposedFunctions, list):
			raise TypeError("Improper implementation of _exposedFunctions(), must return a list of methods to be bundled with TrainedLearner")
		for exposed in exposedFunctions:
			# is callable
			if not hasattr(exposed, '__call__'):
				raise TypeError("Improper implementation of _exposedFunctions, each member of the return must have __call__ attribute")
			# has name attribute
			if not hasattr(exposed, '__name__'):
				raise TypeError("Improper implementation of _exposedFunctions, each member of the return must have __name__ attribute")
			# takes self as attribute
			(args, varargs, keywords, defaults) = inspect.getargspec(exposed)
			if args[0] != 'self':
				raise TypeError("Improper implementation of _exposedFunctions each member's first argument must be 'self', interpreted as a TrainedLearner")

	@property
	def optionNames(self):
		return copy.copy(self._configurableOptionNames())

	def trainAndApply(self, learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', timer=None):
		
		learner = self.train(learnerName, trainX, trainY, arguments, timer)
		if timer is not None:
			timer.start('apply')
		# call TrainedLearner's apply method (which is already wrapped to perform transformation)
		ret = learner.apply(testX, {}, output, scoreMode)
		if timer is not None:
			timer.stop('apply')

		return ret

	def trainAndTest(self, learnerName, trainX, trainY, testX, testY, performanceFunction, arguments={}, output='match', scoreMode='label', negativeLabel=None, timer=None, **kwarguments):
		learner = self.train(learnerName, trainX, trainY, arguments, timer)
		if timer is not None:
			timer.start('test')
		# call TrainedLearner's test method (which is already wrapped to perform transformation)
		ret = learner.test(testX, testY, performanceFunction, {}, output, scoreMode, negativeLabel)
		if timer is not None:
			timer.stop('test')

		return ret

	def train(self, learnerName, trainX, trainY=None, arguments={}, timer=None):
		(trainedBackend, transformedInputs, customDict) = self._trainBackend(learnerName, trainX, trainY, arguments, timer)	
		
		# encapsulate into TrainedLearner object
		return self.TrainedLearner(learnerName, arguments, transformedInputs, customDict, trainedBackend, self)

	def _confirmValidLearner(self, learnerName):
		allLearners = self.listLearners()
		if not learnerName in allLearners:
			raise ArgumentException("" + learnerName + " is not the name of a learner exposed by this interface")
		learnerCall = self.findCallable(learnerName)
		if learnerCall is None:
			raise ArgumentException("" + learnerName + " was not found in this package")
			

	def _trainBackend(self, learnerName, trainX, trainY, arguments, timer):		
		### PLANNING ###

		# verify the learner is available
		self._confirmValidLearner(learnerName)

		#validate argument distributions
		groupedArgsWithDefaults = self._validateArgumentDistribution(learnerName, arguments)

		### INPUT TRANSFORMATION ###

		#recursively work through arguments, doing in-package object instantiation
		instantiatedInputs = self._instantiateArguments(learnerName, groupedArgsWithDefaults)

		# the scratch space dictionary that the package implementor may use to pass information
		# between I/O transformation, the trainer and applier
		customDict = {}

		# separate training data / labels if needed
		if isinstance(trainY, (basestring, int)):
			trainX = trainX.copy()
			trainY = trainX.extractFeatures(toExtract=trainY)

		# execute interface implementor's input transformation.
		transformedInputs = self._inputTransformation(learnerName, trainX, trainY, None, instantiatedInputs, customDict)
		(transTrainX, transTrainY, transTestX, transArguments) = transformedInputs

		### LEARNER CREATION / TRAINING ###

		# train the instantiated learner
		if timer is not None:
			timer.start('train')
		trainedBackend = self._trainer(learnerName, transTrainX, transTrainY, transArguments, customDict)
		if timer is not None:
			timer.stop('train')

		return (trainedBackend, transformedInputs, customDict)

	def setOption(self, option, value):
		if option not in self.optionNames:
			raise ArgumentException(str(option) + " is not one of the accepted configurable option names")
		
		UML.settings.set(self.getCanonicalName(), option, value)

	def getOption(self, option):
		if option not in self.optionNames:
			raise ArgumentException(str(option) + " is not one of the accepted configurable option names")
		
		# empty string is the sentinal value indicating that the configuration
		# file has an option of that name, but the UML user hasn't set a value
		# for it.
		ret = ''
		try:
			ret = UML.settings.get(self.getCanonicalName(), option)
		except:
			# it is possible that the config file doesn't have an option of
			# this name yet. Just pass through and grab the hardcoded default
			pass
		if ret == '':
			ret = self._optionDefaults(option)
		return ret


	def _validateArgumentDistribution(self, learnerName, arguments):
		"""
		We check that each call has all the needed arguments, that in total we are
		using each argument only once, and that we use them all.

		return a copy of the arguments that has been arranged for easy instantiation

		"""
		baseCallName = learnerName
		possibleParamSets = self.getLearnerParameterNames(learnerName)	
		possibleDefaults = self.getLearnerDefaultValues(learnerName)
		bestIndex = self._chooseBestParameterSet(possibleParamSets, possibleDefaults, arguments)
#		if bestSet is None:
#			raise ArgumentException("Missing arguments")
		(neededParams, availableDefaults) = (possibleParamSets[bestIndex], possibleDefaults[bestIndex])
		available = copy.deepcopy(arguments)
		(ret, ignore) = self._validateArgumentDistributionHelper(baseCallName, neededParams, availableDefaults, available, False)
		return ret

	def _isInstantiable(self, val, hasDefault, defVal):
			if hasDefault and isinstance(defVal, basestring):
				return False
			if isinstance(val, basestring) and self.findCallable(val) is not None:
				return True

			return False

	def _validateArgumentDistributionHelper(self, currCallName, currNeededParams, currDefaults, available, sharedPool):
		"""
		Recursive function for actually performing _validateArgumentDistribution. Will recurse
		when encountering shorthand for making in package calls, where the desired arguments
		are in another dictionary.

		"""
		ret = {}
		# key: key value in ret for accessing appropriate subargs set
		# value: list of key value pair to replace in that set
		delayedAllocations = {None:[]}
		# dict where the key matches the param name of the thing that will be
		# instantiated, and the value is a triple, consisting of the the avaiable value,
		# the values consused from available, and the additions to dellayedAllocations
		delayedInstantiations = {}
		#work through this level's needed parameters
		for paramName in currNeededParams:
#			if paramName == 'kernel':
#				import pdb
#				pdb.set_trace()
			# is the param actually there? Is there a default associated with it?
			present = paramName in available
			hasDefault = paramName in currDefaults

			# In each conditional, we have three main tasks: identifying what values will
			# be used, book keeping for that value (removal, delayed allocation / instantiation),
			# and adding values to ret
			if present and hasDefault:				
				paramValue = available[paramName]
				paramDefault = currDefaults[paramName]
				addToDelayedIfNeeded = True
				if self._isInstantiable(paramValue, True, paramDefault) or self._isInstantiable(paramDefault, True, paramDefault):
					availableBackup = copy.deepcopy(available)
					allocationsBackup = copy.deepcopy(delayedAllocations)
				
				if self._isInstantiable(paramDefault, True, paramDefault):
					# try recursive call using default value
					try:
						self._setupValidationRecursiveCall(paramDefault, available, ret, delayedAllocations)
					except:
					# if fail, try recursive call using actual value and copied available
						available = availableBackup
						self._setupValidationRecursiveCall(paramValue, available, ret, delayedAllocations)
						del available[paramName]
						addToDelayedIfNeeded = False
				else:
					ret[paramName] = paramDefault
				
				if not self._isInstantiable(paramValue, True, paramDefault):
					# mark down to use the real value if it isn't allocated elsewhere
					delayedAllocations[None].append((paramName, paramValue))
				else:
					if addToDelayedIfNeeded:
						availableChanges = {}
						for keyBackup in availableBackup:
							valueBackup = availableBackup[keyBackup]
							if keyBackup not in available:
								availableChanges[keyBackup] = valueBackup
						delayedInstantiations[paramName] = (available[paramName], availableChanges, allocationsBackup)

			elif present and not hasDefault:
				paramValue = available[paramName]
				# is it something that needs to be instantiated and therefore needs params of its own?
				if self._isInstantiable(paramValue, False, None):
					self._setupValidationRecursiveCall(paramValue, available, ret, delayedAllocations)
				del available[paramName]
				ret[paramName] = paramValue
			elif not present and hasDefault:
				paramValue = currDefaults[paramName]
				# is it something that needs to be instantiated and therefore needs params of its own?
				# TODO is findCallable really the most reliable trigger for this? maybe we should check
				# that you can get params from it too ....
				#if isInstantiable(paramValue, True, paramValue):		
				#	self._setupValidationRecursiveCall(paramValue, available, ret, delayedAllocations)
				ret[paramName] = currDefaults[paramName]
			# not present and no default
			else:
				raise ArgumentException("Missing parameter named '" + paramName + "' in for call to '" + currCallName + "'")

		# if this pool of arguments is not shared, then this is the last subcall,
		# and we can finalize the allocations
		if not sharedPool:
			# work through list of instantiable arguments which were tentatively called using
			# defaults
			for key in delayedInstantiations.keys():
				(value, used, allocations) = delayedInstantiations[key]
				# check to see if it is still in available
				if key in available:
					# undo the changes made by the default call
					used.update(available)
					# make recursive call instead with the actual value
					#try:
					self._setupValidationRecursiveCall(value, used, ret, delayedAllocations)
					available = used
					ret[key] = value
					del available[key]
					#except:
						# if fail, keep the results of the call with the default
					#	pass

			# work through a list of possible keys for the delayedAllocations dict,
			# if there are allocations associated with that key, perform them. 
			for possibleKey in delayedAllocations.keys():
				changesList = delayedAllocations[possibleKey]
				for (k,v) in changesList:
					if k in available:
						if possibleKey is None:
							ret[k] = v
						else:
							ret[possibleKey][k] = v
						del available[k]

			# at this point, everything should have been used, and then removed.
			if len(available) != 0:
				raise ArgumentException("Extra params in arguments: " + str(available))

			delayedAllocations = {}		

		return (ret, delayedAllocations)

	def _setupValidationRecursiveCall(self, paramValue, available, callingRet, callingAllocations):
		# are the params for this value in a restricted argument pool?
		if paramValue in available:
			subSource = available[paramValue]
			subShared = False
			# We can and should do this here because if there is ever another key with paramVale
			# as the value, then it will be functionally equivalent to save these args for then
			# as it would be to use them here. So, we do the easy thing, and consume them now.
			del available[paramValue]
		# else, they're in the main, shared, pool
		else:
			subSource = available
			subShared = True

		# where we get the wanted parameter set from depends on the kind of thing that we
		# need to instantiate
		if paramValue in self.listLearners():
			subParamGroup = self.getLearnerParameterNames(paramValue)
			subDefaults = self.getLearnerDefaultValues(paramValue)
		else:
			subParamGroup = self._getParameterNames(paramValue)
			subDefaults = self._getDefaultValues(paramValue)

		bestIndex = self._chooseBestParameterSet(subParamGroup, subDefaults, subSource)
		(subParamGroup, subDefaults) = (subParamGroup[bestIndex], subDefaults[bestIndex])

		(ret, allocations) = self._validateArgumentDistributionHelper(paramValue, subParamGroup, subDefaults, subSource, subShared)

		# integrate the returned values into the state of the calling frame
		callingRet[paramValue] = ret
		if subShared:
			for pair in allocations[None]:
				if paramValue not in callingAllocations:
					callingAllocations[paramValue] = []
				callingAllocations[paramValue].append(pair)


	def _instantiateArguments(self, learnerName, arguments):
		"""
		Recursively consumes the contents of the arguments parameter, checking for ones
		that need to be instantiated using in-package calls, and performing that
		action if needed. Returns a new dictionary with the same contents as 'arguments',
		except with the replacement of in-package objects for their string names

		"""
		baseCallName = learnerName
		baseNeededParams = self._getParameterNames(learnerName)	
		toProcess = copy.deepcopy(arguments)
		return self._instantiateArgumentsHelper(toProcess)

	def _instantiateArgumentsHelper(self, toProcess):
		"""
		Recursive function for actually performing _instantiateArguments. Will recurse
		when encountering shorthand for making in package calls, where the desired arguments
		are in another dictionary.

		"""
		ignoreKeys = []
		ret = {}
		for paramName in toProcess:
			paramValue = toProcess[paramName]
			if isinstance(paramValue, basestring):
				ignoreKeys.append(paramValue)
				toCall = self.findCallable(paramValue)
				# if we can find an object for it, and we've prepped the arguments,
				# then we actually instantiate an object
				if toCall is not None and paramValue in toProcess:
					subInitParams = toProcess[paramValue]
					if subInitParams is None:
						ret[paramName] = paramValue
						continue
					instantiatedParams = self._instantiateArgumentsHelper(subInitParams)
					paramValue = toCall(**instantiatedParams)
			
			ret[paramName] = paramValue

		for key in ignoreKeys:
			if key in ret:
				del ret[key]

		return ret 

	def _chooseBestParameterSet(self, possibleParamsSets, matchingDefaults, arguments):
#		import pdb
#		pdb.set_trace()
#		print possibleParamsSets
#		print arguments
		success = False
		missing = []
		bestParams = []
		for i in range(len(possibleParamsSets)):
				missing.append([])
		bestIndex = None
		for i in range(len(possibleParamsSets)):
			currParams = possibleParamsSets[i]
			currDefaults = matchingDefaults[i]
			allIn = True
			for param in currParams:
				if param not in arguments and param not in currDefaults:
					allIn = False
					missing[i].append(param)
			if allIn and len(currParams) >= len(bestParams):
				bestIndex = i
				success = True
		if not success:
			raise ArgumentException("Missing required params in each possible set: " + str(missing))
		return bestIndex

	def _formatScoresToOvA(self, learnerName, learner, testX, applyResults, rawScores, arguments, customDict):
		"""
		Helper that takes raw scores in any of the three accepted formats (binary case best score,
		one vs one pairwise tournament by natural label ordering, or one vs all by natural label
		ordering) and returns them in a one vs all accepted format.

		"""
		order = self._getScoresOrder(learner)
		numLabels = len(order)
		if numLabels == 2 and rawScores.featureCount == 1:
			ret = generateBinaryScoresFromHigherSortedLabelScores(rawScores)
			return UML.createData("Matrix", ret)

		if applyResults is None:
			applyResults = self._applier(learner, testX, arguments, customDict)
			applyResults = self._outputTransformation(learnerName, applyResults, arguments, "match", "label", customDict)
		if rawScores.featureCount != 3:
			strategy = ovaNotOvOFormatted(rawScores, applyResults, numLabels)
		else:
			strategy = checkClassificationStrategy(self, learnerName, arguments)
		# we want the scores to be per label, regardless of the original format, so we
		# check the strategy, and modify it if necessary
		if not strategy:
			scores = []
			for i in xrange(rawScores.pointCount):
				combinedScores = calculateSingleLabelScoresFromOneVsOneScores(rawScores.pointView(i), numLabels)
				scores.append(combinedScores)
			scores = numpy.array(scores)
			return UML.createData("Matrix", scores)
		else:
			return rawScores

	class TrainedLearner():

		def __init__(self, learnerName, arguments, transformedInputs, customDict, backend, interfaceObject):
			"""
			Initialize the object wrapping the trained learner stored in backend, and setting up
			the object methods that may be used to modify or query the backend trained learner.

			learnerName: the name of the learner used in the backend
			arguments: reference to the original arguments parameter to the trainAndApply() function
			transformedArguments: a tuple containing the return value of _inputTransformation() that was called when training the learner in the backend
			customDict: reference to the customizable dictionary that is passed to I/O transformation, training and applying a learner
			backend: the return value from _trainer(), a reference to a some object that is to be used by the package implementor during application
			interfaceObject: a reference to the subclass of UniversalInterface from which this TrainedLearner is being instantiated.

			"""
			self.learnerName = learnerName
			self.arguments = arguments
			self.transformedTrainX = transformedInputs[0]
			self.transformedTrainY = transformedInputs[1]
			self.transforedTestX = transformedInputs[2]
			self.transformedArguments = transformedInputs[3]
			self.customDict = customDict
			self.backend = backend
			self.interface = interfaceObject

			exposedFunctions = self.interface._exposedFunctions()
			for exposed in exposedFunctions:
				methodName = getattr(exposed, '__name__')
				(args, varargs, keywords, defaults) = inspect.getargspec(exposed)
				if 'trainedLearner' in args:
					wrapped = functools.partial(exposed, trainedLearner=self)
					wrapped.__doc__ = 'Wrapped version of the ' + methodName + ' function where the "trainedLearner" parameter has been fixed as this object, and the "self" parameter has been fixed to be ' + str(interfaceObject) 
				else:
					wrapped = functools.partial(exposed)
					wrapped.__doc__ = 'Wrapped version of the ' + methodName + ' function where the "self" parameter has been fixed to be ' + str(interfaceObject) 
				setattr(self, methodName, wrapped)

		def test(self, testX, testY, performanceFunction, arguments={}, output='match', scoreMode='label', negativeLabel=None, **kwarguments):
			"""
			Returns the evaluation of predictions of testX using the argument
			performanceFunction to do the evalutation. Equivalent to having called
			this interface's trainAndTest method, as long as the data and parameter
			setup for training was the same.

			"""
			pred = self.apply(testX, arguments, output, scoreMode, **kwarguments)
			performance = UML.umlHelpers.computeMetrics(testY, None, pred, performanceFunction, negativeLabel)

			return performance

		def _mergeWithTrainArguments(self, newArguments1, newArguments2):
			"""
			When calling apply, merges our fixed arguments with our provided arguments,
			giving the new arguments precedence if needed.

			"""
			ret = _mergeArguments(self.transformedArguments, newArguments1)
			ret = _mergeArguments(ret, newArguments2)
			return ret

		def apply(self, testX, arguments={}, output='match', scoreMode='label', **kwarguments):
			"""
			Returns the application of this learner to the given test data (i.e. performing
			prediction, transformation, etc. as appropriate to the learner). Equivalent to
			having called trainAndApply with the same same setup as this learner was trained
			on.

			"""
#			self.interface._validateOutputFlag(output)
#			self.interface._validateScoreModeFlag(scoreMode)
			usedArguments = self._mergeWithTrainArguments(arguments, kwarguments)

			# input transformation
			(trainX, trainY, transTestX, usedArguments) = self.interface._inputTransformation(self.learnerName, None, None, testX, usedArguments, self.customDict)

			# depending on the mode, we need different information.
			labels = None
			if scoreMode != 'label':
				scores = self.getScores(testX, usedArguments)
			if scoreMode != 'allScores':
				labels = self.interface._applier(self.backend, transTestX, usedArguments, self.customDict)
				labels = self.interface._outputTransformation(self.learnerName, labels, usedArguments, output, "label", self.customDict)

			if scoreMode == 'label':
				return labels
			elif scoreMode == 'allScores':
				return scores
			else:
				scoreOrder = self.interface._getScoresOrder(self.backend)
				scoreOrder = list(scoreOrder)
				# find scores matching predicted labels
				def grabValue(row):
					pointIndex = row.index()
					rowIndex = scoreOrder.index(labels[pointIndex, 0])
					return row[rowIndex]

				scoreVector = scores.applyToPoints(grabValue, inPlace=False)
				labels.appendFeatures(scoreVector)

				return labels

		def retrain(self, trainX, trainY=None):
#			(trainX, trainY, testX, arguments) = self.interface._inputTransformation(self.learnerName,trainX, trainY, None, self.arguments, self.customDict)
			(newBackend, transformedInputs, customDict) = self.interface._trainBackend(self.learnerName, trainX, trainY, self.arguments, None)
			self.backend = newBackend
			self.transformedInputs = transformedInputs
			self.customDict = customDict

		def incrementalTrain(self, trainX, trainY=None):
			(trainX, trainY, testX, arguments) = self.interface._inputTransformation(self.learnerName,trainX, trainY, None, self.arguments, self.customDict)
			self.backend = self.interface._incrementalTrainer(self.backend, trainX, trainY, arguments, self.customDict)

		def getAttributes(self):
			return self.interface._getAttributes(self.backend)

		def getScores(self, testX, arguments={}, **kwarguments):
			"""
			Returns the scores for all labels for each data point. If this TrainedLearner
			is named foo, this operation is equivalent to calling foo.apply with
			'scoreMode="allScores"' 

			"""
			usedArguments = self._mergeWithTrainArguments(arguments, kwarguments)
			(trainX, trainY, testX, usedArguments) = self.interface._inputTransformation(self.learnerName, None, None, testX, usedArguments, self.customDict)
			
			rawScores = self.interface._getScores(self.backend, testX, usedArguments, self.customDict)
			umlTypeRawScores = self.interface._outputTransformation(self.learnerName, rawScores, usedArguments, "Matrix", "allScores", self.customDict)
			formatedRawOrder = self.interface._formatScoresToOvA(self.learnerName, self.backend, testX, None, umlTypeRawScores, usedArguments, self.customDict)
			internalOrder = self.interface._getScoresOrder(self.backend)
			naturalOrder = sorted(internalOrder)
			if numpy.array_equal(naturalOrder, internalOrder):
				return formatedRawOrder
			desiredDict = {}
			for i in range(len(naturalOrder)):
				label = naturalOrder[i]
				desiredDict[label] = i
			def sortScorer(feature):
				label = internalOrder[feature.index()]
				return desiredDict[label]

			formatedRawOrder.sortFeatures(sortHelper=sortScorer)
			return formatedRawOrder

	##############################################
	### CACHING FRONTENDS FOR ABSTRACT METHODS ###
	##############################################

	def listLearners(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		isCustom = isinstance(self, UML.interfaces.CustomLearnerInterface)
		if self._listLearnersCached is None:
			ret = self._listLearnersBackend()
			if not isCustom:
				self._listLearnersCached = ret
		else:
			ret = self._listLearnersCached
		return ret

	@cacheWrapper
	def findCallable(self, name):
		"""
		Find reference to the callable with the given name
		TAKES string name
		RETURNS reference to in-package function or constructor
		"""
		return self._findCallableBackend(name)

	@cacheWrapper
	def _getParameterNames(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		return self._getParameterNamesBackend(name)

	@cacheWrapper
	def getLearnerParameterNames(self, learnerName):
		"""
		Find all parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of list of param names
		"""
		return self._getLearnerParameterNamesBackend(learnerName)	

	@cacheWrapper
	def _getDefaultValues(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""
		return self._getDefaultValuesBackend(name)

	@cacheWrapper
	def getLearnerDefaultValues(self, learnerName):
		"""
		Find all default values for parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of dict of param names to default values
		"""
		return self._getLearnerDefaultValuesBackend(learnerName)

	########################
	### ABSTRACT METHODS ###
	########################

	@abc.abstractmethod
	def accessible(self):
		"""
		Return true if the package underlying this interface is currently accessible,
		False otherwise.

		"""
		pass

	@abc.abstractmethod
	def _listLearnersBackend(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		pass

	@abc.abstractmethod
	def _findCallableBackend(self, name):
		"""
		Find reference to the callable with the given name
		TAKES string name
		RETURNS reference to in-package function or constructor
		"""
		pass

	@abc.abstractmethod
	def _getParameterNamesBackend(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		pass

	@abc.abstractmethod
	def _getLearnerParameterNamesBackend(self, learnerName):
		"""
		Find all parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of list of param names
		"""
		pass	

	@abc.abstractmethod
	def _getDefaultValuesBackend(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""
		pass

	@abc.abstractmethod
	def _getLearnerDefaultValuesBackend(self, learnerName):
		"""
		Find all default values for parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of dict of param names to default values
		"""
		pass

	@abc.abstractmethod
	def learnerType(self, name):
		"""
		Returns a string referring to the action the learner takes out of the possibilities:
		classifier, regressor, featureSelection, dimensionalityReduction 
		TODO

		"""
		pass

	@abc.abstractmethod
	def _getScores(self, learner, testX, arguments, customDict):
		"""
		If the learner is a classifier, then return the scores for each
		class on each data point, otherwise raise an exception.

		"""
		pass

	@abc.abstractmethod
	def _getScoresOrder(self, learner):
		"""
		If the learner is a classifier, then return a list of the the labels corresponding
		to each column of the return from getScores

		"""
		pass

	@abc.abstractmethod
	def isAlias(self, name):
		"""
		Returns true if the name is an accepted alias for this interface

		"""
		pass


	@abc.abstractmethod
	def getCanonicalName(self):
		"""
		Returns the string name that will uniquely identify this interface

		"""
		pass

	@abc.abstractmethod
	def _inputTransformation(self, learnerName, trainX, trainY, testX, arguments, customDict):
		"""
		Method called before any package level function which transforms all
		parameters provided by a UML user.

		trainX, trainY, and testX are filled with the values of the parameters of the same name
		to a call to trainAndApply() or train() and are sometimes empty when being called
		by other functions. For example, a call to apply() will have trainX and trainY be None.
		The arguments parameter is a dictionary mapping names to values of all other
		parameters associated with the learner, each of which may need to be processed.

		The return value of this function must be a tuple mirroring the structure of
		the inputs. Specifically, four values are required: the transformed versions of
		trainX, trainY, testX, and arguments in that specific order.

		"""
		pass

	@abc.abstractmethod
	def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputType, outputFormat, customDict):
		"""
		Method called before any package level function which transforms the returned
		value into a format appropriate for a UML user.

		"""
		pass

	@abc.abstractmethod
	def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
		"""
		build a learner and perform training with the given data
		TAKES name of learner, transformed arguments
		RETURNS an in package object to be wrapped by a TrainedLearner object
		"""
		pass

	@abc.abstractmethod
	def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
		"""
		Given an already trained online learner, extend it's training with the given data
		TAKES trained learner, transformed arguments,
		RETURNS the learner after this batch of training
		"""
		pass


	@abc.abstractmethod
	def _applier(self, learner, testX, arguments, customDict):	
		"""
		use the given learner to do testing/prediction on the given test set
		TAKES a TrainedLearner object that can be tested on 
		RETURNS UML friendly results
		"""
		pass


	@abc.abstractmethod
	def _getAttributes(self, learnerBackend):
		"""
		Returns whatever attributes might be available for the given learner. For
		example, in the case of linear regression, TODO

		"""
		pass

	@abc.abstractmethod
	def _optionDefaults(self, option):
		"""
		Define package default values that will be used for as long as a default
		value hasn't been registered in the UML configuration file. For example,
		these values will always be used the first time an interface is instantiated.

		"""
		pass


	@abc.abstractmethod
	def _configurableOptionNames(self):
		"""
		Returns a list of strings, where each string is the name of a configurable
		option of this interface whose value will be stored in UML's configuration
		file.

		"""
		pass

	@abc.abstractmethod
	def _exposedFunctions(self):
		"""
		Returns a list of references to functions which are to be wrapped
		in I/O transformation, and exposed as attributes of all TrainedLearner
		objects returned by this interface's train() function. If None, or an
		empty list is returned, no functions will be exposed. Each function
		in this list should be a python function, the inspect module will be
		used to retrieve argument names, and the value of the function's
		__name__ attribute will be its name in TrainedLearner.

		"""
		pass


