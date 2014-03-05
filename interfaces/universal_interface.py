"""




"""

import inspect
import copy
import abc
import functools
import numpy

import UML
from UML.exceptions import ArgumentException


class UniversalInterface(object):
	"""

	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		"""

		"""
		### Validate all the information from abstract functions ###

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
		if exposedFunctions is not None and not isinstance(exposedFunctions, list):
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

		### setup configurable options ###

		# TODO
		# query for defaults
		self._configurableOptions = {}
		for optionName in self._configurableOptionNames():
			# if in configuration, use that
			# elif in _optionDefaults(optionName), use that
			# else
			self._configurableOptions[optionName] = None 

	@property
	def optionNames(self):
		return copy.copy(self._configurableOptionNames())

	def trainAndApply(self, learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', timer=None):

		# TODO remove multiclass strategy

		if timer is not None:
			timer.start('train')
		learner = self.train(learnerName, trainX, trainY, arguments)
		if timer is not None:
			timer.stop('train')
			timer.start('test')
		# call TrainedLearner's apply function (which is already wrapped to perform transformation)
		ret = learner.apply(testX, {}, output, scoreMode)
		if timer is not None:
			timer.stop('test')

		return ret

	def train(self, learnerName, trainX, trainY=None, arguments={}):
		copyTrainX = trainX.copy()
		copyTrainY = trainY
		if isinstance(trainY, UML.data.Base):
			copyTrainY = trainY.copy()
		(trainedBackend, transformedInputs, customDict) = self._trainBackend(learnerName, copyTrainX, copyTrainY, arguments)	
		
		# encapsulate into TrainedLearner object
		return self.TrainedLearner(learnerName, arguments, transformedInputs, customDict, trainedBackend, self)

	def _trainBackend(self, learnerName, trainX, trainY, arguments):		
		### PLANNING ###

		# verify the learner is available
		learnerCall = self.findCallable(learnerName)

		#validate argument distributions
		groupedArgsWithDefaults = self._validateArgumentDistribution(learnerName, arguments)

		### INPUT TRANSFORMATION ###

		#recursively work through arguments, doing in-package object instantiation
		instantiatedInputs = self._instantiateArguments(learnerName, groupedArgsWithDefaults)

		# the scratch space dictionary that the package implementor may use to pass information
		# between I/O transformation, the trainer and applier
		customDict = {}

		# separate training data / labels if needed
		if isinstance(trainY, basestring) or isinstance(trainY, int):
			trainY = trainX.extractFeatures(toExtract=trainY)

		# execute interface implementor's input transformation.
		transformedInputs = self._inputTransformation(learnerName, trainX, trainY, None, instantiatedInputs, customDict)
		(transTrainX, transTrainY, transTestX, transArguments) = transformedInputs

		### LEARNER CREATION / TRAINING ###

		# train the instantiated learner
		trainedBackend = self._trainer(learnerName, transTrainX, transTrainY, transArguments, customDict)

		return (trainedBackend, transformedInputs, customDict)

	def setOption(self, option, value):
		if option not in self.optionNames:
			raise ArgumentException(str(option) + " is not one of the accepted configurable option names")
		self._configurableOptions[option] = value

	def getOption(self, option):
		if option not in self.optionNames:
			raise ArgumentException(str(option) + " is not one of the accepted configurable option names")
		return self._configurableOptions[option]

	def setDefaultOption(self, option, value):
		if option not in self.optionNames:
			raise ArgumentException(str(option) + " is not one of the accepted configurable option names")
		# TODO UML.configure set ... getAlias(), option, value
		pass

	def getDefaultOption(self, option):
		if option not in self.optionNames:
			raise ArgumentException(str(option) + " is not one of the accepted configurable option names")
		# TODO UML.configure get ... getAlias(), option, value
		pass

	def allOptions(self):
		return copy.deepcopy(self._configurableOptions)


	def _validateArgumentDistribution(self, learnerName, arguments):
		"""
		We check that each call has all the needed arguments, that in total we are
		using each argument only once, and that we use them all.

		return a copy of the arguments that has been arranged for easy instantiation

		"""
		baseCallName = learnerName
		possibleParamSets = self.getLearnerParameterNames(learnerName)	
		possibleDefaults = self.getLearnerDefaultValues(learnerName)
		(neededParams, availableDefaults) = self._chooseBestParameterSet(possibleParamSets, possibleDefaults, arguments)
		available = copy.deepcopy(arguments)
		(ret, ignore) = self._validateArgumentDistributionHelper(baseCallName, neededParams, availableDefaults, available, False)
		return ret

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
			# is the param actually there? Is there a default associated with it?
			present = paramName in available
			hasDefault = paramName in currDefaults
			if present:
				if hasDefault:
					paramValue = currDefaults[paramName]
					if isinstance(paramValue, basestring) and self.findCallable(paramValue) is not None:
						#copy state
						availableBackup = copy.deepcopy(available)
						allocationsBackup = copy.deepcopy(delayedAllocations)
						# try: recursive call using default
						try:
							self._setupValidationRecursiveCall(paramValue, available, ret, delayedAllocations)
							# if succeed, add to delayedInstantiations
							availableChanges = {}
							for keyBackup in availableBackup:
								valueBackup = availableBackup[keyBackup]
								if keyBackup not in available:
									availableChanges[keyBackup] = valueBackup
							# TODO delayedAllocation changes
							delayedInstantiations[paramName] = (available[paramName], availableChanges, allocationsBackup)
						except:
						# if fail, try recursive call using actual value and copied available
							available = availableBackup
							paramValue = available[paramName]
							self._setupValidationRecursiveCall(paramValue, available, ret, delayedAllocations)
					else:
						# mark down to use the real value if it isn't allocated elsewhere
						delayedAllocations[None].append((paramName, available[paramName]))
				else:
					paramValue = available[paramName]
					# is it something that needs to be instantiated and therefore needs params of its own?
					if isinstance(paramValue, basestring) and self.findCallable(paramValue) is not None:
						self._setupValidationRecursiveCall(paramValue, available, ret, delayedAllocations)
					if paramValue in available:
						del available[paramValue]
					del available[paramName]
			else:
				if hasDefault:
					paramValue = currDefaults[paramName]
					# is it something that needs to be instantiated and therefore needs params of its own?
					if isinstance(paramValue, basestring) and self.findCallable(paramValue) is not None:			
						self._setupValidationRecursiveCall(paramValue, available, ret, delayedAllocations)
					ret[paramName] = currDefaults[paramName]
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
					try:
						self._setupValidationRecursiveCall(value, used, ret, delayedAllocations)
						available = used
						ret[key] = value
						del available[key]
					except:
						# if fail, keep the results of the call with the default
						pass

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

			delayedAllocations = {}			

			# at this point, everything should have been used, and then removed.
			if len(available) != 0:
				raise ArgumentException("Extra params in arguments: " + str(available))

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

		(subParamGroup, subDefaults) = self._chooseBestParameterSet(subParamGroup, subDefaults, subSource)

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
		ret = {}
		for paramName in toProcess:
			paramValue = toProcess[paramName]
			if isinstance(paramValue, basestring):
				toCall = self.findCallable(paramValue)
				if toCall is not None:			
					# this has already been set up to be guaranteed to be in a walled garden
					subInitParams = toProcess[paramValue]
					instantiatedParams = self._instantiateArgumentsHelper(subInitParams)
					paramValue = toCall(**instantiatedParams)
			
			ret[paramName] = paramValue

		return ret 

	def _chooseBestParameterSet(self, possibleParamsSets, matchingDefaults, arguments):
		bestParams = []
		bestDefaults = {}
		for i in range(len(possibleParamsSets)):
			currParams = possibleParamsSets[i]
			currDefaults = matchingDefaults[i]
			for param in currParams:
				if param not in arguments and param not in currDefaults:
					continue
			if len(currParams) > len(bestParams):
				bestParams = currParams
				bestDefaults = currDefaults
		return (bestParams, bestDefaults)

	class TrainedLearner():

		def __init__(self, learnerName, arguments, transformedArguments, customDict, backend, interfaceObject):
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
			self.transformedArguments = transformedArguments
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


		def test(self, testX, testY, arguments, performanceFunction):
			ret = self.apply(testesting in TrainedLearnertX)
			# TODO, call some kind of helper in UML to deal with the actual testing
			pass

		def apply(self, testX, arguments, output='match', scoreMode='label'):
#			self.interface._validateOutputFlag(output)
#			self.interface._validateScoreModeFlag(scoreMode)

			copyTestX = testX
			if isinstance(testX, UML.data.Base):
				copyTestX = testX.copy()

			# input transformation
			(trainX, trainY, transTestX, arguments) = self.interface._inputTransformation(self.learnerName, None, None, copyTestX, arguments, self.customDict)

			# depending on the mode, we need different information.
			if scoreMode != 'label':
				scores = self.getScores(testX, arguments)
			if scoreMode != 'allScores':
				labels = self.interface._applier(self.backend, transTestX, arguments, self.customDict)
				labels = self.interface._outputTransformation(self.learnerName, labels, self.transformedArguments, output, self.customDict)

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
					rowIndex = scoreOrder.index(labels[pointIndex,0])
					return row[rowIndex]

				scoreVector = scores.applyToPoints(grabValue, inPlace=False)
				labels.appendFeatures(scoreVector)

				return labels

		def retrain(self, trainX, trainY=None):
			(trainX, trainY, testX, arguments, extra) = self.interface._inputTransformation(self.learnerName,trainX, trainY, None, self.arguments)
			(newBackend, transformedInputs) = self.interface._trainBackend(self.learnerName, trainX, trainY, arguments)
			self.backend = newBackend
			self.transformedInputs = transformedInputs

		def incrementalTrain(self, trainX, trainY=None):
			(trainX, trainY, testX, arguments, extra) = self.interface._inputTransformation(self.learnerName,trainX, trainY, None, self.arguments)
			self.backend = self.interface._incrementalTrainer(self.backend, trainX, trainY, arguments)

		def getAttributes(self):
			self.interface.getAttributes(self.backend)

		def getScores(self, testX, arguments):
			(trainX, trainY, testX, arguments) = self.interface._inputTransformation(self.learnerName, None, None, testX, None, self.customDict)
			
			rawOrderedScores = self.interface._getScores(self.backend, testX, arguments)
			internalOrder = self.interface._getScoresOrder(self.backend)
			naturalOrder = sorted(internalOrder)
			if numpy.array_equal(naturalOrder, internalOrder):
				return rawOrderedScores
			desiredDict = {}
			for i in range(len(naturalOrder)):
				label = naturalOrder[i]
				desiredDict[label] = i
			def sortScorer(feature):
				label = internalOrder[feature.index()]
				return desiredDict[label]

			return rawOrderedScores.sortFeatures(sortHelper=sortScorer)


	########################
	### ABSTRACT METHODS ###
	########################

	@abc.abstractmethod
	def listLearners(self):
		"""
		Return a list of all learners callable through this interface.

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
	def findCallable(self, name):
		"""
		Find reference to the callable with the given name
		TAKES string name
		RETURNS reference to in-package function or constructor
		"""
		pass

	@abc.abstractmethod
	def _getParameterNames(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		pass

	@abc.abstractmethod
	def getLearnerParameterNames(self, learnerName):
		"""
		Find all parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of list of param names
		"""
		pass	

	@abc.abstractmethod
	def _getDefaultValues(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""
		pass

	@abc.abstractmethod
	def getLearnerDefaultValues(self, learnerName):
		"""
		Find all default values for parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of dict of param names to default values
		"""
		pass

	@abc.abstractmethod
	def _getScores(self, learner, testX, arguments):
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

		trainX, etc. are filled with the values of the parameters of the same name
		to a calls to trainAndApply() or train(), or are empty when being called before other
		functions. arguments is a dictionary mapping names to values of all other
		parameters that need to be processed.

		The return value of this function must be a dictionary mirroring the
		structure of the inputs. Specifically, four keys and values are required:
		keys trainX, trainY, testX, and arguments. For the first three, the associated
		values must be the transformed values, and for the last, the value must be a
		dictionary with the same keys as in the 'arguments' input dictionary, with
		transformed values as the values. However, other information may be added
		by the package implementor, for example to be used in _outputTransformation()

		"""
		pass

	@abc.abstractmethod
	def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputFormat, customDict):
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


