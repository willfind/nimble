
import copy

import UML

from UML.exceptions import ArgumentException
from UML.interfaces.custom_learner import CustomLearner
from UML.interfaces.universal_interface import UniversalInterface

class CustomLearnerInterface(UniversalInterface):
	"""

	"""

	_ignoreNames = ['trainX', 'trainY', 'testX']

	def __init__(self):
		"""

		"""
		self.registeredLearners = {}
		super(CustomLearnerInterface, self).__init__()


	def registerLearnerClass(self, learnerClass):
		CustomLearner.validateSubclass(learnerClass)
		self.registeredLearners[learnerClass.__name__] = learnerClass


	#######################################
	### ABSTRACT METHOD IMPLEMENTATIONS ###
	#######################################


	def listLearners(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		return self.registeredLearners.keys()


	def learnerType(self, name):
		"""
		Returns a string referring to the action the learner takes out of the possibilities:
		classifier, regressor, featureSelection, dimensionalityReduction 
		TODO

		"""
		return self.registeredLearners[name].problemType


	def findCallable(self, name):
		"""
		Find reference to the callable with the given name
		TAKES string name
		RETURNS reference to in-package function or constructor
		"""
		if name in self.registeredLearners:
			return self.registeredLearners[name]
		else:
			return None

	def _getParameterNames(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		return self.getLearnerParameterNames(name)

	def getLearnerParameterNames(self, learnerName):
		"""
		Find all parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of list of param names
		"""
		ret = None
		if learnerName in self.registeredLearners:
			temp = self.registeredLearners[learnerName].getLearnerParameterNames()
			after = []
			for value in temp:
				if value not in self._ignoreNames:
					after.append(value)
			ret = [after]
		
		return ret

	def _getDefaultValues(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""
		return self.getLearnerDefaultValues(name)

	def getLearnerDefaultValues(self, learnerName):
		"""
		Find all default values for parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of dict of param names to default values
		"""
		ret = None
		if learnerName in self.registeredLearners:
			temp = self.registeredLearners[learnerName].getLearnerDefaultValues()
			after = {}
			for key in temp:
				if key not in self._ignoreNames:
					after[key] = temp[key]
			ret = [after]
		
		return ret


	def _getScores(self, learner, testX, arguments, customDict):
		"""
		If the learner is a classifier, then return the scores for each
		class on each data point, otherwise raise an exception.

		"""
		return learner.getScores(testX)

	def _getScoresOrder(self, learner):
		"""
		If the learner is a classifier, then return a list of the the labels corresponding
		to each column of the return from getScores

		"""
		if learner.problemType == 'classification':
			return learner.labelList
		else:
			raise ArgumentException("Can only get scores order for a classifying learner")

	def isAlias(self, name):
		"""
		Returns true if the name is an accepted alias for this interface

		"""
		return name.lower() == 'Custom'.lower()


	def getCanonicalName(self):
		"""
		Returns the string name that will uniquely identify this interface

		"""
		return "Custom"

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
		retTrainX = None
		if trainX is not None:
			retTrainX = trainX.copy()
		retTrainY = None
		if trainY is not None:
			retTrainY = trainY.copy()
		retTestX = None
		if testX is not None:
			retTestX = testX.copy()
		retArgs = None
		if arguments is not None:
			retArgs = copy.copy(arguments)
		return (retTrainX, retTrainY, retTestX, retArgs)

	def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputFormat, customDict):
		"""
		Method called before any package level function which transforms the returned
		value into a format appropriate for a UML user.

		"""
		if isinstance(outputValue, UML.data.Base):
			return outputValue
		else:
			return UML.createData('Matrix', outputValue)

	def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
		"""
		build a learner and perform training with the given data
		TAKES name of learner, transformed arguments
		RETURNS an in package object to be wrapped by a TrainedLearner object
		"""
		ret = self.registeredLearners[learnerName]()
		return ret.trainForInterface(trainX, trainY, arguments)

	def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
		"""
		Given an already trained online learner, extend it's training with the given data
		TAKES trained learner, transformed arguments,
		RETURNS the learner after this batch of training
		"""
		return learner.incrementalTrainForInterface(trainX, trainY, arguments)

	def _applier(self, learner, testX, arguments, customDict):	
		"""
		use the given learner to do testing/prediction on the given test set
		TAKES a TrainedLearner object that can be tested on 
		RETURNS UML friendly results
		"""
		return learner.applyForInterface(testX, arguments)

	def _getAttributes(self, learnerBackend):
		"""
		Returns whatever attributes might be available for the given learner. For
		example, in the case of linear regression, TODO

		"""
		contents = dir(learnerBackend)
		excluded = dir(CustomLearner)
		ret = {}
		for name in contents:
			if not name.startswith('_') and not name in excluded:
				ret[name] = getattr(learnerBackend, name)

		return name

	def _optionDefaults(self, option):
		"""
		Define package default values that will be used for as long as a default
		value hasn't been registered in the UML configuration file. For example,
		these values will always be used the first time an interface is instantiated.

		"""
		{}


	def _configurableOptionNames(self):
		"""
		Returns a list of strings, where each string is the name of a configurable
		option of this interface whose value will be stored in UML's configuration
		file.

		"""
		return []

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
		return []
