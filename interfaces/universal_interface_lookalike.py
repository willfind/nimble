"""




"""


from universal_interface import UniversalInterface

class UniversalInterfaceLookalike(UniversalInterface):
	"""

	"""

	def __init__(self):
		"""

		"""
		super(UniversalInterfaceLookalike, self).__init__()

	########################
	### ABSTRACT METHODS ###
	########################

	def findCallable(self, name):
		"""
		Find reference to the callable with the given name
		TAKES string name
		RETURNS reference to in-package function or constructor
		"""
		pass

	def getScores(self, learner, trainX, trainY=None, arguments={}):
		"""
		If the learner is a classifier, then return the scores for each
		class on each data point, otherwise raise an exception.

		"""
		pass


	def _trainer(self, learnerCall, trainX=None, trainY=None, arguments={}):
		"""
		build a learner and perform training with the given data
		TAKES transformed arguments, tocall
		RETURNS an in package object to be wrapped by a TrainedLearner object
		"""
		pass


	def _applier(self, learner, testX):	
		"""
		use the given learner to do testing/prediction on the given test set
		TAKES a TrainedLearner object that can be tested on 
		RETURNS UML friendly results
		"""
		pass


	def _inputTransformation(self, learnerName, trainX, trainY, testX, arguments):
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
		dictionary with the same keys as in the 'arguments' input dicitonary, with
		transformed values as the values. However, other information may be added
		by the package implementor, for example to be used in _outputTransformation()

		"""
		return {'trainX':trainX, 'trainY':trainY, 'testX':testX, 'arguments':arguments}

	def _outputTransformation(self, learnerName, outputValue, transformedInputs):
		"""
		Method called before any package level function which transforms the returned
		value into a format appropriate for a UML user.

		"""
		return outputValue

	def _configurableOptionNames(self):
		"""
		Returns a list of strings, where each string is the name of a configurable
		option of this interface whose value will be stored in UML's configuration
		file.

		"""
		return []

	def _optionDefaults(self, option):
		"""
		Define package default values that will be used for as long as a default
		value hasn't been registered in the UML configuration file. For example,
		these values will always be used the first time an interface is instantiated.

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

		
	def _getAttributes(self, learnerBackend):
		"""
		Returns whatever attributes might be available for the given learner. For
		example, in the case of linear regression, TODO

		"""
		pass


	def _getScores(self, learner, testX):
		"""
		If the learner is a classifier, then return the scores for each
		class on each data point, otherwise raise an exception.

		"""
		pass

	def _getScoresOrder(self, learner, testX):
		"""
		If the learner is a classifier, then return a list of the the labels corresponding
		to each column of the return from getScores

		"""
		
	def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
		"""
		Given an already trained online learner, extend it's training with the given data
		TAKES trained learner, transformed arguments,
		RETURNS the learner after this batch of training
		"""
		pass

	def learnerType(self, name):
		"""
		Returns a string referring to the action the learner takes out of the possibilities:
		classifier, regressor, featureSelection, dimensionalityReduction 
		TODO

		"""
		pass

