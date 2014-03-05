"""
Relies on being scikit-learn 0.9 or above

"""

# TODO: multinomialHMM requires special input processing for obs param


import importlib
import inspect
import copy
import numpy
import os
import sys

import UML

from UML.exceptions import ArgumentException
from UML.interfaces.interface_helpers import generateBinaryScoresFromHigherSortedLabelScores
from UML.interfaces.interface_helpers import calculateSingleLabelScoresFromOneVsOneScores
from UML.interfaces.interface_helpers import ovaNotOvOFormatted

# Contains path to sciKitLearn root directory
#sciKitLearnDir = '/usr/local/lib/python2.7/dist-packages'
sciKitLearnDir = None

# a dictionary mapping names to learners, or modules
# containing learners. To be used by findInPackage
locationCache = {}


from UML.interfaces.universal_interface import UniversalInterface

class SciKitLearn(UniversalInterface):
	"""

	"""

	def __init__(self):
		"""

		"""
		if sciKitLearnDir is not None:
			sys.path.insert(0, sciKitLearnDir)

		self.skl = importlib.import_module('sklearn')

		super(SciKitLearn, self).__init__()

	#######################################
	### ABSTRACT METHOD IMPLEMENTATIONS ###
	#######################################

	def listLearners(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		ret = []
		subpackages = self.skl.__all__

		exclude = ['BaseDiscreteNB', 'libsvm']

		for sub in subpackages:
			curr = 'sklearn.' + sub
			try:
				exec('import ' + curr)
			except ImportError:
				# no guarantee __all__ is accurate, if something doesn't import, just skip ahead
				continue

			contents = eval('dir(' + curr + ')')
			
			for member in contents:
				if member in exclude:
					continue
				memberContents = eval('dir(' + curr + "." + member + ')')
				if 'fit' in memberContents and 'predict' in memberContents:
					ret.append(member)

		return ret

	def learnerType(self, name):
		"""
		Returns a string referring to the action the learner takes out of the possibilities:
		classifier, regressor, featureSelection, dimensionalityReduction 
		TODO

		"""
		return 'UNKNOWN'

	def findCallable(self, name):
		"""
		Find reference to the callable with the given name
		TAKES string name
		RETURNS reference to in-package function or constructor
		"""
		subpackages = self.skl.__all__

		for sub in subpackages:
			try:
				currSubModule = importlib.import_module('sklearn.' + sub)
			except ImportError:
				# no guarantee __all__ is accurate, if something doesn't import, just skip ahead
				continue

			contents = dir(currSubModule)
			
			for member in contents:
				memberObj = getattr(currSubModule, member)
				if member == name:

					return memberObj

		return None

	def _getParameterNames(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		ret = self._paramQuery(name, None)
		if ret is None:
			return ret
		(objArgs,v,k,d) = ret
		return [objArgs]

	def getLearnerParameterNames(self, learnerName):
		"""
		Find all parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of list of param names
		"""
		ignore = ['self', 'X', 'x', 'Y', 'y', 'obs', 'T']
		init = self._paramQuery('__init__', learnerName, ignore)
		fit = self._paramQuery('fit', learnerName, ignore)
		predict = self._paramQuery('predict', learnerName, ignore)
		transform = self._paramQuery('transform', learnerName, ignore)
		fitPredict = self._paramQuery('fitPredict', learnerName, ignore)
		fitTransform = self._paramQuery('fitTransform', learnerName, ignore)

		if predict is not None:
			ret = init[0] + fit[0] + predict[0]
		elif transform is not None:
			ret = init[0] + fit[0] + transform[0]
		elif fitPredict is not None:
			ret = init[0] + fitPredict[0]
		elif fitTransform is not None:
			ret = init[0] + fitTransform[0]
		else:
			raise ArgumentException("Cannot get parameter names for leaner " + learnerName)

		return [ret]

	def _getDefaultValues(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""
		ret = self._paramQuery(name, None)
		if ret is None:
			return ret
		(objArgs,v,k,d) = ret
		ret = {}
		if d is not None:
			for i in xrange(len(d)):
				ret[objArgs[-(i+1)]] = d[-(i+1)]

		return [ret]

	def getLearnerDefaultValues(self, learnerName):
		"""
		Find all default values for parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of dict of param names to default values
		"""
		ignore = ['self', 'X', 'x', 'Y', 'y', 'T']
		init = self._paramQuery('__init__', learnerName, ignore)
		fit = self._paramQuery('fit', learnerName, ignore)
		predict = self._paramQuery('predict', learnerName, ignore)
		transform = self._paramQuery('transform', learnerName, ignore)
		fitPredict = self._paramQuery('fitPredict', learnerName, ignore)
		fitTransform = self._paramQuery('fitTransform', learnerName, ignore)

		if predict is not None:
			toProcess = [init, fit, predict]
		elif transform is not None:
			toProcess = [init, fit, transform]
		elif fitPredict is not None:
			toProcess = [init, fitPredict]
		else:
			toProcess = [init, fitTransform]

		ret = {}
		for stage in toProcess:
			currNames = stage[0]
			currDefaults = stage[3]
			if stage[3] is not None:
				for i in xrange(len(currDefaults)):
					key = currNames[-(i+1)]
					value = currDefaults[-(i+1)]
					ret[key] = value

		return [ret]

	def _getScores(self, learner, testX, arguments):
		"""
		If the learner is a classifier, then return the scores for each
		class on each data point, otherwise raise an exception.

		"""
		if hasattr(learner, 'decision_function'):
			raw = learner.decision_function(testX)
			order = self._getScoresOrder(learner)
			numLabels = len(order)
			if numLabels == 2:
				ret = generateBinaryScoresFromHigherSortedLabelScores(raw)
				return UML.createData("Matrix", ret)

			predLabels = self._applier(learner, testX, arguments, None)
			strategy = ovaNotOvOFormatted(raw, predLabels, numLabels)
			# we want the scores to be per label, regardless of the original format, so we
			# check the strategy, and modify it if necessary
			if not strategy:
				scores = []
				for i in xrange(len(raw)):
					combinedScores = calculateSingleLabelScoresFromOneVsOneScores(raw[i], numLabels)
					scores.append(combinedScores)
				scores = numpy.array(scores)
				return UML.createData("Matrix", scores)
			else:
				return UML.createData("Matrix", raw)

		else:
			raise ArgumentException('Cannot get scores for this learner')


	def _getScoresOrder(self, learner):
		"""
		If the learner is a classifier, then return a list of the the labels corresponding
		to each column of the return from getScores

		"""
		return learner.UIgetScoreOrder()


	def isAlias(self, name):
		"""
		Returns true if the name is an accepted alias for this interface

		"""
		return name.lower() == self.getCanonicalName().lower()


	def getCanonicalName(self):
		"""
		Returns the string name that will uniquely identify this interface

		"""
		return 'sciKitLearn'


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
		if trainX is not None:
			customDict['match'] = trainX.getTypeString()
			transTrainX = trainX.copyAs('numpy matrix')
		else:
			transTrainX = None

		if trainY is not None:
			transTrainY = (trainY.copyAs('numpy array')).flatten()
		else:
			transTrainY = None

		if testX is not None:
			transTestX = testX.copyAs('numpy matrix')
		else:
			transTestX = None

		return (transTrainX, transTrainY, transTestX, copy.deepcopy(arguments))



	def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputFormat, customDict):
		"""
		Method called before any package level function which transforms the returned
		value into a format appropriate for a UML user.

		"""
		#we are given a row vector, we want a column vector
		outputValue = outputValue.reshape(len(outputValue), 1)

		#TODO correct
		outputFormat = 'Matrix'
		if outputFormat == 'match':
			outputFormat = customDict['match']
		return UML.createData(outputFormat, outputValue)


	def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
		"""
		build a learner and perform training with the given data
		TAKES name of learner, transformed arguments
		RETURNS an in package object to be wrapped by a TrainedLearner object
		"""
		# get parameter names
		initNames = self._paramQuery('__init__', learnerName, ['self'])[0]
		fitNames = self._paramQuery('fit', learnerName, ['self'])[0]

		# pack parameter sets
		initParams = {}
		for name in initNames:
			initParams[name] = arguments[name]
		fitParams = {}
		for name in fitNames:
			if name.lower() == 'x' or name.lower() == 'obs':
				value = trainX
			elif name.lower() == 'y':
				value = trainY
			else:
				value = arguments[name]
			fitParams[name] = value

		learner = self.findCallable(learnerName)(**initParams)
		learner.fit(**fitParams)
		if hasattr(learner, 'decision_function'):
			labelOrder = numpy.unique(trainY)
			def UIgetScoreOrder():
				return labelOrder
			learner.UIgetScoreOrder = UIgetScoreOrder

		return learner


	def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
		"""
		Given an already trained online learner, extend it's training with the given data
		TAKES trained learner, transformed arguments,
		RETURNS the learner after this batch of training
		"""
		# see partial_fit(X, y[, classes, sample_weight])
		pass



	def _applier(self, learner, testX, arguments, customDict):
		"""
		use the given learner to do testing/prediction on the given test set
		TAKES a TrainedLearner object that can be tested on 
		RETURNS UML friendly results
		"""
		if hasattr(learner, 'predict'):
			return self._predict(learner, testX, arguments, customDict)
		elif hasattr(learner, 'transform'):
			return self._transform(learner, testX, arguments, customDict)
		else:
			raise TypeError("Cannot apply this learner to data, no predict or transform function")


	def _getAttributes(self, learnerBackend):
		"""
		Returns whatever attributes might be available for the given learner. For
		example, in the case of linear regression, TODO

		"""
		return learnerBackend.get_params()

	def _optionDefaults(self, option):
		"""
		Define package default values that will be used for as long as a default
		value hasn't been registered in the UML configuration file. For example,
		these values will always be used the first time an interface is instantiated.

		"""
		return {}

	def _configurableOptionNames(self):
		"""
		Returns a list of strings, where each string is the name of a configurable
		option of this interface whose value will be stored in UML's configuration
		file.

		"""
		return ['location']


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
		return [self._predict, self._transform]
		# fit_transform


	def _predict(self, learner, testX, arguments, customDict):
		"""
		Wrapper for the underlying predict function of a scikit-learn learner object
		"""
		return learner.predict(testX)

	def _transform(self, learner, testX, arguments, customDict):
		"""
		Wrapper for the underlying transform function of a scikit-learn learner object
		"""
		return learner.fit(testX)


	###############
	### HELPERS ###
	###############

	def _removeFromArray(self, orig, toIgnore):
		temp = []
		for entry in orig:
			if not entry in toIgnore:
				temp.append(entry)
		return temp

	def _removeFromDict(self, orig, toIgnore):
		for entry in toIgnore:
			if entry in orig:
				del orig[entry]
		return orig

	def _removeFromTailMatchedLists(self, full, matched, toIgnore):
		"""
		full is some list n, matched is a list with length m, where m is less
		than or equal to n, where the last m values of full are matched against
		their positions in matched. If one of those is to be removed, it is to
		be removed in both.
		"""
		temp = {}
		if matched is not None:
			for i in xrange(len(full)):
				if i < len(matched):
					temp[full[len(full)-1-i]] = matched[len(matched)-1-i]
				else:
					temp[full[len(full)-1-i]] = None
		else:
			retFull = self._removeFromArray(full, toIgnore)
			return (retFull, matched)

		for ignoreKey in toIgnore:
			if ignoreKey in temp:
				del temp[ignoreKey]

		retFull = []
		retMatched = []
		for i in xrange(len(full)):
			name = full[i]
			if name in temp:
				retFull.append(name)
				if (i -(len(full) - len(matched))) >= 0:
					retMatched.append(temp[name])

		return (retFull, retMatched)


	def _paramQuery(self, name, parent, ignore=[]):
		"""
		Takes the name of some scikit learn object or function, returns a list
		of parameters used to instantiate that object or run that function, or
		None if the desired thing cannot be found

		"""
		namedModule = self._findInPackage(name,parent)

#		import pdb
#		pdb.set_trace()

		if namedModule is None:
			return None

		try:
			(args, v, k, d) = inspect.getargspec(namedModule)
			(args, d) = self._removeFromTailMatchedLists(args, d, ignore)
			return (args, v, k, d)
		except TypeError:
			try:
				(args, v, k, d) = inspect.getargspec(namedModule.__init__)
				(args, d) = self._removeFromTailMatchedLists(args, d, ignore)
				return (args, v, k, d)
			except TypeError:
				return self._paramQueryHardCoded(name, parent, ignore)


	def _paramQueryHardCoded(self, name, parent, ignore):
		"""
		Returns a list of parameters for in package entities that we have hard coded,
		under the assumption that it is difficult or impossible to find that data
		automatically

		"""
		if parent is not None and parent.lower() == 'GaussianNB'.lower():
#			import pdb
#			pdb.set_trace()
			if name == '__init__':
				ret = ([], None, None, [])
			elif name == 'fit':
				ret = (['X', 'y'], None, None, [])
			elif name == 'predict':
				ret = (['X'], None, None, [])
			else:
				return None

			(newArgs, newDefaults) = self._removeFromTailMatchedLists(ret[0], ret[3], ignore)
			return (newArgs, ret[1], ret[2], newDefaults)

		return None



#		init = self._paramQuery(learnerName, None, ignore)
#		fit = self._paramQuery('fit', learnerName, ignore)
#		predict = self._paramQuery('predict', learnerName, ignore)
#		transform = self._paramQuery('transform', learnerName, ignore)
#		fitPredict = self._paramQuery('fitPredict', learnerName, ignore)
#		fitTransform = self._paramQuery('fitTransform', learnerName, ignore)


	def _findInPackage(self, name, parent=None):
		"""
		Import the desired python package, and search for the module containing
		the wanted learner. For use by interfaces to python packages.

		"""
		packageMod = importlib.import_module('sklearn')

		contents = packageMod.__all__

		searchIn = packageMod
		allowedDepth = 2
		if parent is not None:
			if parent in locationCache:
				searchIn = locationCache[parent]
			else:
				searchIn = self._findInPackageRecursive(parent, allowedDepth, contents, packageMod)
			allowedDepth = 0
			contents = dir(searchIn)
			if searchIn is None:
				return None

		if name in locationCache:
			ret = locationCache[name]
		else:
			ret = self._findInPackageRecursive(name, allowedDepth, contents, searchIn)

		return ret

	
	def _findInPackageRecursive(self, target, allowedDepth, contents, parent):
		for name in contents:
			if name.startswith("__") and name != '__init__':
				continue
			try:
				subMod = getattr(parent, name)
			except AttributeError:
				try:		
					subMod = importlib.import_module(parent.__name__ + "." + name)
				except ImportError:
					continue

			# we want to add learners, and the parents of learners to the cache 
			if hasattr(subMod, 'fit'):
				locationCache[name] = subMod

			if name == target:
				return subMod

			subContents = dir(subMod)

			if allowedDepth > 0:
				ret = self._findInPackageRecursive(target, allowedDepth-1, subContents, subMod)
				if ret is not None:
					return ret

		return None

