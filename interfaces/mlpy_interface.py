"""
Relies on being scikit-learn 0.9 or above

OLS and LARS learners are not allowed as learners. KernelExponential is 
not allowed as a Kernel.

"""

# TODO: multinomialHMM requires special input processing for obs param


import importlib
import inspect
import copy
import numpy
import sys
import copy

import UML

from UML.exceptions import ArgumentException
from UML.interfaces.interface_helpers import generateBinaryScoresFromHigherSortedLabelScores
from UML.interfaces.interface_helpers import calculateSingleLabelScoresFromOneVsOneScores
from UML.interfaces.interface_helpers import ovaNotOvOFormatted

# Contains path to mlpy root directory
mlpyDir = None

# a dictionary mapping names to learners, or modules
# containing learners. To be used by findInPackage
locationCache = {}

from UML.interfaces.universal_interface import UniversalInterface

class Mlpy(UniversalInterface):
	"""

	"""

	_XDataAliases = ['X', 'x', 'T', 't', 'K', 'Kt']
	_YDataAliases = ['Y', 'y']
	_DataAliases = _XDataAliases + _YDataAliases

	def __init__(self):
		"""

		"""
		if mlpyDir is not None:
			sys.path.insert(0, mlpyDir)

		self.mlpy = importlib.import_module('mlpy')

		super(Mlpy, self).__init__()

	#######################################
	### ABSTRACT METHOD IMPLEMENTATIONS ###
	#######################################

	def listLearners(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		ret = []
		contents = dir(self.mlpy)

		toExclude = ['OLS', 'LARS']

		for member in contents:
			subContents = dir(getattr(self.mlpy, member))
			if 'learn' in subContents and ('pred' in subContents or 'transform' in subContents):
				if not member in toExclude:
					ret.append(member)

		ret.append('MFastHCluster')
		ret.append('kmeans')

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
		contents = dir(self.mlpy)

		if name == 'kmeans':
			return _Kmeans
		if name == 'MFastHCluster':
			return _MFastHCluster

		for member in contents:
			if member == name:
				return getattr(self.mlpy, member)

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
		if objArgs[0] == 'self':
			objArgs = objArgs[1:]
		return [objArgs]

	def getLearnerParameterNames(self, learnerName):
		"""
		Find all parameters involved in a trainAndApply() call to the given learner
		TAKES string name of a learner, 
		RETURNS list of list of param names
		"""
		ignore = self._DataAliases + ['self']
		if learnerName == 'MFastHCluster':
			ignore.remove('t')
		init = self._paramQuery('__init__', learnerName, ignore)
		learn = self._paramQuery('learn', learnerName, ignore)
		pred = self._paramQuery('pred', learnerName, ignore)
		transform = self._paramQuery('transform', learnerName, ignore)

		if pred is not None:
			ret = init[0] + learn[0] + pred[0]
		elif transform is not None:
			ret = init[0] + learn[0] + transform[0]
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
		ignore = self._DataAliases + ['self']
		init = self._paramQuery('__init__', learnerName, ignore)
		learn = self._paramQuery('learn', learnerName, ignore)
		pred = self._paramQuery('pred', learnerName, ignore)
		transform = self._paramQuery('transform', learnerName, ignore)


		if pred is not None:
			toProcess = [init, learn, pred]
		elif transform is not None:
			toProcess = [init, learn, transform]

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

	def _getScores(self, learner, testX, arguments, customDict):
		"""
		If the learner is a classifier, then return the scores for each
		class on each data point, otherwise raise an exception.

		"""
		if hasattr(learner, 'pred_values'):
			raw = learner.pred_values(testX)
			order = self._getScoresOrder(learner)
			numLabels = len(order)
			if numLabels == 2:
				ret = generateBinaryScoresFromHigherSortedLabelScores(raw)
				return UML.createData("Matrix", ret)

			predLabels = self._applier(learner, testX, arguments, customDict)
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
		return learner.labels()


	def isAlias(self, name):
		"""
		Returns true if the name is an accepted alias for this interface

		"""
		return name.lower() == self.getCanonicalName().lower()


	def getCanonicalName(self):
		"""
		Returns the string name that will uniquely identify this interface

		"""
		return 'mlpy'


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

		if 'kernel' in arguments:
			if arguments['kernel'] is None and trainX is not None and trainX.pointCount != trainX.featureCount:
				raise ArgumentException("For this learner, in the absence of specifying a kernel, the trainX parameter must be square (representing the inner product space of the features)")

			if isinstance(arguments['kernel'], self.mlpy.KernelExponential):
				raise ArgumentException("This interface disallows the use of KernelExponential; it is bugged in some versions of mlpy")

		customDict['useT'] = False
		if learnerName == 'MFastHCluster':
			customDict['useT'] = True

		return (transTrainX, transTrainY, transTestX, copy.deepcopy(arguments))



	def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputFormat, customDict):
		"""
		Method called before any package level function which transforms the returned
		value into a format appropriate for a UML user.

		"""
		# we are sometimes given a matrix, this will take care of that 
		outputValue = numpy.array(outputValue).flatten()
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
		learnNames = self._paramQuery('learn', learnerName, ['self'])[0]
		predNames = self._paramQuery('pred', learnerName,['self'])
		if predNames is not None:
			customDict['predNames'] = predNames[0]

		# pack parameter sets
		initParams = {}
		for name in initNames:
			initParams[name] = arguments[name]
		learnParams = {}
		for name in learnNames:
			if name in self._XDataAliases:
				value = trainX
			elif name in self._YDataAliases:
				value = trainY
			else:
				value = arguments[name]
			learnParams[name] = value

		learner = self.findCallable(learnerName)(**initParams)
		learner.learn(**learnParams)

		return learner


	def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
		"""
		Given an already trained online learner, extend it's training with the given data
		TAKES trained learner, transformed arguments,
		RETURNS the learner after this batch of training
		"""
		raise RuntimeError()


	def _applier(self, learner, testX, arguments, customDict):
		"""
		use the given learner to do testing/prediction on the given test set
		TAKES a TrainedLearner object that can be tested on 
		RETURNS UML friendly results
		"""
		if hasattr(learner, 'pred'):
			return self._pred(learner, testX, arguments, customDict)
		elif hasattr(learner, 'transform'):
			return self._transform(learner, testX, arguments, customDict)
		else:
			raise TypeError("Cannot apply this learner to data, no predict or transform function")


	def _getAttributes(self, learnerBackend):
		"""
		Returns whatever attributes might be available for the given learner. For
		example, in the case of linear regression, TODO

		"""
		raise RuntimeError()

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
		return [self._pred, self._transform]


	def _pred(self, learner, testX, arguments, customDict):
		"""
		Wrapper for the underlying predict function of a scikit-learn learner object
		"""
		params = customDict['predNames']
		if len(params) > 0:
			if customDict['useT']:
				return learner.pred(arguments['t'])
			else:
				return learner.pred(testX)
		else:
			return learner.pred()

	def _transform(self, learner, testX, arguments, customDict):
		"""
		Wrapper for the underlying transform function of a scikit-learn learner object
		"""
		return learner.transform(testX)


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

		if not namedModule is None:
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
					pass
	
		return self._paramQueryHardCoded(name, parent, ignore)


	def _paramQueryHardCoded(self, name, parent, ignore):
		"""
		Returns a list of parameters for in package entities that we have hard coded,
		under the assumption that it is difficult or impossible to find that data
		automatically

		"""
		pnames = []
		pvarargs = None
		pkeywords = None
		pdefaults = []

		if parent is None:
			return None

		if parent.lower() == 'LibSvm'.lower():
			if name == '__init__':
				pnames = ['svm_type', 'kernel_type', 'degree', 'gamma', 'coef0', 'C', 'nu', 'eps', 'p', 'cache_size', 'shrinking', 'probability', 'weight']
				pdefaults = ['c_svc', 'linear', 3, 0.001, 0, 1, 0.5, 0.001, 0.1, 100, True, False, {}]		
			elif name == 'learn':
				pnames = ['x', 'y']
			elif name == 'pred':
				pnames = ['t']
			else:
				return None
		elif parent.lower() == 'KNN'.lower():
			if name == '__init__':
				pnames = ['k']
			elif name == 'learn':
				pnames = ['x', 'y']
			elif name == 'pred':
				pnames = ['t']
			else:
				return None
		elif parent.lower() == 'LibLinear'.lower():
			if name == '__init__':
				pnames = ['solver_type', 'C', 'eps', 'weight']
				pdefaults = ['l2r_lr', 1, 0.01, {}]
			elif name == 'learn':
				pnames = ['x', 'y']
			elif name == 'pred':
				pnames = ['t']
			else:
				return None
		elif parent.lower() == 'MaximumLikelihoodC'.lower():
			if name == '__init__':
				pass
			elif name == 'learn':
				pnames = ['x', 'y']
			elif name == 'pred':
				pnames = ['t']
			else:
				return None
		elif parent.lower() == 'KernelAdatron'.lower():
			if name == '__init__':
				pnames = ['C', 'maxsteps', 'eps']
				pdefaults = [1000, 1000, 0.01]
			elif name == 'learn':
				pnames = ['K', 'y']
			elif name == 'pred':
				pnames = ['Kt']
			else:
				return None
		elif parent.lower() == 'ClassTree'.lower():
			if name == '__init__':
				pnames = ['stumps', 'minsize']
				pdefaults = [0, 1]
			elif name == 'learn':
				pnames = ['x', 'y']
			elif name == 'pred':
				pnames = ['t']
			else:
				return None
		elif parent.lower() == 'MFastHCluster'.lower():
			if name == '__init__':
				pnames = ['method']
				pdefaults = ['single']
			elif name == 'learn':
				pnames = ['x']
			elif name == 'pred':
				pnames = ['t']
			else:
				return None
		elif parent.lower() == 'kmeans'.lower():
			if name == '__init__':
				pnames = ['k', 'plus', 'seed']
				pdefaults = [False, 0]
			elif name == 'learn':
				pnames = ['x']
			elif name == 'pred':
				pnames = []
			else:
				return None

		else:
			return None

		ret = (pnames, pvarargs, pkeywords, pdefaults)
		(newArgs, newDefaults) = self._removeFromTailMatchedLists(ret[0], ret[3], ignore)
		return (newArgs, ret[1], ret[2], newDefaults)


	def _findInPackage(self, name, parent=None):
		"""
		Import the desired python package, and search for the module containing
		the wanted learner. For use by interfaces to python packages.

		"""

		if name == 'kmeans':
			return _Kmeans
		if parent == 'kmeans':
			if not hasattr(_Kmeans, name):
				return None
			return getattr(_Kmeans, name)
		if name == 'MFastHCluster':
			return _MFastHCluster
		if parent == 'MFastHCluster':
			if not hasattr(_MFastHCluster, name):
				return None
			return getattr(_MFastHCluster, name)


		packageMod = self.mlpy

		contents = dir(packageMod)

		searchIn = packageMod
		allowedDepth = 0
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
			if hasattr(subMod, 'learn'):
				locationCache[name] = subMod

			if name == target:
				return subMod

			subContents = dir(subMod)

			if allowedDepth > 0:
				ret = self._findInPackageRecursive(target, allowedDepth-1, subContents, subMod)
				if ret is not None:
					return ret

		return None


class _Kmeans(object):
	def __init__(self, k, plus=False, seed=0):
		self.k = k
		self.plus = plus
		self.seed = seed

	def learn(self, x):
		self.x = x

	def pred(self):
		import mlpy
		(self.clusters, self.means, self.steps) = mlpy.kmeans(self.x, self.k, self.plus, self.seed)
		return self.clusters


class _MFastHCluster(object):
	def __init__(self, method='single'):
		import mlpy
		self.obj = mlpy.MFastHCluster(method)

	def learn(self, x):
		self.obj.linkage(x)

	def pred(self, t):
		return self.obj.cut(t)
