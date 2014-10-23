"""




"""

# TODO?
# * online learning
# * different feature types (streaming, for other problem types)
# *

import importlib
import numpy
import copy
import sys
import os
import json
import distutils.version


import UML

from UML.interfaces.universal_interface import UniversalInterface
from UML.interfaces.interface_helpers import PythonSearcher
from UML.exceptions import ArgumentException

# Interesting alias cases:
# * DomainAdaptionMulticlassLibLinear  -- or probably any nested machine


trainXAliases = ['traindat', 'f', 'features', 'feats', 'feat', 'training_data', 'train_features', 'data']
trainYAliases = ['trainlab', 'lab', 'labs', 'training_labels', 'train_labels']

# kernel : k, kernel
# distance : d

class Shogun(UniversalInterface):
	"""

	"""

	def __init__(self):
		"""

		"""
		super(Shogun, self).__init__()

		self.allowDiscovery = True
		try:
			self.clang.cindex = importlib.import_module("clang.cindex")
		except ImportError as ie:
			self.allowDiscovery = False
			#raise ie

		self.shogun = importlib.import_module('shogun')
		self.versionString = None

		def isLearner(obj):
			hasTrain = hasattr(obj, 'train')
			hasApply = hasattr(obj, 'apply')
			
			if not (hasTrain and hasApply):
				return False

			if obj.__name__ in excludedLearners2:
				return False

			# needs more to be able to distinguish between things that are runnable
			# and partial implementations. try get_machine_problem_type()?
			try:
				instantiated = obj()
			except TypeError:
				# if we can't even instantiate it, then its not a usable class
				return False
			try:
				instantiated.get_machine_problem_type()
			except Exception:
				return False

#				if name.startswith("get_"):
#					try:
#						getattr(instantiated, name)()
#					except Exception:
#						return False

			return True

		self._searcher = PythonSearcher(self.shogun, self.shogun.__all__, {}, isLearner, 2)

		self.setupParameterManifest()


	def accessible(self):
		try:
			import shogun
		except ImportError:
			return False
		return True

	def _listLearnersBackend(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		return self._searcher.allLearners()

	def _getMatchineProblemType(self, learnerName):
		""" SystemError """
		learner = self.findCallable(learnerName)

		try:
			learner = learner()
		except TypeError:
			pass

		ptVal = learner.get_machine_problem_type()
		return ptVal

	def learnerType(self, name):
		"""
		Returns a string referring to the action the learner takes out of the possibilities:
		classification, regression, featureSelection, dimensionalityReduction 
		TODO

		"""
		try:
			ptVal = self._getMatchineProblemType(name)
		except SystemError:
			return 'UNKNOWN'

		if ptVal == self.shogun.Classifier.PT_BINARY or ptVal == self.shogun.Classifier.PT_MULTICLASS:
			return 'classification'
		if ptVal == self.shogun.Classifier.PT_REGRESSION:
			return 'regression'
		if ptVal == self.shogun.Classifier.PT_STRUCTURED:
			return 'UNKNOWN'
		if ptVal == self.shogun.Classifier.PT_LATENT:
			return 'UNKNOWN'

		# TODO warning, unknown problem type code

		return 'UNKNOWN'

	def _getLearnerParameterNamesBackend(self, name):
		base = self._getParameterNamesBackend(name)

		#remove aliases
		ret = []
		for group in base:
			curr = []
			for paramName in group:
				if paramName not in trainXAliases and paramName not in trainYAliases:
					curr.append(paramName)
			ret.append(curr)

		return ret

	def _getLearnerDefaultValuesBackend(self, name):
		allNames = self._getLearnerParameterNamesBackend(name)
		return self._setupDefaultsGivenBaseNames(name, allNames)

	def _getParameterNamesBackend(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		query = self._queryParamManifest(name)
		base = query if query is not None else [[]]

#		return base
		ret = []
		for group in base:
			backend = self.findCallable(name)
			params = group
			for funcname in dir(backend):
				if funcname.startswith('set_'):
					funcname = funcname[4:]
					if funcname not in params:
						params.append(funcname)
			ret.append(params)

		return ret

	def _getDefaultValuesBackend(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""		
		allNames = self._getParameterNamesBackend(name)
		return self._setupDefaultsGivenBaseNames(name, allNames)

	def _setupDefaultsGivenBaseNames(self, name, allNames):
		allValues = []
		for index in range(len(allNames)):
			group = allNames[index]
			curr = {}
			for paramName in group:
				curr[paramName] = self.HiddenDefault(paramName)
			allValues.append(curr)

		ret = []
		query = self._queryParamManifest(name)
		if query is not None:
			base = query 
		else:
			base = []
			for i in range(len(allValues)):
				base.append([])
		for index in range(len(base)):
			group = base[index]
			curr = allValues[index]
			for paramName in group:
				if paramName in curr:
					del curr[paramName]
			ret.append(curr)

		return ret

	def _getScores(self, learner, testX, arguments, customDict):
		"""
		If the learner is a classifier, then return the scores for each
		class on each data point, otherwise raise an exception.

		"""
		predObj = self._applier(learner, testX, arguments, customDict)
		predLabels = predObj.get_labels()
		numLabels = customDict['numLabels']
		if hasattr(predObj, 'get_multiclass_confidences'):
			# setup an array in the right shape, number of predicted labels by number of possible labels
			scoresPerPoint = numpy.empty((len(predLabels), numLabels))
			for i in xrange(len(predLabels)):
				currConfidences = predObj.get_multiclass_confidences(i)
				scoresPerPoint[i, :] = currConfidences
		# otherwise we must be dealing with binary classification
		else:
			# we get a 1d array containing the winning label's confidence value
			scoresPerPoint = predObj.get_values()
			scoresPerPoint.resize(scoresPerPoint.size,1)

		return scoresPerPoint

	def _getScoresOrder(self, learner):
		"""
		If the learner is a classifier, then return a list of the the labels corresponding
		to each column of the return from getScores

		"""
		return learner.get_unique_labels

	def isAlias(self, name):
		"""
		Returns true if the name is an accepted alias for this interface

		"""
		if name.lower() in ['shogun']:
			return True
		else:
			return False

	def getCanonicalName(self):
		"""
		Returns the string name that will uniquely identify this interface

		"""
		return "shogun"

	def _findCallableBackend(self, name):
		"""
		Find reference to the callable with the given name
		TAKES string name
		RETURNS reference to in-package function or constructor
		"""
		return self._searcher.findInPackage(None, name)


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

		# check something that we know won't work, but shogun will not report intelligently
		if trainX is not None or testX is not None:
			if 'pointLen' not in customDict:
				customDict['pointLen'] = trainX.featureCount if trainX is not None else testX.featureCount
			if trainX is not None and trainX.featureCount != customDict['pointLen']:
				raise ArgumentException("Length of points in the training data and testing data must be the same")
			if testX is not None and testX.featureCount != customDict['pointLen']:
				raise ArgumentException("Length of points in the training data and testing data must be the same")

		trainXTrans = None
		if trainX is not None:
			customDict['match'] = trainX.getTypeString()
			trainXTrans = self._inputTransDataHelper(trainX, learnerName)
			
		trainYTrans = None
		if trainY is not None:
			trainYTrans = self._inputTransLabelHelper(trainY, learnerName, customDict)
			
		testXTrans = None
		if testX is not None:
			testXTrans = self._inputTransDataHelper(testX, learnerName)

		delkeys = []
		for key in arguments:
			val = arguments[key]
			if isinstance(val, self.HiddenDefault):
				delkeys.append(key)
		for key in delkeys:
			del arguments[key]

		# TODO copiedArguments = copy.deepcopy(arguments) ?
		# copiedArguments = copy.copy(arguments)
		copiedArguments = arguments

		return (trainXTrans, trainYTrans, testXTrans, copiedArguments)



	def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputType, outputFormat, customDict):
		"""
		Method called before any package level function which transforms the returned
		value into a format appropriate for a UML user.

		"""
		# often, but not always, we have to unpack a Labels object
		if isinstance(outputValue, self.shogun.Classifier.Labels):
			# outputValue is a labels object, have to pull out the raw values with a function call
			retRaw = outputValue.get_labels()
			# prep for next call
			retRaw = numpy.atleast_2d(retRaw)
			# we are given a column organized return, we want row first organization, to match up
			# with our input rows as points standard
			retRaw = retRaw.transpose()
		else:
			retRaw = outputValue

		outputType = 'Matrix'
		if outputType == 'match':
			outputType = customDict['match']
		ret = UML.createData(outputType, retRaw)

		if outputFormat == 'label':
			remap = customDict['remap']
			if remap is not None:
				def makeInverseMapper(inverseMappingParam):
					def inverseMapper(value):
						return inverseMappingParam[int(value)]
					return inverseMapper
				ret.applyToElements(makeInverseMapper(remap), features=0)

		return ret


	def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
		"""
		build a learner and perform training with the given data
		TAKES name of learner, transformed arguments
		RETURNS an in package object to be wrapped by a TrainedLearner object
		"""
		toCall = self.findCallable(learnerName)

		# Figure out which, if any, aliases for trainX or trainY were ignored
		# during argument validation and instantiation
#		setX = True
#		setY = True
		learnerParams = self._getLearnerParameterNamesBackend(learnerName)
		rawParams = self._getParameterNamesBackend(learnerName)
		learnerDefaults = self._getLearnerDefaultValuesBackend(learnerName)
		rawDefaults = self._getDefaultValuesBackend(learnerName)
		bestIndex = self._chooseBestParameterSet(learnerParams, learnerDefaults, arguments)
		diffNames = list(set(rawParams[bestIndex]) - set(learnerParams[bestIndex]))

		# Figure out which params have to be set using setters, instead of passed in.
		setterArgs = {}
		initArgs = {}
		kernels = []
		for name in arguments:
			if name in learnerDefaults[bestIndex]:
				setterArgs[name] = arguments[name]
			else:
				initArgs[name]= arguments[name]

			if isinstance(arguments[name], self.shogun.Classifier.Kernel):
				kernels.append(name)

		# if we've ignored aliased names (as demonstrated by the difference between
		# the raw parameter name list and the learner parameter name list) then this
		# is where we have to add them back in.
		for name in diffNames:
			if name in trainXAliases:
				if name in rawDefaults[bestIndex]:
					pass
					#setterArgs[name] = trainX  TODO -- do we actually want this?
				else:
					initArgs[name] = trainX
			if name in trainYAliases:
				if name in rawDefaults[bestIndex]:
					pass
					#setterArgs[name] = trainY  TODO -- do we actually want this?
				else:
					initArgs[name] = trainY

		# it may be the case that the UML passed the args needed to initialize
		# a kernel. If not, it isn't useable until we do it ourselves
		for name in kernels:
			currKern = arguments[name]
			if not currKern.has_features():
				currKern.init(trainX, trainX)

		# actually pack args for init. c++ backend with a thin python layer means
		# the crucial information is where in a list the values are, NOT the associated
		# name.
		initArgsList = []
		for name in rawParams[bestIndex]:
			if name not in rawDefaults[bestIndex]:
				initArgsList.append(initArgs[name])
		learner = toCall(*initArgsList)

		for name in setterArgs:
			setter = getattr(learner, 'set_' + name)
			setter(setterArgs[name])

		if trainY is not None:
			learner.set_labels(trainY)

		learner.train(trainX)

		# TODO online training prep learner.start_train()
		# batch training if data is passed

		learner.get_unique_labels = numpy.unique(trainY.get_labels())
		customDict['numLabels'] = len(learner.get_unique_labels)

		return learner


	def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
		"""
		Given an already trained online learner, extend it's training with the given data
		TAKES trained learner, transformed arguments,
		RETURNS the learner after this batch of training
		"""
		# StreamingDotFeatures?
		raise NotImplementedError



	def _applier(self, learner, testX, arguments, customDict):
		"""
		use the given learner to do testing/prediction on the given test set
		TAKES a TrainedLearner object that can be tested on 
		RETURNS UML friendly results
		"""
		try:
			retLabels = learner.apply(testX)
		except Exception as e:
			print e
			return None
		return retLabels


	def _getAttributes(self, learnerBackend):
		"""
		Returns whatever attributes might be available for the given learner. For
		example, in the case of linear regression, TODO

		"""
		# check for everything start with 'get_'?
		raise NotImplementedError


	def _optionDefaults(self, option):
		"""
		Define package default values that will be used for as long as a default
		value hasn't been registered in the UML configuration file. For example,
		these values will always be used the first time an interface is instantiated.

		"""
		return None


	def _configurableOptionNames(self):
		"""
		Returns a list of strings, where each string is the name of a configurable
		option of this interface whose value will be stored in UML's configuration
		file.

		"""
		return ['location', 'sourceLocation', 'libclangLocation']


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

	def version(self):
		"""
		Return a string designating the version of the package underlying this interface
		"""
		if self.versionString is None:
			shogunLib = importlib.import_module('shogun.Library')
#			import pdb
#			pdb.set_trace()
			self.versionString = shogunLib.Version_get_version_release()

		return self.versionString

	######################
	### METHOD HELPERS ###
	######################

	def setupParameterManifest(self):
		"""
		Load manifest containing parameter names and defaults for all relevant objects
		in shogun. If manifest is missing, empty, or outdated then run the discovery code.
		The discovery code attempts to parse header and source files in the directory
		associated with the 'location' option.

		"""
		# issue: protecting users from a failed clang import
		location = self.getOption('libclangLocation')

		try:
			clang.cindex.Config.set_library_file(location)
			clang.cindex.Index.create()
		except Exception as e:
			#print str(e) # send it to warning log?
			self.allowDiscovery = False

		# find most likely manifest file
		metadataPath = os.path.join(UML.UMLPath, 'interfaces', 'metadata')
		best = self._findBestManifest(metadataPath)
		exists = os.path.exists(best)

		shogunSourcePath = self.getOption('sourceLocation')

		ranDiscovery = False
		# if file missing:
		if not exists and self.allowDiscovery:
			self._paramsManifest = discoverConstructors(shogunSourcePath)
			ranDiscovery = True
		# manifest file present
		else:
			with open(best, 'r') as fp:
				self._paramsManifest = json.load(fp, object_hook=_enforceNonUnicodeStrings)
			accurate = None
			empty = (self._paramsManifest == {})
			# grab version data
			if not empty:
				accurate = True # TODO: actually do some checks
				if os.path.basename(best).split('_', 1)[1] != self.version():
					accurate = False
			# if empty or different version:
			if (empty or not accurate) and self.allowDiscovery:
				self._paramsManifest = discoverConstructors(shogunSourcePath)
				ranDiscovery = True

		modified = False
		# check params for each learner in listLearner
			# if no params:
				# modify manifest to have empty param name list for that learner
		# but wait, do we really want this ???

		# has it been written to a file? did we modify the manifest in memory?
		if ranDiscovery or modified:
			writePath = os.path.join(metadataPath, ('shogunParameterManifest_' + self.version()))
			with open(writePath, 'w') as fp:
				json.dump(self._paramsManifest, fp, indent=4)


	def _findBestManifest(self, metadataPath):
		"""
		Returns absolute path to the manifest file that is the closest match for
		this version of shogun

		"""
		ourVersion = self.version()
		ourVersion = distutils.version.LooseVersion(ourVersion.split('_')[0])

		possible = os.listdir(metadataPath)
		if len(possible) == 0:
			return None
	
		ours = (ourVersion, None, None)
		toSort = [ours]
		for name in possible:
			if name.startswith("shogunParameterManifest"):
				pieces = name.split('_')
				currVersion = distutils.version.LooseVersion(pieces[1])
				toSort.append((currVersion, name, ))
		sortedPairs = sorted(toSort, key=(lambda p: p[0]))
		ourIndex = sortedPairs.index(ours)

		left, right = None, None
		if ourIndex != 0:
			left = sortedPairs[ourIndex - 1]
		if ourIndex != len(sortedPairs) -1:
			right = sortedPairs[ourIndex + 1]

		best = None
		if left is None:
			best = right
		elif right is None:
			best = left
		# at least one of them must be non-None, so this must mean both
		# are non None
		else:
			best = left[1]
			for index in range(len(ourVersion.version)):
				currOurs = ourVersion.version[index]
				currL = left[0].version[index]
				currR = right[0].version[index]

				if currL == currOurs and currR != currOurs:
					best = left[1]
					break
				if currL !=currOurs and currR == currOurs:
					best = right[1]
					break

		return os.path.join(metadataPath, best[1])



	def _inputTransLabelHelper(self, labelsObj, learnerName, customDict):
		customDict['remap'] = None
		try:
			inverseMapping = None
			#if labelsObj.getTypeString() != 'Matrix':
			labelsObj = labelsObj.copyAs('Matrix')
			problemType = self._getMatchineProblemType(learnerName)
			if problemType == self.shogun.Classifier.PT_MULTICLASS:
				inverseMapping = _remapLabelsRange(labelsObj)
				customDict['remap'] = inverseMapping
				if len(inverseMapping) == 1:
					raise ArgumentException("Cannot train a classifier with data containing only one label")
				flattened = labelsObj.copyAs('numpyarray', outputAs1D=True)
				labels = self.shogun.Features.MulticlassLabels(flattened.astype(float))
			elif problemType == self.shogun.Classifier.PT_BINARY:
				inverseMapping = _remapLabelsSpecific(labelsObj, [-1,1])
				customDict['remap'] = inverseMapping
				if len(inverseMapping) == 1:
					raise ArgumentException("Cannot train a classifier with data containing only one label")
				flattened = labelsObj.copyAs('numpyarray', outputAs1D=True)
				labels = self.shogun.Features.BinaryLabels(flattened.astype(float))
			elif problemType == self.shogun.Classifier.PT_REGRESSION:
				flattened = labelsObj.copyAs('numpyarray', outputAs1D=True)
				labels = self.shogun.Features.RegressionLabels(flattened.astype(float))
			else:
				raise ArgumentException("Learner problem type (" + str(problemType) + ") not supported")
		except ImportError:
			from shogun.Features import Labels
			flattened = labelsObj.copyAs('numpyarray', outputAs1D=True)
			labels = Labels(labelsObj.astype(float))

		return labels

	def _inputTransDataHelper(self, dataObj, learnerName):
		typeString = dataObj.getTypeString()
		if typeString == 'Sparse':
			#raw = dataObj.data.tocsc().astype(numpy.float)
			#raw = raw.transpose()
			raw = dataObj.copyAs("scipy csc", rowsArePoints=False)
			trans = self.shogun.Features.SparseRealFeatures()
			trans.set_sparse_feature_matrix(raw)
			if 'Online' in learnerName:
				trans = self.shogun.Features.StreamingSparseRealFeatures(trans)
		else:
			#raw = dataObj.copyAs('numpyarray').astype(numpy.float)
			#raw = raw.transpose()
			raw = dataObj.copyAs('numpyarray', rowsArePoints=False)
			trans = self.shogun.Features.RealFeatures()
			trans.set_feature_matrix(raw)
			if 'Online' in learnerName:
				trans = self.shogun.Features.StreamingRealFeatures()
		return trans

	def _queryParamManifest(self, name):
		"""
		Checks the param manifest for an entry associated with the given name.
		Returns a list of list of parameter names if an entry is found, None
		otherwise. The parameter manifest is the raw output of constructor parsing,
		so there are some idiosyncrasies in the naming that this helper
		navigates
		"""
		ret = None
		# exactly correct key
		if name in self._paramsManifest:
			ret = copy.deepcopy(self._paramsManifest[name])
		# some objects have names which start with a capital C and then
		# are followed by the name they would have in the documentation
		if 'C' + name in self._paramsManifest:
			ret = copy.deepcopy(self._paramsManifest['C' + name])

		return ret

	class HiddenDefault(object):
		def __init__(self, name, typeString='UNKNOWN'):
			self.name = name
			self.typeString = typeString
		def __eq__(self, other):
			if isinstance(other, Shogun.HiddenDefault):
				return self.name == other.name and self.typeString == other.typeString
			return False
		def __copy__(self):
			return Shogun.HiddenDefault(self.name, self.typeString)
		def __deepcopy(self):
			return self.__copy__()

#######################
### GENERIC HELPERS ###
#######################

excludedLearners2 = [# parent classes, not actually runnable
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
			#'BalancedConditionalProbabilityTree', # streaming dense features input
			#'ConditionalProbabilityTree', 	 # requires streaming features
			'DomainAdaptationSVMLinear', #segfault
			'DomainAdaptationMulticlassLibLinear', # segFault
			'DomainAdaptationSVM',
			#'DualLibQPBMSOSVM',  # problem type 3
			'FeatureBlockLogisticRegression',  # remapping
			'KernelRidgeRegression', #segfault
			#'KernelStructuredOutputMachine', # problem type 3
			#'LatentSVM', # problem type 4
			'LibLinearRegression',
			#'LibSVMOneClass',
			#'LinearMulticlassMachine', # mixes machines. is this even possible to run?
			#'LinearStructuredOutputMachine', # problem type 3
			#'MKLMulticlass', # needs combined kernel type?
			#'MKLClassification', # compute by subkernel not implemented
			#'MKLOneClass', # Interleaved MKL optimization is currently only supported with SVMlight
			#'MKLRegression',  # kernel stuff?
			'MultitaskClusteredLogisticRegression',   # assertion error
			'MultitaskCompositeMachine',  # takes machine as input?
			#'MultitaskL12LogisticRegression',  # assertion error
			'MultitaskLeastSquaresRegression',  #core dump
			'MultitaskLogisticRegression',  #core dump
			#'MultitaskTraceLogisticRegression',  # assertion error
			'OnlineLibLinear', # needs streaming dot features
			'OnlineSVMSGD', # needs streaming dot features
			#'PluginEstimate', # takes string inputs?
			#'RandomConditionalProbabilityTree',  # takes streaming dense features
			#'RelaxedTree', # [ERROR] Call set_machine_for_confusion_matrix before training
			#'ShareBoost', # non standard input
			#'StructuredOutputMachine', # problem type 3
			#'SubGradientSVM', #doesn't terminate
			'VowpalWabbit',  # segfault
			#'WDSVMOcas', # string input

			# functioning learners
			#'AveragedPerceptron'
			'GaussianNaiveBayes', # something wonky with getting scores
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
			'NewtonSVM',
			#'Perceptron',
			#'SGDQN',
			#'SVMLight',
			#'SVMLightOneClass',
			#'SVMLin',
			#'SVMOcas', 
			#'SVMSGD',
			#'SVRLight',
			]

def _enforceNonUnicodeStrings(manifest):
	for name in manifest:
		groupList = manifest[name]
		for group in groupList:
			for i in xrange(len(group)):
				group[i] = str(group[i])
	return manifest


def _remapLabelsRange(toRemap):
	"""
	Takes the object toRemap, which must be a data representation with a single point,
	and maps the values in that point into the range between 0 and n-1, where n is
	the number of distinct values in that feature. The object is modified, and the
	inverse mapping is returned a length n list

	"""

	mapping = {}
	inverse = []
	invIndex = 0

	view = toRemap.featureView(0)

	for x in xrange(toRemap.pointCount):
		value = view[x]
		if value not in mapping:
			mapping[value] = invIndex
			inverse.append(value)
			invIndex += 1
		view[x] = mapping[value]

	return inverse

def _remapLabelsSpecific(toRemap, space):
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

	view = toRemap.featureView(0)

	for x in xrange(toRemap.pointCount):
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

	for x in xrange(toRemap.pointCount):
		value = view[x]
		view[x] = space[mapping[value]]

	return inverse



def discoverConstructors(path, desiredFile=None, desiredExt=['.cpp']):
	"""
	Recursively visit all directories in the given path, calling
	findConstructors for each cpp source file

	"""

	results = {}
	contents = []
	for (folderPath, subFolders, contents) in os.walk(path):
		for fileName in contents:
			filePath = os.path.join(folderPath,fileName)
			(rootName, ext) = os.path.splitext(fileName)
			(rootPath, ext) = os.path.splitext(filePath)
			if desiredFile is None or rootName in desiredFile:
				if ext in desiredExt:
					findConstructors(filePath, results, rootPath)

	return results

def findConstructors(fileName, results, targetDirectory):
	""" Find all constructors and list their params in the given file """
#	pdb.set_trace()
#	print fileName
	index = clang.cindex.Index.create()
	tu = index.parse(fileName)
	findConstructorsBackend(tu.cursor, results, targetDirectory)


def findConstructorsBackend(node, results, targetDirectory):
	""" Recursively visit all nodes, checking if it is a constructor """
	if node.location.file is not None:
		if not node.location.file.name.startswith(targetDirectory):
			return

#	hasConstructor = False
#	for child in node.get_children():
#		if child.kind == clang.cindex.CursorKind.CONSTRUCTOR:
#			hasConstructor = True
#	if hasConstructor:
#		import pdb
#		pdb.set_trace()
#		for child in node.get_children():
#			pass

#	print node.kind
#	print node.spelling
#	if node.kind == clang.cindex.CursorKind.CLASS_DECL:
#		print node.spelling
#		for child in node.get_children():
#			print child.kind
	if node.kind == clang.cindex.CursorKind.CONSTRUCTOR:
		constructorName = node.spelling
		args = []
		for value in node.get_arguments():
			args.append(value.spelling)
			# TODO value.type.spelling
		
		#print "%s%s" % (constructorName, str(args))
		if not constructorName in results:
			results[constructorName] = []
		if args not in results[constructorName]:
			results[constructorName].append(args)
	# Recurse for children of this node if it isn't a constructor
	else:
		for child in node.get_children():
			findConstructorsBackend(child, results, targetDirectory)

