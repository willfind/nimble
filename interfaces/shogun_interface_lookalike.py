"""




"""

import importlib

import shogun_interface_old as shogun
from universal_interface_lookalike import UniversalInterfaceLookalike
from interface_helpers import cacheWrapper

locationCache = {}

class Shogun(UniversalInterfaceLookalike):
	"""

	"""

	

	def __init__(self):
		"""

		"""
		super(Shogun, self).__init__()

	def trainAndApply(self, learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', sendToLog=True):
		return shogun.shogun(learnerName, trainX, trainY, testX, arguments, output, scoreMode, 'default', sendToLog)


	@cacheWrapper
	def _listLearnersBackend(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		return shogun.listShogunLearners()

	def _getLearnerParameterNamesBackend(self, name):
		return shogun.getParameters(name)

	def _getLearnerDefaultValuesBackend(self, name):
		return shogun.getDefaultValues(name)

	def _getParameterNamesBackend(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		return shogun.getParameters(name)


	def _getDefaultValuesBackend(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""
		return shogun.getDefaultValues(name)


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
		return self._findInPackage(name)

	def _findInPackage(self, name, parent=None):
		"""
		Import the desired python package, and search for the module containing
		the wanted learner. For use by interfaces to python packages.

		"""
		packageMod = importlib.import_module('shogun')

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

