"""




"""

import scikit_learn_interface_old as skl
from UML.interfaces.universal_interface_lookalike import UniversalInterfaceLookalike

class SciKitLearnLookalike(UniversalInterfaceLookalike):
	"""

	"""

	def __init__(self):
		"""

		"""
		super(SciKitLearnLookalike, self).__init__()

	def run(self, learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', sendToLog=True):
		return skl.sciKitLearn(learnerName, trainX, trainY, testX, arguments, output, scoreMode, multiClassStrategy, sendToLog)


	def listLearners(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		return skl.listSciKitLearnLearners()


	def getLearnerParameterNames(self, name):
		return skl.getParameters(name)

	def getLearnerDefaultValues(self, name):
		return skl.getDefaultValues(name)

	def _getParameterNames(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		return skl.getParameters(name)


	def _getDefaultValues(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""
		return skl.getDefaultValues(name)

	def isAlias(self, name):
		"""
		Returns true if the name is an accepted alias for this interface

		"""
		if name.lower() in ['skl', 'sklearn', 'scikitlearn', 'sci_kit_learn']:
			return True
		else:
			return False


	def getCanonicalName(self):
		"""
		Returns the string name that will uniquely identify this interface

		"""
		return "sciKitLearnOld"
