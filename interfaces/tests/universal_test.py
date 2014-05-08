"""
Unit tests for the universal interface function.

"""

from nose.tools import raises

import UML

from UML.exceptions import ArgumentException
from UML.interfaces.universal_interface import UniversalInterface

class TestInterface(UniversalInterface):

	def __init__(self):
		super(TestInterface, self).__init__()

	def _listLearnersBackend(self):
		return ['l0', 'l1', 'l2', 'exposeTest']

	def _getLearnerParameterNamesBackend(self, name):
		return self._getParameterNames(name)

	def _getParameterNamesBackend(self, name):
		if name == 'l0':
			return [['l0a0', 'l0a1']]
		elif name == 'l1':
			return [['l1a0']]
		elif name == 'l2':
			return [['dup', 'sub']]
		elif name == 'l1a0':
			return [['l1a0a0']]
		elif name == 'subFunc':
			return [['dup']]
		elif name == 'foo':
			return [['estimator']]
		elif name == 'initable':
			return [['C', 'thresh']]
		else:
			return [[]]

	def _getLearnerDefaultValuesBackend(self, name):
		return self._getDefaultValues(name)

	def _getDefaultValuesBackend(self, name):
		if name == 'bar':
			return [{'estimator':'initable'}, {'estimater2':'initable'}]
		if name == 'foo':
			return [{'estimator':'initable'}]
		if name == 'initable':
			return [{'C':1, 'thresh':0.5}]
		return [{}]

	def isAlias(self, name):
		if name.lower() in ['test']:
			return True
		else:
			return False

	def getCanonicalName(self):
		return "Test"

	def _inputTransformation(self, learnerName, trainX, trainY, testX, arguments, customDict):
		return (trainX, trainY, testX, arguments)

	def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputType, outputFormat, customDict):
		return outputValue

	def _configurableOptionNames(self):
		return []

	def _optionDefaults(self, option):
		return []

	def _exposedFunctions(self):
		return [self.exposedOne, self.exposedTwo, self.exposedThree]

	def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
		return (learnerName, trainX, trainY, arguments)

	def _applier(self, learner, testX, customDict):	
		return testX

	def _findCallableBackend(self, name):
		available = ['l0', 'l1', 'l2', 'l1a0', 'subFunc', 'initable', 'foo', 'bar', 'exposeTest']
		if name in available:
			return name
		else:
			return None

	def _getScores(self, learner, trainX, trainY, arguments):
		pass

	def _getScoresOrder(self, learner, testX):
		pass

	def learnerType(self, name):
		pass

	def _getAttributes(self, learnerBackend):
		pass

	def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
		pass

	def exposedOne(self):
		return 1

	def exposedTwo(self):
		return 2

	def exposedThree(self, trainedLearner):
		assert trainedLearner.learnerName == 'exposeTest'
		return 3



TestObject = TestInterface()

#######################################
### _validateArgumentDistribution() ###
#######################################

#def test__validateArgumentDistribution

# TODO tests involving default arguments

@raises(ArgumentException)
def test__validateArgumentDistributionMissingArgument():
	learner = 'l0'
	arguments = {'l0a0':1}
	TestObject._validateArgumentDistribution(learner, arguments)

@raises(ArgumentException)
def test__validateArgumentDistributionOverlappingArgumentsFlat():
	learner = 'l2'
	arguments = {'dup':1, 'sub':'subFunc'}
	TestObject._validateArgumentDistribution(learner, arguments)

@raises(ArgumentException)
def test__validateArgumentDistributionExtraArgument():
	learner = 'l1'
	arguments = {'l1a0':1, 'l5a100':11}
	TestObject._validateArgumentDistribution(learner, arguments)

def test__validateArgumentDistributionWorking():
	learner = 'l1'
	arguments = {'l1a0':1}
	ret = TestObject._validateArgumentDistribution(learner, arguments)
	assert ret == {'l1a0':1}

def test__validateArgumentDistributionOverlappingArgumentsNested():
	learner = 'l2'
	arguments = {'dup':1, 'sub':'subFunc', 'subFunc':{'dup':2}}
	ret = TestObject._validateArgumentDistribution(learner, arguments)
	assert ret == {'dup':1, 'sub':'subFunc', 'subFunc':{'dup':2}}

def test__validateArgumentDistributionInstantiableDefaultValue():
	learner = 'foo'
	arguments = {}
	ret = TestObject._validateArgumentDistribution(learner, arguments)
	assert ret == {'estimator':'initable', 'initable':{'C':1, 'thresh':0.5}}

def test__validateArgumentDistributionInstantiableDelayedAllocationOfSubArgsSeparate():
	learner = 'foo'
	arguments = {'initable':{'C':11}}
	ret = TestObject._validateArgumentDistribution(learner, arguments)
	assert ret == {'estimator':'initable', 'initable':{'C':11, 'thresh':0.5}}

def test__validateArgumentDistributionInstantiableDelayedAllocationOfSubArgsInFlat():
	learner = 'foo'
	arguments = {'C':11}
	ret = TestObject._validateArgumentDistribution(learner, arguments)
	assert ret == {'estimator':'initable', 'initable':{'C':11, 'thresh':0.5}}

def test__validateArgumentDistributionInstantiableArgWithDefaultValue():
	learner = 'foo'
	arguments = {'estimator':'initable', 'C':.11, 'thresh':15}
	ret = TestObject._validateArgumentDistribution(learner, arguments)
	assert ret == {'estimator':'initable', 'initable':{'C':.11, 'thresh':15}}



#########################
### exposed functions ###
#########################

def test_eachExposedPresent():
	dummyUML = UML.createData('Matrix', [[1,1],[2,2]])
	tl = TestObject.train('exposeTest', dummyUML, dummyUML)
	assert hasattr(tl, 'exposedOne')
	assert tl.exposedOne() == 1
	assert hasattr(tl, 'exposedTwo')
	assert tl.exposedTwo() == 2
	assert hasattr(tl, 'exposedThree')
	assert tl.exposedThree() == 3





# import each test suite


# one test for each interface, recalls all the tests in the suite?




