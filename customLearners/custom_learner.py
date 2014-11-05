
import abc
import inspect
import numpy

class CustomLearner(object):
	"""
	Base class defining a hierarchy of objects which encapsulate what is needed
	to be a single learner callable through the Custom Universal Interface. At
	minimum, a subclass must provide an implementation for the method apply()
	and at least one out of train() or incrementalTrain(). If incrementalTrain()
	is implemented yet train() is not, then incremental train is used in place
	of calls to train(). Furthermore, a subclass must not require any arguments
	for its __init__() method.

	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		pass

	@classmethod
	def validateSubclass(cls, check):
		# check learnerType 
		accepted = ["unknown", 'regression', 'classification', 'featureselection', 'dimensionalityreduction']
		if not hasattr(check, 'learnerType') or check.learnerType not in accepted:
			raise TypeError("The custom learner must have a class variable named 'learnerType' with a value from the list " + str(accepted))

		# check train / apply params
		trainInfo = inspect.getargspec(check.train)
		incInfo = inspect.getargspec(check.incrementalTrain)
		applyInfo = inspect.getargspec(check.apply)
		if trainInfo[0][0] != 'self' or trainInfo[0][1] != 'trainX' or trainInfo[0][2] != 'trainY':
			raise TypeError("The train method of a CustomLearner must have 'trainX' and 'trainY' as its first two (non 'self') parameters")
		if incInfo[0][0] != 'self' or incInfo[0][1] != 'trainX' or incInfo[0][2] != 'trainY':
			raise TypeError("The incrementalTrain method of a CustomLearner must have 'trainX' and 'trainY' as its first two (non 'self') parameters")
		if applyInfo[0][0] != 'self' or applyInfo[0][1] != 'testX':
			raise TypeError("The apply method of a CustomLearner must have 'testX' as its first (non 'self') parameter")

		# need either train or incremental train
		def overridden(func):
			for checkBases in inspect.getmro(check):
				if func in checkBases.__dict__ and checkBases == check:
					return True
			return False
		
		incrementalImplemented = overridden('incrementalTrain')
		trainImplemented = overridden('train')
		if not trainImplemented:
			if not incrementalImplemented:
				raise TypeError("Must provide an implementation for train()")
			else:
				check.train = check.incrementalTrain
				newVal = check.__abstractmethods__ - frozenset(['train'])
				check.__abstractmethods__ = newVal
			
		# getScores has same params as apply if overridden
		getScoresImplemented = overridden('getScores')
		if getScoresImplemented:
			getScoresInfo = inspect.getargspec(check.getScores)
			if getScoresInfo != applyInfo:
				raise TypeError("The signature for the getScores() method must be the same as the apply() method")

		# check the return type of options() is legit
		options = check.options()
		if not isinstance(options, list):
			raise TypeError("The classmethod options must return a list of stings")
		for name in options:
			if not isinstance(name, basestring):
				raise TypeError("The classmethod options must return a list of stings")

		# check that we can instantiate this subclass
		initInfo = inspect.getargspec(check.__init__)
		if len(initInfo[0]) > 1 or initInfo[0][0] != 'self':
			raise TypeError("The __init__() method for this class must only have self as an argument")


		# instantiate it so that the abc stuff gets validated
		check()	

	@classmethod
	def getLearnerParameterNames(cls):
		return cls.getTrainParameters() + cls.getApplyParameters()

	@classmethod
	def getLearnerDefaultValues(cls):
		return dict(cls.getTrainDefaults().items() + cls.getApplyDefaults().items())

	@classmethod
	def getTrainParameters(cls):
		info = inspect.getargspec(cls.train)
		return info[0][2:]

	@classmethod
	def getTrainDefaults(cls):
		info = inspect.getargspec(cls.train)
		(objArgs,v,k,d) = info
		ret = {}
		if d is not None:
			for i in xrange(len(d)):
				ret[objArgs[-(i+1)]] = d[-(i+1)]
		return ret

	@classmethod
	def getApplyParameters(cls):
		info = inspect.getargspec(cls.apply)
		return info[0][1:]

	@classmethod
	def getApplyDefaults(cls):
		info = inspect.getargspec(cls.apply)
		(objArgs,v,k,d) = info
		ret = {}
		if d is not None:
			for i in xrange(len(d)):
				ret[objArgs[-(i+1)]] = d[-(i+1)]
		return ret

	@classmethod
	def options(self):
		return []

	def getScores(self, testX):
		"""
		If this learner is a classifier, then return the scores for each
		class on each data point, otherwise raise an exception. The scores
		must be returned in the natural ordering of the classes.

		This method may be optionally overridden by a concrete subclass. 
		"""
		raise NotImplementedError

	def trainForInterface(self, trainX, trainY, arguments):
		self.trainArgs = arguments

		# TODO store list of classes in trainY if classifying
		if self.__class__.learnerType == 'classification':
			self.labelList = numpy.unique(trainY.copyAs('numpyarray'))

		self.train(trainX, trainY, **arguments)

		return self

	def incrementalTrainForInterface(self, trainX, trainY, arguments):
		if self.__class__.learnerType == 'classification':
			self.labelList = numpy.union1d(self.labelList, trainY.copyAs('numpyarray').flatten())
		self.incrementalTrain(trainX, trainY)
		return self

	def applyForInterface(self, testX, arguments):
		self.applyArgs = arguments
		useArgs = {}
		for value in self.__class__.getApplyParameters():
			if value in arguments:
				useArgs[value] = arguments[value]
		return self.apply(testX, **useArgs)

	def incrementalTrain(self, trainX, trainY):
		raise RuntimeError("This learner does not support incremental training")

	@abc.abstractmethod
	def train(self, trainX, trainY):
		pass

	@abc.abstractmethod
	def apply(self, testX):
		pass


