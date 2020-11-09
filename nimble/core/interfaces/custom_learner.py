"""
Contains the interface and object for CustomLearner.
"""

import abc
import inspect
import copy

import numpy

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble._utility import inheritDocstringsFactory
from nimble._utility import inspectArguments
from nimble._utility import dtypeConvert
from .universal_interface import UniversalInterface
from .universal_interface import captureOutput


@inheritDocstringsFactory(UniversalInterface)
class CustomLearnerInterface(UniversalInterface):
    """
    Base interface class for any CustomLearner.
    """

    _ignoreNames = ['trainX', 'trainY', 'testX']

    def __init__(self, packageName):
        self.name = packageName
        self.registeredLearners = {}
        super(CustomLearnerInterface, self).__init__()

    def registerLearnerClass(self, learnerClass):
        """
        Record the given learnerClass as being accessible through this
        particular interface.
        """
        validateCustomLearnerSubclass(learnerClass)
        self.registeredLearners[learnerClass.__name__] = learnerClass


    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
        return True

    def getCanonicalName(self):
        return self.name

    @captureOutput
    def listLearners(self):
        """
        Return a list of all learners callable through this interface.
        """
        # can't cache because new learners can be registered
        return self._listLearnersBackend()

    def _listLearnersBackend(self):
        return list(self.registeredLearners.keys())


    def learnerType(self, name):
        return self.registeredLearners[name].learnerType


    def _findCallableBackend(self, name):
        if name in self.registeredLearners:
            return self.registeredLearners[name]

        return None

    def _getParameterNamesBackend(self, name):
        return self.getLearnerParameterNames(name)

    def _getLearnerParameterNamesBackend(self, learnerName):
        ret = None
        if learnerName in self.registeredLearners:
            learner = self.registeredLearners[learnerName]
            temp = learner.getLearnerParameterNames()
            after = []
            for value in temp:
                if value not in self._ignoreNames:
                    after.append(value)
            ret = [after]

        return ret

    def _getDefaultValuesBackend(self, name):
        return self.getLearnerDefaultValues(name)

    def _getLearnerDefaultValuesBackend(self, learnerName):
        ret = None
        if learnerName in self.registeredLearners:
            learner = self.registeredLearners[learnerName]
            temp = learner.getLearnerDefaultValues()
            after = {}
            for key in temp:
                if key not in self._ignoreNames:
                    after[key] = temp[key]
            ret = [after]

        return ret


    def _getScores(self, learnerName, learner, testX, newArguments,
                   storedArguments, customDict):
        return learner.getScores(testX)

    def _getScoresOrder(self, learner):
        if learner.learnerType == 'classification':
            return learner.labelList

        msg = "Can only get scores order for a classifying learner"
        raise InvalidArgumentValue(msg)

    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             arguments, customDict):
        retArgs = None
        if arguments is not None:
            retArgs = copy.copy(arguments)
        return (trainX, trainY, testX, retArgs)

    def _outputTransformation(self, learnerName, outputValue,
                              transformedInputs, outputType, outputFormat,
                              customDict):
        if isinstance(outputValue, nimble.core.data.Base):
            return outputValue

        return nimble.data('Matrix', outputValue, useLog=False)

    def _trainer(self, learnerName, trainX, trainY, arguments, randomSeed,
                 customDict):
        ret = self.registeredLearners[learnerName]()
        return ret.trainForInterface(trainX, trainY, arguments)

    def _incrementalTrainer(self, learnerName, learner, trainX, trainY,
                            arguments, customDict):
        return learner.incrementalTrainForInterface(trainX, trainY, arguments)

    def _applier(self, learnerName, learner, testX, newArguments,
                 storedArguments, customDict):
        backendArgs = learner.getApplyParameters()[1:] # ignore testX param
        applyArgs = self._getMethodArguments(backendArgs, newArguments,
                                             storedArguments)
        return learner.applyForInterface(testX, applyArgs)

    def _getAttributes(self, learnerBackend):
        contents = dir(learnerBackend)
        excluded = dir(CustomLearner)
        ret = {}
        for name in contents:
            if not name.startswith('_') and not name in excluded:
                ret[name] = getattr(learnerBackend, name)

        return ret

    def _exposedFunctions(self):
        return []

    def version(self):
        pass


class CustomLearner(metaclass=abc.ABCMeta):
    """
    Base class defining a hierarchy of objects which encapsulate what is
    needed to be a single learner callable through the Custom Universal
    Interface. At minimum, a subclass must provide an implementation for
    the method apply() and at least one out of train() or
    incrementalTrain(). If incrementalTrain() is implemented yet train()
    is not, then incremental train is used in place of calls to train().
    Furthermore, a subclass must not require any arguments for its
    __init__() method.
    """
    learnerType = None

    def __init__(self):
        self.labelList = None

    @classmethod
    def getLearnerParameterNames(cls):
        """
        Class method used by the a custom learner interface to supply
        learner parameters to the user through the standard nimble
        functions.
        """
        return cls.getTrainParameters() + cls.getApplyParameters()

    @classmethod
    def getLearnerDefaultValues(cls):
        """
        Class method used by the a custom learner interface to supply
        learner parameter default values to the user through the
        standard nimble functions.
        """
        trainDefaults = cls.getTrainDefaults().items()
        applyDefaults = cls.getApplyDefaults().items()
        return dict(list(trainDefaults) + list(applyDefaults))

    @classmethod
    def getTrainParameters(cls):
        """
        Class method used to determine the parameters of only the train
        method.
        """
        info = inspectArguments(cls.train)
        return info[0][2:]

    @classmethod
    def getTrainDefaults(cls):
        """
        Class method used to determine the default values of only the
        train method.
        """
        info = inspectArguments(cls.train)
        (objArgs, _, _, d) = info
        ret = {}
        if d is not None:
            for i in range(len(d)):
                ret[objArgs[-(i + 1)]] = d[-(i + 1)]
        return ret

    @classmethod
    def getApplyParameters(cls):
        """
        Class method used to determine the parameters of only the apply
        method.
        """
        info = inspectArguments(cls.apply)
        return info[0][1:]

    @classmethod
    def getApplyDefaults(cls):
        """
        Class method used to determine the default values of only the
        apply method.
        """
        info = inspectArguments(cls.apply)
        (objArgs, _, _, d) = info
        ret = {}
        if d is not None:
            for i in range(len(d)):
                ret[objArgs[-(i + 1)]] = d[-(i + 1)]
        return ret

    def getScores(self, testX):
        """
        If this learner is a classifier, then return the scores for each
        class on each data point, otherwise raise an exception. The
        scores must be returned in the natural ordering of the classes.

        This method may be optionally overridden by a concrete subclass.
        """
        msg = "This custom learner has not implemented the getScores method"
        raise NotImplementedError(msg)

    def trainForInterface(self, trainX, trainY, arguments):
        if self.__class__.learnerType == 'classification':
            labels = dtypeConvert(trainY.copy(to='numpyarray'))
            self.labelList = numpy.unique(labels)

        self.train(trainX, trainY, **arguments)

        return self

    def incrementalTrainForInterface(self, trainX, trainY, arguments):
        if self.__class__.learnerType == 'classification':
            flattenedY = dtypeConvert(trainY.copy(to='numpyarray').flatten())
            if self.labelList is None: # no previous training
                self.labelList = []
            self.labelList = numpy.union1d(self.labelList, flattenedY)
        self.incrementalTrain(trainX, trainY, **arguments)
        return self

    def applyForInterface(self, testX, arguments):
        return self.apply(testX, **arguments)

    def incrementalTrain(self, trainX, trainY):
        msg = "This learner does not support incremental training"
        raise RuntimeError(msg)

    @abc.abstractmethod
    def train(self, trainX, trainY):
        pass

    @abc.abstractmethod
    def apply(self, testX):
        pass


def validateCustomLearnerSubclass(check):
    """
    Ensures the the given class conforms to the custom learner
    specification.
    """
    # check learnerType
    accepted = ["unknown", 'regression', 'classification',
                'featureselection', 'dimensionalityreduction']
    if check.learnerType not in accepted:
        msg = "The custom learner must have a class variable named "
        msg += "'learnerType' with a value from the list " + str(accepted)
        raise TypeError(msg)

    # check train / apply params
    trainInfo = inspectArguments(check.train)
    incInfo = inspectArguments(check.incrementalTrain)
    applyInfo = inspectArguments(check.apply)
    if (trainInfo[0][0] != 'self'
            or trainInfo[0][1] != 'trainX'
            or trainInfo[0][2] != 'trainY'):
        msg = "The train method of a CustomLearner must have 'trainX' and "
        msg += " 'trainY' as its first two (non 'self') parameters"
        raise TypeError(msg)
    if (incInfo[0][0] != 'self'
            or incInfo[0][1] != 'trainX'
            or incInfo[0][2] != 'trainY'):
        msg = "The incrementalTrain method of a CustomLearner must have "
        msg += "'trainX' and 'trainY' as its first two (non 'self') "
        msg += "parameters"
        raise TypeError(msg)
    if applyInfo[0][0] != 'self' or applyInfo[0][1] != 'testX':
        msg = "The apply method of a CustomLearner must have 'testX' as "
        msg += "its first (non 'self') parameter"
        raise TypeError(msg)

    def overridden(func):
        # Determine if a function was redefined outside of its CustomLearner
        # definition. [:-2] ignores CustomLearner and object
        for checkBase in inspect.getmro(check)[:-2]:
            if func in checkBase.__dict__:
                return True
        return False

    # need either train or incremental train
    incrementalImplemented = overridden('incrementalTrain')
    trainImplemented = overridden('train')
    if not trainImplemented:
        if not incrementalImplemented:
            raise TypeError("Must provide an implementation for train()")

        check.train = check.incrementalTrain
        newVal = check.__abstractmethods__ - frozenset(['train'])
        check.__abstractmethods__ = newVal

    # getScores has same params as apply if overridden
    getScoresImplemented = overridden('getScores')
    if getScoresImplemented:
        getScoresInfo = inspectArguments(check.getScores)
        if getScoresInfo != applyInfo:
            msg = "The signature for the getScores() method must be the "
            msg += "same as the apply() method"
            raise TypeError(msg)

    # check that we can instantiate this subclass
    initInfo = inspectArguments(check.__init__)
    if len(initInfo[0]) > 1 or initInfo[0][0] != 'self':
        msg = "The __init__() method for this class must only have self "
        msg += "as an argument"
        raise TypeError(msg)


    # instantiate it so that the abc stuff gets validated
    check()
