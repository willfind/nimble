"""
Contains the CustomLearner class.
"""

from __future__ import absolute_import
import abc
import inspect

import numpy
import six
from six.moves import range

from nimble.helpers import inspectArguments


class CustomLearner(six.with_metaclass(abc.ABCMeta, object)):
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

    def __init__(self):
        pass

    @classmethod
    def validateSubclass(cls, check):
        """
        Class method called during custom learner registration which
        ensures the the given class conforms to the custom learner
        specification.
        """
        # check learnerType
        accepted = ["unknown", 'regression', 'classification',
                    'featureselection', 'dimensionalityreduction']
        if (not hasattr(check, 'learnerType')
                or check.learnerType not in accepted):
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
            getScoresInfo = inspectArguments(check.getScores)
            if getScoresInfo != applyInfo:
                msg = "The signature for the getScores() method must be the "
                msg += "same as the apply() method"
                raise TypeError(msg)

        # check the return type of options() is legit
        options = check.options()
        if not isinstance(options, list):
            msg = "The classmethod options must return a list of stings"
            raise TypeError(msg)
        for name in options:
            if not isinstance(name, six.string_types):
                msg = "The classmethod options must return a list of stings"
                raise TypeError(msg)

        # check that we can instantiate this subclass
        initInfo = inspectArguments(check.__init__)
        if len(initInfo[0]) > 1 or initInfo[0][0] != 'self':
            msg = "The __init__() method for this class must only have self "
            msg += "as an argument"
            raise TypeError(msg)


        # instantiate it so that the abc stuff gets validated
        check()

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

    @classmethod
    def options(self):
        """
        Class function which supplies the names of the configuration
        options associated with this learner.
        """
        return []

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

        self.trainArgs = arguments

        # TODO store list of classes in trainY if classifying
        if self.__class__.learnerType == 'classification':
            self.labelList = numpy.unique(trainY.copy(to='numpyarray'))

        self.train(trainX, trainY, **arguments)

        return self

    def incrementalTrainForInterface(self, trainX, trainY, arguments):
        if self.__class__.learnerType == 'classification':
            flattenedY = trainY.copy(to='numpyarray').flatten()
            self.labelList = numpy.union1d(self.labelList, flattenedY)
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
        msg = "This learner does not support incremental training"
        raise RuntimeError(msg)

    @abc.abstractmethod
    def train(self, trainX, trainY):
        pass

    @abc.abstractmethod
    def apply(self, testX):
        pass
