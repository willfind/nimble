"""
Contains the CustomLearner class.
"""

import abc
import inspect

import numpy

from .helpers import inspectArguments
from .utility import dtypeConvert


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

    def __init__(self):
        pass

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
            labels = dtypeConvert(trainY.copy(to='numpyarray'))
            self.labelList = numpy.unique(labels)

        self.train(trainX, trainY, **arguments)

        return self

    def incrementalTrainForInterface(self, trainX, trainY, arguments):
        if self.__class__.learnerType == 'classification':
            flattenedY = dtypeConvert(trainY.copy(to='numpyarray').flatten())
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
