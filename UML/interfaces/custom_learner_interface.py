"""
Base class for a CustomLearner.
"""

from __future__ import absolute_import
import copy

import UML

from UML.exceptions import InvalidArgumentValue
from UML.customLearners import CustomLearner
from UML.interfaces.universal_interface import UniversalInterface
from UML.docHelpers import inheritDocstringsFactory


@inheritDocstringsFactory(UniversalInterface)
class CustomLearnerInterface(UniversalInterface):
    """
    Base interface class for customLearners.
    """

    _ignoreNames = ['trainX', 'trainY', 'testX']

    def __init__(self, packageName):
        self.name = packageName
        self.registeredLearners = {}
        self._configurableOptionNamesAvailable = []
        super(CustomLearnerInterface, self).__init__()


    def registerLearnerClass(self, learnerClass):
        """
        Record the given learnerClass as being accessible through this
        particular interface. The parameter is assumed to be class
        object which is a subclass of CustomLearner and has been
        validated by CustomLearner.validateSubclass(); no sanity
        checking is performed in this method.
        """
        self.registeredLearners[learnerClass.__name__] = learnerClass

        options = learnerClass.options()

        #		if isinstance(options, list):
        #			temp = {}
        #			for name in options:
        #				temp[name] = ''
        #			options = temp

        #		for (k,v) in options.items():
        #			fullKey = learnerClass.__name__ + '.' + k
        #			self._configurableOptionNamesAvailable[fullKey] = v
        for name in options:
            fullName = learnerClass.__name__ + '.' + name
            self._configurableOptionNamesAvailable.append(fullName)

    def deregisterLearner(self, learnerName):
        """
        Remove accessibility of the learner with the given name from
        this interface.

        Returns True of there are other learners still accessible
        through this interface, False if there are not.
        """
        if not learnerName in self.registeredLearners:
            msg = "Given learnerName does not refer to a learner accessible "
            msg += "through this interface"
            raise InvalidArgumentValue(msg)

        # TODO remove option names
        toRemove = self.registeredLearners[learnerName].options()
        fullNameToRemove = [learnerName + '.' + x for x in toRemove]

        temp = []
        for opName in self._configurableOptionNamesAvailable:
            if not opName in fullNameToRemove:
                temp.append(opName)

        self._configurableOptionNamesAvailable = temp

        del self.registeredLearners[learnerName]

        return len(self.registeredLearners) == 0

    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
        return True

    def _listLearnersBackend(self):
        return list(self.registeredLearners.keys())


    def learnerType(self, name):
        return self.registeredLearners[name].learnerType


    def _findCallableBackend(self, name):
        if name in self.registeredLearners:
            return self.registeredLearners[name]
        else:
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


    def _getScores(self, learner, testX, arguments, customDict):
        return learner.getScores(testX)

    def _getScoresOrder(self, learner):
        if learner.learnerType == 'classification':
            return learner.labelList
        else:
            msg = "Can only get scores order for a classifying learner"
            raise InvalidArgumentValue(msg)

    def isAlias(self, name):
        return name.lower() == self.getCanonicalName().lower()


    def getCanonicalName(self):
        return self.name

    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             arguments, customDict):
        retArgs = None
        if arguments is not None:
            retArgs = copy.copy(arguments)
        return (trainX, trainY, testX, retArgs)

    def _outputTransformation(self, learnerName, outputValue,
                              transformedInputs, outputType, outputFormat,
                              customDict):
        if isinstance(outputValue, UML.data.Base):
            return outputValue
        else:
            return UML.createData('Matrix', outputValue)

    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        ret = self.registeredLearners[learnerName]()
        return ret.trainForInterface(trainX, trainY, arguments)

    def _incrementalTrainer(self, learner, trainX, trainY, arguments,
                            customDict):
        return learner.incrementalTrainForInterface(trainX, trainY, arguments)

    def _applier(self, learner, testX, arguments, customDict):
        return learner.applyForInterface(testX, arguments)

    def _getAttributes(self, learnerBackend):
        contents = dir(learnerBackend)
        excluded = dir(CustomLearner)
        ret = {}
        for name in contents:
            if not name.startswith('_') and not name in excluded:
                ret[name] = getattr(learnerBackend, name)

        return ret

    def _optionDefaults(self, option):
        return None


    def _configurableOptionNames(self):
        return self._configurableOptionNamesAvailable

    def _exposedFunctions(self):
        return []

    def version(self):
        pass
