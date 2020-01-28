"""
Interface to autoimpute package.
"""

import warnings
import types
import logging
import re

import numpy

import nimble
from nimble.interfaces.universal_interface import UniversalInterface
from nimble.interfaces.scikit_learn_interface import SciKitLearn
from nimble.exceptions import InvalidArgumentValue
from nimble.interfaces.interface_helpers import modifyImportPathAndImport
from nimble.interfaces.interface_helpers import removeFromTailMatchedLists
from nimble.interfaces.interface_helpers import PythonSearcher
from nimble.helpers import inspectArguments
from nimble.utility import inheritDocstringsFactory
from nimble.configuration import configErrors

# Contains path to sciKitLearn root directory
#sciKitLearnDir = '/usr/local/lib/python2.7/dist-packages'
autoimputeDir = None

# a dictionary mapping names to learners, or modules
# containing learners. To be used by findInPackage
locationCache = {}


@inheritDocstringsFactory(UniversalInterface)
class Autoimpute(SciKitLearn):
    """
    This class is an interface to autoimpute.
    """

    def __init__(self):
        """

        """
        # suppress DeprecationWarnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.autoimpute = modifyImportPathAndImport(autoimputeDir,
                                                        'autoimpute')

        # disable logging output from pymc3 unless specified in config
        logging.getLogger('pymc3').disabled = True

        def isLearner(obj):
            try:
                # if object cannot be instantiated without additional
                # arguments, we cannot support it at this time
                init = obj()
            except TypeError:
                return False
            # only support learners with a predict, transform,
            # fit_predict or fit_transform, all have fit attribute
            hasPred = hasattr(obj, 'predict')
            hasTrans = hasattr(obj, 'transform')
            hasFitPred = hasattr(obj, 'fit_predict')
            hasFitTrans = hasattr(obj, 'fit_transform')

            return hasPred or hasTrans or hasFitPred or hasFitTrans

        self.randomParam = 'seed'

        self._searcher = PythonSearcher(
            self.autoimpute, self.autoimpute.__all__, {}, isLearner, 1)
        _ = self._searcher.allLearners() # populate locationCache
        self.allEstimators = self._searcher._locationCache

        super(SciKitLearn, self).__init__()

    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
        try:
            import autoimpute
        except ImportError:
            return False
        return True

    @classmethod
    def getCanonicalName(cls):
        return 'autoimpute'

    @classmethod
    def isAlias(cls, name):
        return name.lower() == cls.getCanonicalName().lower()

    @classmethod
    def _installInstructions(cls):
        msg = """
To install autoimpute
---------------------
    Installation instructions for autoimpute can be found at:
    https://autoimpute.readthedocs.io/en/latest/user_guide/getting_started.html"""
        return msg

    def learnerType(self, name):
        obj = self.findCallable(name)
        if issubclass(obj, self.autoimpute.imputations.BaseImputer):
            return 'transformation'
        if issubclass(obj, self.autoimpute.analysis.MiBaseRegressor):
            if 'LogisticRegression' in name:
                return 'classification'
            return 'regression'

        return 'UNKNOWN'

    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             arguments, customDict):

        if trainX is not None:
            customDict['match'] = trainX.getTypeString()
            trainX = trainX.copy(to='pandasdataframe')

        if trainY is not None:
            trainY = trainY.copy(to='pandasdataframe')
            # trainY columns cannot be same as trainX in autoimpute
            if (trainX is not None
                    and any(col in trainX.columns for col in trainY.columns)):
                yCols = [str(col) + '_trainY_' for col in trainY.columns]
                trainY.columns = yCols

        if testX is not None:
            testX = testX.copy(to='pandasdataframe')

        instantiatedArgs = {}
        for arg, val in arguments.items():
            if isinstance(val, nimble.Init):
                val = self._argumentInit(val)
            instantiatedArgs[arg] = val

        return (trainX, trainY, testX, instantiatedArgs)

    def _outputTransformation(self, learnerName, outputValue,
                              transformedInputs, outputType, outputFormat,
                              customDict):
        # MultipleImputer outputs a generator of predicted dataframes
        # we will average the predictions together for our final fill values
        if isinstance(outputValue, types.GeneratorType):
            finalDF = None
            for i, df in outputValue:
                if finalDF is None:
                    finalDF = df
                else:
                    finalDF += df
            outputValue = finalDF / i

        # In the case of prediction we are given a row vector,
        # yet we want a column vector
        if outputFormat == "label" and len(outputValue.shape) == 1:
            outputValue = outputValue.reshape(len(outputValue), 1)

        # TODO correct
        outputType = 'Matrix'
        if outputType == 'match':
            outputType = customDict['match']
        return nimble.createData(outputType, outputValue, useLog=False)

    def _initLearner(self, learnerName, trainX, trainY, arguments):
        initNames = self._paramQuery('__init__', learnerName, ['self'])[0]
        initParams = {name: arguments[name] for name in initNames
                      if name in arguments}
        defaults = self.getLearnerDefaultValues(learnerName)[0]
        if self.randomParam in defaults and self.randomParam not in arguments:
            initParams[self.randomParam] = defaults[self.randomParam]
        learner = self.findCallable(learnerName)(**initParams)

        # need to enforce strategy as a required parameter for the imputer
        if (isinstance(learner, self.autoimpute.imputations.BaseImputer)
                and 'strategy' not in initParams):
            strategies = list(learner.strategies.keys())
            msg = "Due to the level of complexity of {learnerName}'s default "
            msg += "arguments, nimble requires the 'strategy' argument. "
            msg += "The options for strategy are {strategies}"
            msg = msg.format(learnerName=learnerName, strategies=strategies)
            raise InvalidArgumentValue(msg)
        # for regressors, also need to check that MultipleImputer strategy is
        # defined if the user did not provide a MultipleImputer directly
        elif (isinstance(learner, self.autoimpute.analysis.MiBaseRegressor)
              and ('mi' not in initParams or initParams['mi'] is None)
              and ('mi_kwgs' not in initParams
                   or 'strategy' not in initParams['mi_kwgs'])):
            strategies = list(learner.mi.strategies.keys())
            msg = "Due to the level of complexity of {learnerName}'s "
            msg += "default arguments, nimble requires the MultipleImputer "
            msg += "strategy argument. This can be accomplished 1) by "
            msg += "providing a dictionary with a 'strategy' key as the "
            msg += "'mi_kwgs' argument or 2) providing a MultipleImputer "
            msg += "as the 'mi' argument using nimble.Init('MultipleImputer', "
            msg += "strategy=...). The options for strategy are {strategies}"
            msg = msg.format(learnerName=learnerName, strategies=strategies)
            raise InvalidArgumentValue(msg)

        return learner

    def _fitLearner(self, learner, learnerName, trainX, trainY, arguments):
        # autoimpute validation fails if all fit arguments are passed via
        # **kwargs, so need to pack required arguments separately
        fitNames = self._paramQuery('fit', learnerName, ['self'])
        allFitNames = fitNames[0]
        defaultIdx = len(allFitNames) - len(fitNames[1])
        fitArgs = []
        fitKwargs = {}
        for i, name in enumerate(allFitNames):
            if name.lower() == 'x':
                value = trainX
            elif name.lower() == 'y':
                value = trainY
            elif name in arguments:
                value = arguments[name]
            else:
                continue
            if i < defaultIdx:
                fitArgs.append(value)
            else:
                fitKwargs[name] = value

        try:
            learner.fit(*fitArgs, **fitKwargs)
        except ValueError as ve:
            # these occur when the learner requires different input data
            # (multi-dimensional, non-negative)
            raise InvalidArgumentValue(str(ve))

    def version(self):
        return self.autoimpute.__version__

    def _argumentInit(self, toInit):
        # override universal method to require strategy param for imputers
        initObj = super(Autoimpute, self)._argumentInit(toInit)
        if 'Imputer' in toInit.name and 'strategy' not in toInit.kwargs:
            strategies = list(initObj.strategies.keys())
            msg = "Due to the complexity of the default arguments, nimble "
            msg += "does not allow {name} to be instantiated without "
            msg += "specifying the 'strategy' keyword argument. The options "
            msg += "for strategy are {strategies}"
            msg = msg.format(name=toInit.name, strategies=strategies)
            raise InvalidArgumentValue(msg)
        return initObj
