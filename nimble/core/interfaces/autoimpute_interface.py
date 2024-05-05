
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Interface to autoimpute package.
"""

# pylint: disable=unused-argument

import types
import logging
import importlib
import numpy as np
from packaging.version import Version

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble._utility import inheritDocstringsFactory
from .scikit_learn_interface import _SciKitLearnAPI
from ._interface_helpers import modifyImportPathAndImport
from ._interface_helpers import PythonSearcher
from ._interface_helpers import validInitParams

logging.getLogger('theano.configdefaults').setLevel(logging.ERROR)


@inheritDocstringsFactory(_SciKitLearnAPI)
class Autoimpute(_SciKitLearnAPI):
    """
    This class is an interface to autoimpute.

    Autoimpute follows the sci-kit learn api so this class inherits from
    our SciKitLearn interface so we do not need to define every method.
    """

    def __init__(self):
        """

        """
        self.package = modifyImportPathAndImport('autoimpute', 'autoimpute')

        # Need to import the analysis module so that it's readily available
        # in the other methods. BUT autoimpute experiences an import failure
        # when scipy>=1.8.0 so we patch the problem attribute with its new
        # location.
        if nimble._utility.scipy.nimbleAccessible():
            verStr = nimble._utility.scipy.__version__
            ver = Version(verStr)
            if ver >= Version("1.8.0"):
                sigTools = importlib.import_module("scipy.signal.signaltools")
                _sigTools = importlib.import_module("scipy.signal._signaltools")
                sigTools._centered = _sigTools._centered
                # import the submodule that accesses the changed attribute
                importlib.import_module(".analysis", "autoimpute")

        def isLearner(obj):
            try:
                # if object cannot be instantiated without additional
                # arguments, we cannot support it at this time
                _ = obj()
            except TypeError:
                return False
            # only support learners with a predict, transform,
            # fit_predict or fit_transform, all have fit attribute
            hasPred = hasattr(obj, 'predict')
            hasTrans = hasattr(obj, 'transform')
            hasFitPred = hasattr(obj, 'fit_predict')
            hasFitTrans = hasattr(obj, 'fit_transform')

            return hasPred or hasTrans or hasFitPred or hasFitTrans

        # Issue relating to dependency pymc3 (< v3.1.1.5) relying on dead
        # project theano, which incorrectly accesses an attribute in numpy
        # for >= v1.22. Since it causes an attribute error during import,
        # we perform a patch to present the info irrespective of the
        # potential pymc3 version.
        npVer = Version(np.__version__)
        if Version("1.22") <= npVer and npVer < Version("1.26"):
            from numpy.distutils import system_info as npdusi
            patchValue = npdusi.get_info("blas_opt", 0)
            np.distutils.__config__.blas_opt_info = patchValue

        self._searcher = PythonSearcher(self.package, isLearner, 1)

        super().__init__('seed')

    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
        try:
            _ = modifyImportPathAndImport('autoimpute', 'autoimpute')
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

    def _learnerNamesBackend(self, onlyTrained=False):
        if onlyTrained:
            return []

        return self._searcher.allLearners()

    def _learnerType(self, learnerBackend):
        if isinstance(learnerBackend, self.package.imputations.BaseImputer):
            return 'transformation'
        if isinstance(learnerBackend, self.package.analysis.MiBaseRegressor):
            if learnerBackend.__class__.__name__ == 'LogisticRegression':
                return 'classification'
            return 'regression'
        return "UNKNOWN"
        # TODO MissingnessClassifier can be identified as a classifier using
        # super method, but its scoring method is currently failing tests if
        # categorized as classification
        # return super()._learnerType(learnerBackend)

    def _findCallableBackend(self, name):
        return self._searcher.findInPackage(None, name)

    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             randomSeed, arguments, customDict):

        def dtypeConvert(dataframe):
            for idx, ser in dataframe.items():
                try:
                    dataframe.loc[:, idx] = ser.astype(float)
                except ValueError:
                    pass

        if trainX is not None:
            customDict['match'] = trainX.getTypeString()
            trainX = trainX.copy(to='pandasdataframe')
            dtypeConvert(trainX)

        if trainY is not None:
            trainY = trainY.copy(to='pandasdataframe')
            dtypeConvert(trainY)
            # trainY columns cannot be same as trainX in autoimpute
            if (trainX is not None
                    and any(col in trainX.columns for col in trainY.columns)):
                yCols = [str(col) + '_trainY_' for col in trainY.columns]
                trainY.columns = yCols

        if testX is not None:
            testX = testX.copy(to='pandasdataframe')
            dtypeConvert(testX)

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
        if isinstance(outputValue, (list, types.GeneratorType)):
            finalDF = None
            iMax = 0
            for i, dataframe in outputValue:
                if finalDF is None:
                    finalDF = dataframe
                else:
                    finalDF += dataframe
                iMax = max(iMax, i)
            outputValue = finalDF / iMax

        # In the case of prediction we are given a row vector,
        # yet we want a column vector
        if outputFormat == "label" and len(outputValue.shape) == 1:
            outputValue = outputValue.reshape(len(outputValue), 1)

        # TODO correct
        outputType = 'Matrix'
        if outputType == 'match':
            outputType = customDict['match']
        return nimble.data(outputValue, returnType=outputType, useLog=False)

    def _initLearner(self, learnerName, trainX, trainY, arguments, randomSeed):
        initNames = self._paramQuery('__init__', learnerName, ['self'])[0]
        initParams = validInitParams(initNames, arguments, randomSeed,
                                     self.randomParam)
        learner = self.findCallable(learnerName)(**initParams)

        # need to enforce strategy as a required parameter for the imputer
        if (isinstance(learner, self.package.imputations.BaseImputer)
                and 'strategy' not in initParams):
            strategies = list(learner.strategies.keys())
            msg = f"Due to the level of complexity of {learnerName}'s default "
            msg += "arguments, nimble requires the 'strategy' argument. "
            msg += f"The options for strategy are {strategies}"
            raise InvalidArgumentValue(msg)
        # for regressors, also need to check that MultipleImputer strategy is
        # defined if the user did not provide a MultipleImputer directly
        if (isinstance(learner, self.package.analysis.MiBaseRegressor)
                and ('mi' not in initParams or initParams['mi'] is None)
                and ('mi_kwgs' not in initParams
                     or 'strategy' not in initParams['mi_kwgs'])):
            strategies = list(learner.mi.strategies.keys())
            msg = f"Due to the level of complexity of {learnerName}'s "
            msg += "default arguments, nimble requires the MultipleImputer "
            msg += "strategy argument. This can be accomplished 1) by "
            msg += "providing a dictionary with a 'strategy' key as the "
            msg += "'mi_kwgs' argument or 2) providing a MultipleImputer "
            msg += "as the 'mi' argument using nimble.Init('MultipleImputer', "
            msg += f"strategy=...). The options for strategy are {strategies}"
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
        except ValueError as e:
            raise InvalidArgumentValue(str(e)) from e

    def _argumentInit(self, toInit):
        # override universal method to require strategy param for imputers
        initObj = super()._argumentInit(toInit)
        if 'Imputer' in toInit.name and 'strategy' not in toInit.kwargs:
            strategies = list(initObj.strategies.keys())
            msg = "Due to the complexity of the default arguments, nimble "
            msg += f"does not allow {toInit.name} to be instantiated without "
            msg += "specifying the 'strategy' keyword argument. The options "
            msg += f"for strategy are {strategies}"
            raise InvalidArgumentValue(msg)
        return initObj
