
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
Relies on being keras 2.4 or greater (tf integration) but less
than 3.0 (return to multibackend)
"""

import os
import logging
import warnings
import inspect
import copy

import numpy as np
from packaging.version import parse

import nimble
from nimble._dependencies import checkVersion
from nimble._utility import inspectArguments, dtypeConvert
from nimble._utility import inheritDocstringsFactory
from nimble._utility import mergeArguments
from nimble.exceptions import InvalidArgumentValue
from .universal_interface import PredefinedInterfaceMixin
from ._interface_helpers import PythonSearcher
from ._interface_helpers import modifyImportPathAndImport
from ._interface_helpers import collectAttributes
from ._interface_helpers import removeFromTailMatchedLists
from ._interface_helpers import notCallable, notABCAssociated
from ._interface_helpers import checkArgsForRandomParam


LEARNERTYPES = {
    'classification': [
        'BinaryCrossentropy', 'binary_crossentropy',
        'BinaryFocalCrossentropy', 'binary_focal_crossentropy',
        'CategoricalCrossentropy', 'categorical_crossentropy',
        'CategoricalFocalCrossentropy', 'categorical_focal_crossentropy',
        'SparseCategoricalCrossentropy', 'sparse_categorical_crossentropy',
        'KLDivergence', 'kl_divergence',
        'Hinge', 'hinge',
        'CategoricalHinge', 'categorical_hinge',
        'SquaredHinge', 'squared_hinge',
        'CTC', 'ctc',
    ],
    'regression': [
        'MeanSquaredError', 'mean_squared_error',
        'MeanAbsoluteError', 'mean_absolute_error',
        'MeanAbsolutePercentageError', 'mean_absolute_percentage_error',
        'MeanSquaredLogarithmicError', 'mean_squared_logarithmic_error',
        'CosineSimilarity', 'cosine_similarity',
        'Huber', 'huber',
        'LogCosh', 'log_cosh',
        'Poisson', 'poisson',  # Seems to be for regression of 'counting' data?
    ]
}

# Hard coded params and defaults to be used for those versions of keras that
# can't report them
APPSBASELINEPARAMS = {
    "include_top": True,
    "weights": "imagenet",
    "input_tensor": None,
    "input_shape": None,
    "pooling": None,
    "classes": 1000,
    "classifier_activation":"softmax"
}

# Hard coded params and defaults beyond the baseline shared by all apps,
# to be used on top of APPSBASELINEPARAMS
APPSNAMESTOEXTRAPARAMS = {
    'MobileNet': {"alpha": 1.0,
                  "depth_multiplier": 1,
                  "dropout": 0.001},
    'MobileNetV2': {"alpha": 1.0},
    'EfficientNetV2B0': {"include_preprocessing":True},
    'EfficientNetV2B1': {"include_preprocessing":True},
    'EfficientNetV2B2': {"include_preprocessing":True},
    'EfficientNetV2B3': {"include_preprocessing":True},
    'EfficientNetV2S': {"include_preprocessing":True},
    'EfficientNetV2M': {"include_preprocessing":True},
    'EfficientNetV2L': {"include_preprocessing":True},

}


@inheritDocstringsFactory(PredefinedInterfaceMixin)
class Keras(PredefinedInterfaceMixin):
    """
    This class is an interface to keras.
    """
    def __init__(self):
        # tensorflow has a tremendous quantity of informational outputs
        # that drown out anything else on standard out
        logging.getLogger('tensorflow').disabled = True

        # we want to grab certain metadata direct from keras
        self.keras = modifyImportPathAndImport('keras', 'keras')
        # needed for certain configuration
        self.tensorflow = modifyImportPathAndImport('tensorflow', 'tensorflow')

        # this is the api we'll access, even through it is backed by
        # the keras package itself.
        self.package = self.keras

        # setup __all__ so that learner and object searching can proceed as
        # anticipated, and we can more easily filter certain things.
        if not hasattr(self.package, '__all__'):
            self.package.__all__ = []

        # extend __all__ as needed
        names = os.listdir(self.package.__path__[0])
        possibilities = []
        for name in names:
            splitList = name.split('.')
            if len(splitList) == 1 or splitList[1] in ['py', 'pyc']:
                if (splitList[0] not in self.package.__all__
                        and not splitList[0].startswith('_')):
                    possibilities.append(splitList[0])

        possibilities = np.unique(possibilities).tolist()
        self.package.__all__.extend(possibilities)
        unwanted = ['utils', 'experimental']
        for subMod in unwanted:
            if subMod in self.package.__all__:
                self.package.__all__.remove(subMod)

        def isLearner(obj):
            """
            In Keras, there are two classes of learners: from scratch,
            and known structures from Keras Applications. From scratch
            must have the full pipeline of functions (so long as it
            isn't one of the excluded ones), whereas any of the loader
            functions in the keras.applications submodle are wanted.
            """
            # The basic Model object and the Functional object aren't allowed.
            # WideDeepModel and Linear model are currently experimental, and
            # it isn't clear how we might want to handle them.
            excluded = ["Model", "Functional", "SharpnessAwareMinimization",
                        "WideDeepModel", "LinearModel"]
            if obj.__name__ in excluded:
                return False

            hasFit = hasattr(obj, 'fit')
            hasPred = hasattr(obj, 'predict')
            hasCompile = hasattr(obj, 'compile')

            # This combined with the correct depth limit is sufficient to
            # grab the application loading functions.
            inApps = False
            if hasattr(obj, "__module__"):
                # the exact location changes depending on the version,
                # we just need one to be present to demonstrate it's
                # a learner.
                possible = [
                    "keras.applications",
                    "keras.src.applications"
                ]
                inApps = any(appLoc in obj.__module__ for appLoc in possible)

            if (hasFit and hasPred and hasCompile) or inApps:
                return True

            return False

        self._searcher = PythonSearcher(self.package, isLearner, 1)

        super().__init__()

    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
        try:
            _ = modifyImportPathAndImport('keras', 'keras')
        except ImportError:
            return False
        return True

    def _checkVersion(self):
        checkVersion(self.keras)

    def version(self):
        return self.keras.__version__

    @classmethod
    def getCanonicalName(cls):
        return 'keras'

    @classmethod
    def isAlias(cls, name):
        if name.lower() in ['tensorflow.keras', 'tf.keras']:
            return True
        return name.lower() == cls.getCanonicalName().lower()

    @classmethod
    def _installInstructions(cls):
        msg = """
To install keras
----------------
    The latest versions of keras come packaged with tensorflow as
    tensorflow.keras. Tensorflow install instructions can be found at:
    https://www.tensorflow.org/install
    And the latest keras documentation is available at:
    https://keras.io/

    Nimble supports older (multi-backend) versions of keras as well, but these
    will not continue to be supported so installing them is NOT recommended.
    More information regarding these changes is available at:
    https://github.com/keras-team/keras/releases
    """
        return msg

    def _learnerNamesBackend(self, onlyTrained=False):
        possibilities = self._searcher.allLearners()
        # mismatch between weights='imagenet' and network structure, so we
        # disable in all cases
        exclude = ["NASNetMobile", "NASNetLarge",]
        if onlyTrained:
            exclude.append("Sequential")
        ret = []
        for name in possibilities:
            if not name in exclude:
                ret.append(name)

        return ret

    def learnerType(self, name): # pylint: disable=unused-argument
        """
        Keras learner types cannot be defined until compiled.
        """
        return 'undefined'

    def _learnerType(self, learnerBackend):
        for lType, losses in LEARNERTYPES.items():
            if not hasattr(learnerBackend, 'loss'):
                return 'UNKNOWN'
            loss = getattr(learnerBackend, 'loss')
            if hasattr(loss, 'name'):
                if loss.name == "LossFunctionWrapper":
                    loss = loss.fn.__name__
                loss = loss.name
            if loss in losses:
                return lType
        return 'UNKNOWN'

    def _findCallableBackend(self, name):
        return self._searcher.findInPackage(None, name)

    def _getParameterNamesBackend(self, name):
        ret = self._paramQuery(name, None, ignore=['self', 'seed'])
        if ret is None:
            return ret
        (objArgs, _, _, _) = ret
        return [objArgs]

    def _getLearnerParameterNamesBackend(self, learnerName):
        ignore = ['self', 'X', 'x', 'Y', 'y']
        isApp = (learnerName != "Sequential")
        # Checking object init or keras apps loader func respectively
        start = self._paramQuery('__init__', learnerName, ignore)
        if isApp:
            # want info on the function itself
            start = self._paramQuery(learnerName, None, ignore)

        # All models share the same compile / fit /predict API
        compile_ = self._paramQuery('compile', "Model", ignore)
        fit = self._paramQuery('fit', "Model", ignore)
        predict = self._paramQuery('predict', "Model", ignore)

        ret = start[0] + compile_[0] + fit[0] + predict[0]

        return [ret]

    def _getDefaultValuesBackend(self, name):
        ret = self._paramQuery(name, None, ignore=['seed'])
        if ret is None:
            return ret
        (objArgs, _, _, d) = ret
        ret = {}
        if d is not None:
            for i in range(len(d)):
                ret[objArgs[-(i + 1)]] = d[-(i + 1)]

        return [ret]

    def _getLearnerDefaultValuesBackend(self, learnerName):
        ignore = ['self', 'X', 'x', 'Y', 'y']
        isApp = (learnerName != "Sequential")
        # Checking object init or keras apps loader func respectively
        start = self._paramQuery('__init__', learnerName, ignore)
        if isApp:
            # want info on the function itself
            start = self._paramQuery(learnerName, None, ignore)

        # All models share the same compile / fit /predict API
        compile_ = self._paramQuery('compile', "Model", ignore)
        fit = self._paramQuery('fit', "Model", ignore)
        predict = self._paramQuery('predict', "Model", ignore)

        toProcess = [start, compile_, fit, predict]

        ret = {}
        for stage in [stg for stg in toProcess if stg is not None]:
            currNames = stage[0]
            currDefaults = stage[3]
            if stage[3] is not None:
                for i in range(len(currDefaults)):
                    key = currNames[-(i + 1)]
                    value = currDefaults[-(i + 1)]
                    ret[key] = value

        return [ret]

    def _getScores(self, learnerName, learner, testX, newArguments,
                   storedArguments, customDict):
        if self._learnerType(learner) == 'classification':
            return self._applyBackend(learnerName, learner, testX,
                                      newArguments, storedArguments,
                                      customDict)

        raise NotImplementedError('Cannot get scores for this learner')


    def _getScoresOrder(self, learner):
        return learner.UIgetScoreOrder

    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             randomSeed, arguments, customDict):
        if trainX is not None:
            dataType = trainX.getTypeString()
            if dataType == 'Sparse':
                trainX = trainX.copy(to="scipycsr")
            else:
                trainX = trainX.copy(to='numpy array')
                trainX = dtypeConvert(trainX)
        if trainY is not None:
            if len(trainY.features) > 1:
                trainY = (trainY.copy(to='numpy array'))
            else:
                trainY = trainY.copy(to='numpy array', outputAs1D=True)
            trainY = dtypeConvert(trainY)

        if testX is not None:
            dataType = testX.getTypeString()
            if dataType == 'Sparse':
                testX = testX.copy(to="scipycsr")
            else:
                testX = testX.copy(to='numpy array')
                testX = dtypeConvert(testX)

        instantiatedArgs = {}
        for arg, val in arguments.items():
            if isinstance(val, nimble.Init):
                val = self._seedingArgumentInit(val, randomSeed)
            elif not isinstance(val, str) and (hasattr(val, '__iter__') or
                                               hasattr(val, '__getitem__')):
                val = copy.copy(val)
                for i, v in enumerate(val):
                    if isinstance(v, nimble.Init):
                        val[i] = self._seedingArgumentInit(v, randomSeed)
                    elif isinstance(v, nimble.core.data.Base):
                        val[i] = v.copy('numpy array')
            instantiatedArgs[arg] = val

        isApp = (learnerName != "Sequential")
        if isApp:
            learnerModule = self._findAppsSubModule(learnerName)

            preprocessor = learnerModule.preprocess_input
            if trainX is not None:
                trainX = preprocessor(trainX)
            if testX is not None:
                testX = preprocessor(testX)

        return (trainX, trainY, testX, instantiatedArgs)


    def _outputTransformation(self, learnerName, outputValue,
                              transformedInputs, outputType, outputFormat,
                              customDict):
        # In the case of prediction we are given a row vector,
        # yet we want a column vector
        if outputFormat == "label" and len(outputValue.shape) == 1:
            outputValue = outputValue.reshape(len(outputValue), 1)

        # In the case of predicting with one of the pretrained on imagenet
        # learners, the outputs are the scores for all 1000 categories.
        # We use decode outputs to pick the highest confidence "label"
        isPreTrained = learnerName != "Sequential"
        if outputFormat == "label" and isPreTrained:
            learnerModule = self._findAppsSubModule(learnerName)

            decoder = learnerModule.decode_predictions
            outputValue = decoder(outputValue, top=1)
            # Output now in form of a list of lists of (ID, plain-text-name, score).
            # We're defining the ID as the label, and redefining the result as a list
            # of lists to match the shape of a feature.
            for i in range(len(outputValue)):
                outputValue[i] = [outputValue[i][0][0]]

        outputType = None
        if outputType == 'match':
            outputType = customDict['match']
        return nimble.data(outputValue, returnType=outputType, useLog=False)


    def _setRandomness(self, arguments, randomSeed):
        checkArgsForRandomParam(arguments, 'seed')
        self.tensorflow.random.set_seed(randomSeed)


    def _trainer(self, learnerName, trainX, trainY, arguments, randomSeed,
                 customDict):
        initNames = self._paramQuery('__init__', learnerName, ['self'])[0]
        compileNames = self._paramQuery('compile', "Model", ['self'])[0]

        self._setRandomness(arguments, randomSeed)
        fitNames = self._paramQuery('fit', "Model", ['self'])[0]

        # pack parameter sets
        initParams = {name: arguments[name] for name in initNames
                      if name in arguments}
        learner = self.findCallable(learnerName)(**initParams)

        compileParams = {name: arguments[name] for name in compileNames
                         if name in arguments}
        learner.compile(**compileParams)

        fitParams = {}
        for name in fitNames:
            if name.lower() in ['x']:
                value = trainX
            elif name.lower() == 'y':
                value = trainY
            elif name in arguments:
                value = arguments[name]
            else:
                continue
            fitParams[name] = value

        learner.fit(**fitParams)

        if self._learnerType(learner) == 'classification':
            learner.UIgetScoreOrder = np.unique(trainY)

        return learner


    def _incrementalTrainer(self, learnerName, learner, trainX, trainY,
                            arguments, randomSeed, customDict):
        self._setRandomness(arguments, randomSeed)
        param = 'train_on_batch'
        trainOnBatchNames = self._paramQuery(param, learnerName, ['self'])[0]
        trainOnBatchParams = {}
        for name in trainOnBatchNames:
            value = None
            if name.lower() == 'x':
                value = trainX
            elif name.lower() == 'y':
                value = trainY
            elif name in arguments:
                value = arguments[name]
            if value is not None:
                trainOnBatchParams[name] = value
        learner.train_on_batch(**trainOnBatchParams)
        return learner

    def _applyBackend(self, learnerName, learner, testX, newArguments,
                      storedArguments, customDict):
        if not hasattr(learner, 'predict'):
            msg = f"Cannot apply {learnerName} to data, no predict function"
            raise TypeError(msg)

        ignore = ['X', 'x', 'self']
        backendArgs = self._paramQuery('predict', "Model", ignore)[0]

        applyArgs = self._getMethodArguments(backendArgs, newArguments,
                                             storedArguments)
        return self._predict(learner, testX, applyArgs, customDict)

    def _applier(self, learnerName, learner, testX, newArguments,
                 storedArguments, customDict):
        ret = self._applyBackend(learnerName, learner, testX, newArguments,
                                 storedArguments, customDict)

        # for classification, convert to labels
        if self._learnerType(learner) == 'classification':
            # This indicates we're in the distribution case; argmax will
            # grab the right category regardless of encoding via probability
            # or logits
            if len(ret) > 1 and len(ret[0]) > 1:
                ret = np.argmax(ret, axis=1)
            # means we must be in the single value case and have to distinguish
            # between logits and probability
            else:
                lrnLoss = learner.loss
                if hasattr(lrnLoss, "from_logits") and lrnLoss.from_logits:
                    ret = ret >= 0
                else:
                    ret = ret >= 0.5
        return ret


    def _getAttributes(self, learnerBackend):
        obj = learnerBackend
        # the loss attribute of learners, when an object, was being excluded
        # since it was callable
        def notCallableExceptLoss(obj, name, val):
            if name == "loss":
                return True
            return notCallable(obj, name, val)

        checkers = [notCallableExceptLoss, notABCAssociated]

        def wrappedDir(obj):
            ret = {}
            keys = dir(obj)
            func = lambda n: not n.startswith("_") and n != "submodules"
            acceptedKeys = filter(func, keys)
            for k in acceptedKeys:
                try:
                    val = getattr(obj, k)
                    ret[k] = val
                # safety against any sort of error someone may have in their
                # property code.
                except (AttributeError, ValueError):
                    pass
            return ret

        ret = collectAttributes(obj, [wrappedDir], checkers)
        return ret

    @classmethod
    def _optionDefaults(cls, option):
        return None

    @classmethod
    def _configurableOptionNames(cls):
        return ['location']

    def _predict(self, learner, testX, arguments, _):
        """
        Wrapper for the underlying predict function of a keras learner
        object.
        """
        return learner.predict(testX, **arguments)

    def _loadTrainedLearnerBackend(self, learnerName, arguments):
        loadParams = self._paramQuery(learnerName, None, 'self')[0]

        loadArgs = {name: arguments[name] for name in loadParams
                         if name in arguments}

        toInit = self._findCallableBackend(learnerName)
        return toInit(**loadArgs)

    ###############
    ### HELPERS ###
    ###############

    def _seedingArgumentInit(self, toInit, seed):
        """
        Precurser to the _argumentInit method to handle keras specific objects
        that require random seeding.
        """
        unpack = self._paramQuery('__init__', toInit.name, ['self'])
        params, _, _, defaults = unpack

        # We only need to wrangle with those arguments that represent
        # initializers.
        initializers = filter(lambda x: 'initializer' in x, params)

        # Assumption: all initializer params will have non-None defaults
        for name in initializers:
            lizerIdx = params.index(name)
            lizerNegIdx = lizerIdx - len(params)
            defaultVal = defaults[lizerNegIdx]

            present = name in toInit.kwargs
            # user is specifying a initializer with nimble.Init
            if present and isinstance(toInit.kwargs[name], nimble.Init):
                lizerInit = toInit.kwargs[name]
                lizerName = lizerInit.name
                lizerArgs = lizerInit.kwargs.copy()
                # verify arguments before we subsequently init the object
                self._validateArgumentValues(lizerName, lizerArgs)
            # user is specifying a initializer by string
            elif present and isinstance(toInit.kwargs[name], str):
                lizerName = toInit.kwargs[name]
                lizerArgs = {}
            # nothing provided, use default value
            else:
                lizerName = defaultVal
                lizerArgs = {}

            # not all initializers take seeds, only take action if they do
            checkNames = self._paramQuery('__init__', lizerName,['self'])[0]
            if 'seed' in checkNames:
                lizerArgs['seed'] = seed
                initObject = self.findCallable(lizerName)
                if initObject is None:
                    msg = f'Unable to locate "{lizerName}" in this interface'
                    raise InvalidArgumentValue(msg)
                lizerObj = initObject(**lizerArgs)

                toInit.kwargs[name] = lizerObj

        # Sub-object seeding handled if needed, can proceed with the standard
        # recursive helper
        return self._argumentInit(toInit)


    def _paramQuery(self, name, parent, ignore=None):
        """
        Takes the name of some keras learn object or function, returns
        a list of parameters used to instantiate that object or run that
        function, or None if the desired thing cannot be found.
        """
        if ignore is None:
            ignore = []

        found = self._searcher.findInPackage(parent, name)

        if found is None:
            return None

        class InheritedEmptyInit(object):
            """
            Class with an empty __init__ (no parameters)
            """

        if isinstance(found,
                      type(getattr(InheritedEmptyInit, '__init__'))):
            return ([], None, None, None)

        # Keras uses metaclassing and __new__ extensively to select backends
        # for objects, and Signature targets __new__ over __init__. As such,
        # for all objects we explicitly target __init__ to get the allowed
        # parameters.
        if inspect.isclass(found):
            found = found.__init__

        try:
            (args, v, k, d) = inspectArguments(found)
            args, d = removeFromTailMatchedLists(args, d, ignore)
            ret = (args, v, k, d)
        except TypeError:
            ret = None

        if name in self._learnerNamesBackend(True) and not ret[0]:
            ret = self._appsParamQuery(name)
        return ret

    def _findAppsSubModule(self, learnerName):
        learnerModuleName = self.findCallable(learnerName).__module__.split('.')[-1]
        if "applications" not in learnerModuleName:
            return getattr(self.package.applications, learnerModuleName)

        for entry in dir(self.package.applications):
            temp = getattr(self.package.applications, entry)
            if learnerName in dir(temp):
                return temp

        return None

    def _appsParamQuery(self, name):
        extra = {}
        if name in APPSNAMESTOEXTRAPARAMS:
            extra = APPSNAMESTOEXTRAPARAMS[name]

        merged = mergeArguments(APPSBASELINEPARAMS, extra)

        a = []
        d = []
        for k,v in merged.items():
            a.append(k)
            d.append(v)

        return [a, None, None, d]
