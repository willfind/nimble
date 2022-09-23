"""
Relies on being keras 2.0.8
"""

import os
import logging
import warnings
import inspect
import copy

import numpy as np
from packaging.version import parse

import nimble
from nimble._utility import inspectArguments, dtypeConvert
from nimble._utility import inheritDocstringsFactory
from nimble._dependencies import checkVersion
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
        'SparseCategoricalCrossentropy', 'sparse_categorical_crossentropy',
        'KLDivergence', 'kl_divergence',
        'Hinge', 'hinge',
        'CategoricalHinge', 'categorical_hinge',
        'SquaredHinge', 'squared_hinge',
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


@inheritDocstringsFactory(PredefinedInterfaceMixin)
class Keras(PredefinedInterfaceMixin):
    """
    This class is an interface to keras.
    """
    def __init__(self):
        # modify path if another directory provided
        self._tfVersion2 = False
        try:
            # keras recommends using tensorflow.keras when possible
            # need to set tensorflow random seed before importing keras
            self.tensorflow = modifyImportPathAndImport('tensorflow',
                                                        'tensorflow')
            checkVersion(self.tensorflow)
            # tensorflow has a tremendous quantity of informational outputs
            # that drown out anything else on standard out
            logging.getLogger('tensorflow').disabled = True
            if parse(self.tensorflow.__version__) < parse('2'):
                msg = "Randomness is outside of Nimble's control for version "
                msg += "1.x of Tensorflow. Reproducible results cannot be "
                msg += "guaranteed"
                warnings.warn(msg, UserWarning)
            else:
                self._tfVersion2 = True
            self.package = modifyImportPathAndImport('keras',
                                                     'tensorflow.keras')

        except ImportError:
            self.package = modifyImportPathAndImport('keras', 'keras')


        # keras 2.0.8 has no __all__
        names = os.listdir(self.package.__path__[0])
        possibilities = []
        if not hasattr(self.package, '__all__'):
            self.package.__all__ = []
        for name in names:
            splitList = name.split('.')
            if len(splitList) == 1 or splitList[1] in ['py', 'pyc']:
                if (splitList[0] not in self.package.__all__
                        and not splitList[0].startswith('_')):
                    possibilities.append(splitList[0])

        possibilities = np.unique(possibilities).tolist()
        if 'utils' in possibilities:
            possibilities.remove('utils')
        self.package.__all__.extend(possibilities)

        def isLearner(obj):
            """
            In Keras, there are two classes of learners: from scratch,
            and known structures from Keras Applications. From scratch
            must have the full pipeline of functions (so long as it
            isn't one of the excluded ones), whereas any of the loader
            functions in the keras.applications submodle are wanted.
            """
            # The basic Model object isn't allowed. WideDeepModel is
            # currently experimental, and it isn't clear how we might
            # want to handle it.
            excluded = ["Model", "WideDeepModel"]
            if obj.__name__ in excluded:
                return False

            hasFit = hasattr(obj, 'fit')
            hasPred = hasattr(obj, 'predict')
            hasCompile = hasattr(obj, 'compile')

            # This combined with the correct depth limit is sufficient to
            # grab the application loading functions.
            inApps = "keras.applications" in obj.__module__

            if (hasFit and hasPred and hasCompile) or inApps:
                return True

            return False

        self._searcher = PythonSearcher(self.package, isLearner, 1)

        super().__init__()

    def _checkVersion(self):
        savedName = self.package.__name__
        if savedName != 'keras':
            try:
                self.package.__name__ = 'keras'
                super()._checkVersion()
            finally:
                self.package.__name__ = savedName
        else:
            super()._checkVersion()

    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
        try:
            try:
                _ = modifyImportPathAndImport('keras', 'tensorflow.keras')
            except ImportError:
                _ = modifyImportPathAndImport('keras', 'keras')
        except ImportError:
            return False
        return True

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
        exclude = []
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
        for lt, losses in LEARNERTYPES.items():
            attrs = self._getAttributes(learnerBackend)
            if 'loss' not in attrs:
                return 'UNKNOWN'
            loss = attrs['loss']
            if hasattr(loss, 'name'):
                loss = loss.name
            if loss in losses:
                return lt
        return 'UNKNOWN'

    def _findCallableBackend(self, name):
        return self._searcher.findInPackage(None, name)

    def _getParameterNamesBackend(self, name):
        ret = self._paramQuery(name, None, ignore=['self'])
        if ret is None:
            return ret
        (objArgs, _, _, _) = ret
        return [objArgs]

    def _getLearnerParameterNamesBackend(self, learnerName):
        ignore = ['self', 'X', 'x', 'Y', 'y']
        start = self._paramQuery('__init__', learnerName, ignore)

        # All models share the same compile / fit /predict API
        compile_ = self._paramQuery('compile', "Model", ignore)
        fit = self._paramQuery('fit', "Model", ignore)
        predict = self._paramQuery('predict', "Model", ignore)

        ret = start[0] + compile_[0] + fit[0] + predict[0]

        return [ret]

    def _getDefaultValuesBackend(self, name):
        ret = self._paramQuery(name, None)
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

        start = self._paramQuery('__init__', learnerName, ignore)

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
                             arguments, customDict):
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
                val = self._argumentInit(val)
            elif not isinstance(val, str) and (hasattr(val, '__iter__') or
                                               hasattr(val, '__getitem__')):
                val = copy.copy(val)
                try:
                    for i, v in enumerate(val):
                        if isinstance(v, nimble.Init):
                            val[i] = self._argumentInit(v)
                        elif isinstance(v, nimble.core.data.Base):
                            val[i] = v.copy('numpy array')
                except TypeError:
                    pass
            instantiatedArgs[arg] = val

        return (trainX, trainY, testX, instantiatedArgs)


    def _outputTransformation(self, learnerName, outputValue,
                              transformedInputs, outputType, outputFormat,
                              customDict):
        # In the case of prediction we are given a row vector,
        # yet we want a column vector
        if outputFormat == "label" and len(outputValue.shape) == 1:
            outputValue = outputValue.reshape(len(outputValue), 1)

        # TODO correct
        outputType = 'Matrix'
        if outputType == 'match':
            outputType = customDict['match']
        return nimble.data(outputValue, returnType=outputType, useLog=False)


    def _setRandomness(self, arguments, randomSeed):
        if self._tfVersion2:
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


    ###############
    ### HELPERS ###
    ###############

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
            return (args, v, k, d)
        except TypeError:
            return None

