"""
Relies on being keras 2.0.8
"""

import os
import logging
import warnings

import numpy

import nimble
from nimble._utility import inspectArguments
from nimble._utility import inheritDocstringsFactory, numpy2DArray
from nimble.exceptions import InvalidArgumentValue, _prettyListString
from .universal_interface import UniversalInterface
from .universal_interface import PredefinedInterface
from ._interface_helpers import PythonSearcher
from ._interface_helpers import modifyImportPathAndImport
from ._interface_helpers import collectAttributes
from ._interface_helpers import removeFromTailMatchedLists
from ._interface_helpers import noLeading__, notCallable, notABCAssociated


class _TensorFlowRandom:
    """
    Wrap tensorflow's randomness module to make adjustments that allow
    it to work with nimble randomness control.
    """
    def __init__(self, tensorflowRandom):
        self.tensorflowRandom = tensorflowRandom
        self.setSeed(42)
        self.state = self.tensorflowRandom.create_rng_state(42, 1)

    def setSeed(self, seed):
        if seed is None:
            seed = int(numpy.random.randint(2 ** 32))
        self.tensorflowRandom.set_seed(seed)
        self.state = self.tensorflowRandom.create_rng_state(seed, 1)

    def getState(self):
        return self.state

    def setState(self, state):
        self.setSeed(int(state[0]))


@inheritDocstringsFactory(UniversalInterface)
class Keras(PredefinedInterface, UniversalInterface):
    """
    This class is an interface to keras.
    """
    def __init__(self):
        # modify path if another directory provided
        try:
            # keras recommends using tensorflow.keras when possible
            # need to set tensorflow random seed before importing keras
            tensorflow = modifyImportPathAndImport('tensorflow', 'tensorflow')
            # tensorflow has a tremendous quantity of informational outputs
            # that drown out anything else on standard out
            logging.getLogger('tensorflow').disabled = True
            # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            if tensorflow.__version__[:2] == '1.':
                msg = "Randomness is outside of Nimble's control for version "
                msg += "1 of Tensorflow. Reproducible results cannot be "
                msg += "guaranteed"
                warnings.warn(msg, UserWarning)
            else:
                randomObj = _TensorFlowRandom(tensorflow.random)
                randomInfo = {'state': None,
                              'methods': ('setSeed', 'getState', 'setState')}
                nimble.random._saved[randomObj] =  randomInfo

            self.keras = modifyImportPathAndImport('keras', 'tensorflow.keras')
        except ImportError:
            self.keras = modifyImportPathAndImport('keras', 'keras')

        # keras 2.0.8 has no __all__
        names = os.listdir(self.keras.__path__[0])
        possibilities = []
        if not hasattr(self.keras, '__all__'):
            self.keras.__all__ = []
        for name in names:
            splitList = name.split('.')
            if len(splitList) == 1 or splitList[1] in ['py', 'pyc']:
                if (splitList[0] not in self.keras.__all__
                        and not splitList[0].startswith('_')):
                    possibilities.append(splitList[0])

        possibilities = numpy.unique(possibilities).tolist()
        if 'utils' in possibilities:
            possibilities.remove('utils')
        self.keras.__all__.extend(possibilities)

        def isLearner(obj):
            """
            In Keras, there are 2 learners: Sequential and Model.
            """
            hasFit = hasattr(obj, 'fit')
            hasPred = hasattr(obj, 'predict')
            hasCompile = hasattr(obj, 'compile')

            if not (hasFit and hasPred and hasCompile):
                return False

            return True

        self._searcher = PythonSearcher(self.keras, self.keras.__all__, {},
                                        isLearner, 2)

        super(Keras, self).__init__()

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

    def _listLearnersBackend(self):
        possibilities = self._searcher.allLearners()
        exclude = []
        ret = []
        for name in possibilities:
            if not name in exclude:
                ret.append(name)

        return ret

    def learnerType(self, name):
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
        ignore = ['self', 'X', 'x', 'Y', 'y', 'obs', 'T']
        init = self._paramQuery('__init__', learnerName, ignore)
        fit = self._paramQuery('fit', learnerName, ignore)
        fitGenerator = self._paramQuery('fit_generator', learnerName, ignore)
        predict = self._paramQuery('predict', learnerName, ignore)
        compile_ = self._paramQuery('compile', learnerName, ignore)

        ret = init[0] + fit[0] + predict[0]
        if fitGenerator is not None:
            ret += fitGenerator[0]
        if compile_ is not None:
            ret += compile_[0]
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
        ignore = ['self', 'X', 'x', 'Y', 'y', 'T']
        init = self._paramQuery('__init__', learnerName, ignore)
        fit = self._paramQuery('fit', learnerName, ignore)
        fitGenerator = self._paramQuery('fit_generator', learnerName, ignore)
        predict = self._paramQuery('predict', learnerName, ignore)
        compile_ = self._paramQuery('compile', learnerName, ignore)

        toProcess = [init, fit, fitGenerator, compile_, predict]

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
        if hasattr(learner, 'decision_function'):
            method = 'decision_function'
            toCall = learner.decision_function
        elif hasattr(learner, 'predict_proba'):
            method = 'predict_proba'
            toCall = learner.predict_proba
        else:
            raise NotImplementedError('Cannot get scores for this learner')
        ignore = ['X', 'x', 'self']
        backendArgs = self._paramQuery(method, learnerName, ignore)[0]
        scoreArgs = self._getMethodArguments(backendArgs, newArguments,
                                             storedArguments)
        raw = toCall(testX, **scoreArgs)
        # in binary classification, we return a row vector. need to reshape
        if len(raw.shape) == 1:
            return raw.reshape(len(raw), 1)
        else:
            return raw


    def _getScoresOrder(self, learner):
        return learner.UIgetScoreOrder

    def _validateFitArguments(self, dataType, learnerName, arguments):
        fitArgs = self._paramQuery('fit', learnerName)[0]
        fitGenArgs = self._paramQuery('fit_generator', learnerName)[0]
        if fitGenArgs is not None and dataType == 'Sparse':
            useArgs = fitGenArgs
            ignoreArgs = fitArgs
        elif dataType == 'Sparse':
            useArgs = fitArgs
            ignoreArgs = []
        else:
            # when not Sparse, ignore fit args that only apply to generators
            genArgs = ['max_queue_size', 'workers', 'use_multiprocessing']
            useArgs = [arg for arg in fitArgs if arg not in genArgs]
            ignoreArgs = fitGenArgs if fitGenArgs is not None else []

        invalid = frozenset(ignoreArgs) - frozenset(useArgs)
        extra = []
        for arg in invalid:
            if arg in arguments:
                extra.append(arg)
        if extra:
            msg = "EXTRA LEARNER PARAMETER! "
            msg += "When trying to validate arguments for "
            msg += learnerName + ", the following list of parameter "
            msg += "names were not matched: "
            msg += _prettyListString(extra, useAnd=True)
            msg += ". Those parameters are only suitable for "
            if dataType == 'Sparse':
                msg += "dense types (List, Matrix, DataFrame) and this object "
                msg += "is Sparse"
            else:
                msg += "Sparse objects and this a " + dataType
            raise InvalidArgumentValue(msg)

    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             arguments, customDict):

        if trainX is not None:
            dataType = trainX.getTypeString()
            self._validateFitArguments(dataType, learnerName, arguments)
            if dataType != 'Sparse':
            #for sparse cases, keep it untouched here.
                trainX = trainX.copy(to='numpy array')
        if trainY is not None:
            if len(trainY.features) > 1:
                trainY = (trainY.copy(to='numpy array'))
            else:
                trainY = trainY.copy(to='numpy array', outputAs1D=True)

        if testX is not None:
            if testX.getTypeString() != 'Sparse':
            #for sparse cases, keep it untouched here.
                testX = testX.copy(to='numpy array')

        instantiatedArgs = {}
        for arg, val in arguments.items():
            if arg == 'layers':
                for i, v in enumerate(val):
                    if isinstance(v, nimble.Init):
                        val[i] = self.findCallable(v.name)(**v.kwargs)
            elif isinstance(val, nimble.Init):
                val = self._argumentInit(val)
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
        return nimble.data(outputType, outputValue, useLog=False)


    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        initNames = self._paramQuery('__init__', learnerName, ['self'])[0]
        compileNames = self._paramQuery('compile', learnerName, ['self'])[0]
        isSparse = isinstance(trainX, nimble.core.data.Sparse)
        # keras 2.2.5+ fit_generator functionality merged into fit.
        # fit_generator may be removed, but will be used when possible
        # to support earlier versions.
        if isSparse:
            param = 'fit_generator'
        else:
            param = 'fit'
        fitNames = self._paramQuery(param, learnerName, ['self'])[0]
        if fitNames is None: # fit_generator has been removed
            param = 'fit'
            fitNames = self._paramQuery('fit', learnerName, ['self'])[0]

        # pack parameter sets
        initParams = {name: arguments[name] for name in initNames
                      if name in arguments}
        learner = self.findCallable(learnerName)(**initParams)

        compileParams = {name: arguments[name] for name in compileNames
                         if name in arguments}
        learner.compile(**compileParams)

        def sparseGenerator():
            while True:
                for i in range(len(trainX.points)):
                    tmpData = (trainX.pointView(i).copy(to='numpyarray'),
                               numpy2DArray(trainY[i]))
                    yield tmpData

        fitForGenerator = param == 'fit' and isSparse
        fitParams = {}
        for name in fitNames:
            if name.lower() in ['x', 'obs'] and fitForGenerator:
                value = sparseGenerator()
            elif name.lower() in ['x', 'obs']:
                value = trainX
            elif name.lower() == 'y' and fitForGenerator:
                continue # y not allowed when using fit with generator
            elif name.lower() == 'y':
                value = trainY
            elif name in arguments:
                value = arguments[name]
            else:
                continue
            fitParams[name] = value

        if param == 'fit_generator':
            fitParams['generator'] = sparseGenerator()
            learner.fit_generator(**fitParams)
        else:
            learner.fit(**fitParams)

        if (hasattr(learner, 'decision_function')
                or hasattr(learner, 'predict_proba')):
            if trainY is not None:
                labelOrder = numpy.unique(trainY)
            else:
                allLabels = learner.predict(trainX)
                labelOrder = numpy.unique(allLabels)

            learner.UIgetScoreOrder = labelOrder

        return learner


    def _incrementalTrainer(self, learnerName, learner, trainX, trainY,
                            arguments, customDict):
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


    def _applier(self, learnerName, learner, testX, newArguments,
                 storedArguments, customDict):
        if not hasattr(learner, 'predict'):
            msg = "Cannot apply this learner to data, no predict function"
            raise TypeError(msg)
        # keras 2.2.5+ predict_generator functionality merged into predict.
        # predict_generator may be removed but will be used when possible
        # to support earlier versions.
        if isinstance(testX, nimble.core.data.Sparse):
            method = 'predict_generator'
        else:
            method = 'predict'
        ignore = ['X', 'x', 'self']
        backendArgs = self._paramQuery(method, learnerName, ignore)[0]
        if backendArgs is None: # predict_generator has been removed
            method = 'predict'
            backendArgs = self._paramQuery(method, learnerName, ignore)[0]
        customDict['predictMethod'] = method
        applyArgs = self._getMethodArguments(backendArgs, newArguments,
                                             storedArguments)
        return self._predict(learner, testX, applyArgs, customDict)


    def _getAttributes(self, learnerBackend):
        obj = learnerBackend
        generators = None
        checkers = []
        checkers.append(noLeading__)
        checkers.append(notCallable)
        checkers.append(notABCAssociated)

        ret = collectAttributes(obj, generators, checkers)
        return ret

    def _optionDefaults(self, option):
        return None

    def _configurableOptionNames(self):
        return ['location']


    def _exposedFunctions(self):
        return [self._predict]


    def version(self):
        return self.keras.__version__


    def _predict(self, learner, testX, arguments, customDict):
        """
        Wrapper for the underlying predict function of a keras learner
        object.
        """
        def sparseGenerator():
            while True:
                for i in range(len(testX.points)):
                    tmpData = testX.pointView(i).copy(to='numpy array')
                    yield tmpData

        predGenerator = customDict['predictMethod'] == 'predict_generator'
        isSparse = isinstance(testX, nimble.core.data.Sparse)
        if predGenerator and isSparse:
            arguments['generator'] = sparseGenerator()
            return learner.predict_generator(**arguments)
        elif isSparse:
            return learner.predict(sparseGenerator(), **arguments)
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
        if name == 'fit_generator':
            return (['steps_per_epoch', 'epochs', 'verbose', 'callbacks',
                     'validation_data', 'validation_steps', 'class_weight',
                     'max_queue_size', 'workers', 'use_multiprocessing',
                     'initial_epoch'],
                    'args', 'kwargs',
                    [None, 1, 1, None, None, None, None, 10, 1, False, 0])
        elif name == 'predict_generator':
            return (['steps', 'max_queue_size', 'workers',
                     'use_multiprocessing', 'verbose'],
                    'args', 'kwargs', [None, 10, 1, False, 0])

        namedModule = self._searcher.findInPackage(parent, name)

        if namedModule is None:
            return None

        class InheritedEmptyInit(object):
            """
            Class with an empty __init__ (no parameters)
            """
            pass

        if isinstance(namedModule,
                      type(getattr(InheritedEmptyInit, '__init__'))):
            return ([], None, None, None)

        try:
            (args, v, k, d) = inspectArguments(namedModule)
            args, d = removeFromTailMatchedLists(args, d, ignore)
            return (args, v, k, d)
        except TypeError:
            return self._paramQueryHardCoded(name, parent, ignore)

    def _paramQueryHardCoded(self, name, parent, ignore):
        """
        Returns a list of parameters for in package entities that we
        have hard coded, under the assumption that it is difficult or
        impossible to find that data automatically.
        """
        return None
