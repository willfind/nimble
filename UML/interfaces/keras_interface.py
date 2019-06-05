"""
Relies on being keras 2.0.8
"""

from __future__ import absolute_import
import copy
import os
import sys
import logging

import numpy
from six.moves import range

import nimble
from nimble.interfaces.universal_interface import UniversalInterface
from nimble.interfaces.interface_helpers import PythonSearcher
from nimble.interfaces.interface_helpers import collectAttributes
from nimble.interfaces.interface_helpers import removeFromTailMatchedLists
from nimble.helpers import inspectArguments
from nimble.docHelpers import inheritDocstringsFactory


# Contains path to keras root directory
#kerasDir = '/usr/local/lib/python2.7/dist-packages'
kerasDir = None

# a dictionary mapping names to learners, or modules
# containing learners. To be used by findInPackage
locationCache = {}


@inheritDocstringsFactory(UniversalInterface)
class Keras(UniversalInterface):
    """
    This class is an interface to keras.
    """
    def __init__(self):
        if kerasDir is not None:
            sys.path.insert(0, kerasDir)

        self.keras = nimble.importModule('keras')

        backendName = self.keras.backend.backend()
        # tensorflow has a tremendous quantity of informational outputs which
        # drown out anything else on standard out
        if backendName == 'tensorflow':
            logging.getLogger('tensorflow').disabled = True

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

            if not (hasFit and hasPred):
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
            import keras
        except ImportError:
            return False
        return True

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

        ret = init[0] + fit[0] + fitGenerator[0] + compile_[0] + predict[0]

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
        for stage in toProcess:
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
        return learner.UIgetScoreOrder()


    def isAlias(self, name):
        if name.lower() == 'keras':
            return True
        return name.lower() == self.getCanonicalName().lower()


    def getCanonicalName(self):
        return 'keras'


    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             arguments, customDict):
        if 'layers' in arguments:
            #this is to check if layers has been processed or not
            if isinstance(arguments['layers'][0], dict):
                if learnerName == 'Sequential':
                    layersObj = []
                    for layer in arguments['layers']:
                        layerType = layer.pop('type')
                        layersObj.append(self.findCallable(layerType)(**layer))
                else:
                    layersObj = {}
                    for layer in arguments['layers']:
                        layerType = layer.pop('type')
                        layerName = layer.pop('layerName')
                        toCall = self.findCallable(layerType)(**layer)
                        if 'inputs' in layer:
                            inputName = layer.pop('inputs')
                            layersObj[layerName] = toCall(layersObj[inputName])
                        else:
                            layersObj[layerName] = toCall
                    arguments['inputs'] = layersObj[arguments['inputs']]
                    arguments['outputs'] = layersObj[arguments['outputs']]
                arguments['layers'] = layersObj

        if trainX is not None:
            if trainX.getTypeString() != 'Sparse':
            #for sparse cases, keep it untouched here.
                trainX = trainX.copy(to='numpy matrix')

        if trainY is not None:
            if len(trainY.features) > 1:
                trainY = (trainY.copy(to='numpy array'))
            else:
                trainY = trainY.copy(to='numpy array', outputAs1D=True)

        if testX is not None:
            if testX.getTypeString() == 'Matrix':
                testX = testX.data
            elif testX.getTypeString() != 'Sparse':
            #for sparse cases, keep it untouched here.
                testX = testX.copy(to='numpy matrix')
        #
        # # this particular learner requires integer inputs
        # if learnerName == 'MultinomialHMM':
        #     if trainX is not None:
        #         trainX = numpy.array(trainX, numpy.int32)
        #     if trainY is not None:
        #         trainY = numpy.array(trainY, numpy.int32)
        #     if testX is not None:
        #         testX = numpy.array(testX, numpy.int32)

        return (trainX, trainY, testX, copy.copy(arguments))


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
        return nimble.createData(outputType, outputValue, useLog=False)


    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        # get parameter names
        # if learnerName == 'Model':
        #     initNames = ['inputs', 'outputs']
        # else:
        self.learnerName = learnerName
        initNames = self._paramQuery('__init__', learnerName, ['self'])[0]
        compileNames = self._paramQuery('compile', learnerName, ['self'])[0]
        if isinstance(trainX, nimble.data.Sparse):
            param = 'fit_generator'
        else:
            param = 'fit'
        fitNames = self._paramQuery(param, learnerName, ['self'])[0]

        # pack parameter sets
        initParams = {}
        for name in initNames:
            initParams[name] = arguments[name]
        learner = self.findCallable(learnerName)(**initParams)

        compileParams = {}
        for name in compileNames:
            compileParams[name] = arguments[name]
        learner.compile(**compileParams)

        fitParams = {}
        for name in fitNames:
            if name.lower() == 'x' or name.lower() == 'obs':
                value = trainX
            elif name.lower() == 'y':
                value = trainY
            else:
                value = arguments[name]
            fitParams[name] = value

        if isinstance(trainX, nimble.data.Sparse):
            def sparseGenerator():
                while True:
                    for i in range(len(trainX.points)):
                        tmpData = (trainX.pointView(i).copy(to='numpy matrix'),
                                   numpy.matrix(trainY[i]))
                        yield tmpData
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

            def UIgetScoreOrder():
                return labelOrder

            learner.UIgetScoreOrder = UIgetScoreOrder

        return learner


    def _incrementalTrainer(self, learner, trainX, trainY, arguments,
                            customDict):
        param = 'train_on_batch'
        learnerName = self.learnerName
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
        if hasattr(learner, 'predict'):
            if isinstance(testX, nimble.data.Sparse):
                method = 'predict_generator'
            else:
                method = 'predict'
            ignore = ['X', 'x', 'self']
            backendArgs = self._paramQuery(method, learnerName, ignore)[0]
            applyArgs = self._getMethodArguments(backendArgs, newArguments,
                                                 storedArguments)
            return self._predict(learner, testX, applyArgs, customDict)
        else:
            msg = "Cannot apply this learner to data, no predict function"
            raise TypeError(msg)


    def _getAttributes(self, learnerBackend):
        obj = learnerBackend
        generators = None
        checkers = []
        checkers.append(nimble.interfaces.interface_helpers.noLeading__)
        checkers.append(nimble.interfaces.interface_helpers.notCallable)
        checkers.append(nimble.interfaces.interface_helpers.notABCAssociated)

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
        if isinstance(testX, nimble.data.Sparse):
            def sparseGenerator():
                while True:
                    for i in range(len(testX.points)):
                        tmpData = testX.pointView(i).copy(to='numpy matrix')
                        yield tmpData
            arguments['generator'] = sparseGenerator()
            return learner.predict_generator(**arguments)
        else:
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
            (args, d) = removeFromTailMatchedLists(args, d, ignore)
            if 'random_state' in args:
                index = args.index('random_state')
                negdex = index - len(args)
                d[negdex] = nimble.randomness.generateSubsidiarySeed()
            return (args, v, k, d)
        except TypeError:
            try:
                (args, v, k, d) = inspectArguments(namedModule.__init__)
                (args, d) = removeFromTailMatchedLists(args, d, ignore)
                if 'random_state' in args:
                    index = args.index('random_state')
                    negdex = index - len(args)
                    d[negdex] = nimble.randomness.generateSubsidiarySeed()
                return (args, v, k, d)
            except TypeError:
                return self._paramQueryHardCoded(name, parent, ignore)


    def _paramQueryHardCoded(self, name, parent, ignore):
        """
        Returns a list of parameters for in package entities that we
        have hard coded, under the assumption that it is difficult or
        impossible to find that data automatically.
        """
        #if needed, this function should be rewritten for keras.
        if parent is not None and parent.lower() == 'KernelCenterer'.lower():
            if name == '__init__':
                ret = ([], None, None, [])
            (newArgs, newDefaults) = removeFromTailMatchedLists(ret[0], ret[3],
                                                                ignore)
            return (newArgs, ret[1], ret[2], newDefaults)

        if parent is not None and parent.lower() == 'LabelEncoder'.lower():
            if name == '__init__':
                ret = ([], None, None, [])
            (newArgs, newDefaults) = removeFromTailMatchedLists(ret[0], ret[3],
                                                                ignore)
            return (newArgs, ret[1], ret[2], newDefaults)

        if parent is not None and parent.lower() == 'DummyRegressor'.lower():
            if name == '__init__':
            #	ret = (['strategy', 'constant'], None, None, ['mean', None])
                ret = ([], None, None, [])
            (newArgs, newDefaults) = removeFromTailMatchedLists(ret[0], ret[3],
                                                                ignore)
            return (newArgs, ret[1], ret[2], newDefaults)
        if parent is not None and parent.lower() == 'ZeroEstimator'.lower():
            if name == '__init__':
                return ([], None, None, [])

        if parent is not None and parent.lower() == 'GaussianNB'.lower():
            if name == '__init__':
                ret = ([], None, None, [])
            elif name == 'fit':
                ret = (['X', 'y'], None, None, [])
            elif name == 'predict':
                ret = (['X'], None, None, [])
            else:
                return None

            (newArgs, newDefaults) = removeFromTailMatchedLists(ret[0], ret[3],
                                                                ignore)
            return (newArgs, ret[1], ret[2], newDefaults)

        return None
