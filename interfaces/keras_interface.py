"""
Relies on being keras 2.0.8

"""

import importlib
import inspect
import copy
import numpy
import os
import sys
import functools

import UML

keras = UML.importModule('keras')
modelFunc = keras.models.Model.__init__
def tmpFunc(self, inputs=None, outputs=None, layers=None, **kwarguments):
    modelFunc(self, inputs=inputs, outputs=outputs)
keras.models.Model.__init__ = tmpFunc

from UML.exceptions import ArgumentException
from UML.interfaces.interface_helpers import PythonSearcher
from UML.interfaces.interface_helpers import collectAttributes

# Contains path to sciKitLearn root directory
#sciKitLearnDir = '/usr/local/lib/python2.7/dist-packages'
kerasDir = None

# a dictionary mapping names to learners, or modules
# containing learners. To be used by findInPackage
locationCache = {}

from UML.interfaces.universal_interface import UniversalInterface


class Keras(UniversalInterface):
    """

    """
    def __init__(self):
        """

        """
        if kerasDir is not None:
            sys.path.insert(0, kerasDir)
        self.keras = keras

        # keras 2.0.8 has no __all__
        names = os.listdir(self.keras.__path__[0])
        possibilities = []
        if not hasattr(self.keras, '__all__'):
            self.keras.__all__ = []
        for name in names:
            splitList = name.split('.')
            if len(splitList) == 1 or splitList[1] in ['py', 'pyc']:
                if splitList[0] not in self.keras.__all__ and not splitList[0].startswith('_'):
                    possibilities.append(splitList[0])

        possibilities = numpy.unique(possibilities).tolist()
        if 'utils' in possibilities:
            possibilities.remove('utils')
        self.keras.__all__.extend(possibilities)

        def isLearner(obj):
            """
            in Keras, there are 2 learners: Sequential and Model.
            """
            hasFit = hasattr(obj, 'fit')
            hasPred = hasattr(obj, 'predict')
            hasTrans = hasattr(obj, 'transform')
            hasFitPred = hasattr(obj, 'fit_predict')
            hasFitTrans = hasattr(obj, 'fit_transform')

            if not ((hasFit and (hasPred or hasTrans)) or hasFitPred or hasFitTrans):
                return False


            # try:
            #     obj()
            # except TypeError:
            #     # We're using a failed init call as a cue that object in question
            #     # is a kind of intermediate class (which we want to ignore). All
            #     # the working estimators seem to have full defaults for all params
            #     # to __init__
            #     return False

            return True

        self._searcher = PythonSearcher(self.keras, self.keras.__all__, {}, isLearner, 2)

        super(Keras, self).__init__()

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
        """
        Returns a string referring to the action the learner takes out of the possibilities:
        optimization

        """

        return 'OPTIMIZER'

    def _findCallableBackend(self, name):
        """
        Find reference to the callable with the given name
        TAKES string name
        RETURNS reference to in-package function or constructor
        """
        return self._searcher.findInPackage(None, name)

    def _getParameterNamesBackend(self, name):
        """
        Find params for instantiation and function calls
        TAKES string name,
        RETURNS list of list of param names to make the chosen call
        """
        ret = self._paramQuery(name, None, ignore=['self'])
        if ret is None:
            return ret
        (objArgs, v, k, d) = ret
        return [objArgs]

    def _getLearnerParameterNamesBackend(self, learnerName):
        """
        Find all parameters involved in a trainAndApply() call to the given learner
        TAKES string name of a learner,
        RETURNS list of list of param names
        """
        ignore = ['self', 'X', 'x', 'Y', 'y', 'obs', 'T']
        init = self._paramQuery('__init__', learnerName, ignore)
        fit = self._paramQuery('fit', learnerName, ignore)
        predict = self._paramQuery('predict', learnerName, ignore)
        compile = self._paramQuery('compile', learnerName, ignore)

        ret = init[0] + fit[0] + compile[0] + predict[0]
        # if predict is not None:
        #     ret = init[0] + fit[0] + predict[0]
        # elif compile is not None:
        #     ret = init[0] + fit[0] + compile[0]
        # else:
        #     raise ArgumentException("Cannot get parameter names for leaner " + learnerName)

        return [ret]

    def _getDefaultValuesBackend(self, name):
        """
        Find default values
        TAKES string name,
        RETURNS list of dict of param names to default values
        """
        ret = self._paramQuery(name, None)
        if ret is None:
            return ret
        (objArgs, v, k, d) = ret
        ret = {}
        if d is not None:
            for i in xrange(len(d)):
                ret[objArgs[-(i + 1)]] = d[-(i + 1)]

        return [ret]

    def _getLearnerDefaultValuesBackend(self, learnerName):
        """
        Find all default values for parameters involved in a trainAndApply() call to the given learner
        TAKES string name of a learner,
        RETURNS list of dict of param names to default values
        """
        ignore = ['self', 'X', 'x', 'Y', 'y', 'T']
        init = self._paramQuery('__init__', learnerName, ignore)
        fit = self._paramQuery('fit', learnerName, ignore)
        predict = self._paramQuery('predict', learnerName, ignore)
        compile = self._paramQuery('compile', learnerName, ignore)

        toProcess = [init, fit, compile, predict]

        ret = {}
        for stage in toProcess:
            currNames = stage[0]
            currDefaults = stage[3]
            if stage[3] is not None:
                for i in xrange(len(currDefaults)):
                    key = currNames[-(i + 1)]
                    value = currDefaults[-(i + 1)]
                    ret[key] = value

        return [ret]

    def _getScores(self, learner, testX, arguments, customDict):
        """
        If the learner is a classifier, then return the scores for each
        class on each data point, otherwise raise an exception.

        """
        if hasattr(learner, 'decision_function'):
            toCall = learner.decision_function
        elif hasattr(learner, 'predict_proba'):
            toCall = learner.predict_proba
        else:
            raise ArgumentException('Cannot get scores for this learner')
        raw = toCall(testX)
        # in binary classification, we return a row vector. need to reshape
        if len(raw.shape) == 1:
            return raw.reshape(len(raw), 1)
        else:
            return raw


    def _getScoresOrder(self, learner):
        """
        If the learner is a classifier, then return a list of the the labels corresponding
        to each column of the return from getScores

        """
        return learner.UIgetScoreOrder()


    def isAlias(self, name):
        """
        Returns true if the name is an accepted alias for this interface

        """
        if name.lower() == 'skl':
            return True
        return name.lower() == self.getCanonicalName().lower()


    def getCanonicalName(self):
        """
        Returns the string name that will uniquely identify this interface

        """
        return 'keras'


    def _inputTransformation(self, learnerName, trainX, trainY, testX, arguments, customDict):
        """
        Method called before any package level function which transforms all
        parameters provided by a UML user.

        trainX, etc. are filled with the values of the parameters of the same name
        to a calls to trainAndApply() or train(), or are empty when being called before other
        functions. arguments is a dictionary mapping names to values of all other
        parameters that need to be processed.

        The return value of this function must be a dictionary mirroring the
        structure of the inputs. Specifically, four keys and values are required:
        keys trainX, trainY, testX, and arguments. For the first three, the associated
        values must be the transformed values, and for the last, the value must be a
        dictionary with the same keys as in the 'arguments' input dictionary, with
        transformed values as the values. However, other information may be added
        by the package implementor, for example to be used in _outputTransformation()

        """
        if 'layers' in arguments:
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
                    if 'inputs' in layer:
                        inputName = layer.pop('inputs')
                        layersObj[layerName] = self.findCallable(layerType)(**layer)(layersObj[inputName])
                    else:
                        layersObj[layerName] = self.findCallable(layerType)(**layer)
                arguments['inputs'] = layersObj[arguments['inputs']]
                arguments['outputs'] = layersObj[arguments['outputs']]
            arguments['layers'] = layersObj

        if trainX is not None:
            if trainX.getTypeString() == 'Sparse':
                trainX = trainX.copyAs('scipycsr')
            else:
                trainX = trainX.copyAs('numpy matrix')

        if trainY is not None:
            if trainY.featureCount > 1:
                trainY = (trainY.copyAs('numpy array'))
            else:
                trainY = trainY.copyAs('numpy array', outputAs1D=True)
        #
        # if testX is not None:
        #     if testX.getTypeString() == 'Matrix':
        #         testX = testX.data
        #     elif testX.getTypeString() == 'Sparse':
        #         testX = testX.copyAs('scipycsr')
        #     else:
        #         testX = testX.copyAs('numpy matrix')
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


    def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputType, outputFormat, customDict):
        """
        Method called before any package level function which transforms the returned
        value into a format appropriate for a UML user.

        """
        #In the case of prediction we are given a row vector, yet we want a column vector
        if outputFormat == "label" and len(outputValue.shape) == 1:
            outputValue = outputValue.reshape(len(outputValue), 1)

        #TODO correct
        outputType = 'Matrix'
        if outputType == 'match':
            outputType = customDict['match']
        return UML.createData(outputType, outputValue)


    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        """
        build a learner and perform training with the given data
        TAKES name of learner, transformed arguments
        RETURNS an in package object to be wrapped by a TrainedLearner object
        """
        # get parameter names
        # if learnerName == 'Model':
        #     initNames = ['inputs', 'outputs']
        # else:
        initNames = self._paramQuery('__init__', learnerName, ['self'])[0]
        compileNames = self._paramQuery('compile', learnerName, ['self'])[0]
        fitNames = self._paramQuery('fit', learnerName, ['self'])[0]

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
        learner.fit(**fitParams)

        if hasattr(learner, 'decision_function') or hasattr(learner, 'predict_proba'):
            if trainY is not None:
                labelOrder = numpy.unique(trainY)
            else:
                allLabels = learner.predict(trainX)
                labelOrder = numpy.unique(allLabels)

            def UIgetScoreOrder():
                return labelOrder

            learner.UIgetScoreOrder = UIgetScoreOrder

        return learner


    def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
        """
        Given an already trained online learner, extend it's training with the given data
        TAKES trained learner, transformed arguments,
        RETURNS the learner after this batch of training
        """
        # see partial_fit(X, y[, classes, sample_weight])
        pass


    def _applier(self, learner, testX, arguments, customDict):
        """
        use the given learner to do testing/prediction on the given test set
        TAKES a TrainedLearner object that can be tested on
        RETURNS UML friendly results
        """
        if hasattr(learner, 'predict'):
            return self._predict(learner, testX, arguments, customDict)
        elif hasattr(learner, 'transform'):
            return self._transform(learner, testX, arguments, customDict)
        else:
            raise TypeError("Cannot apply this learner to data, no predict or transform function")


    def _getAttributes(self, learnerBackend):
        """
        Returns whatever attributes might be available for the given learner,
        in the form of a dictionary. For example, in the case of linear
        regression, one might expect to find an intercept and a list of
        coefficients in the output. Makes use of the
        UML.interfaces.interface_helpers.collectAttributes function to
        automatically generate the results.

        """
        obj = learnerBackend
        generators = None
        checkers = []
        checkers.append(UML.interfaces.interface_helpers.noLeading__)
        checkers.append(UML.interfaces.interface_helpers.notCallable)
        checkers.append(UML.interfaces.interface_helpers.notABCAssociated)

        ret = collectAttributes(obj, generators, checkers)
        return ret

    def _optionDefaults(self, option):
        """
        Define package default values that will be used for as long as a default
        value hasn't been registered in the UML configuration file. For example,
        these values will always be used the first time an interface is instantiated.

        """
        return None

    def _configurableOptionNames(self):
        """
        Returns a list of strings, where each string is the name of a configurable
        option of this interface whose value will be stored in UML's configuration
        file.

        """
        return ['location']


    def _exposedFunctions(self):
        """
        Returns a list of references to functions which are to be wrapped
        in I/O transformation, and exposed as attributes of all TrainedLearner
        objects returned by this interface's train() function. If None, or an
        empty list is returned, no functions will be exposed. Each function
        in this list should be a python function, the inspect module will be
        used to retrieve argument names, and the value of the function's
        __name__ attribute will be its name in TrainedLearner.

        """
        return [self._predict, self._transform]

    # fit_transform


    def _predict(self, learner, testX, arguments, customDict):
        """
        Wrapper for the underlying predict function of a scikit-learn learner object
        """
        return learner.predict(testX)

    def _transform(self, learner, testX, arguments, customDict):
        """
        Wrapper for the underlying transform function of a scikit-learn learner object
        """
        return learner.transform(testX)


    ###############
    ### HELPERS ###
    ###############

    def _removeFromArray(self, orig, toIgnore):
        temp = []
        for entry in orig:
            if not entry in toIgnore:
                temp.append(entry)
        return temp

    def _removeFromDict(self, orig, toIgnore):
        for entry in toIgnore:
            if entry in orig:
                del orig[entry]
        return orig

    def _removeFromTailMatchedLists(self, full, matched, toIgnore):
        """
        full is some list n, matched is a list with length m, where m is less
        than or equal to n, where the last m values of full are matched against
        their positions in matched. If one of those is to be removed, it is to
        be removed in both.
        """
        temp = {}
        if matched is not None:
            for i in xrange(len(full)):
                if i < len(matched):
                    temp[full[len(full) - 1 - i]] = matched[len(matched) - 1 - i]
                else:
                    temp[full[len(full) - 1 - i]] = None
        else:
            retFull = self._removeFromArray(full, toIgnore)
            return (retFull, matched)

        for ignoreKey in toIgnore:
            if ignoreKey in temp:
                del temp[ignoreKey]

        retFull = []
        retMatched = []
        for i in xrange(len(full)):
            name = full[i]
            if name in temp:
                retFull.append(name)
                if (i - (len(full) - len(matched))) >= 0:
                    retMatched.append(temp[name])

        return (retFull, retMatched)


    def _paramQuery(self, name, parent, ignore=[]):
        """
        Takes the name of some scikit learn object or function, returns a list
        of parameters used to instantiate that object or run that function, or
        None if the desired thing cannot be found

        """
        namedModule = self._searcher.findInPackage(parent, name)

        if namedModule is None:
            return None

        class InheritedEmptyInit(object):
            pass

        if type(namedModule) == type(getattr(InheritedEmptyInit, '__init__')):
            return ([], None, None, None)

        try:
            (args, v, k, d) = inspect.getargspec(namedModule)
            (args, d) = self._removeFromTailMatchedLists(args, d, ignore)
            if 'random_state' in args:
                index = args.index('random_state')
                negdex = index - len(args)
                d[negdex] = UML.randomness.generateSubsidiarySeed()
            return (args, v, k, d)
        except TypeError:
            try:
                (args, v, k, d) = inspect.getargspec(namedModule.__init__)
                (args, d) = self._removeFromTailMatchedLists(args, d, ignore)
                if 'random_state' in args:
                    index = args.index('random_state')
                    negdex = index - len(args)
                    d[negdex] = UML.randomness.generateSubsidiarySeed()
                return (args, v, k, d)
            except TypeError:
                return self._paramQueryHardCoded(name, parent, ignore)


    def _paramQueryHardCoded(self, name, parent, ignore):
        """
        Returns a list of parameters for in package entities that we have hard coded,
        under the assumption that it is difficult or impossible to find that data
        automatically

        """
        if parent is not None and parent.lower() == 'KernelCenterer'.lower():
            if name == '__init__':
                ret = ([], None, None, [])
            (newArgs, newDefaults) = self._removeFromTailMatchedLists(ret[0], ret[3], ignore)
            return (newArgs, ret[1], ret[2], newDefaults)
        if parent is not None and parent.lower() == 'LabelEncoder'.lower():
            if name == '__init__':
                ret = ([], None, None, [])
            (newArgs, newDefaults) = self._removeFromTailMatchedLists(ret[0], ret[3], ignore)
            return (newArgs, ret[1], ret[2], newDefaults)
        if parent is not None and parent.lower() == 'DummyRegressor'.lower():
            if name == '__init__':
            #				ret = (['strategy', 'constant'], None, None, ['mean', None])
                ret = ([], None, None, [])
            (newArgs, newDefaults) = self._removeFromTailMatchedLists(ret[0], ret[3], ignore)
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

            (newArgs, newDefaults) = self._removeFromTailMatchedLists(ret[0], ret[3], ignore)
            return (newArgs, ret[1], ret[2], newDefaults)

        return None

