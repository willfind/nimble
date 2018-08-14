"""
Relies on being scikit-learn 0.9 or above

"""

# TODO: multinomialHMM requires special input processing for obs param


from __future__ import absolute_import
import importlib
import inspect
import copy
import numpy
import os
import sys
import functools

import UML
import warnings
from UML.exceptions import ArgumentException
from UML.interfaces.interface_helpers import PythonSearcher
from UML.interfaces.interface_helpers import collectAttributes
from six.moves import range

# Contains path to sciKitLearn root directory
#sciKitLearnDir = '/usr/local/lib/python2.7/dist-packages'
sciKitLearnDir = None

# a dictionary mapping names to learners, or modules
# containing learners. To be used by findInPackage
locationCache = {}

from UML.interfaces.universal_interface import UniversalInterface


class SciKitLearn(UniversalInterface):
    """

    """

    def __init__(self):
        """

        """
        if sciKitLearnDir is not None:
            sys.path.insert(0, sciKitLearnDir)

        self.skl = importlib.import_module('sklearn')

        version = self.skl.__version__
        self._version = version
        self._versionSplit = list(map(int,version.split('.')))

        # __all__ has been known to not have some subpackages that we want
        # so we check the root directory of sklearn for names that we can
        # attempt to import
        names = os.listdir(self.skl.__path__[0])
        possibilities = []
        for name in names:
            splitList = name.split('.')
            if len(splitList) == 1 or splitList[1] in ['py', 'pyc']:
                if splitList[0] not in self.skl.__all__ and not splitList[0].startswith('_'):
                    possibilities.append(splitList[0])

        possibilities = numpy.unique(possibilities).tolist()
        if 'utils' in possibilities:
            possibilities.remove('utils')
        self.skl.__all__.extend(possibilities)

        #		self.newLearners = []
        #		for name in self._listLearnersBackend():
        #			if name not in oldList:
        #				self.newLearners.append(name)

        def isLearner(obj):
            if not inspect.isclass(obj):
                return False
            if not issubclass(obj, self.skl.base.BaseEstimator):
                return False

            hasFit = hasattr(obj, 'fit')
            hasPred = hasattr(obj, 'predict')
            hasTrans = hasattr(obj, 'transform')
            hasFitPred = hasattr(obj, 'fit_predict')
            hasFitTrans = hasattr(obj, 'fit_transform')

            if not ((hasFit and (hasPred or hasTrans)) or hasFitPred or hasFitTrans):
                return False

            try:
                instantiated = obj()
            except TypeError:
                # We're using a failed init call as a cue that object in question
                # is a kind of intermediate class (which we want to ignore). All
                # the working estimators seem to have full defaults for all params
                # to __init__
                return False

            return True

        self._searcher = PythonSearcher(self.skl, self.skl.__all__, {}, isLearner, 2)

        super(SciKitLearn, self).__init__()

    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
        try:
            import sklearn
        except ImportError:
            return False
        return True

    def _listLearnersBackend(self):
        ret = []
        # # TODO sklearn.utils.testing.all_estimators will be implemented in 0.20
        # try:
        #     all_estimators = self.skl.util.testing.all_estimators
        #     estimators = all_estimators()
        #     for name, module in estimators:
        #         ret.append(name)
        # except:
        possibilities = self._searcher.allLearners()

        exclude = [
            'BaseDiscreteNB', 'libsvm', 'GMMHMM', 'GaussianHMM', 'MultinomialHMM',
            'GridSearchCV', 'RandomizedSearchCV', 'IsotonicRegression',
            'LogOddsEstimator', 'PriorProbabilityEstimator', 'MeanEstimator',
            'BaggingClassifier', 'PatchExtractor', 'DummyClassifier',
            'CountVectorizer', 'TfidfVectorizer', 'LabelBinarizer',
            'MultiLabelBinarizer', 'LabelEncoder', 'FeatureHasher',
            'DictVectorizer',
            'FeatureAgglomeration', 'LocalOutlierFactor', 'NMF', 'KernelCenterer',]
        ret = []
        for name in possibilities:
            if not name in exclude:
                ret.append(name)

        return ret

    def learnerType(self, name):
        """
        Returns a string referring to the action the learner takes out of the possibilities:
        classification, regression, featureSelection, dimensionalityReduction
        TODO

        """
        obj = self.findCallable(name)
        if issubclass(obj, self.skl.base.ClassifierMixin):
            return 'classification'
        if issubclass(obj, self.skl.base.RegressorMixin):
            return 'regression'
        if issubclass(obj, self.skl.base.ClusterMixin):
            return 'cluster'
        if issubclass(obj, self.skl.base.TransformerMixin):
            return 'transformation'
        # if hasattr(obj, 'classes_') or hasattr(obj, 'label_') or hasattr(obj, 'labels_'):
        #     return 'classification'
        # if "Classifier" in obj.__name__:
        #     return 'classification'
        #
        # if "Regressor" in obj.__name__:
        #     return 'regression'

        return 'UNKNOWN'

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
        ret = self._paramQuery(name, None)
        if ret is None:
            return ret
        (objArgs, d) = ret
        return [objArgs]

    def _getLearnerParameterNamesBackend(self, learnerName):
        """
        Find all parameters involved in a trainAndApply() call to the given learner
        TAKES string name of a learner,
        RETURNS list of list of param names
        """
        #		if learnerName == 'KernelCenterer':
        #			import pdb
        #			pdb.set_trace()
        ignore = ['self', 'X', 'x', 'Y', 'y', 'obs', 'T']
        init = self._paramQuery('__init__', learnerName, ignore)
        fit = self._paramQuery('fit', learnerName, ignore)
        predict = self._paramQuery('predict', learnerName, ignore)
        transform = self._paramQuery('transform', learnerName, ignore)
        fitPredict = self._paramQuery('fit_predict', learnerName, ignore)
        fitTransform = self._paramQuery('fit_transform', learnerName, ignore)

        if predict is not None:
            ret = init[0] + fit[0] + predict[0]
        elif transform is not None:
            ret = init[0] + fit[0] + transform[0]
        elif fitPredict is not None:
            ret = init[0] + fitPredict[0]
        elif fitTransform is not None:
            ret = init[0] + fitTransform[0]
        else:
            raise ArgumentException("Cannot get parameter names for leaner " + learnerName)

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
        (objArgs, d) = ret
        ret = {}
        if d is not None:
            for i in range(len(d)):
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
        transform = self._paramQuery('transform', learnerName, ignore)
        fitPredict = self._paramQuery('fit_predict', learnerName, ignore)
        fitTransform = self._paramQuery('fit_transform', learnerName, ignore)

        if predict is not None:
            toProcess = [init, fit, predict]
        elif transform is not None:
            toProcess = [init, fit, transform]
        elif fitPredict is not None:
            toProcess = [init, fitPredict]
        else:
            toProcess = [init, fitTransform]

        ret = {}
        for stage in toProcess:
            currNames = stage[0]
            currDefaults = stage[1]
            if stage[1] is not None:
                for i in range(len(currDefaults)):
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
            raise NotImplementedError('Cannot get scores for this learner')
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
        return 'sciKitLearn'


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
        mustCopy = ['PLSRegression']
        if trainX is not None:
            customDict['match'] = trainX.getTypeString()
            if trainX.getTypeString() == 'Matrix' and learnerName not in mustCopy:
                trainX = trainX.data
            elif trainX.getTypeString() == 'Sparse':
                trainX = trainX.copyAs('scipycsr')
            else:
                trainX = trainX.copyAs('numpy matrix')

        if trainY is not None:
            if trainY.features > 1:
                trainY = (trainY.copyAs('numpy array'))
            else:
                trainY = trainY.copyAs('numpy array', outputAs1D=True)

        if testX is not None:
            if testX.getTypeString() == 'Matrix':
                testX = testX.data
            elif testX.getTypeString() == 'Sparse':
                testX = testX.copyAs('scipycsr')
            else:
                testX = testX.copyAs('numpy matrix')

        # this particular learner requires integer inputs
        if learnerName == 'MultinomialHMM':
            if trainX is not None:
                trainX = numpy.array(trainX, numpy.int32)
            if trainY is not None:
                trainY = numpy.array(trainY, numpy.int32)
            if testX is not None:
                testX = numpy.array(testX, numpy.int32)

        return (trainX, trainY, testX, copy.deepcopy(arguments))


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
        if self._versionSplit[1] < 19:
            msg = "UML was tested using sklearn 0.19.1 and above, we cannot be "
            msg += "sure of success for version {0}".format(self._version)
            warnings.warn(msg)
        # get parameter names
        initNames = self._paramQuery('__init__', learnerName, ['self'])[0]
        fitNames = self._paramQuery('fit', learnerName, ['self'])[0]

        # pack parameter sets
        initParams = {}
        for name in initNames:
            initParams[name] = arguments[name]
        fitParams = {}
        for name in fitNames:
            if name.lower() == 'x' or name.lower() == 'obs':
                value = trainX
            elif name.lower() == 'y':
                value = trainY
            else:
                value = arguments[name]
            fitParams[name] = value

        learner = self.findCallable(learnerName)(**initParams)
        try:
            learner.fit(**fitParams)
        except ValueError as ve:
            # these occur when the learner requires different input data (multi-dimensional, non-negative)
            raise ArgumentException(str(ve))
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
        elif hasattr(learner, 'labels_'):
            return learner.labels_
        elif hasattr(learner, 'embedding_'):
            return learner.embedding_
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
            for i in range(len(full)):
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
        for i in range(len(full)):
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

        if inspect.isclass(namedModule) or name == '__init__':
            if parent is None:
                toCall = self.findCallable(name)
            else:
                toCall = self.findCallable(parent)
            obj = toCall()
            initDefaults = obj.get_params()
            initParams = list(initDefaults.keys())
            initValues = list(initDefaults.values())
            return (initParams, initValues)

        (args, v, k, d) = inspect.getargspec(namedModule)
        (args, d) = self._removeFromTailMatchedLists(args, d, ignore)

        return (args, d)
