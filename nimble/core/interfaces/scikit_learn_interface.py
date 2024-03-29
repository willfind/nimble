
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
Relies on being scikit-learn 1.0 or above

TODO: multinomialHMM requires special input processing for obs param
"""

# pylint: disable=unused-argument

import warnings
from unittest import mock
import pkgutil
import abc
import importlib

import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentValue, ImproperObjectAction
from nimble._utility import inspectArguments
from nimble._utility import inheritDocstringsFactory, dtypeConvert
from .universal_interface import PredefinedInterfaceMixin
from ._interface_helpers import modifyImportPathAndImport
from ._interface_helpers import collectAttributes
from ._interface_helpers import removeFromTailMatchedLists
from ._interface_helpers import noLeading__, notCallable, notABCAssociated
from ._interface_helpers import validInitParams


MANUAL_LEARNERTYPES = {
    'BayesianGaussianMixture': 'cluster',
    'CountVectorizer': 'transformation',
    # 'EllipticEnvelope': 'classification', # TODO outliers
    'GaussianMixture': 'cluster',
    # 'IsolationForest': 'classification', # TODO outliers
    'MDS': 'transformation',
    # 'OneClassSVM': 'classification', # TODO outliers
    'PatchExtractor': 'transformation',
    'RandomTreesEmbedding': 'transformation',
    # 'SGDOneClassSVM': 'classification', # TODO outliers
    'SpectralEmbedding': 'transformation',
    'TSNE': 'transformation',
    'TfidfVectorizer': 'transformation',
}

@inheritDocstringsFactory(PredefinedInterfaceMixin)
class _SciKitLearnAPI(PredefinedInterfaceMixin):
    """
    Base class for interfaces following the scikit-learn api.
    """
    def __init__(self, randomParam):
        self._skl = modifyImportPathAndImport('sklearn', 'sklearn')
        self.randomParam = randomParam
        super().__init__()

    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def _learnerType(self, learnerBackend):

        # Operating on Clusters supercedes all other mixin considerations
        if isinstance(learnerBackend, self._skl.base.ClusterMixin):
            return 'cluster'
        if isinstance(learnerBackend, self._skl.base.ClassifierMixin):
            return 'classification'
        if isinstance(learnerBackend, self._skl.base.RegressorMixin):
            return 'regression'
        # There are a number of classifiers and regressors which also have the
        # TransformerMixin. They are prefered to be what _applier (according
        # to the current implementation) treats them as. As such, this is
        # checked last to short circuit to the other options.
        if isinstance(learnerBackend, self._skl.base.TransformerMixin):
            return 'transformation'

        return 'UNKNOWN'

    def _getParameterNamesBackend(self, name):
        ret = self._paramQuery(name, None)
        if ret is None:
            return ret
        (objArgs, _) = ret
        return [objArgs]

    def _getLearnerParameterNamesBackend(self, learnerName):
        ignore = ['self', 'X', 'x', 'Y', 'y', 'obs', 'T', 'raw_documents',
                  self.randomParam]
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
            msg = "Cannot get parameter names for learner " + learnerName
            raise InvalidArgumentValue(msg)

        return [ret]

    def _getDefaultValuesBackend(self, name):
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
        ignore = ['self', 'X', 'x', 'Y', 'y', 'T', 'raw_documents',
                  self.randomParam]
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
        ignore = ['self', 'X', 'x', 'Y', 'y', 'T', 'raw_documents']
        backendArgs = self._paramQuery(method, learnerName, ignore)[0]
        scoreArgs = self._getMethodArguments(backendArgs, newArguments,
                                             storedArguments)
        raw = toCall(testX, **scoreArgs)
        # in binary classification, we return a row vector. need to reshape
        if len(raw.shape) == 1:
            return raw.reshape(len(raw), 1)

        return raw


    def _getScoresOrder(self, learner):
        return learner.UIgetScoreOrder


    def _trainer(self, learnerName, trainX, trainY, arguments, randomSeed,
                 customDict):
        # init learner
        learner = self._initLearner(learnerName, trainX, trainY, arguments,
                                    randomSeed)
        # fit learner
        self._fitLearner(learner, learnerName, trainX, trainY, arguments)

        if (hasattr(learner, 'decision_function')
                or hasattr(learner, 'predict_proba')):
            if trainY is not None:
                labelOrder = np.unique(trainY)
            else:
                allLabels = learner.predict(trainX)
                labelOrder = np.unique(allLabels)

            learner.UIgetScoreOrder = labelOrder

        return learner

    def _incrementalTrainer(self, learnerName, learner, trainX, trainY,
                            arguments, randomSeed, customDict):
        # see partial_fit(X, y[, classes, sample_weight])
        msg = 'this interface does not implement incremental training'
        raise ImproperObjectAction(msg)

    def _applier(self, learnerName, learner, testX, newArguments,
                 storedArguments, customDict):
        if hasattr(learner, 'predict'):
            method = 'predict'
            toCall = self._predict
        elif hasattr(learner, 'transform'):
            method = 'transform'
            toCall = self._transform
        elif hasattr(learner, 'fit_predict'):
            method = 'fit_predict'
            toCall = self._fit_predict
        elif hasattr(learner, 'fit_transform'):
            method = 'fit_transform'
            toCall = self._fit_transform
        else:
            msg = "Cannot apply this learner to data, no predict or "
            msg += "transform function"
            raise TypeError(msg)
        ignore = ['self', 'X', 'x', 'Y', 'y', 'T', 'raw_documents']
        backendArgs = self._paramQuery(method, learnerName, ignore)[0]
        applyArgs = self._getMethodArguments(backendArgs, newArguments,
                                             storedArguments)
        if 'raw_documents' in self._paramQuery(method, learnerName)[0]:
            testX = testX.tolist()[0] # 1D list
        return toCall(learner, testX, applyArgs, customDict)


    def _getAttributes(self, learnerBackend):
        obj = learnerBackend
        generators = None
        checkers = []
        checkers.append(noLeading__)
        checkers.append(notCallable)
        checkers.append(notABCAssociated)

        ret = collectAttributes(obj, generators, checkers)
        return ret

    @classmethod
    def _optionDefaults(cls, option):
        return None

    @classmethod
    def _configurableOptionNames(cls):
        return ['location']

    # fit_transform

    def _predict(self, learner, testX, arguments, customDict):
        """
        Wrapper for the underlying predict function of a scikit-learn
        learner object.
        """
        return learner.predict(testX, **arguments)

    def _transform(self, learner, testX, arguments, customDict):
        """
        Wrapper for the underlying transform function of a scikit-learn
        learner object.
        """
        return learner.transform(testX, **arguments)

    def _fit_predict(self, learner, testX, arguments, customDict):
        """
        Wrapper for the underlying fit_predict function of a
        scikit-learn learner object.
        """
        return learner.labels_

    def _fit_transform(self, learner, testX, arguments, customDict):
        """
        Wrapper for the underlying fit_transform function of a
        scikit-learn learner object.
        """
        return learner.embedding_

    def _loadTrainedLearnerBackend(self, learnerName, arguments):
        msg = "This interface offers no pre-trained Learners"
        raise InvalidArgumentValue(msg)

    ###############
    ### HELPERS ###
    ###############

    def _paramQuery(self, name, parent, ignore=None):
        """
        Takes the name of some scikit learn object or function, returns
        a list of parameters used to instantiate that object or run that
        function, or None if the desired thing cannot be found.
        """
        if ignore is None:
            ignore = []
        if parent is None:
            namedModule = self.findCallable(name)
        else:
            namedModule = self.findCallable(parent)

        if parent is None or name == '__init__':
            obj = namedModule()
            initDefaults = obj.get_params()
            initParams = list(initDefaults.keys())
            initValues = list(initDefaults.values())
            return (initParams, initValues)
        if not hasattr(namedModule, name):
            return None
        (args, _, _, d) = inspectArguments(getattr(namedModule, name))
        (args, d) = removeFromTailMatchedLists(args, d, ignore)
        return (args, d)

    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    @abc.abstractmethod
    def accessible(self):
        pass

    @classmethod
    @abc.abstractmethod
    def getCanonicalName(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def isAlias(cls, name):
        pass


    @classmethod
    @abc.abstractmethod
    def _installInstructions(cls):
        pass

    @abc.abstractmethod
    def _findCallableBackend(self, name):
        pass

    @abc.abstractmethod
    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             randomSeed, arguments, customDict):
        pass

    @abc.abstractmethod
    def _outputTransformation(self, learnerName, outputValue,
                              transformedInputs, outputType, outputFormat,
                              customDict):
        pass

    @abc.abstractmethod
    def _initLearner(self, learnerName, trainX, trainY, arguments, randomSeed):
        pass

    @abc.abstractmethod
    def _fitLearner(self, learner, learnerName, trainX, trainY, arguments):
        pass

@inheritDocstringsFactory(_SciKitLearnAPI)
class SciKitLearn(_SciKitLearnAPI):
    """
    This class is an interface to scikit-learn.
    """

    def __init__(self):
        self.package = modifyImportPathAndImport('sklearn', 'sklearn')

        walkPackages = pkgutil.walk_packages

        def mockWalkPackages(*args, **kwargs):
            packages = walkPackages(*args, **kwargs)
            ret = []
            # ignore anything that imports libraries that are not installed,
            # test modules, and experimental modules.
            for pkg in packages:
                # each pkg is a tuple (importer, moduleName, isPackage)
                module = pkg[1]
                # no need to search tests and can fail in unexpected ways
                if 'tests' in module or 'experimental' in module:
                    continue
                try:
                    _ = importlib.import_module(module)
                    ret.append(pkg)
                except ImportError:
                    pass

            return ret

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', (DeprecationWarning,
                                             FutureWarning))
            with mock.patch('pkgutil.walk_packages', mockWalkPackages):
                utils = modifyImportPathAndImport('sklearn', 'sklearn.utils')
                estimatorDict = utils.all_estimators()

            self.allEstimators = {}
            for name, obj in estimatorDict:
                if name.startswith('_'):
                    continue
                try:
                    # if object cannot be instantiated without additional
                    # arguments, we cannot support it at this time
                    _ = obj()
                except TypeError:
                    continue
                # only support learners with a predict, transform,
                # fit_predict or fit_transform, all have fit attribute
                hasPred = hasattr(obj, 'predict')
                hasTrans = hasattr(obj, 'transform')
                hasFitPred = hasattr(obj, 'fit_predict')
                hasFitTrans = hasattr(obj, 'fit_transform')

                if hasPred or hasTrans or hasFitPred or hasFitTrans:
                    self.allEstimators[name] = obj

        super().__init__('random_state')

    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
        try:
            _ = modifyImportPathAndImport('sklearn', 'sklearn')
        except ImportError:
            return False
        return True

    @classmethod
    def getCanonicalName(cls):
        return 'sklearn'

    @classmethod
    def isAlias(cls, name):
        if name.lower() in ['skl', 'scikitlearn', 'scikit-learn']:
            return True
        return name.lower() == cls.getCanonicalName().lower()

    @classmethod
    def _installInstructions(cls):
        msg = """
To install scikit-learn
-----------------------
    Installation instructions for scikit-learn can be found at:
    https://scikit-learn.org/stable/install.html"""
        return msg

    def _learnerNamesBackend(self, onlyTrained=False):
        if onlyTrained:
            return []

        possibilities = []
        exclude = [
            'DictVectorizer', 'FeatureHasher', 'HashingVectorizer',
            'LabelBinarizer', 'LabelEncoder', 'MultiLabelBinarizer',
            'FeatureAgglomeration', 'LocalOutlierFactor',
            # the above do not take the standard X, [y] inputs
            'KernelCenterer', # fit takes K param not X
            'LassoLarsIC', # modifies original data
            'IsotonicRegression', # requires 1D input data
            ]

        for name in self.allEstimators:
            if name not in exclude:
                possibilities.append(name)

        return possibilities

    def _learnerType(self, learnerBackend):
        ret = super()._learnerType(learnerBackend)
        if ret == 'UNKNOWN':
            learnerClass = learnerBackend.__class__.__name__
            return MANUAL_LEARNERTYPES.get(learnerClass, 'UNKNOWN')
        return ret

    def _findCallableBackend(self, name):
        try:
            return self.allEstimators[name]
        except KeyError:
            return None

    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             randomSeed, arguments, customDict):

        mustCopyTrainX = ['PLSRegression']
        if trainX is not None:
            customDict['match'] = trainX.getTypeString()
            if trainX.shape != trainX.dimensions:
                trainX = trainX.copy(to='numpy array')
            elif (trainX.getTypeString() == 'Matrix'
                    and learnerName not in mustCopyTrainX):
                trainX = trainX._data
            elif trainX.getTypeString() == 'Sparse':
                trainX = trainX.copy()._data
            else:
                trainX = trainX.copy(to='numpy array')
            trainX = dtypeConvert(trainX)

        if trainY is not None:
            if len(trainY.features) > 1 or trainY.shape != trainY.dimensions:
                trainY = trainY.copy(to='numpy array')
            else:
                trainY = trainY.copy(to='numpy array', outputAs1D=True)
            trainY = dtypeConvert(trainY)

        if testX is not None:
            mustCopyTestX = ['StandardScaler']
            if testX.shape != testX.dimensions:
                testX = testX.copy(to='numpy array')
            elif (testX.getTypeString() == 'Matrix'
                    and learnerName not in mustCopyTestX):
                testX = testX._data
            elif testX.getTypeString() == 'Sparse':
                testX = testX.copy()._data
            else:
                testX = testX.copy(to='numpy array')
            testX = dtypeConvert(testX)

        # this particular learner requires integer inputs
        if learnerName == 'MultinomialHMM':
            if trainX is not None:
                trainX = np.array(trainX, np.int32)
            if trainY is not None:
                trainY = np.array(trainY, np.int32)
            if testX is not None:
                testX = np.array(testX, np.int32)

        instantiatedArgs = {}
        for arg, val in arguments.items():
            if isinstance(val, nimble.Init):
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
        return nimble.data(outputValue, returnType=outputType, useLog=False)

    def _initLearner(self, learnerName, trainX, trainY, arguments, randomSeed):
        initNames = self._paramQuery('__init__', learnerName, ['self'])[0]
        initParams = validInitParams(initNames, arguments, randomSeed,
                                     self.randomParam)
        learner = self.findCallable(learnerName)(**initParams)

        return learner

    def _fitLearner(self, learner, learnerName, trainX, trainY, arguments):
        fitNames = self._paramQuery('fit', learnerName, ['self'])[0]
        fitParams = {}
        for name in fitNames:
            if name.lower() == 'x' or name.lower() == 'obs':
                value = trainX
            elif name.lower() == 'y':
                value = trainY
            elif name.lower() == 'raw_documents':
                value = trainX.tolist()[0] #1D list
            elif name in arguments:
                value = arguments[name]
            else:
                continue
            fitParams[name] = value

        try:
            learner.fit(**fitParams)
        except ValueError as e:
            # these occur when the learner requires different input data
            # (multi-dimensional, non-negative)
            raise InvalidArgumentValue(str(e)) from e

    def version(self):
        return self.package.__version__
