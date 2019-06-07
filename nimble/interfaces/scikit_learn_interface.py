"""
Relies on being scikit-learn 0.19 or above

TODO: multinomialHMM requires special input processing for obs param
"""

from __future__ import absolute_import
import importlib
import copy
import sys
import warnings
from unittest import mock

import numpy
from six.moves import range

import nimble
from nimble.interfaces.universal_interface import UniversalInterface
from nimble.interfaces.universal_interface import BuiltinInterface
from nimble.exceptions import InvalidArgumentValue
from nimble.interfaces.interface_helpers import modifyImportPath
from nimble.interfaces.interface_helpers import collectAttributes
from nimble.interfaces.interface_helpers import removeFromTailMatchedLists
from nimble.helpers import inspectArguments
from nimble.docHelpers import inheritDocstringsFactory

# Contains path to sciKitLearn root directory
#sciKitLearnDir = '/usr/local/lib/python2.7/dist-packages'
sciKitLearnDir = None

# a dictionary mapping names to learners, or modules
# containing learners. To be used by findInPackage
locationCache = {}


@inheritDocstringsFactory(UniversalInterface)
class SciKitLearn(BuiltinInterface, UniversalInterface):
    """
    This class is an interface to scikit-learn.
    """

    def __init__(self):
        """

        """
        # modify path if another directory provided
        modifyImportPath(sciKitLearnDir, 'sciKitLearn')
        # suppress DeprecationWarnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.skl = importlib.import_module('sklearn')

        version = self.version()
        self._versionSplit = list(map(int, version.split('.')))

        from sklearn.utils.testing import all_estimators
        import pkgutil
        pkgutil.walk_packages_ = pkgutil.walk_packages
        def mockWalkPackages(*args, **kwargs):
            packages = pkgutil.walk_packages_(*args, **kwargs)
            return [pkg for pkg in packages if not 'conftest' in pkg[1]]

        with mock.patch('pkgutil.walk_packages', mockWalkPackages):
            all_estimators = all_estimators()

        self.allEstimators = {}
        for name, obj in all_estimators:
            if name.startswith('_'):
                continue
            try:
                # if object cannot be instantiated without additional
                # arguments, we cannot support it at this time
                init = obj()
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

    @classmethod
    def getCanonicalName(cls):
        return 'sciKitLearn'


    @classmethod
    def isAlias(cls, name):
        if name.lower() in ['skl', 'sklearn']:
            return True
        return name.lower() == cls.getCanonicalName().lower()

    def _listLearnersBackend(self):
        possibilities = []
        exclude = [
            'CountVectorizer', 'PatchExtractor', 'TfidfVectorizer',
            'DictVectorizer', 'FeatureHasher', 'HashingVectorizer',
            'LabelBinarizer', 'LabelEncoder', 'MultiLabelBinarizer',
            'FeatureAgglomeration', 'LocalOutlierFactor',
            # the above do not take the standard X, [y] inputs
            'KernelCenterer', # fit takes K param not X
            'LassoLarsIC', # modifies original data
            'IsotonicRegression', # requires 1D input data
            ]

        for name in self.allEstimators.keys():
            if name not in exclude:
                possibilities.append(name)

        return possibilities

    def learnerType(self, name):
        obj = self.findCallable(name)
        if issubclass(obj, self.skl.base.ClassifierMixin):
            return 'classification'
        if issubclass(obj, self.skl.base.RegressorMixin):
            return 'regression'
        if issubclass(obj, self.skl.base.ClusterMixin):
            return 'cluster'
        if issubclass(obj, self.skl.base.TransformerMixin):
            return 'transformation'
        # if (hasattr(obj, 'classes_') or hasattr(obj, 'label_')
        #         or hasattr(obj, 'labels_')):
        #     return 'classification'
        # if "Classifier" in obj.__name__:
        #     return 'classification'
        #
        # if "Regressor" in obj.__name__:
        #     return 'regression'

        return 'UNKNOWN'

    def _findCallableBackend(self, name):
        try:
            return self.allEstimators[name]
        except KeyError:
            return None

    def _getParameterNamesBackend(self, name):
        ret = self._paramQuery(name, None)
        if ret is None:
            return ret
        (objArgs, _) = ret
        return [objArgs]

    def _getLearnerParameterNamesBackend(self, learnerName):
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
        ignore = ['self', 'X', 'x', 'Y', 'y', 'T']
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


    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             arguments, customDict):
        mustCopyTrainX = ['PLSRegression']
        if trainX is not None:
            customDict['match'] = trainX.getTypeString()
            if (trainX.getTypeString() == 'Matrix'
                    and learnerName not in mustCopyTrainX):
                trainX = trainX.data
            elif trainX.getTypeString() == 'Sparse':
                trainX = trainX.copy().data
            else:
                trainX = trainX.copy(to='numpy matrix')

        if trainY is not None:
            if len(trainY.features) > 1:
                trainY = (trainY.copy(to='numpy array'))
            else:
                trainY = trainY.copy(to='numpy array', outputAs1D=True)
            if trainY.dtype == numpy.object_:
                try:
                    trainY = trainY.astype(numpy.float)
                except ValueError:
                    pass

        if testX is not None:
            mustCopyTestX = ['StandardScaler']
            if (testX.getTypeString() == 'Matrix'
                    and learnerName not in mustCopyTestX):
                testX = testX.data
            elif testX.getTypeString() == 'Sparse':
                testX = testX.copy().data
            else:
                testX = testX.copy(to='numpy matrix')

        # this particular learner requires integer inputs
        if learnerName == 'MultinomialHMM':
            if trainX is not None:
                trainX = numpy.array(trainX, numpy.int32)
            if trainY is not None:
                trainY = numpy.array(trainY, numpy.int32)
            if testX is not None:
                testX = numpy.array(testX, numpy.int32)

        return (trainX, trainY, testX, copy.deepcopy(arguments))


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
        if self._versionSplit[1] < 19:
            msg = "nimble was tested using sklearn 0.19 and above, we cannot be "
            msg += "sure of success for version {0}".format(self.version())
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
            # these occur when the learner requires different input data
            # (multi-dimensional, non-negative)
            raise InvalidArgumentValue(str(ve))
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
        # see partial_fit(X, y[, classes, sample_weight])
        pass


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
        ignore = ['self', 'X', 'x', 'Y', 'y', 'T']
        backendArgs = self._paramQuery(method, learnerName, ignore)[0]
        applyArgs = self._getMethodArguments(backendArgs, newArguments,
                                             storedArguments)
        return toCall(learner, testX, applyArgs, customDict)

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
        return [self._predict, self._transform]


    def version(self):
        return self.skl.__version__

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
            if 'random_state' in initParams:
                index = initParams.index('random_state')
                negdex = index - len(initParams)
                initValues[negdex] = nimble.randomness.generateSubsidiarySeed()
            return (initParams, initValues)
        elif not hasattr(namedModule, name):
            return None
        else:
            (args, _, _, d) = inspectArguments(getattr(namedModule, name))
            if 'random_state' in args:
                index = args.index('random_state')
                negdex = index - len(args)
                d[negdex] = nimble.randomness.generateSubsidiarySeed()
            (args, d) = removeFromTailMatchedLists(args, d, ignore)
            return (args, d)
