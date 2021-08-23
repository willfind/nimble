"""
A nimble interface building off of the modular python interface
for Shogun ML.

"""

import importlib
import multiprocessing
import warnings

import numpy as np

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble._utility import inheritDocstringsFactory
from .universal_interface import PredefinedInterfaceMixin
from ._interface_helpers import PythonSearcher
from ._interface_helpers import modifyImportPathAndImport

# TODO?
# * online learning
# * different feature types (streaming, for other problem types)
# *

# Interesting alias cases:
# * DomainAdaptionMulticlassLibLinear  -- or probably any nested machine

trainXAliases = ['traindat', 'f', 'features', 'feats', 'feat', 'training_data',
                 'train_features', 'data']
trainYAliases = ['trainlab', 'lab', 'labs', 'labels', 'training_labels',
                 'train_labels']


@inheritDocstringsFactory(PredefinedInterfaceMixin)
class Shogun(PredefinedInterfaceMixin):
    """
    This class is an interface to shogun.
    """

    def __init__(self):
        self.package = modifyImportPathAndImport('shogun', 'shogun')
        self._hasAll = hasattr(self.package, '__all__')
        if not hasattr(self.package, '__version__'):
            version = self._access('Version', 'get_version_release')()
            self.package.__version__ = version
        def isLearner(obj):
            hasTrain = hasattr(obj, 'train')
            hasApply = hasattr(obj, 'apply')

            if not (hasTrain and hasApply):
                return False

            # MKL seem to require some sort of dependency (SVMlight?)
            # Labels setting not currently set up to handle Multitask learners
            # TODO Online
            ignore = ['MKL', 'Multitask', 'Online']
            if any(partial in obj.__name__ for partial in ignore):
                return False
            if obj.__name__ in excludedLearners:
                return False

            # needs more to be able to distinguish between things that are
            # runnable and partial implementations. get_machine_problem_type()?
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    instantiated = obj()
                instantiated.get_machine_problem_type()
            except (SystemError, TypeError):
                return False

            return True

        self._searcher = PythonSearcher(self.package, isLearner, 2)

        super().__init__()

    def _access(self, module, target):
        """
        Helper to automatically search the correct locations for target
        objects in different historical versions of shogun that have
        different module structures. Due to the automated component that
        attempts to smooth over multiple competing versions of the
        package, this should be used with caution.
        """
        # If shogun has an __all__ attribute, it is in the old style of
        # organization, where things are separated into submodules. They need
        # to be loaded before access.
        if self._hasAll:
            if hasattr(self.package, module):
                submod = getattr(self.package, module)
            else:
                submod = importlib.import_module('shogun.' + module)
        # If there is no __all__ attribute, we're in the newer, flatter style
        # of module organization. Some things will still be in the submodule
        # they were previously,others will up at the top level. This will check
        # both locations
        else:
            if hasattr(self.package, target):
                submod = self.package
            else:
                submod = getattr(self.package, module)

        return getattr(submod, target)


    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
        try:
            _ = modifyImportPathAndImport('shogun', 'shogun')
        except ImportError:
            return False
        return True


    @classmethod
    def getCanonicalName(cls):
        return 'shogun'


    @classmethod
    def _installInstructions(cls):
        msg = """
To install shogun
-----------------
    shogun for Python can be built from source or installed through the conda
    package manager. conda requires installing Anaconda or Miniconda. Once
    conda is available, shogun can be installed by running the command:"
        conda install -c conda-forge shogun
    Further installation instructions for shogun can be found at:
    https://www.shogun-toolbox.org/install"""
        return msg

    def _learnerNamesBackend(self):
        return self._searcher.allLearners()


    def _getMachineProblemType(self, learnerName):
        """
        Get learner type
        """
        learner = self.findCallable(learnerName)
        learner = learner()
        ptVal = learner.get_machine_problem_type()
        return ptVal


    def learnerType(self, name):
        ptVal = self._getMachineProblemType(name)

        if (ptVal == self._access('Classifier', 'PT_BINARY')
                or ptVal == self._access('Classifier', 'PT_MULTICLASS')
                or ptVal == self._access('Classifier', 'PT_CLASS')):
            return 'classification'
        if ptVal == self._access('Classifier', 'PT_REGRESSION'):
            return 'regression'
        if ptVal == self._access('Classifier', 'PT_STRUCTURED'):
            return 'UNKNOWN'
        if ptVal == self._access('Classifier', 'PT_LATENT'):
            return 'UNKNOWN'

        # TODO warning, unknown problem type code

        return 'UNKNOWN'


    def _getLearnerParameterNamesBackend(self, learnerName):
        base = self._getParameterNamesBackend(learnerName)

        #remove aliases
        ret = []
        for group in base:
            curr = []
            for paramName in group:
                if paramName not in trainXAliases + trainYAliases:
                    curr.append(paramName)
            ret.append(curr)

        return ret


    def _getLearnerDefaultValuesBackend(self, learnerName):
        allNames = self._getLearnerParameterNamesBackend(learnerName)
        return self._setupDefaultsGivenBaseNames(allNames)


    def _getParameterNamesBackend(self, name):
        ret = []
        backend = self._searcher.findInPackage(None, name)
        for funcname in dir(backend):
            if funcname.startswith('set_'):
                funcname = funcname[4:]
                ret.append(funcname)

        return [ret]

    def _getDefaultValuesBackend(self, name):
        allNames = self._getParameterNamesBackend(name)
        return self._setupDefaultsGivenBaseNames(allNames)

    def _setupDefaultsGivenBaseNames(self, allNames):
        allValues = []
        for group in allNames:
            curr = {}
            for paramName in group:
                curr[paramName] = ShogunDefault(paramName)
            allValues.append(curr)

        return allValues

    def _getScores(self, learnerName, learner, testX, newArguments,
                   storedArguments, customDict):
        # TODO deal with merging stored vs new arguments
        # is this even a thing for shogun?
        predObj = self._applier(learnerName, learner, testX, newArguments,
                                storedArguments, customDict)
        predLabels = predObj.get_labels()
        numLabels = customDict['numLabels']
        if hasattr(predObj, 'get_multiclass_confidences'):
            # setup an array in the right shape, number of predicted labels
            # by number of possible labels
            scoresPerPoint = np.empty((len(predLabels), numLabels))
            for i in range(len(predLabels)):
                currConfidences = predObj.get_multiclass_confidences(i)
                if len(currConfidences) == 0:
                    msg = "The shogun learner {0} ".format(learner)
                    msg += "doesn't provide confidence scores"
                    raise NotImplementedError(msg)
                scoresPerPoint[i, :] = currConfidences
        # otherwise we must be dealing with binary classification
        else:
            # we get a 1d array containing the winning label's confidence value
            scoresPerPoint = predObj.get_values()
            scoresPerPoint.resize(scoresPerPoint.size, 1)

        return scoresPerPoint

    def _getScoresOrder(self, learner):
        return learner.UIgetScoreOrder


    def _findCallableBackend(self, name):
        def shogunToPython(cls):

            class WrappedShogun(cls):
                """
                Set only non-default arguments for the class.
                """
                def __init__(self, **kwargs):
                    super().__init__()

                    for name, arg in kwargs.items():
                        if not isinstance(arg, ShogunDefault):
                            setter = getattr(self, 'set_' + name)
                            setter(arg)

                    self.__name__ = cls.__name__
                    self.__doc__ = cls.__doc__

            return WrappedShogun

        callableObj = self._searcher.findInPackage(None, name)

        shogunWarn = ['LibLinearRegression', 'SGDQN']
        if name in shogunWarn:
            msg = "nimble cannot guarantee replicable results for " + name
            warnings.warn(msg)

        return shogunToPython(callableObj)


    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             arguments, customDict):
        # check something that we know won't work, but shogun will not report
        # intelligently
        if trainX is not None or testX is not None:
            if 'pointLen' not in customDict:
                if trainX is not None:
                    customDict['pointLen'] = len(trainX.features)
                else:
                    customDict['pointLen'] = len(testX.features)
            if (trainX is not None
                    and len(trainX.features) != customDict['pointLen']):
                msg = "Length of points in the training data and testing data "
                msg += "must be the same"
                raise InvalidArgumentValueCombination(msg)
            if (testX is not None
                    and len(testX.features) != customDict['pointLen']):
                msg = "Length of points in the training data and testing data "
                msg += "must be the same"
                raise InvalidArgumentValueCombination(msg)

        trainXTrans = None
        if trainX is not None:
            customDict['match'] = trainX.getTypeString()
            trainXTrans = self._inputTransDataHelper(trainX, learnerName)

        trainYTrans = None
        if trainY is not None:
            trainYTrans = self._inputTransLabelHelper(trainY, learnerName,
                                                      customDict)

        testXTrans = None
        if testX is not None:
            testXTrans = self._inputTransDataHelper(testX, learnerName)

        delkeys = []
        for key, val in arguments.items():
            if isinstance(val, ShogunDefault):
                delkeys.append(key)
        for key in delkeys:
            del arguments[key]

        instantiatedArgs = {}
        for arg, val in arguments.items():
            if isinstance(val, nimble.Init):
                allParams = self._getParameterNames(val.name)[0]
                # add trainX and/or trainY if needed for object instantiation
                for yParam in [p for p in trainYAliases if p in allParams]:
                    val.kwargs[yParam] = trainYTrans
                for xParam in [p for p in trainXAliases if p in allParams]:
                    val.kwargs[xParam] = trainXTrans
                val = self._argumentInit(val)
            instantiatedArgs[arg] = val

        return (trainXTrans, trainYTrans, testXTrans, instantiatedArgs)


    def _outputTransformation(self, learnerName, outputValue,
                              transformedInputs, outputType, outputFormat,
                              customDict):
        # often, but not always, we have to unpack a Labels object
        if isinstance(outputValue, self._access('Classifier', 'Labels')):
            # outputValue is a labels object, have to pull out the raw values
            # with a function call
            retRaw = outputValue.get_labels()
            # prep for next call
            retRaw = np.atleast_2d(retRaw)
            # we are given a column organized return, we want row first
            # organization, to match up with our input rows as points standard
            retRaw = retRaw.transpose()
        else:
            retRaw = outputValue

        outputType = customDict['match']
        ret = nimble.data(outputType, retRaw, useLog=False)

        if outputFormat == 'label' and 'remap' in customDict:
            remap = customDict['remap']
            if remap is not None:
                ret.transformElements(remap, useLog=False)

        return ret


    def _trainer(self, learnerName, trainX, trainY, arguments, randomSeed,
                 customDict):
        toCall = self.findCallable(learnerName)
        learnerDefaults = self._getDefaultValuesBackend(learnerName)[0]

        self.package.Math.init_random(randomSeed)
        # Figure out which argument values are needed for training
        setterArgs = {}
        for name, arg in arguments.items():
            # use setter for everything in learnerDefaults
            if name in learnerDefaults:
                setterArgs[name] = arg

        learner = toCall(**setterArgs)

        if trainY is not None:
            checkProcessFailure('labels', learner.set_labels, trainY)
            learner.set_labels(trainY)
        checkProcessFailure('train', learner.train, trainX)
        learner.train(trainX)

        # TODO online training prep learner.start_train()
        # batch training if data is passed
        if trainY is not None:
            labelOrder = np.unique(trainY.get_labels())
            if customDict['remap'] is None:
                learner.UIgetScoreOrder = labelOrder
            else:
                remap = customDict['remap']
                learner.UIgetScoreOrder = [remap[v] for v in labelOrder]
            customDict['numLabels'] = len(learner.UIgetScoreOrder)

        return learner


    def _incrementalTrainer(self, learnerName, learner, trainX, trainY,
                            arguments, randomSeed, customDict):
        # StreamingDotFeatures?
        raise NotImplementedError


    def _applier(self, learnerName, learner, testX, newArguments,
                 storedArguments, customDict):
        # shogun does not appear to allow apply time arguments
        ptVal = learner.get_machine_problem_type()
        if ptVal == self._access('Classifier', 'PT_CLASS'):
            ptVal = customDict['problemType']
        if ptVal == self._access('Classifier', 'PT_BINARY'):
            retFunc = learner.apply_binary
        elif ptVal == self._access('Classifier', 'PT_MULTICLASS'):
            retFunc = learner.apply_multiclass
        elif ptVal == self._access('Classifier', 'PT_REGRESSION'):
            retFunc = learner.apply_regression
        elif ptVal == self._access('Classifier', 'PT_STRUCTURED'):
            retFunc = learner.apply_structured
        elif ptVal == self._access('Classifier', 'PT_LATENT'):
            retFunc = learner.apply_latent
        else:
            retFunc = learner.apply
        checkProcessFailure('apply', retFunc, testX)
        retLabels = retFunc(testX)
        return retLabels


    def _getAttributes(self, learnerBackend):
        # check for everything start with 'get_'?
        raise NotImplementedError


    def _optionDefaults(self, option):
        return None


    def _configurableOptionNames(self):
        return ['location', 'sourceLocation', 'libclangLocation']

    ######################
    ### METHOD HELPERS ###
    ######################

    def _inputTransLabelHelper(self, labelsObj, learnerName, customDict):
        customDict['remap'] = None
        try:
            inverseMapping = None
            #if labelsObj.getTypeString() != 'Matrix':
            labelsObj = labelsObj.copy(to='Matrix')
            problemType = self._getMachineProblemType(learnerName)
            if problemType == self._access('Classifier', 'PT_CLASS'):
                # could be either binary or multiclass
                flattened = labelsObj.copy(to='numpyarray', outputAs1D=True)
                if len(np.unique(flattened)) == 2:
                    problemType = self._access('Classifier', 'PT_BINARY')
                else:
                    problemType = self._access('Classifier', 'PT_MULTICLASS')
                customDict['problemType'] = problemType
            if problemType == self._access('Classifier', 'PT_MULTICLASS'):
                inverseMapping = _remapLabels(labelsObj)
                customDict['remap'] = inverseMapping
                flattened = labelsObj.copy(to='numpyarray', outputAs1D=True)
                labeler = self._access('Features', 'MulticlassLabels')
            elif problemType == self._access('Classifier', 'PT_BINARY'):
                inverseMapping = _remapLabels(labelsObj, [-1, 1])
                customDict['remap'] = inverseMapping
                flattened = labelsObj.copy(to='numpyarray', outputAs1D=True)
                labeler = self._access('Features', 'BinaryLabels')
            elif problemType == self._access('Classifier', 'PT_REGRESSION'):
                flattened = labelsObj.copy(to='numpyarray', outputAs1D=True)
                labeler = self._access('Features', 'RegressionLabels')
            else:
                msg = "Learner problem type (" + str(problemType) + ") "
                msg += "not supported"
                raise InvalidArgumentValue(msg)
            labels = labeler(flattened.astype(float))
        except ImportError:
            from shogun.Features import Labels # pylint: disable=import-outside-toplevel

            flattened = labelsObj.copy(to='numpyarray', outputAs1D=True)
            labels = Labels(labelsObj.astype(float))

        return labels

    def _inputTransDataHelper(self, dataObj, learnerName):
        typeString = dataObj.getTypeString()
        dataObj._convertToNumericTypes(allowInt=False, allowBool=False)
        if typeString == 'Sparse':
            raw = dataObj.copy(to="scipy csc", rowsArePoints=False)
            trans = self._access('Features', 'SparseRealFeatures')()
            trans.set_sparse_feature_matrix(raw)
            if 'Online' in learnerName:
                sprf = self._access('Features', 'StreamingSparseRealFeatures')
                trans = sprf(trans)
        else:
            raw = dataObj.copy(to='numpyarray', rowsArePoints=False)
            trans = self._access('Features', 'RealFeatures')()
            trans.set_feature_matrix(raw)
            if 'Online' in learnerName:
                trans = self._access('Features', 'StreamingRealFeatures')()
        return trans

class ShogunDefault(object):
    """
    Sentinel value to take the place of default arguments in shogun.
    """
    def __init__(self, name, typeString='UNKNOWN'):
        self.name = name
        self.typeString = typeString

    def __eq__(self, other):
        if isinstance(other, ShogunDefault):
            return self.name == other.name
        return False

    def __copy__(self):
        return ShogunDefault(self.name, self.typeString)

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __str__(self):
        return "ShogunDefault({0})".format(self.name)

    def __repr__(self):
        return self.__str__()


#######################
### GENERIC HELPERS ###
#######################

excludedLearners = [
    # parent classes, not actually runnable
    'CSVM',
    'MulticlassSVM',

    # Deliberately unsupported
    'ScatterSVM',  # experimental method
    ]

def _remapLabels(toRemap, space=None):
    """
    Transform toRemap so its values are mapped into a given space.

    If space is None, the space is set to the range of n unique values,
    0 to n - 1.  Otherwise, the space must provide a value for each
    unique value in toRemp. The return is the inverse map for use when
    mapping these values back to their original values.
    """
    assert len(toRemap.features) == 1
    uniqueVals = list(toRemap.countUniqueElements().keys())
    if space is None:
        space = range(len(uniqueVals))
    remap = dict(zip(uniqueVals, space))
    if len(remap) == 1:
        msg = "Cannot train a classifier with data containing only one label"
        raise InvalidArgumentValue(msg)
    if len(uniqueVals) != len(space):
        if space == [-1, 1]:
            spaceStr = "binary"
        else:
            spaceStr = "space " + str(space)
        msg = "Cannot map label values to " + spaceStr
        raise InvalidArgumentValue(msg)
    toRemap.transformElements(remap, useLog=False)
    inverse = {mapped: orig for orig, mapped in remap.items()}
    return inverse


def catchSignals(target, args, kwargs):
    """
    Run shogun in a separate process to prevent signal failures.

    If this process exits due to a signal, this will trigger an
    exception to be raised in the parent process.
    """
    try:
        target(*args, **kwargs)
    # Exceptions are acceptable, we are only concerned with signals
    except Exception: # pylint: disable=broad-except
        pass


def checkProcessFailure(process, target, *args, **kwargs):
    """
    Determine if a call to shogun function or method will be successful.

    When data is incompatible in shogun, it may throw a signal to exit
    rather than raising an exception. These seem to occur almost
    immediately upon calling the function, so we try running the
    function in a separate process for 0.2s and raise an exception
    (SystemError for consistency with shogun) if the process exits.
    """
    allArgs = (target, args, kwargs)
    proc = multiprocessing.Process(target=catchSignals, args=allArgs)
    proc.start()
    proc.join(timeout=0.2)
    exitcode = proc.exitcode
    proc.terminate()
    if exitcode:
        msg = "shogun encountered an error while attempting "
        if process == 'labels':
            msg += "to set the labels with the provided trainY data "
        elif process == 'train':
            msg += "to train with the provided trainX data "
        elif process == 'apply':
            msg += "to apply the learner to the provided testX data "
        msg += "and exited with code " + str(abs(exitcode))
        raise SystemError(msg)
