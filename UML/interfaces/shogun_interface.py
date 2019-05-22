"""
A nimble interface building off of the modular python interface
for Shogun ML.

"""

# TODO?
# * online learning
# * different feature types (streaming, for other problem types)
# *

from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
try:
    import clang
    import clang.cindex

    clangAvailable = True
except ImportError:
    clangAvailable = False

import importlib
import numpy
import copy
import sys
import os
import json
import distutils.version

import UML as nimble
from UML.interfaces.universal_interface import UniversalInterface
from UML.interfaces.interface_helpers import PythonSearcher
from UML.exceptions import InvalidArgumentValue
from UML.exceptions import InvalidArgumentValueCombination
from UML.docHelpers import inheritDocstringsFactory

# Interesting alias cases:
# * DomainAdaptionMulticlassLibLinear  -- or probably any nested machine


trainXAliases = ['traindat', 'f', 'features', 'feats', 'feat', 'training_data', 'train_features', 'data']
trainYAliases = ['trainlab', 'lab', 'labs', 'training_labels', 'train_labels']

# kernel : k, kernel
# distance : d

@inheritDocstringsFactory(UniversalInterface)
class Shogun(UniversalInterface):
    """
    This class is an interface to shogun.
    """

    def __init__(self):
        super(Shogun, self).__init__()

        self.shogun = importlib.import_module('shogun')
        self.versionString = None

        def isLearner(obj):
            hasTrain = hasattr(obj, 'train')
            hasApply = hasattr(obj, 'apply')

            if not (hasTrain and hasApply):
                return False

            if obj.__name__ in excludedLearners:
                return False

            # needs more to be able to distinguish between things that are runnable
            # and partial implementations. try get_machine_problem_type()?
            try:
                instantiated = obj()
            except TypeError:
                # if we can't even instantiate it, then its not a usable class
                return False
            try:
                instantiated.get_machine_problem_type()
            except Exception:
                return False

            #				if name.startswith("get_"):
            #					try:
            #						getattr(instantiated, name)()
            #					except Exception:
            #						return False

            return True

        self.hasAll = hasattr(self.shogun, '__all__')
        contents = self.shogun.__all__ if self.hasAll else dir(self.shogun)
        self._searcher = PythonSearcher(self.shogun, contents, {}, isLearner, 2)

        self.loadParameterManifest()

    def _access(self, module, target):
        """
        Helper to automatically search the correct locations for target objects
        in different historical versions of shogun that have different module
        structures. Due to the automated component that attempts to smooth
        over multiple competing versions of the package, this should be
        used with caution.

        """
        # If shogun has an __all__ attribute, it is in the old style of organization,
        # where things are separated into submodules. They need to be loaded before
        # access.
        if self.hasAll:
            if hasattr(self.shogun, module):
                submod = getattr(self.shogun, module)
            else:
                submod = importlib.import_module('shogun.' + module)
        # If there is no __all__ attribute, we're in the newer, flatter style of module
        # organization. Some things will still be in the submodule they were previously,
        # others will up at the top level. This will check both locations
        else:
            if hasattr(self.shogun, target):
                submod = self.shogun
            else:
                submod = getattr(self.shogun, module)

        return getattr(submod, target)

    def accessible(self):
        try:
            import shogun
        except ImportError:
            return False
        return True


    def _listLearnersBackend(self):
        return self._searcher.allLearners()


    def _getMachineProblemType(self, learnerName):
        """
        SystemError
        """
        learner = self.findCallable(learnerName)

        try:
            learner = learner()
        except TypeError:
            pass

        ptVal = learner.get_machine_problem_type()
        return ptVal


    def learnerType(self, name):
        try:
            ptVal = self._getMachineProblemType(name)
        except SystemError:
            return 'UNKNOWN'

        if ptVal == self._access('Classifier', 'PT_BINARY') or ptVal == self._access('Classifier', 'PT_MULTICLASS'):
            return 'classification'
        if ptVal == self._access('Classifier', 'PT_REGRESSION'):
            return 'regression'
        if ptVal == self._access('Classifier', 'PT_STRUCTURED'):
            return 'UNKNOWN'
        if ptVal == self._access('Classifier', 'PT_LATENT'):
            return 'UNKNOWN'

        # TODO warning, unknown problem type code

        return 'UNKNOWN'


    def _getLearnerParameterNamesBackend(self, name):
        base = self._getParameterNamesBackend(name)

        #remove aliases
        ret = []
        for group in base:
            curr = []
            for paramName in group:
                if paramName not in trainXAliases and paramName not in trainYAliases:
                    curr.append(paramName)
            ret.append(curr)

        return ret


    def _getLearnerDefaultValuesBackend(self, name):
        allNames = self._getLearnerParameterNamesBackend(name)
        return self._setupDefaultsGivenBaseNames(name, allNames)


    def _getParameterNamesBackend(self, name):
        query = self._queryParamManifest(name)
        base = query if query is not None else [[]]

        #		return base
        ret = []
        for group in base:
            backend = self.findCallable(name)
            params = group
            for funcname in dir(backend):
                if funcname.startswith('set_'):
                    funcname = funcname[4:]
                    if funcname not in params:
                        params.append(funcname)
            ret.append(params)

        return ret

    def _getDefaultValuesBackend(self, name):
        allNames = self._getParameterNamesBackend(name)
        return self._setupDefaultsGivenBaseNames(name, allNames)

    def _setupDefaultsGivenBaseNames(self, name, allNames):
        allValues = []
        for index in range(len(allNames)):
            group = allNames[index]
            curr = {}
            for paramName in group:
                curr[paramName] = self.HiddenDefault(paramName)
            allValues.append(curr)

        ret = []
        query = self._queryParamManifest(name)
        if query is not None:
            base = query
        else:
            base = []
            for i in range(len(allValues)):
                base.append([])
        for index in range(len(base)):
            group = base[index]
            curr = allValues[index]
            for paramName in group:
                if paramName in curr:
                    del curr[paramName]
            ret.append(curr)

        return ret

    def _getScores(self, learnerName, learner, testX, newArguments,
                   storedArguments, customDict):
        # TODO deal with merging stored vs new arguments
        # is this even a thing for shogun?
        predObj = self._applier(learnerName, learner, testX, newArguments,
                                storedArguments, customDict)
        predLabels = predObj.get_labels()
        numLabels = customDict['numLabels']
        if hasattr(predObj, 'get_multiclass_confidences'):
            # setup an array in the right shape, number of predicted labels by number of possible labels
            scoresPerPoint = numpy.empty((len(predLabels), numLabels))
            for i in range(len(predLabels)):
                currConfidences = predObj.get_multiclass_confidences(i)
                if len(currConfidences) == 0:
                    msg = "The shogun learner %s doesn't provide confidence scores" % str(learner)
                    raise NotImplementedError(msg)
                scoresPerPoint[i, :] = currConfidences
        # otherwise we must be dealing with binary classification
        else:
            # we get a 1d array containing the winning label's confidence value
            scoresPerPoint = predObj.get_values()
            scoresPerPoint.resize(scoresPerPoint.size, 1)

        return scoresPerPoint

    def _getScoresOrder(self, learner):
        return learner.get_unique_labels

    def isAlias(self, name):
        if name.lower() in ['shogun']:
            return True
        else:
            return False

    def getCanonicalName(self):
        return "shogun"

    def _findCallableBackend(self, name):
        return self._searcher.findInPackage(None, name)


    def _inputTransformation(self, learnerName, trainX, trainY, testX, arguments, customDict):
        # check something that we know won't work, but shogun will not report intelligently
        if trainX is not None or testX is not None:
            if 'pointLen' not in customDict:
                customDict['pointLen'] = len(trainX.features) if trainX is not None else len(testX.features)
            if trainX is not None and len(trainX.features) != customDict['pointLen']:
                msg = "Length of points in the training data and testing data must be the same"
                raise InvalidArgumentValueCombination(msg)
            if testX is not None and len(testX.features) != customDict['pointLen']:
                msg = "Length of points in the training data and testing data must be the same"
                raise InvalidArgumentValueCombination(msg)

        trainXTrans = None
        if trainX is not None:
            customDict['match'] = trainX.getTypeString()
            trainXTrans = self._inputTransDataHelper(trainX, learnerName)

        trainYTrans = None
        if trainY is not None:
            trainYTrans = self._inputTransLabelHelper(trainY, learnerName, customDict)

        testXTrans = None
        if testX is not None:
            testXTrans = self._inputTransDataHelper(testX, learnerName)

        delkeys = []
        for key in arguments:
            val = arguments[key]
            if isinstance(val, self.HiddenDefault):
                delkeys.append(key)
        for key in delkeys:
            del arguments[key]

        # TODO copiedArguments = copy.deepcopy(arguments) ?
        # copiedArguments = copy.copy(arguments)
        copiedArguments = arguments

        return (trainXTrans, trainYTrans, testXTrans, copiedArguments)


    def _outputTransformation(self, learnerName, outputValue, transformedInputs, outputType, outputFormat, customDict):
        # often, but not always, we have to unpack a Labels object
        if isinstance(outputValue, self._access('Classifier', 'Labels')):
            # outputValue is a labels object, have to pull out the raw values with a function call
            retRaw = outputValue.get_labels()
            # prep for next call
            retRaw = numpy.atleast_2d(retRaw)
            # we are given a column organized return, we want row first organization, to match up
            # with our input rows as points standard
            retRaw = retRaw.transpose()
        else:
            retRaw = outputValue

        outputType = 'Matrix'
        if outputType == 'match':
            outputType = customDict['match']
        ret = nimble.createData(outputType, retRaw, useLog=False)

        if outputFormat == 'label':
            remap = customDict['remap']
            if remap is not None:
                def makeInverseMapper(inverseMappingParam):
                    def inverseMapper(value):
                        return inverseMappingParam[int(value)]

                    return inverseMapper

                ret.elements.transform(makeInverseMapper(remap), features=0,
                                       useLog=False)

        return ret


    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        toCall = self.findCallable(learnerName)

        # Figure out which, if any, aliases for trainX or trainY were ignored
        # during argument validation and instantiation
        #		setX = True
        #		setY = True
        learnerParams = self._getLearnerParameterNamesBackend(learnerName)
        rawParams = self._getParameterNamesBackend(learnerName)
        learnerDefaults = self._getLearnerDefaultValuesBackend(learnerName)
        rawDefaults = self._getDefaultValuesBackend(learnerName)
        bestIndex = self._chooseBestParameterSet(learnerParams, learnerDefaults, arguments)
        diffNames = list(set(rawParams[bestIndex]) - set(learnerParams[bestIndex]))

        # Figure out which params have to be set using setters, instead of passed in.
        setterArgs = {}
        initArgs = {}
        kernels = []
        for name in arguments:
            if name in learnerDefaults[bestIndex]:
                setterArgs[name] = arguments[name]
            else:
                initArgs[name] = arguments[name]

            if isinstance(arguments[name], self._access('Classifier', 'Kernel')):
                kernels.append(name)

        # if we've ignored aliased names (as demonstrated by the difference between
        # the raw parameter name list and the learner parameter name list) then this
        # is where we have to add them back in.
        for name in diffNames:
            if name in trainXAliases:
                if name in rawDefaults[bestIndex]:
                    pass
                #setterArgs[name] = trainX  TODO -- do we actually want this?
                else:
                    initArgs[name] = trainX
            if name in trainYAliases:
                if name in rawDefaults[bestIndex]:
                    pass
                #setterArgs[name] = trainY  TODO -- do we actually want this?
                else:
                    initArgs[name] = trainY

        # it may be the case that the nimble passed the args needed to initialize
        # a kernel. If not, it isn't useable until we do it ourselves
        for name in kernels:
            currKern = arguments[name]
            if not currKern.has_features():
                currKern.init(trainX, trainX)

        # actually pack args for init. c++ backend with a thin python layer means
        # the crucial information is where in a list the values are, NOT the associated
        # name.
        initArgsList = []
        for name in rawParams[bestIndex]:
            if name not in rawDefaults[bestIndex]:
                initArgsList.append(initArgs[name])
        learner = toCall(*initArgsList)

        for name in setterArgs:
            setter = getattr(learner, 'set_' + name)
            setter(setterArgs[name])

        if trainY is not None:
            learner.set_labels(trainY)

        learner.train(trainX)

        # TODO online training prep learner.start_train()
        # batch training if data is passed

        learner.get_unique_labels = numpy.unique(trainY.get_labels())
        customDict['numLabels'] = len(learner.get_unique_labels)

        return learner


    def _incrementalTrainer(self, learner, trainX, trainY, arguments, customDict):
        # StreamingDotFeatures?
        raise NotImplementedError


    def _applier(self, learnerName, learner, testX, newArguments,
                 storedArguments, customDict):
        # TODO does shogun allow apply time arguments?
        try:
            ptVal = learner.get_machine_problem_type()
            if ptVal == self._access('Classifier', 'PT_BINARY'):
                retLabels = learner.apply_binary(testX)
            elif ptVal == self._access('Classifier', 'PT_MULTICLASS'):
                retLabels = learner.apply_multiclass(testX)
            elif ptVal == self._access('Classifier', 'PT_REGRESSION'):
                retLabels = learner.apply_regression(testX)
            elif ptVal == self._access('Classifier', 'PT_STRUCTURED'):
                retLabels = learner.apply_structured(testX)
            elif ptVal == self._access('Classifier', 'PT_LATENT'):
                retLabels = learner.apply_latent(testX)
            else:
                retLabels = learner.apply(testX)
        except Exception as e:
            print(e)
            return None
        return retLabels


    def _getAttributes(self, learnerBackend):
        # check for everything start with 'get_'?
        raise NotImplementedError


    def _optionDefaults(self, option):
        return None


    def _configurableOptionNames(self):
        return ['location', 'sourceLocation', 'libclangLocation']


    def _exposedFunctions(self):

        return []

    def version(self):
        if self.versionString is None:
            self.versionString = self._access('Version', 'get_version_release')()

        return self.versionString

    ######################
    ### METHOD HELPERS ###
    ######################

    def loadParameterManifest(self):
        """
        Load manifest containing parameter names and defaults for all relevant objects
        in shogun. If no manifest is available, then instantiate to an empty object.

        """
        # find most likely manifest file
        metadataPath = os.path.join(nimble.nimblePath, 'interfaces', 'metadata')
        best, exact = self._findBestManifest(metadataPath)
        exists = os.path.exists(best) if best is not None else False
        # default to empty if no best manifest exists
        self._paramsManifest = {}
        if exists:
            with open(best, 'r') as fp:
                self._paramsManifest = json.load(fp, object_hook=_enforceNonUnicodeStrings)


    def _findBestManifest(self, metadataPath):
        """
        Returns a double. The first value is the absolute path to the manifest
        file that is the closest match for this version of shogun, or None if
        there is no such file. The second value is a boolean stating whether
        the first value is an exact match for the desired version.

        """
        def _getSignificantVersion(versionString):
            return distutils.version.LooseVersion(versionString.split('_')[0]).version

        ourVersion = _getSignificantVersion(self.version())

        possible = os.listdir(metadataPath)
        if len(possible) == 0:
            return None, False

        ours = (ourVersion, None, None)
        toSort = [ours]
        for name in possible:
            if name.startswith("shogunParameterManifest"):
                pieces = name.split('_')
                currVersion = _getSignificantVersion(pieces[1])
                toSort.append((currVersion, name, ))
        if len(toSort) == 1:
            return None, False

        sortedPairs = sorted(toSort, key=(lambda p: p[0]))
        ourIndex = sortedPairs.index(ours)

        # If what we're looking for is present, it has to be next to
        # our seeded version at ourIndex
        left, right = None, None
        if ourIndex != 0:
            left = sortedPairs[ourIndex - 1]
        if ourIndex != len(sortedPairs) - 1:
            right = sortedPairs[ourIndex + 1]

        best = None
        if left is None:
            best = right
        elif right is None:
            best = left
        # at least one of them must be non-None, so this must mean both
        # are non None
        else:
            best = left[1]
            for index in range(len(ourVersion)):
                currOurs = ourVersion[index]
                currL = left[0][index]
                currR = right[0][index]

                if currL == currOurs and currR != currOurs:
                    best = left
                    break
                if currL != currOurs and currR == currOurs:
                    best = right
                    break

        return os.path.join(metadataPath, best[1]), ourVersion == best[0]


    def _inputTransLabelHelper(self, labelsObj, learnerName, customDict):
        customDict['remap'] = None
        try:
            inverseMapping = None
            #if labelsObj.getTypeString() != 'Matrix':
            labelsObj = labelsObj.copy(to='Matrix')
            problemType = self._getMachineProblemType(learnerName)
            if problemType == self._access('Classifier', 'PT_MULTICLASS'):
                inverseMapping = _remapLabelsRange(labelsObj)
                customDict['remap'] = inverseMapping
                if len(inverseMapping) == 1:
                    raise InvalidArgumentValue("Cannot train a classifier with data containing only one label")
                flattened = labelsObj.copy(to='numpyarray', outputAs1D=True)
                labels = self._access('Features', 'MulticlassLabels')(flattened.astype(float))
            elif problemType == self._access('Classifier', 'PT_BINARY'):
                inverseMapping = _remapLabelsSpecific(labelsObj, [-1, 1])
                customDict['remap'] = inverseMapping
                if len(inverseMapping) == 1:
                    raise InvalidArgumentValue("Cannot train a classifier with data containing only one label")
                flattened = labelsObj.copy(to='numpyarray', outputAs1D=True)
                labels = self._access('Features', 'BinaryLabels')(flattened.astype(float))
            elif problemType == self._access('Classifier', 'PT_REGRESSION'):
                flattened = labelsObj.copy(to='numpyarray', outputAs1D=True)
                labels = self._access('Features', 'RegressionLabels')(flattened.astype(float))
            else:
                raise InvalidArgumentValue("Learner problem type (" + str(problemType) + ") not supported")
        except ImportError:
            from shogun.Features import Labels

            flattened = labelsObj.copy(to='numpyarray', outputAs1D=True)
            labels = Labels(labelsObj.astype(float))

        return labels

    def _inputTransDataHelper(self, dataObj, learnerName):
        typeString = dataObj.getTypeString()
        if typeString == 'Sparse':
            #raw = dataObj.data.tocsc().astype(numpy.float)
            #raw = raw.transpose()
            raw = dataObj.copy(to="scipy csc", rowsArePoints=False)
            trans = self._access('Features', 'SparseRealFeatures')()
            trans.set_sparse_feature_matrix(raw)
            if 'Online' in learnerName:
                trans = self._access('Features', 'StreamingSparseRealFeatures')(trans)
        else:
            #raw = dataObj.copy(to='numpyarray').astype(numpy.float)
            #raw = raw.transpose()
            raw = dataObj.copy(to='numpyarray', rowsArePoints=False)
            trans = self._access('Features', 'RealFeatures')()
            trans.set_feature_matrix(raw)
            if 'Online' in learnerName:
                trans = self._access('Features', 'StreamingRealFeatures')()
        return trans

    def _queryParamManifest(self, name):
        """
        Checks the param manifest for an entry associated with the given name.
        Returns a list of list of parameter names if an entry is found, None
        otherwise. The parameter manifest is the raw output of constructor parsing,
        so there are some idiosyncrasies in the naming that this helper
        navigates
        """
        ret = None
        # exactly correct key
        if name in self._paramsManifest:
            ret = copy.deepcopy(self._paramsManifest[name])
        # some objects have names which start with a capital C and then
        # are followed by the name they would have in the documentation
        if 'C' + name in self._paramsManifest:
            ret = copy.deepcopy(self._paramsManifest['C' + name])

        return ret

    class HiddenDefault(object):
        def __init__(self, name, typeString='UNKNOWN'):
            self.name = name
            self.typeString = typeString

        def __eq__(self, other):
            if isinstance(other, Shogun.HiddenDefault):
                return self.name == other.name and self.typeString == other.typeString
            return False

        def __copy__(self):
            return Shogun.HiddenDefault(self.name, self.typeString)

        def __deepcopy(self):
            return self.__copy__()

#######################
### GENERIC HELPERS ###
#######################

excludedLearners = [  # parent classes, not actually runnable
                    'BaseMulticlassMachine',
                    'CDistanceMachine',
                    'CSVM',
                    'KernelMachine',
                    'KernelMulticlassMachine',
                    'LinearLatentMachine',
                    'LinearMachine',
                    'MKL',
                    'Machine',
                    'MulticlassMachine',
                    'MulticlassSVM',
                    'MultitaskLinearMachineBase',
                    'NativeMulticlassMachine',
                    'OnlineLinearMachine',
                    'TreeMachineWithConditionalProbabilityTreeNodeData',
                    'TreeMachineWithRelaxedTreeNodeData',

                    # Deliberately unsupported
                    'ScatterSVM',  # experimental method

                    # Should be implemented, but don't work
                    #'BalancedConditionalProbabilityTree',  # streaming dense features input
                    #'ConditionalProbabilityTree', 	 # requires streaming features
                    'DomainAdaptationSVMLinear',  # segfault
                    'DomainAdaptationMulticlassLibLinear',  # segFault
                    'DomainAdaptationSVM',
                    #'DualLibQPBMSOSVM',  # problem type 3
                    'FeatureBlockLogisticRegression',  # remapping
                    'GaussianProcessRegression',  # segfault in testDataIntegrity
                    'KernelRidgeRegression',  # segfault
                    #'KernelStructuredOutputMachine',  # problem type 3
                    'KRRNystrom',  # segfault on train - strict kern on init requirement?
                    #'LatentSVM',  # problem type 4
                    'LibLinearRegression',
                    #'LibSVMOneClass',
                    #'LinearMulticlassMachine',  # mixes machines. is this even possible to run?
                    #'LinearStructuredOutputMachine',  # problem type 3
                    #'MKLMulticlass',  # needs combined kernel type?
                    #'MKLClassification',  # compute by subkernel not implemented
                    #'MKLOneClass',  # Interleaved MKL optimization is currently only supported with SVMlight
                    #'MKLRegression',  # kernel stuff?
                    'MultitaskClusteredLogisticRegression',  # assertion error
                    'MultitaskCompositeMachine',  # takes machine as input?
                    #'MultitaskL12LogisticRegression',  # assertion error
                    'MultitaskLeastSquaresRegression',  # core dump
                    'MultitaskLogisticRegression',  # core dump
                    #'MultitaskTraceLogisticRegression',  # assertion error
                    'OnlineLibLinear',  # needs streaming dot features
                    'OnlineSVMSGD',  # needs streaming dot features
                    #'PluginEstimate',  # takes string inputs?
                    #'RandomConditionalProbabilityTree',  # takes streaming dense features
                    #'RelaxedTree',  # [ERROR] Call set_machine_for_confusion_matrix before training
                    #'ShareBoost',  # non standard input
                    #'StructuredOutputMachine',  # problem type 3
                    #'SubGradientSVM',  #doesn't terminate
                    'VowpalWabbit',  # segfault
                    #'WDSVMOcas',  # string input

                    # functioning learners
                    #'AveragedPerceptron'
                    'GaussianNaiveBayes',  # something wonky with getting scores
                    #'GMNPSVM',
                    #'GNPPSVM',
                    #'GPBTSVM',
                    #'Hierarchical',
                    #'KMeans',
                    #'KNN',
                    #'LaRank',
                    #'LibLinear',
                    #'LibSVM',
                    #'LibSVR',
                    #'MPDSVM',
                    #'MulticlassLibLinear',
                    #'MulticlassLibSVM',
                    'NewtonSVM',
                    #'Perceptron',
                    #'SGDQN',
                    #'SVMLight',
                    #'SVMLightOneClass',
                    #'SVMLin',
                    #'SVMOcas',
                    #'SVMSGD',
                    #'SVRLight',
]


def _enforceNonUnicodeStrings(manifest):
    for name in manifest:
        groupList = manifest[name]
        for group in groupList:
            for i in range(len(group)):
                group[i] = str(group[i])
    return manifest


def _remapLabelsRange(toRemap):
    """
    Transform toRemap so its n unique values are mapped into the value range 0 to n-1

    toRemap: must be a nimble Base object with a single feature. The contained values
    will be transformed, on a first come first serve basis, into the values 0 to n-1
    where n is the number of unique values. This object is modified by this function.

    Returns: list encoding the inverse mapping, where the value at index i was the
    value originally in toRemap that was replaced with the value i.

    """
    assert len(toRemap.features) == 1
    assert len(toRemap.features) > 0

    mapping = {}
    inverse = []

    def remap(fView):
        invIndex = 0
        ret = []
        for value in fView:
            if value not in mapping:
                mapping[value] = invIndex
                inverse.append(value)
                invIndex += 1
            ret.append(mapping[value])
        return ret

    toRemap.features.transform(remap, useLog=False)

    return inverse


def _remapLabelsSpecific(toRemap, space):
    """
    Transform toRemap so its values are mapped into the provided space

    toRemap: must be a data representation with a single feature containing as many
    unique values as the length of the list-typed parameter named 'space'.
    The contained values will be transformed, on a first come first serve basis, into
    the values contained in space. This object is modified by this function.

    space: list containing the values you want to be contained by toRemap.

    Returns: dict encoding the inverse mapping where the value at i was the
    value originally in toRemap that was replaced with the value i. The only valid
    keys in the return will be those also present in the paramater space

    Raises: InvalidArgumentValue if there are more than unique values than values in space

    """
    assert len(toRemap.points) > 0
    assert len(toRemap.features) == 1

    mapping = {}
    inverse = []
    invIndex = 0
    maxLength = len(space)

    for value in toRemap:
        if value not in mapping:
            mapping[value] = invIndex
            inverse.append(value)
            invIndex += 1
            if invIndex > maxLength:
                if space == [-1, 1]:
                    msg = "Multiclass training data cannot be used by a binary-only classifier"
                    raise InvalidArgumentValue(msg)
                else:
                    msg = "toRemap contains more values than can be mapped into the provided space."
                    raise InvalidArgumentValue(msg)

    def remap(fView):
        ret = []
        for value in fView:
            ret.append(space[mapping[value]])
        return ret

    toRemap.features.transform(remap, useLog=False)

    return inverse
