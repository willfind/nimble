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
import multiprocessing
import re

import nimble
from nimble.interfaces.universal_interface import UniversalInterface
from nimble.interfaces.universal_interface import PredefinedInterface
from nimble.interfaces.interface_helpers import PythonSearcher
from nimble.interfaces.interface_helpers import modifyImportPathAndImport
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.docHelpers import inheritDocstringsFactory

# Interesting alias cases:
# * DomainAdaptionMulticlassLibLinear  -- or probably any nested machine
shogunDir = None

trainXAliases = ['traindat', 'f', 'features', 'feats', 'feat', 'training_data',
                 'train_features', 'data']
trainYAliases = ['trainlab', 'lab', 'labs', 'training_labels', 'train_labels']

# kernel : k, kernel
# distance : d

@inheritDocstringsFactory(UniversalInterface)
class Shogun(PredefinedInterface, UniversalInterface):
    """
    This class is an interface to shogun.
    """

    def __init__(self):

        self.shogun = modifyImportPathAndImport(shogunDir, 'shogun')
        self.versionString = None

        def isLearner(obj):
            hasTrain = hasattr(obj, 'train')
            hasApply = hasattr(obj, 'apply')

            if not (hasTrain and hasApply):
                return False

            if obj.__name__ in excludedLearners or 'Online' in obj.__name__:
                # Online TODO
                return False

            # needs more to be able to distinguish between things that are runnable
            # and partial implementations. try get_machine_problem_type()?
            try:
                instantiated = obj()
                instantiated.get_machine_problem_type()
            except Exception:
                return False

            return True

        self.hasAll = hasattr(self.shogun, '__all__')
        contents = self.shogun.__all__ if self.hasAll else dir(self.shogun)
        self._searcher = PythonSearcher(self.shogun, contents, {}, isLearner, 2)

        self.loadParameterManifest()

        super(Shogun, self).__init__()

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


    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
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
        ptVal = self._getMachineProblemType(name)

        if (ptVal == self._access('Classifier', 'PT_BINARY')
                or ptVal == self._access('Classifier', 'PT_MULTICLASS')):
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
                if paramName not in trainXAliases + trainYAliases:
                    curr.append(paramName)
            ret.append(curr)

        return ret


    def _getLearnerDefaultValuesBackend(self, name):
        allNames = self._getLearnerParameterNamesBackend(name)
        return self._setupDefaultsGivenBaseNames(name, allNames)


    def _getParameterNamesBackend(self, name):
        ret = []
        backend = self.findCallable(name)
        for funcname in dir(backend):
            if funcname.startswith('set_'):
                funcname = funcname[4:]
                ret.append(funcname)

        return [ret]

    def _getDefaultValuesBackend(self, name):
        allNames = self._getParameterNamesBackend(name)
        return self._setupDefaultsGivenBaseNames(name, allNames)

    def _setupDefaultsGivenBaseNames(self, name, allNames):
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
            if isinstance(val, ShogunDefault):
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

                ret.elements.transform(remap, useLog=False)

        return ret


    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        toCall = self.findCallable(learnerName)
        learnerDefaults = self._getDefaultValuesBackend(learnerName)[0]

        # Figure out which params have to be set using setters, instead of passed in.
        setterArgs = {}
        for name, arg in arguments.items():
            # use setter for everything in learnerDefaults
            if name in learnerDefaults:
                setterArgs[name] = arg

        learner = toCall()

        for name, arg in setterArgs.items():
            setter = getattr(learner, 'set_' + name)
            setter(arg)

        if trainY is not None:
            runSuccessful(learner.set_labels, trainY)
            learner.set_labels(trainY)
        runSuccessful(learner.train, trainX)
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
        ptVal = learner.get_machine_problem_type()
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
        runSuccessful(retFunc, testX)
        retLabels = retFunc(testX)
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
                inverseMapping = _remapLabels(labelsObj)
                customDict['remap'] = inverseMapping
                flattened = labelsObj.copy(to='numpyarray', outputAs1D=True)
                labels = self._access('Features', 'MulticlassLabels')(flattened.astype(float))
            elif problemType == self._access('Classifier', 'PT_BINARY'):
                inverseMapping = _remapLabels(labelsObj, [-1, 1])
                customDict['remap'] = inverseMapping
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

class ShogunDefault(object):
    def __init__(self, name, typeString='UNKNOWN'):
        self.name = name
        self.typeString = typeString

    def __eq__(self, other):
        if isinstance(other, ShogunDefault):
            return self.name == other.name and self.typeString == other.typeString
        return False

    def __copy__(self):
        return ShogunDefault(self.name, self.typeString)

    def __deepcopy(self):
        return self.__copy__()

    def __str__(self):
        return "ShogunDefault({0})".format(self.name)

    def __repr__(self):
        return self.__str__()


#######################
### GENERIC HELPERS ###
#######################

excludedLearners = [ # parent classes, not actually runnable
                    'BaseMulticlassMachine',
                    'CDistanceMachine',
                    'CSVM',
                    'KernelMachine',
                    'KernelMulticlassMachine',
                    'KMeansBase',
                    'LinearLatentMachine',
                    'LinearMachine',
                    'LinearMulticlassMachine',
                    'MKL',
                    'Machine',
                    'MulticlassMachine',
                    'MulticlassSVM',
                    'MultitaskLinearMachineBase',
                    'NativeMulticlassMachine',
                    'OnlineLinearMachine',
                    'TreeMachineWithC45TreeNodeData',
                    'TreeMachineWithCARTreeNodeData',
                    'TreeMachineWithCHAIDTreeNodeData',
                    'TreeMachineWithID3TreeNodeData',
                    'TreeMachineWithConditionalProbabilityTreeNodeData',
                    'TreeMachineWithRelaxedTreeNodeData',

                    # Deliberately unsupported
                    'ScatterSVM',  # experimental method
                    ]


def _enforceNonUnicodeStrings(manifest):
    for name in manifest:
        groupList = manifest[name]
        for group in groupList:
            for i in range(len(group)):
                group[i] = str(group[i])
    return manifest

def _remapLabels(toRemap, space=None):
    """
    Transform toRemap so its values are mapped into a given space.

    If space is None, the space is set to the range of n unique values,
    0 to n - 1.  Otherwise, the space must provide a value for each
    unique value in toRemp. The return is the inverse map for use when
    mapping these values back to their original values.
    """
    assert len(toRemap.features) == 1
    uniqueVals = list(toRemap.elements.countUnique().keys())
    if space is None:
        space = range(len(uniqueVals))
    remap = {orig: mapped for orig, mapped in zip(uniqueVals, space)}
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
    toRemap.elements.transform(remap, useLog=False)
    inverse = {mapped: orig for orig, mapped in remap.items()}
    return inverse


def catchSignals(conn, target, args, kwargs):
    """
    Run Shogun in a separate process to prevent signal failures.

    If this process exits due to a signal, nothing will be sent to the
    parent conn indicating that this process failed to run sucessfully.
    Exceptions are caught and sent to the parent conn to allow the
    exception to be raised outside this process.
    """
    try:
        target(*args, **kwargs)
        conn.send(True)
    # Process will catch exception, send through conn to raise later
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()


def runSuccessful(target, *args, **kwargs):
    """
    Determine if a call to shogun function or method will be successful.

    The function runs as a separate process through a child connection.
    The process can either exit or run successfully. If successful, the
    parent conn may be True (we can run the function) or an Exception to
    be raised. If unsuccessful, we cannot be sure why the process exited
    so a SystemError is raised.
    """
    pConn, cConn = multiprocessing.Pipe()
    allArgs = (cConn, target, args, kwargs)
    p = multiprocessing.Process(target=catchSignals, args=allArgs)
    p.start()
    p.join()
    if pConn.poll():
        ret = pConn.recv()
        if isinstance(ret, Exception):
            raise ret
        return
    raise SystemError("shogun encountered an unidentifiable error")
