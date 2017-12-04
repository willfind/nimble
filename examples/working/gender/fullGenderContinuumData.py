"""
Example script using a sparse, incomplete dataset defining personality
temperment responses to do gender classification.

"""

from __future__ import absolute_import
from __future__ import print_function
from .allowImports import boilerplate
from six.moves import range
boilerplate()

import os
import sys
import time
import numpy

import UML
from UML.calculate import fractionIncorrect
from UML.customLearners import CustomLearner
from UML.examples.working.logisticRegressionCoefficientSelection import LogisticRegressionSelectByRegularization
from UML.examples.working.logisticRegressionCoefficientSelection import LogisticRegressionSelectByOmission

scipy = UML.importModule('scipy')
decomposition = UML.importModule('sklearn.decomposition')

missingPath = "/home/tpburns/gimbel_tech/python_workspace/missingValues"
sys.path.append(missingPath)
missing = UML.importModule("missing")


class MissingAwareSVD(CustomLearner):
    """
    Singular Value Decomposition calculated using missing-aware covariance.

    """

    learnerType = "dimensionalityreduction"

    def __init__(self):
        pass

    def train(self, trainX, trainY, k):
        if k < 0 or k > trainX.features:
            msg = "k must be greater than 0 and less than or equal to "
            msg += "the number of features."
            raise ValueError(msg)

        # in the correct format for missing code, features are now rows
        csrFormated = trainX.copyAs("scipycsr", rowsArePoints=False)

        covMatrix = missing.covarianceOfRows(csrFormated)
        
        (w, vr) = scipy.linalg.eig(covMatrix)

        # we want to find the largest eigen values so we know which
        # eigen vectors capture the most variance
        eigSortIndices = numpy.argsort(w)

        # sort the eigen vectors in ascending order of how much variance
        # they capture
        sortedEigVals = vr[:,eigSortIndices]

        # we have an n x n matrix, we want to grab the n x k portion
        # that is furtherest along the second axis since w was sorted
        # in ascending order
        toApplyRaw = sortedEigVals[:, vr.shape[1]-k:]

        self.toApply = UML.createData("Matrix", toApplyRaw)

    def apply(self, testX):
        return testX * self.toApply


def test_MASVD_vs_skl_noMissing():
    return
    UML.registerCustomLearner("Custom", MissingAwareSVD)

    raw = numpy.random.rand(10,4)

    data = UML.createData("Sparse", raw)
    tl = UML.train("Custom.MissingAwareSVD", trainX=data, trainY=None, k=2)
    transUML = tl.apply(data)
    transUML = transUML.copyAs("numpyarray")

    sklEst = decomposition.PCA(n_components=2)
    transSKL = sklEst.fit_transform(raw)
#   print sklEst.components_

    print(transUML)
    print(transSKL)

    numpy.testing.assert_array_almost_equal(transUML, transSKL)
#   assert False


def fileLoadHelper(retType, defaultFile, allowRemLess50Load):
    if len(sys.argv) <= 1:
        if os.path.exists(defaultFile + '_nz50Rem.mtx') and allowRemLess50Load:
            defaultFile += '_nz50Rem.mtx'
        elif os.path.exists(defaultFile + '.mtx'):
            defaultFile += '.mtx'
        elif os.path.exists(defaultFile + '.csv'):
            defaultFile += '.csv'
        else:
            raise RuntimeError("Cannot find file")
        inName = defaultFile
#       print "Must provide an argument defining the path to the input file"
#       exit(0)
    else:
        inName = sys.argv[1]

    print(time.asctime(time.localtime()))
    print("loading: " + inName)

    if inName.endswith('csv'):
        allData = loadTransformSave(inName)
        allData = allData.copyAs(retType)
    else:
        allData = UML.createData(retType, inName)

    print(time.asctime(time.localtime()))
    print("load complete")

    # remove hard points if needed
    if not inName.endswith('_nz50Rem.mtx'):
        print("Removing points with <50 non-zero entries")
        preprocess_RemoveLowNZ(allData)
        print(time.asctime(time.localtime()))
        print("removal complete")

    return allData


def removeDemographics(obj):
    """
    From the given object, remove those features that are demographic
    information, not personality response questions. The only exception
    being gender information, which we are predicting for.

    """
    removed = obj.extractFeatures(start=3, end=20)
    print(removed.getFeatureNames())

    removed = obj.extractFeatures(end=1)
    print(removed.getFeatureNames())

def replaceNAs(obj):
    """
    Given a List object that contains values of the string "NA"
    replace all such instances with the integer 0

    """
    def NAtoZero(value):
        if value == "NA":
            return 0
        else:
            return value

    obj.transformEachElement(NAtoZero)

def keepFeaturesNoDemographics():
    """
    Generate a list of those features which are not demographic,
    to be used in the keepFeatures argument of a createData call.

    """
    allIDs = list(range(716))
    results = []
    for val in allIDs:
        if val not in [0,1] and val not in range(3,21):
            results.append(val)

    return results


def batchCreateFromOrigFile(name):
    """
    Do a 5 batch load of the given file path, returning the combined
    object that has had its "NA" values replaced with zeros and
    its gender feature transformed to numerical values

    """
    kf = keepFeaturesNoDemographics()

    batch1 = UML.createData(
        "List", name, featureNames=True, keepPoints=range(5000),
        keepFeatures=kf)

    batch2 = UML.createData(
        "List", name, featureNames=True, keepPoints=range(5000,10000),
        keepFeatures=kf)

    batch3 = UML.createData(
        "List", name, featureNames=True, keepPoints=range(10000,15000),
        keepFeatures=kf)

    batch4 = UML.createData(
        "List", name, featureNames=True, keepPoints=range(15000,20000),
        keepFeatures=kf)

    batch5 = UML.createData(
        "List", name, featureNames=True, keepPoints=range(20000,23681),
        keepFeatures=kf)

    batches = [batch1, batch2, batch3, batch4, batch5]
    trainY = batch1.extractFeatures(0)

    print("batches loaded")

    for index, obj in enumerate(batches):
        if index > 0:
            trainY.appendPoints(obj.extractFeatures(0))
        assert "gender" not in obj.getFeatureNames()
        replaceNAs(obj)
        batches[index] = obj.copyAs("List")

    print(batches[0].getTypeString())
    print(batches[1].getTypeString())
    print(batches[2].getTypeString())
    print(batches[3].getTypeString())
    print(batches[4].getTypeString())

    print("batches NA replaced")

    dataAll = batches[0]
    dataAll.appendPoints(batches[1])
    dataAll.appendPoints(batches[2])
    dataAll.appendPoints(batches[3])
    dataAll.appendPoints(batches[4])

    print("batches combined")

#   trainY.transformFeatureToIntegers(0)
#   print "transformed trainY"

    makeGenderIntegers(trainY)
    trainY = trainY.copyAs("List")
    print("transformed trainY to integers")
    print(trainY.getFeatureNames())

    dataAll.appendFeatures(trainY)

    return dataAll

def makeGenderIntegers(obj):
    """
    In the given object transform the feature named 'gender'
    into numerical values.

    """
    def maleToOne(point):
        if point[0] == "male":
            return 1
        else:
            return 0
    obj.transformEachPoint(maleToOne)

def featNormalize_stdScore_applyTo(train, test):
    vals = []

    def fn(feat):
        workspace = numpy.array(feat)
        mn = numpy.mean(workspace)
        std = numpy.std(workspace)
        vals.append((mn,std))

        return (workspace - mn) / std

    train.transformEachPoint(fn)

    def fnLookup(feat):
        workspace = numpy.array(feat)
        index = test.getFeatureIndex(feat.getFeatureName(0))
        (mn,std) = vals[index]

        return (workspace - mn) / std

    test.transformEachFeature(fnLookup)


def featNormalize_stdScoreMissing_applyTo(train, test):
    vals = []

    def fn(feat):
        workspace = numpy.array(feat)
        mask = workspace != 0
        mn = missing.mean(workspace)
        std = missing.standardDeviation(workspace)
        vals.append((mn,std))

        onAll = (workspace - mn) / std
        return numpy.multiply(onAll, mask)

    train.transformEachFeature(fn)

#   def fnLookup(feat):
#       workspace = numpy.array(feat)
#       index = test.getFeatureIndex(feat.getFeatureName(0))
#       (mn,std) = vals[index]

#       return (workspace - mn) / std

#   test.transformEachFeature(fnLookup)


def pointNormalizeViaApplyTo_DivNZ(obj):
    """
    Normalize each point by dividing by the number of
    non-zero values in that point. This is an implementation
    that uses a single call to transformEachPoint.

    """
#   nzRecord = []
    def pn(point):
        workspace = numpy.array(point)
        nz = numpy.count_nonzero(workspace)
        return workspace / nz

    obj.transformEachPoint(pn)

#   print max(nzRecord)
#   print numpy.average(nzRecord)
#   print numpy.median(nzRecord)
#   print min(nzRecord)

def pointNormalize_DivNZ(obj):
    """
    Normalize each point by dividing by the number of
    non-zero values in that point. This is an implementation
    that uses the normalizePoints method.

    """
    nzPerPoint = obj.calculateForEachPoint(numpy.count_nonzeroNZ)
    nzPerPoint = nzPerPoint.copyAs("Matrix")

    obj.normalizePoints(divide=nzPerPoint)

def loadTransformSave(origFilePath):
    """
    Load and clean the original csv file using the batch load helper,
    then write it back out as an mtx file, returning the loaded data.

    """
    allData = batchCreateFromOrigFile(origFilePath)
    sparsifiedPath = origFilePath[:-3] + 'mtx'
    allData.writeFile(sparsifiedPath, format='mtx', includeNames=True)
    return allData

def applyNZRemovalAndSave(allData, origFilePath):
    preprocess_RemoveLowNZ(allData)
    nzRemPath = origFilePath[:-4] + '_nz50Rem' + '.mtx'
    allData.writeFile(nzRemPath, format='mtx', includeNames=True)


def checkNumberNZ_point(obj):
    """
    Plot the distribution of non-zero values of each point
    in the given object.
    """
    nzObj = obj.calculateForEachPoint(numpy.count_nonzero)
#   nzObj.show("nz", maxHeight=100)

    path = "/home/tpburns/gimbel_tech/sparse-gender-nz.png"
    nzObj.plotFeatureDistribution(0, outPath=path)

def checkNumberNZ_feature(obj):
    """
    Plot the distribution of non-zero values of each feature (question)
    in the given object.
    """
    nzObj = obj.calculateForEachFeature(numpy.count_nonzero)
#   nzObj.show("nz", maxHeight=100)

    path = "/home/tpburns/gimbel_tech/sparse-gender-nz-Qs.png"
    nzObj.plotPointDistribution(0, outPath=path)


def printInAndOutSampleError(trainedLearner, trainX, testX, testY):
    #Now measure the accuracy of the model
#   print "\n\n"
    print("")
    errorOutSample = trainedLearner.test(testX, testY, performanceFunction=fractionIncorrect)
    print("Out sample error: " + str(round(errorOutSample*100,1)) + "%")
    errorInSample = trainedLearner.test(trainX, trainY, performanceFunction=fractionIncorrect)
    print("In sample error: " + str(round(errorInSample*100,1)) + "%")
    print("")


def writeOutCoefficientsAndNames(trainedLearner, trialType):
    coefs = trainedLearner.getAttributes()['origCoefs'].flatten()
    coefsObj = UML.createData("List", coefs)
    if coefsObj.points == 1:
        coefsObj.transpose()

    namesObj = UML.createData("List", [trainX.getFeatureNames()])

    namesObj.appendFeatures(coefsObj)

    path = "/home/tpburns/gimbel_tech/gender-sparse-coefs-" + trialType
    namesObj.writeFile(path, format='csv', includeNames=False)

def preprocess_RemoveLowNZ(dataObj, labelObj=None):
    nzObj = dataObj.calculateForEachPoint(numpy.count_nonzero)
    nzArr = nzObj.copyAs('numpyarray', outputAs1D=True)
    remove = list(numpy.where(nzArr >= 50))

    dataObj.extractPoints(remove)
    if labelObj is not None:
        labelObj.extractPoints(remove)

def preprocess_RemoveLowNZOld(dataObj, labelObj=None):
    def countNZ(point):
        nz = 0
        for val in point:
            if val != 0:
                nz += 1
        return nz

    nzObj = dataObj.calculateForEachPoint(countNZ)

    def removeLessThan50(point):
        index = dataObj.getPointIndex(point.getPointName(0))
        i = int(index)
        if nzObj[i] < 50:
            return True
        else:
            return False

    dataObj.extractPoints(removeLessThan50)
    if labelObj is not None:
        labelObj.extractPoints(removeLessThan50)



def featureNorm_stdScore(trainX, testX):
    trainX.normalizeFeatures('mean', 'std', testX)




def trial_LogRegL2(trainX, trainY, testX, testY):

    print("\nStarting LogRegL2")
#   print time.asctime(time.localtime())

    name = "scikitlearn.LogisticRegression"
    cVals = tuple([100. / (10**n) for n in range(7)])
    print("Cross validated over C with values of: " + str(cVals))
    trainedLearner = UML.train(
        name, trainX, trainY, C=cVals, penalty='l2',
        performanceFunction=fractionIncorrect)

#   print "Trained!"
#   print time.asctime(time.localtime())
    
    printInAndOutSampleError(trainedLearner, trainX, testX, testY)

    return trainedLearner

def trial_RegularizationSelection(trainX, trainY, testX, testY, wantedNZcoefs):
    print("Starting LogisticRegressionSelectByRegularization")
    print("with num wanted=" + str(wantedNZcoefs))
#   print time.asctime(time.localtime())

    UML.registerCustomLearner("custom", LogisticRegressionSelectByRegularization)

    name = "custom.LogisticRegressionSelectByRegularization"
    trainedLearner = UML.train(
        name, trainX, trainY, desiredNonZero=wantedNZcoefs,
        performanceFunction=fractionIncorrect)

#   print "Trained!"
#   print time.asctime(time.localtime())
    
    printInAndOutSampleError(trainedLearner, trainX, testX, testY)

    return trainedLearner

def trial_SelectionByOmission(trainX, trainY, testX, testY, wantedNZcoefs,
        method):
    print("Starting LogisticRegressionSelectByOmission")
    print("with num wanted=" + str(wantedNZcoefs) + " and method=" + method)
#   print time.asctime(time.localtime())

    UML.registerCustomLearner("custom", LogisticRegressionSelectByOmission)

    numToOmit = trainX.features - wantedNZcoefs

    name = "custom.LogisticRegressionSelectByOmission"
    cVals = tuple([100. / (10**n) for n in range(7)])
    print("Cross validated over C with values of: " + str(cVals))
    trainedLearner = UML.train(
        name, trainX, trainY, numberToOmit=numToOmit,
        performanceFunction=fractionIncorrect, C=cVals,
        method=method)

#   print "Trained!"
#   print time.asctime(time.localtime())
    
    printInAndOutSampleError(trainedLearner, trainX, testX, testY)

    return trainedLearner

def trial_SVMClassifier_linear(trainX, trainY, testX, testY):
    print("Starting SVM classifier with linear kernel")
#   print time.asctime(time.localtime())

    name = "scikitlearn.SVC"
    cVals = tuple([100. / (10**n) for n in range(7)])
    print("Cross validated over C with values of: " + str(cVals))
    trainedLearner = UML.train(
        name, trainX, trainY, C=cVals, kernel='linear',
        performanceFunction=fractionIncorrect,
        max_iter=200)

#   print "Trained!"
#   print time.asctime(time.localtime())
    
    printInAndOutSampleError(trainedLearner, trainX, testX, testY)

    return trainedLearner

def trial_SVMClassifier_poly(trainX, trainY, testX, testY, degree):
    print("Starting SVM classifier with polynomial kernel")
    print("with degree=" + str(degree))
#   print time.asctime(time.localtime())

    name = "scikitlearn.SVC"
    cVals = tuple([100. / (10**n) for n in range(7)])
    print("Cross validated over C with values of: " + str(cVals))
    coef0vals = (0,10,100)
    print("Cross Validated over coef0 values of:" + str(coef0vals))
    trainedLearner = UML.train(
        name, trainX, trainY, C=cVals, kernel='poly',
        degree=degree, coef0=coef0vals,
        performanceFunction=fractionIncorrect,
        max_iter=200)

#   print "Trained!"
#   print time.asctime(time.localtime())
    
    printInAndOutSampleError(trainedLearner, trainX, testX, testY)

    return trainedLearner



if __name__ == "__main__":
    # Some variables to control the flow of the program
    retType = "Matrix"
    defaultFile = "/home/tpburns/gimbel_tech/data/gender/sapaTempData696items08dec2013thru26jul2014"
    allowRemLess50Load = True
    UML.registerCustomLearner("Custom", MissingAwareSVD)

    allData = fileLoadHelper(retType, defaultFile, allowRemLess50Load)

#       wantedData = allData.extractPoints(end=ptsSel)
        # separate train and test data.     
#       numPoints = wantedData.points
    #   testPortion = 3000./numPoints
    #   testPortion = 1500./numPoints
#       print "total Data: " + str(numPoints) + " x " + str(wantedData.features)
#       print "test portion: " + str(testPortion)

#       trainX, trainY, testX, testY = wantedData.trainAndTestSets(
#           testPortion, labels="gender", randomOrder=True)

#       trainX.name = "Training Data"
#       trainY.name = "Training Labels"
#       testX.name = "Testing Data"
#       testY.name = "Testing Labels"



    # trial run with less data
    testPortion = 3000
#   for (ptsSel, testSel) in [(1250,157.),(2500,325.),(5000,750.),(10000,1500.)]:
    for ptsSel in [2500,5000,10000]:
#   for ptsSel in [250,500,1000]:
#   for ptsSel in [20000]:
        print("total Data: " + str(ptsSel) + " x " + str(allData.features))

        trainX = allData.copyPoints(end=ptsSel)
        trainY = trainX.extractFeatures("gender")
        testX = allData.copyPoints(start=allData.points-3000)
        testY = testX.extractFeatures("gender")

        trainX.name = "Training Data"
        trainY.name = "Training Labels"
        testX.name = "Testing Data"
        testY.name = "Testing Labels"

        # normalize data
        #   print trainX.data[0:5,0:20]
        #   pointNormalizeViaApplyTo_DivNZ(trainX)
        #   print trainX.data[0:5,0:20]
        #   pointNormalizeViaApplyTo_DivNZ(testX)

#       print trainX.data[0:5, 0:15]
#       print time.asctime(time.localtime())
#       featNormalize_stdScore_applyTo(trainX, testX)
        featNormalize_stdScoreMissing_applyTo(trainX, testX)
#       trainX.normalizeFeatures('mean', 'std', testX)
#       print time.asctime(time.localtime()) + "\n"
#       print trainX.data[0:5, 0:15]

        trial_LogRegL2(trainX, trainY, testX, testY)
#       trial_SVMClassifier_linear(trainX, trainY, testX, testY)
        trial_SVMClassifier_poly(trainX, trainY, testX, testY, degree=2)
#       trial_SVMClassifier_poly(trainX, trainY, testX, testY, degree=3)

        for kToUse in [50, 100, 200]:
#       for kToUse in [5, 10, 20]:
            print(time.asctime(time.localtime()))
            print("\nMASVD with k=" + str(kToUse))
            svdTL = UML.train("Custom.MissingAwareSVD", trainX, trainY, k=kToUse)
            redTrainX = svdTL.apply(trainX)
            redTestX = svdTL.apply(testX)
#           redTrainX = trainX
#           redTestX = testX

            # Run trials

            trainedLearner = trial_LogRegL2(redTrainX, trainY, redTestX, testY)

            trial_SVMClassifier_linear(redTrainX, trainY, redTestX, testY)
            trial_SVMClassifier_poly(redTrainX, trainY, redTestX, testY, degree=2)
#           trial_SVMClassifier_poly(redTrainX, trainY, redTestX, testY, degree=3)

    exit(0)

#   for wanted in [50, 100, 150]:
#       print time.asctime(time.localtime())
#       trainedLearner = trial_RegularizationSelection(trainX, trainY, testX, testY, wanted)

#   for wanted in [50, 100, 150]:
#       print time.asctime(time.localtime())
#       trial_SelectionByOmission(trainX, trainY, testX, testY, wanted, method="least magnitude")

#   for wanted in [50, 100, 150]:
#       print time.asctime(time.localtime())
#       trial_SelectionByOmission(trainX, trainY, testX, testY, wanted, method="least value")



    
    # lasso?
    # pca preprocess, with mean norm? into LogReg, SVM with RBF


#   writeOutCoefficientsAndNames(trainedLearner, '')

    print(time.asctime(time.localtime()))
    exit(0)
