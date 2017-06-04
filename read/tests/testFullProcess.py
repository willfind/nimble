import math
import os

import UML
from UML.read.data_loading import *
from UML.read.dok_data_set import DokDataSet
from UML.read.convert_to_base import convertToCooBaseData

testDirectory = os.path.join(UML.UMLPath, 'read', 'tests', 'testDirectory')
testDirectory2 = os.path.join(UML.UMLPath, 'read', 'tests', 'testDirectory2')


def test_DokDataSetFrequencyFloor():
    """
    Unit test for DokDataSet object's minimum frequency
    filter
    """
    testSet = DokDataSet()
    testSet.loadDirectory(testDirectory2, featureMergeMode='multiTyped')
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com',
                    '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '003': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    cooDataSet = convertToCooBaseData(testDirectory2, dirMappingMode='multiTyped', minTermFrequency=3,
                                      attributeMaps=attributeMapMap, docIdClassLabelMaps=classLabelMapMap)

    featureNames = cooDataSet.getFeatureNames()

    assert len(featureNames) == 9
    assert 'body/cat' in featureNames
    assert 'body/dog' in featureNames
    assert 'body/account' in featureNames
    assert 'body/abl' in featureNames
    assert 'head/cat' in featureNames
    assert 'domain/bing.com' in featureNames
    assert 'domain/google.com' in featureNames

    assert cooDataSet.data.shape[0] == 8
    assert cooDataSet.data.shape[1] == 9

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    doc1Map = {'classLabel': 1, 'body/dog': 1, 'body/cat': 1, 'body/account': 1, 'body/abl': 1}
    doc2Map = {'classLabel': 2, 'body/cat': 2, 'body/dog': 1, 'body/account': 1, 'head/cat': 1, 'domain/google.com': 1}
    doc3Map = {'classLabel': 2, 'body/abl': 2, 'body/dog': 1, 'domain/bing.com': 1}
    doc4Map = {'classLabel': 3, 'body/cat': 2, 'body/dog': 1, 'domain/google.com': 1, 'head/cat': 2}
    doc5Map = {'classLabel': 1, 'head/cat': 1, 'head/account': 1, 'domain/bing.com': 1}
    doc7Map = {'classLabel': 2, 'body/dog': 1, 'body/pant': 1, 'body/cat': 1, 'domain/bing.com': 1}
    doc8Map = {'classLabel': 1, 'body/dog': 1, 'body/abl': 1, 'body/hors': 1, 'domain/bing.com': 1}
    doc9Map = {'classLabel': 1, 'body/account': 2, 'body/abl': 1, 'body/dog': 1, 'domain/google.com': 1}

    docMap = {1: doc1Map, 2: doc2Map, 3: doc3Map, 4: doc4Map, 5: doc5Map, 7: doc7Map, 8: doc8Map, 9: doc9Map}

    for i in range(numRows):
        docIdNumber = dokVersion[i, 0]
        classLabel = dokVersion[i, 1]
        assert classLabel == docMap[docIdNumber]['classLabel']
        if i != 5:
            featureCountMap = docMap[int(docIdNumber)]
        else:
            continue
        print "docIdNumber: " + str(docIdNumber)
        for j in range(2, numColumns):
            feature = featureNames[j]
            if dokVersion[i, j] > 0:
                print "processed feature: " + str(feature)
                print "processed feature count: " + str(dokVersion[i, j])
                if feature in featureCountMap:
                    print "manually computed feature count: " + str(featureCountMap[feature])
                assert int(dokVersion[i, j]) == featureCountMap[feature]


def test_DokDataSetTypeWeighting():
    """
    Unit test for DokDataSet object's type weighting
    functions
    """
    testSet = DokDataSet()
    testSet.loadDirectory(testDirectory2, featureMergeMode='multiTyped')
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com',
                    '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '003': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    typeWeightScheme = {'domain': 3}
    cooDataSet = convertToCooBaseData(testDirectory2, dirMappingMode='multiTyped', minTermFrequency=3,
                                      featureTypeWeightScheme=typeWeightScheme, attributeMaps=attributeMapMap,
                                      docIdClassLabelMaps=classLabelMapMap)

    featureNames = cooDataSet.getFeatureNames()

    assert len(featureNames) == 9
    assert 'body/cat' in featureNames
    assert 'body/dog' in featureNames
    assert 'body/account' in featureNames
    assert 'body/abl' in featureNames
    assert 'head/cat' in featureNames
    assert 'domain/bing.com' in featureNames
    assert 'domain/google.com' in featureNames

    assert cooDataSet.data.shape[0] == 8
    assert cooDataSet.data.shape[1] == 9

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    doc1Map = {'classLabel': 1, 'body/dog': 1, 'body/cat': 1, 'body/account': 1, 'body/abl': 1}
    doc2Map = {'classLabel': 2, 'body/cat': 2, 'body/dog': 1, 'body/account': 1, 'head/cat': 1, 'domain/google.com': 3}
    doc3Map = {'classLabel': 2, 'body/abl': 2, 'body/dog': 1, 'domain/bing.com': 3}
    doc4Map = {'classLabel': 3, 'body/cat': 2, 'body/dog': 1, 'domain/google.com': 1, 'head/cat': 2}
    doc5Map = {'classLabel': 1, 'head/cat': 1, 'head/account': 1, 'domain/bing.com': 3}
    doc7Map = {'classLabel': 2, 'body/dog': 1, 'body/pant': 1, 'body/cat': 1, 'domain/bing.com': 3}
    doc8Map = {'classLabel': 1, 'body/dog': 1, 'body/abl': 1, 'body/hors': 1, 'domain/bing.com': 3}
    doc9Map = {'classLabel': 1, 'body/account': 2, 'body/abl': 1, 'body/dog': 1, 'domain/google.com': 3}

    docMap = {1: doc1Map, 2: doc2Map, 3: doc3Map, 4: doc4Map, 5: doc5Map, 7: doc7Map, 8: doc8Map, 9: doc9Map}

    # nameBuffer = ''
    # for featureNameIndex in xrange(cooDataSet.featureCount):
    #     nameBuffer += str(featureNames[featureNameIndex]) + " "*(20 - len(featureNames[featureNameIndex]))
    # print nameBuffer

    # for i in range(numRows):
    #     printBuffer = ''
    #     for j in range(numColumns):
    #         printBuffer += str(dokVersion[i, j]) + " "*19
    #     print printBuffer

    for i in range(numRows):
        docIdNumber = dokVersion[i, 0]
        classLabel = dokVersion[i, 1]
        assert classLabel == docMap[docIdNumber]['classLabel']
        if i != 5:
            featureCountMap = docMap[int(docIdNumber)]
        else:
            continue
            # print "docIdNumber: " + str(docIdNumber)
        for j in range(2, numColumns):
            feature = featureNames[j]
            if dokVersion[i, j] > 0:
                # print "processed feature: " + str(feature)
                # print "processed feature count: " + str(dokVersion[i, j])
                if feature in featureCountMap:
                    # print "manually computed feature count: " + str(featureCountMap[feature])
                    pass
                assert int(dokVersion[i, j]) == featureCountMap[feature]


def test_convertToCooBaseDataFreqRepMultiTyped():
    """
    Unit test for convertToCooBaseData function in text_processing
    """
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com',
                    '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '003': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    cooDataSet = convertToCooBaseData(testDirectory2, dirMappingMode='multiTyped', minTermFrequency=1,
                                      attributeMaps=attributeMapMap, docIdClassLabelMaps=classLabelMapMap)
    featureNames = cooDataSet.getFeatureNames()

    # print "featureNames: " + str(featureNames)
    assert len(featureNames) == 15
    assert 'body/cat' in featureNames
    assert 'body/dog' in featureNames
    assert 'body/account' in featureNames
    assert 'body/abl' in featureNames
    assert 'body/hors' in featureNames
    assert 'body/pant' in featureNames
    assert 'head/sauc' in featureNames
    assert 'head/cat' in featureNames
    assert 'head/account' in featureNames
    assert 'head/dog' in featureNames

    assert cooDataSet.data.shape[0] == 8
    assert cooDataSet.data.shape[1] == 15

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    doc1Map = {'classLabel': 1, 'body/dog': 1, 'body/cat': 1, 'body/account': 1, 'body/abl': 1, 'domain/yahoo.com': 1}
    doc2Map = {'classLabel': 2, 'body/cat': 2, 'body/dog': 1, 'body/hors': 1, 'body/account': 1, 'head/cat': 1,
               'head/account': 1, 'head/sauc': 1, 'domain/google.com': 1}
    doc3Map = {'classLabel': 2, 'body/abl': 2, 'body/dog': 1, 'domain/bing.com': 1}
    doc4Map = {'classLabel': 3, 'body/cat': 2, 'body/dog': 1, 'domain/google.com': 1, 'head/cat': 2, 'head/dog': 1}
    doc5Map = {'classLabel': 1, 'head/cat': 1, 'head/account': 1, 'domain/bing.com': 1}
    doc7Map = {'classLabel': 2, 'body/dog': 1, 'body/pant': 1, 'body/cat': 1, 'domain/bing.com': 1}
    doc8Map = {'classLabel': 1, 'body/dog': 1, 'body/abl': 1, 'body/hors': 1, 'domain/bing.com': 1}
    doc9Map = {'classLabel': 1, 'body/account': 2, 'body/abl': 1, 'body/dog': 1, 'domain/google.com': 1}

    docMap = {1: doc1Map, 2: doc2Map, 3: doc3Map, 4: doc4Map, 5: doc5Map, 7: doc7Map, 8: doc8Map, 9: doc9Map}

    for i in range(numRows):
        docIdNumber = dokVersion[i, 0]
        classLabel = dokVersion[i, 1]
        assert classLabel == docMap[docIdNumber]['classLabel']
        if i != 5:
            featureCountMap = docMap[int(docIdNumber)]
        else:
            continue
            #print "docIdNumber: " + str(docIdNumber)
        for j in range(2, numColumns):
            feature = featureNames[j]
            if dokVersion[i, j] > 0:
                #print "processed feature: " + str(feature)
                #print "processed feature count: " + str(dokVersion[i, j])
                if feature in featureCountMap:
                    #print "manually computed feature count: " + str(featureCountMap[feature])
                    pass
                assert int(dokVersion[i, j]) == featureCountMap[feature]

    return


def test_convertToCooBaseDataBinaryMultiTyped():
    """
    Unit test for convertToCooBaseData function in text_processing
    """
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com',
                    '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '003': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    cooDataSet = convertToCooBaseData(testDirectory2, dirMappingMode='multiTyped', minTermFrequency=1,
                                      attributeMaps=attributeMapMap, docIdClassLabelMaps=classLabelMapMap,
                                      featureRepresentation='binary')
    featureNames = cooDataSet.getFeatureNames()

    assert len(featureNames) == 15
    assert 'body/cat' in featureNames
    assert 'body/dog' in featureNames
    assert 'body/account' in featureNames
    assert 'body/abl' in featureNames
    assert 'body/hors' in featureNames
    assert 'body/pant' in featureNames
    assert 'head/sauc' in featureNames
    assert 'head/cat' in featureNames
    assert 'head/account' in featureNames
    assert 'head/dog' in featureNames

    assert cooDataSet.data.shape[0] == 8
    assert cooDataSet.data.shape[1] == 15

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    doc1Map = {'classLabel': 1, 'body/dog': 1, 'body/cat': 1, 'body/account': 1, 'body/abl': 1, 'domain/yahoo.com': 1}
    doc2Map = {'classLabel': 2, 'body/cat': 1, 'body/dog': 1, 'body/hors': 1, 'body/account': 1, 'head/cat': 1,
               'head/account': 1, 'head/sauc': 1, 'domain/google.com': 1}
    doc3Map = {'classLabel': 2, 'body/abl': 1, 'body/dog': 1, 'domain/bing.com': 1}
    doc4Map = {'classLabel': 3, 'body/cat': 1, 'body/dog': 1, 'domain/google.com': 1, 'head/cat': 1, 'head/dog': 1}
    doc5Map = {'classLabel': 1, 'head/cat': 1, 'head/account': 1, 'domain/bing.com': 1}
    doc7Map = {'classLabel': 2, 'body/dog': 1, 'body/pant': 1, 'body/cat': 1, 'domain/bing.com': 1}
    doc8Map = {'classLabel': 1, 'body/dog': 1, 'body/abl': 1, 'body/hors': 1, 'domain/bing.com': 1}
    doc9Map = {'classLabel': 1, 'body/account': 1, 'body/abl': 1, 'body/dog': 1, 'domain/google.com': 1}

    docMap = {1: doc1Map, 2: doc2Map, 3: doc3Map, 4: doc4Map, 5: doc5Map, 7: doc7Map, 8: doc8Map, 9: doc9Map}

    for i in range(numRows):
        docIdNumber = dokVersion[i, 0]
        classLabel = dokVersion[i, 1]
        assert classLabel == docMap[docIdNumber]['classLabel']
        if i != 5:
            featureCountMap = docMap[int(docIdNumber)]
        else:
            continue
            #print "docIdNumber: " + str(docIdNumber)
        for j in range(2, numColumns):
            feature = featureNames[j]
            if dokVersion[i, j] > 0:
                #print "processed feature: " + str(feature)
                #print "processed feature count: " + str(dokVersion[i, j])
                if feature in featureCountMap:
                    #print "manually computed feature count: " + str(featureCountMap[feature])
                    pass
                assert int(dokVersion[i, j]) == featureCountMap[feature]

    return


def test_convertToCooBaseDataTfIdfMultiTyped():
    """
    Unit test for convertToCooBaseData function in text_processing
    """
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com',
                    '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '003': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    cooDataSet = convertToCooBaseData(testDirectory2, dirMappingMode='multiTyped', minTermFrequency=1,
                                      attributeMaps=attributeMapMap, docIdClassLabelMaps=classLabelMapMap,
                                      featureRepresentation='tfidf')
    featureNames = cooDataSet.getFeatureNames()

    assert len(featureNames) == 15
    assert 'body/cat' in featureNames
    assert 'body/dog' in featureNames
    assert 'body/account' in featureNames
    assert 'body/abl' in featureNames
    assert 'body/hors' in featureNames
    assert 'body/pant' in featureNames
    assert 'head/sauc' in featureNames
    assert 'head/cat' in featureNames
    assert 'head/account' in featureNames
    assert 'head/dog' in featureNames

    assert cooDataSet.data.shape[0] == 8
    assert cooDataSet.data.shape[1] == 15

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    doc1Map = {'classLabel': 1, 'body/dog': 0.053349, 'body/cat': 0.27693, 'body/account': 0.391867,
               'body/abl': 0.27693, 'domain/yahoo.com': 0.830791}
    doc2Map = {'classLabel': 2, 'body/cat': 0.2982506, 'body/dog': 0.0383061, 'body/hors': 0.39766752,
               'body/account': 0.2813577, 'head/cat': 0.2813577, 'head/account': 0.39766752, 'head/sauc': 0.59650128,
               'domain/google.com': 0.281357723}
    doc3Map = {'classLabel': 2, 'body/abl': 0.82734, 'body/dog': 0.106255, 'domain/bing.com': 0.55156}
    doc4Map = {'classLabel': 3, 'body/cat': 0.355587, 'body/dog': 0.045668, 'domain/google.com': 0.335446,
               'head/cat': 0.50317, 'head/dog': 0.711174}
    doc5Map = {'classLabel': 1, 'head/cat': 0.534745, 'head/account': 0.755803, 'domain/bing.com': 0.377901}
    doc7Map = {'classLabel': 2, 'body/dog': 0.057987, 'body/pant': .903012, 'body/cat': .301004,
               'domain/bing.com': .301004}
    doc8Map = {'classLabel': 1, 'body/dog': .078405, 'body/abl': .406992, 'body/hors': .813983,
               'domain/bing.com': .406992}
    doc9Map = {'classLabel': 1, 'body/account': 0.772749, 'body/abl': 0.364065, 'body/dog': 0.070135,
               'domain/google.com': 0.515166}

    docMap = {1: doc1Map, 2: doc2Map, 3: doc3Map, 4: doc4Map, 5: doc5Map, 7: doc7Map, 8: doc8Map, 9: doc9Map}

    for i in range(numRows):
        docIdNumber = dokVersion[i, 0]
        classLabel = dokVersion[i, 1]
        assert classLabel == docMap[docIdNumber]['classLabel']
        if i != 5:
            featureCountMap = docMap[int(docIdNumber)]
        else:
            continue
            #print "docIdNumber: " + str(docIdNumber)
        for j in range(2, numColumns):
            feature = featureNames[j]
            if dokVersion[i, j] > 0:
            #print "processed feature: " + str(feature)
            #print "processed feature count: " + str(dokVersion[i, j])
            #if feature in featureCountMap:
            #print "manually computed feature count: " + str(featureCountMap[feature])
                assert math.fabs(dokVersion[i, j] - featureCountMap[feature]) < 0.01

    return


def test_convertToCooBaseDataRequiredClassLabel():
    """
    Unit test for requiredClassLabel functionality in text_processing
    """
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com',
                    '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    requiredClassLabelTypes = set(['classLabel'])
    cooDataSet = convertToCooBaseData(testDirectory2, dirMappingMode='multiTyped', minTermFrequency=1,
                                      attributeMaps=attributeMapMap, docIdClassLabelMaps=classLabelMapMap,
                                      requiredClassLabelTypes=requiredClassLabelTypes)
    #print "featureReport:\n" + cooDataSet.featureReport()
    featureNames = cooDataSet.getFeatureNames()

    #print "featureNames: " + str(featureNames)
    assert len(featureNames) == 15
    assert 'body/cat' in featureNames
    assert 'body/dog' in featureNames
    assert 'body/account' in featureNames
    assert 'body/abl' in featureNames
    assert 'body/hors' in featureNames
    assert 'body/pant' in featureNames
    assert 'head/sauc' in featureNames
    assert 'head/cat' in featureNames
    assert 'head/account' in featureNames
    assert 'head/dog' in featureNames

    assert cooDataSet.data.shape[0] == 7
    assert cooDataSet.data.shape[1] == 15

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    doc1Map = {'classLabel': 1, 'body/dog': 1, 'body/cat': 1, 'body/account': 1, 'body/abl': 1, 'domain/yahoo.com': 1}
    doc2Map = {'classLabel': 2, 'body/cat': 2, 'body/dog': 1, 'body/hors': 1, 'body/account': 1, 'head/cat': 1,
               'head/account': 1, 'head/sauc': 1, 'domain/google.com': 1}
    #doc3Map = {'classLabel': 2, 'body/abl': 2, 'body/dog': 1, 'domain/bing.com': 1}
    doc4Map = {'classLabel': 3, 'body/cat': 2, 'body/dog': 1, 'domain/google.com': 1, 'head/cat': 2, 'head/dog': 1}
    doc5Map = {'classLabel': 1, 'head/cat': 1, 'head/account': 1, 'domain/bing.com': 1}
    doc7Map = {'classLabel': 2, 'body/dog': 1, 'body/pant': 1, 'body/cat': 1, 'domain/bing.com': 1}
    doc8Map = {'classLabel': 1, 'body/dog': 1, 'body/abl': 1, 'body/hors': 1, 'domain/bing.com': 1}
    doc9Map = {'classLabel': 1, 'body/account': 2, 'body/abl': 1, 'body/dog': 1, 'domain/google.com': 1}

    docMap = {1: doc1Map, 2: doc2Map, 4: doc4Map, 5: doc5Map, 7: doc7Map, 8: doc8Map, 9: doc9Map}

    for i in range(numRows):
        docIdNumber = dokVersion[i, 0]
        classLabel = dokVersion[i, 1]
        assert classLabel == docMap[docIdNumber]['classLabel']
        if i != 5 and i != 3:
            featureCountMap = docMap[int(docIdNumber)]
        else:
            continue
            #print "docIdNumber: " + str(docIdNumber)
        for j in range(2, numColumns):
            feature = featureNames[j]
            if dokVersion[i, j] > 0:
                #print "processed feature: " + str(feature)
                #print "processed feature count: " + str(dokVersion[i, j])
                if feature in featureCountMap:
                    #print "manually computed feature count: " + str(featureCountMap[feature])
                    pass
                assert int(dokVersion[i, j]) == featureCountMap[feature]

    return


if __name__ == "__main__":
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com',
                    '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '003': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    cooDataSet = convertToCooBaseData(testDirectory2, dirMappingMode='multiTyped', attributeMaps=attributeMapMap,
                                      docIdClassLabelMaps=classLabelMapMap, minTermFrequency=2,
                                      featureRepresentation='frequency')
    featureNames = cooDataSet.getFeatureNames()

    print "featureNames: " + str(featureNames)

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    headerRow = "     "
    for k in range(1, numColumns):
        feature = featureNames[k]
        headerLength = len(feature)
        leftoverLength = 20 - headerLength
        headerRow += ' ' * leftoverLength + feature

    print headerRow

    for i in range(numRows):
        rowString = ''
        for j in range(numColumns):
            if j == 0:
                rowString += str(dokVersion[i, j]) + ' ' * (5 - len(str(dokVersion[i, j])))
            else:
                leftoverLength = 20 - len(str(dokVersion[i, j]))
                rowString += ' ' * leftoverLength + str(dokVersion[i, j])
        print rowString

