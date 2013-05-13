import math

from UML.uml_loading.data_loading import *
from UML.uml_loading.convert_to_basedata import convertToCooBaseData


def test_convertToCooBaseDataFreqRepMultiTyped():
    """
    Unit test for convertToCooBaseData function in text_processing
    """
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com', '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '003': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    cooDataSet = convertToCooBaseData('uml_loading/tests/testDirectory2', dirMappingMode='multiTyped', attributeMaps=attributeMapMap, docIdClassLabelMaps=classLabelMapMap)
    featureNameMap = cooDataSet.featureNames
    inverseFeatureNameMap = cooDataSet.featureNamesInverse

    assert len(featureNameMap) == 15
    assert 'body_cat' in featureNameMap
    assert 'body_dog' in featureNameMap
    assert 'body_account' in featureNameMap
    assert 'body_abl' in featureNameMap
    assert 'body_hors' in featureNameMap
    assert 'body_pant' in featureNameMap
    assert 'head_sauc' in featureNameMap
    assert 'head_cat' in featureNameMap
    assert 'head_account' in featureNameMap
    assert 'head_dog' in featureNameMap

    assert cooDataSet.data.shape[0] == 8
    assert cooDataSet.data.shape[1] == 15

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    doc1Map = {'classLabel': 1, 'body_dog': 1, 'body_cat': 1, 'body_account': 1, 'body_abl': 1, 'domain_yahoo.com': 1}
    doc2Map = {'classLabel': 2, 'body_cat': 2, 'body_dog': 1, 'body_hors': 1, 'body_account': 1, 'head_cat': 1, 'head_account': 1, 'head_sauc': 1, 'domain_google.com': 1}
    doc3Map = {'classLabel': 2, 'body_abl': 2, 'body_dog': 1, 'domain_bing.com': 1}
    doc4Map = {'classLabel': 3, 'body_cat': 2, 'body_dog': 1, 'domain_google.com': 1, 'head_cat': 2, 'head_dog': 1}
    doc5Map = {'classLabel': 1, 'head_cat': 1, 'head_account': 1, 'domain_bing.com': 1}
    doc7Map = {'classLabel': 2, 'body_dog': 1, 'body_pant': 1, 'body_cat': 1, 'domain_bing.com': 1}
    doc8Map = {'classLabel': 1, 'body_dog': 1, 'body_abl': 1, 'body_hors': 1, 'domain_bing.com': 1}
    doc9Map = {'classLabel': 1, 'body_account': 2, 'body_abl': 1, 'body_dog': 1, 'domain_google.com': 1}

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
            feature = inverseFeatureNameMap[j]
            if dokVersion[i, j] > 0:
                print "processed feature: " + str(feature)
                print "processed feature count: " + str(dokVersion[i, j])
                if feature in featureCountMap:
                    print "manually computed feature count: " + str(featureCountMap[feature])
                assert int(dokVersion[i, j]) == featureCountMap[feature]

    return

def test_convertToCooBaseDataBinaryMultiTyped():
    """
    Unit test for convertToCooBaseData function in text_processing
    """
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com', '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '003': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    cooDataSet = convertToCooBaseData('uml_loading/tests/testDirectory2', dirMappingMode='multiTyped', attributeMaps=attributeMapMap, docIdClassLabelMaps=classLabelMapMap, featureRepresentation='binary')
    featureNameMap = cooDataSet.featureNames
    inverseFeatureNameMap = cooDataSet.featureNamesInverse

    assert len(featureNameMap) == 15
    assert 'body_cat' in featureNameMap
    assert 'body_dog' in featureNameMap
    assert 'body_account' in featureNameMap
    assert 'body_abl' in featureNameMap
    assert 'body_hors' in featureNameMap
    assert 'body_pant' in featureNameMap
    assert 'head_sauc' in featureNameMap
    assert 'head_cat' in featureNameMap
    assert 'head_account' in featureNameMap
    assert 'head_dog' in featureNameMap

    assert cooDataSet.data.shape[0] == 8
    assert cooDataSet.data.shape[1] == 15

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    doc1Map = {'classLabel': 1, 'body_dog': 1, 'body_cat': 1, 'body_account': 1, 'body_abl': 1, 'domain_yahoo.com': 1}
    doc2Map = {'classLabel': 2, 'body_cat': 1, 'body_dog': 1, 'body_hors': 1, 'body_account': 1, 'head_cat': 1, 'head_account': 1, 'head_sauc': 1, 'domain_google.com': 1}
    doc3Map = {'classLabel': 2, 'body_abl': 1, 'body_dog': 1, 'domain_bing.com': 1}
    doc4Map = {'classLabel': 3, 'body_cat': 1, 'body_dog': 1, 'domain_google.com': 1, 'head_cat': 1, 'head_dog': 1}
    doc5Map = {'classLabel': 1, 'head_cat': 1, 'head_account': 1, 'domain_bing.com': 1}
    doc7Map = {'classLabel': 2, 'body_dog': 1, 'body_pant': 1, 'body_cat': 1, 'domain_bing.com': 1}
    doc8Map = {'classLabel': 1, 'body_dog': 1, 'body_abl': 1, 'body_hors': 1, 'domain_bing.com': 1}
    doc9Map = {'classLabel': 1, 'body_account': 1, 'body_abl': 1, 'body_dog': 1, 'domain_google.com': 1}

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
            feature = inverseFeatureNameMap[j]
            if dokVersion[i, j] > 0:
                print "processed feature: " + str(feature)
                print "processed feature count: " + str(dokVersion[i, j])
                if feature in featureCountMap:
                    print "manually computed feature count: " + str(featureCountMap[feature])
                assert int(dokVersion[i, j]) == featureCountMap[feature]

    return

def test_convertToCooBaseDataTfIdfMultiTyped():
    """
    Unit test for convertToCooBaseData function in text_processing
    """
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com', '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '003': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    cooDataSet = convertToCooBaseData('uml_loading/tests/testDirectory2', dirMappingMode='multiTyped', attributeMaps=attributeMapMap, docIdClassLabelMaps=classLabelMapMap, featureRepresentation='tfidf')
    featureNameMap = cooDataSet.featureNames
    inverseFeatureNameMap = cooDataSet.featureNamesInverse

    assert len(featureNameMap) == 15
    assert 'body_cat' in featureNameMap
    assert 'body_dog' in featureNameMap
    assert 'body_account' in featureNameMap
    assert 'body_abl' in featureNameMap
    assert 'body_hors' in featureNameMap
    assert 'body_pant' in featureNameMap
    assert 'head_sauc' in featureNameMap
    assert 'head_cat' in featureNameMap
    assert 'head_account' in featureNameMap
    assert 'head_dog' in featureNameMap

    assert cooDataSet.data.shape[0] == 8
    assert cooDataSet.data.shape[1] == 15

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    doc1Map = {'classLabel': 1, 'body_dog': 0.053349, 'body_cat': 0.27693, 'body_account': 0.391867, 'body_abl': 0.27693, 'domain_yahoo.com': 0.830791}
    doc2Map = {'classLabel': 2, 'body_cat': 0.2982506, 'body_dog': 0.0383061, 'body_hors': 0.39766752, 'body_account': 0.2813577, 'head_cat': 0.2813577, 'head_account': 0.39766752, 'head_sauc': 0.59650128, 'domain_google.com': 0.281357723}
    doc3Map = {'classLabel': 2, 'body_abl': 0.82734, 'body_dog': 0.106255, 'domain_bing.com': 0.55156}
    doc4Map = {'classLabel': 3, 'body_cat': 0.355587, 'body_dog': 0.045668, 'domain_google.com': 0.335446, 'head_cat': 0.50317, 'head_dog': 0.711174}
    doc5Map = {'classLabel': 1, 'head_cat': 0.534745, 'head_account': 0.755803, 'domain_bing.com': 0.377901}
    doc7Map = {'classLabel': 2, 'body_dog': 0.057987, 'body_pant': .903012, 'body_cat': .301004, 'domain_bing.com': .301004}
    doc8Map = {'classLabel': 1, 'body_dog': .078405, 'body_abl': .406992, 'body_hors': .813983, 'domain_bing.com': .406992}
    doc9Map = {'classLabel': 1, 'body_account': 0.772749, 'body_abl': 0.364065, 'body_dog': 0.070135, 'domain_google.com': 0.515166}

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
            feature = inverseFeatureNameMap[j]
            if dokVersion[i, j] > 0:
                print "processed feature: " + str(feature)
                print "processed feature count: " + str(dokVersion[i, j])
                if feature in featureCountMap:
                    print "manually computed feature count: " + str(featureCountMap[feature])
                assert math.fabs(dokVersion[i, j] -featureCountMap[feature]) < 0.01

    return


if __name__ == "__main__":
    attributeMap = {'001': 'yahoo.com', '002': 'google.com', '003': 'bing.com', '004': 'google.com', '005': 'bing.com', '007': 'bing.com', '008': 'bing.com', '009': 'google.com'}
    classLabelMap = {'001': 1, '002': 2, '003': 2, '004': 3, '005': 1, '007': 2, '008': 1, '009': 1}
    attributeMapMap = {'domain': attributeMap}
    classLabelMapMap = {'classLabel': classLabelMap}
    cooDataSet = convertToCooBaseData('UML/uml_loading/tests/testDirectory2', dirMappingMode='multiTyped', attributeMaps=attributeMapMap, docIdClassLabelMaps=classLabelMapMap, featureRepresentation='tfidf')
    featureNameMap = cooDataSet.featureNames
    inverseFeatureNameMap = cooDataSet.featureNamesInverse

    dokVersion = cooDataSet.data.todok()

    dokShape = dokVersion.shape
    numRows = dokShape[0]
    numColumns = dokShape[1]

    headerRow = "     "
    for k in range(1, numColumns):
        feature = inverseFeatureNameMap[k]
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

