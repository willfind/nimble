from allowImports import boilerplate
boilerplate()

def readMapFile(filePath):
    """
    Read in a file mapping id to attribute, formatted as such:
    id=attribute, with one entry per line
    """
    mapFile = open(filePath, 'r')

    idAttrMap = {}

    for line in mapFile:
        line = line.strip('\n')
        lineList = line.split('=', 1)
        idAttrMap[lineList[0].lower()] = lineList[1].lower()

    return idAttrMap

if __name__ == "__main__":
    from UML.read.convert_to_base import convertToCooBaseData


    rawTextDirPath = '/home/ross/library/LaddersData'
    jobTitleMapPath = 'UML/datasets/rawData/jobTitlesMapAll.txt'
    urlMapPath = 'UML/datasets/rawData/urlMapAll.txt'
    companyNamePath = 'UML/datasets/rawData/companyNameMapAll.txt'
    approvalMapPath = 'UML/datasets/rawData/approvalMap207K.txt'
    timeStampPath = '/home/ross/library/LaddersData/individualFeatures/'

    jobTitleMap = readMapFile(jobTitleMapPath)
    urlMap = readMapFile(urlMapPath)
    companyNameMap = readMapFile(companyNamePath)
    approvalMap = readMapFile(approvalMapPath)
    timeStampMap = readMapFile(timeStampPath)
    #convert string labels ('A', 'R') to ints
    for docId, approvalClass in approvalMap.iteritems():
        if approvalClass.lower() == 'a':
            approvalMap[docId] = '1'
        else:
            approvalMap[docId] = '2'

    labelMaps = {'approval':approvalMap, 'postDate':timeStampMap}

    attributeMaps = {'jobTitle':jobTitleMap, 'url':urlMap, 'companyName':companyNameMap}

    dataObj = convertToCooBaseData(rawTextDirPath, dirMappingMode='multiTyped', attributeMaps=attributeMaps, docIdClassLabelMaps=labelMaps, minTermFrequency=3, featureRepresentation='tfidf')

    dataObj.writeFile('/home/ross/library/LaddersData/umlApproval50KTfIdfRounded.mtx', includeFeatureNames=True)
