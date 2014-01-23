"""
Script that uses 50K points of job posts to try to predict approved/rejected status
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
    import os.path
    import UML
    from UML import crossValidate
    from UML import createData
    from UML import splitData

    pathIn = os.path.join(UML.UMLPath, "datasets/10points2columns.mtx")
    allData = createData("Sparse", pathIn, fileType="mtx")
    trainX, trainY, testX, testY = splitData(allData, labelID=1, fractionForTestSet=.2)

    # sparse types aren't playing nice with the error metrics currently, so convert
    trainY = trainY.copyAs(format="Matrix")
    testY = testY.copyAs(format="Matrix")

    trainYList = []
    
    for i in range(len(trainY.data)):
        label = trainY.data[i][0]
        trainYList.append([int(label)])

    testYList = []
    for i in range(len(testY.data)):
        label = testY.data[i][0]
        testYList.append([int(label)])

    trainY = createData('Matrix', trainYList)
    testY = createData('Matrix', testYList)


    # setup parameters we want to cross validate over, and the functions and metrics to evaluate
    toRun = 'dataPrinter(trainX, testX, trainY, testY)'
    extraParams = {'dataPrinter':dataPrinter}
    results = crossValidate(trainX, trainY, [toRun], numFolds=2, extraParams=extraParams, sendToLog=False)
