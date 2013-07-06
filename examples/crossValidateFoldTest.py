"""
Script that uses 50K points of job posts to try to predict approved/rejected status
"""

from allowImports import boilerplate
boilerplate()

if __name__ == "__main__":
    from UML import crossValidate
    from UML import create
    from UML import loadTrainingAndTesting

    pathIn = "UML/datasets/10points2columns.mtx"
    trainX, trainY, testX, testY = loadTrainingAndTesting(pathIn, labelID=1, fractionForTestSet=.2, loadType="CooSparseData", fileType="mtx")

    # sparse types aren't playing nice with the error metrics currently, so convert
    trainY = trainY.toDenseMatrixData()
    testY = testY.toDenseMatrixData()

    trainYList = []
    
    for i in range(len(trainY.data)):
        label = trainY.data[i][0]
        trainYList.append([int(label)])

    testYList = []
    for i in range(len(testY.data)):
        label = testY.data[i][0]
        testYList.append([int(label)])

    trainY = create('dense', trainYList)
    testY = create('dense', testYList)


    # setup parameters we want to cross validate over, and the functions and metrics to evaluate
    toRun = 'dataPrinter(trainX, testX, trainY, testY)'
    extraParams = {'dataPrinter':dataPrinter}
    results = crossValidate(trainX, trainY, [toRun], numFolds=2, extraParams=extraParams, sendToLog=False)
