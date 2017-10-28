"""
Short module demonstrating a call to normalizeData and the effect on the passed
datasets.

"""

from allowImports import boilerplate

boilerplate()

if __name__ == "__main__":
    from UML import trainAndApply
    from UML import normalizeData
    from UML import createData

    # we separate into classes accoring to whether x1 is positive or negative
    variables = ["y", "x1", "x2", "x3"]
    data1 = [[1, 6, 0, 0], [1, 3, 0, 0], [0, -5, 0, 0], [0, -3, 0, 0]]
    trainObj = createData('Matrix', data=data1, featureNames=variables)
    trainObjY = trainObj.extractFeatures('y')

    # data we're going to classify
    variables2 = ["x1", "x2", "x3"]
    data2 = [[1, 0, 0], [4, 0, 0], [-1, 0, 0], [-2, 0, 0]]
    testObj = createData('Matrix', data=data2, featureNames=variables2)

    # baseline check
    assert trainObj.features == 3
    assert testObj.features == 3

    # reserve the original data for comparison
    trainObjOrig = trainObj.copy()
    testObjOrig = testObj.copy()

    # use normalize to modify our data; we call a dimentionality reduction algorithm to
    # simply our mostly redundant points. k is the desired number of dimensions in the output
    normalizeData('mlpy.PCA', trainObj, testX=testObj, arguments={'k': 1})

    # assert that we actually do have fewer dimensions
    assert trainObj.features == 1
    assert testObj.features == 1

    # assert we can predict the correct classes
    ret = trainAndApply('mlpy.KNN', trainObj, trainObjY, testObj, arguments={'k': 1})
    assert ret[0, 0] == 1
    assert ret[1, 0] == 1
    assert ret[2, 0] == 0
    assert ret[3, 0] == 0

    # demonstrate that the results have not changed, when compared to the original data;
    # uses python's **kwargs based argument passing.
    retOrig = trainAndApply('mlpy.KNN', trainObjOrig, trainObjY, testObjOrig, k=1)
    assert ret == retOrig
