"""
Short module demonstrating the flatter imports and user facing functions provided
in the root of the UML package

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
    assert trainObj.data[0].size == 3
    assert testObj.data[0].size == 3

    # use normalize to modify our data; we call a dimentionality reduction algorithm to
    # simply our mostly redundant points. k is the desired number of dimensions in the output
    normalizeData('mlpy.PCA', trainObj, testX=testObj, arguments={'k': 1})

    # assert that we actually do have fewer dimensions
    assert trainObj.data[0].size == 1
    assert testObj.data[0].size == 1

    ret = trainAndApply('mlpy.KNN', trainObj, trainObjY, testObj, arguments={'k': 1})

    # assert we get the correct classes
    assert ret.data[0, 0] == 1
    assert ret.data[1, 0] == 1
    assert ret.data[2, 0] == 0
    assert ret.data[3, 0] == 0
