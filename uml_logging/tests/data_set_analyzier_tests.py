import numpy as np

from nose.tools import assert_almost_equal, assert_equal

from UML import create
from UML.uml_logging.data_set_analyzer import *


def testDense():
    """
        Test the functionality of calculating statistical/informational 
        funcs on a dense Base object using the produceInfoTable.
    """
    data1 = np.array([[1, 2, 3, 1], [3, 3, 1, 5], [1, 1, 5, 2]])
    names1 = ['var1', 'var2', 'var3', 'var4']

    trainObj = create('Dense', data1, names1)
    funcs = featurewiseFunctionGenerator()
    rawTable = produceFeaturewiseInfoTable(trainObj, funcs)
    funcNames = rawTable[0]
    for i in range(len(funcNames)):
        funcName = funcNames[i]
        if funcName == "mean_" or funcName == "mean":
            assert_almost_equal(rawTable[1][i], 1.6667, 3)
            assert_almost_equal(rawTable[2][i], 2.000, 3)
            assert_almost_equal(rawTable[3][i], 3.000, 3)
            assert_almost_equal(rawTable[4][i], 2.6667, 3)
        elif funcName == "min_" or funcName == "min":
            assert_equal(rawTable[1][i], 1)
            assert_equal(rawTable[2][i], 1)
            assert_equal(rawTable[3][i], 1)
            assert_equal(rawTable[4][i], 1)
        elif funcName == "max_" or funcName == "max":
            assert_equal(rawTable[1][i], 3)
            assert_equal(rawTable[2][i], 3)
            assert_equal(rawTable[3][i], 5)
            assert_equal(rawTable[4][i], 5)
        elif funcName == "numUnique":
            assert_equal(rawTable[1][i], 2)
            assert_equal(rawTable[2][i], 3)
            assert_equal(rawTable[3][i], 3)
            assert_equal(rawTable[4][i], 3)
        elif funcName == "standardDeviation":
            assert_almost_equal(rawTable[1][i], 0.9428, 3)
            assert_almost_equal(rawTable[2][i], 0.8165, 3)
            assert_almost_equal(rawTable[3][i], 1.633, 3)
            assert_almost_equal(rawTable[4][i], 1.6997, 3)
        elif funcName == "median_" or funcName == "median":
            assert_equal(rawTable[1][i], 1.0)
            assert_equal(rawTable[2][i], 2.0)
            assert_equal(rawTable[3][i], 3.0)
            assert_equal(rawTable[4][i], 2.0)


def testSparse():
    """
        Test the functionality of calculating statistical/informational 
        funcs on a dense Base object using the produceInfoTable.

        Testing matrix is 6 x 6, with 11 non-zero values, 1 None/Missing value,
        and 24 zero values.
    """
    row = np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5])
    col = np.array([0, 4, 2, 3, 1, 3, 4, 0, 1, 3, 4, 5])
    vals = np.array([1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1])

    testObj = create('coo', (vals, (row, col)))
    funcs = featurewiseFunctionGenerator()
    rawTable = produceFeaturewiseInfoTable(testObj, funcs)

    funcNames = rawTable[0]
    for i in range(len(funcNames)):
        funcName = funcNames[i]
        if funcName == "mean_" or funcName == "mean":
            assert_almost_equal(rawTable[1][i], 0.3333, 3)
            assert_almost_equal(rawTable[2][i], 0.3333, 3)
            assert_almost_equal(rawTable[3][i], 0.1667, 3)
            assert_almost_equal(rawTable[4][i], 0.4000, 3)
            assert_almost_equal(rawTable[5][i], 0.5000, 3)
            assert_almost_equal(rawTable[6][i], 0.1667, 3)
        elif funcName == "min_" or funcName == "min":
            assert_equal(rawTable[1][i], 0)
            assert_equal(rawTable[2][i], 0)
            assert_equal(rawTable[3][i], 0)
            assert_equal(rawTable[4][i], 0)
            assert_equal(rawTable[5][i], 0)
            assert_equal(rawTable[6][i], 0)
        elif funcName == "max_" or funcName == "max":
            assert_equal(rawTable[1][i], 1)
            assert_equal(rawTable[2][i], 1)
            assert_equal(rawTable[3][i], 1)
            assert_equal(rawTable[4][i], 1)
            assert_equal(rawTable[5][i], 1)
            assert_equal(rawTable[6][i], 1)
        elif funcName == "numUnique":
            assert_equal(rawTable[1][i], 2)
            assert_equal(rawTable[2][i], 2)
            assert_equal(rawTable[3][i], 2)
            assert_equal(rawTable[4][i], 2)
            assert_equal(rawTable[5][i], 2)
            assert_equal(rawTable[6][i], 2)
        elif funcName == "standardDeviation":
            assert_almost_equal(rawTable[1][i], 0.4714, 3)
            assert_almost_equal(rawTable[2][i], 0.4714, 3)
            assert_almost_equal(rawTable[3][i], 0.3727, 3)
            assert_almost_equal(rawTable[4][i], 0.4899, 3)
            assert_almost_equal(rawTable[5][i], 0.500, 3)
            assert_almost_equal(rawTable[6][i], 0.3727, 3)
        elif funcName == "median_" or funcName == "median":
            assert_equal(rawTable[1][i], 0)
            assert_equal(rawTable[2][i], 0)
            assert_equal(rawTable[3][i], 0)
            assert_equal(rawTable[4][i], 0)
            assert_equal(rawTable[5][i], 0.5)
            assert_equal(rawTable[6][i], 0)


def testRowList():
    data1 = np.array([[1, 2, 3, 1], [3, 3, 1, 5], [1, 1, 5, 2]])
    names1 = ['var1', 'var2', 'var3', 'var4']

    trainObj = create('List', data1, names1)
    funcs = featurewiseFunctionGenerator()
    rawTable = produceFeaturewiseInfoTable(trainObj, funcs)
    funcNames = rawTable[0]
    for i in range(len(funcNames)):
        funcName = funcNames[i]
        if funcName == "mean_" or funcName == "mean":
            assert_almost_equal(rawTable[1][i], 1.6667, 3)
            assert_almost_equal(rawTable[2][i], 2.000, 3)
            assert_almost_equal(rawTable[3][i], 3.000, 3)
            assert_almost_equal(rawTable[4][i], 2.6667, 3)
        elif funcName == "min_" or funcName == "min":
            assert_equal(rawTable[1][i], 1)
            assert_equal(rawTable[2][i], 1)
            assert_equal(rawTable[3][i], 1)
            assert_equal(rawTable[4][i], 1)
        elif funcName == "max_" or funcName == "max":
            assert_equal(rawTable[1][i], 3)
            assert_equal(rawTable[2][i], 3)
            assert_equal(rawTable[3][i], 5)
            assert_equal(rawTable[4][i], 5)
        elif funcName == "numUnique":
            assert_equal(rawTable[1][i], 2)
            assert_equal(rawTable[2][i], 3)
            assert_equal(rawTable[3][i], 3)
            assert_equal(rawTable[4][i], 3)
        elif funcName == "standardDeviation":
            assert_almost_equal(rawTable[1][i], 0.9428, 3)
            assert_almost_equal(rawTable[2][i], 0.8165, 3)
            assert_almost_equal(rawTable[3][i], 1.633, 3)
            assert_almost_equal(rawTable[4][i], 1.6997, 3)
        elif funcName == "median_" or funcName == "median":
            assert_equal(rawTable[1][i], 1.0)
            assert_equal(rawTable[2][i], 2.0)
            assert_equal(rawTable[3][i], 3.0)
            assert_equal(rawTable[4][i], 2.0)

def testAppendColumns():
    """
        Unit test for appendColumns function in data_set_analyzer
    """
    table1 = [[1, 2, 3], [7, 8, 9]]
    table2 = [[4, 5, 6], [10, 11, 12]]

    table3 = [[1], [2], [3], [4]]
    table4 = [[5], [6], [7], [8]]

    table5 = [["one"], ["two"], ["three"], ["four"]]
    table6 = [[1], [2], [3], [4]]

    appendColumns(table1, table2)
    appendColumns(table3, table4)
    appendColumns(table5, table6)

    assert len(table1) == 2
    assert len(table1[0]) == 6
    assert table1[0][2] == 3
    assert table1[0][3] == 4
    assert table1[1][2] == 9
    assert table1[1][3] == 10

    assert len(table3) == 4
    assert len(table3[0]) == 2
    assert table3[0][0] == 1
    assert table3[0][1] == 5
    assert table3[1][0] == 2
    assert table3[1][1] == 6
    assert table3[2][0] == 3
    assert table3[2][1] == 7
    assert table3[3][0] == 4
    assert table3[3][1] == 8

    assert len(table5) == 4
    assert len(table5[0]) == 2
    assert table5[0][0] == 'one'
    assert table5[0][1] == 1
    assert table5[1][0] == 'two'
    assert table5[1][1] == 2
    assert table5[2][0] == 'three'
    assert table5[2][1] == 3
    assert table5[3][0] == 'four'
    assert table5[3][1] == 4

def testProduceAggregateTable():
    """
    TODO: add docstring
    """
    data1 = np.array([[1, 2, 3, 1], [3, 3, 1, 5], [1, 1, 5, 2]])
    names1 = ['var1', 'var2', 'var3', 'var4']

    trainObj = create('List', data1, names1)
    rawTable = produceAggregateTable(trainObj)

    for i in range(len(rawTable[0])):
        funcName = rawTable[0][i]
        if funcName == "proportionZero":
            assert rawTable[1][i] == 0.0
        elif funcName == "proportionMissing":
            assert rawTable[1][i] == 0.0
        elif funcName == "Points":
            assert rawTable[1][i] == 12
        elif funcName == "Features":
            assert rawTable[1][i] == 4


def testStDev():
    testRowList = create('List', np.array([[1], [1], [3], [4], [2], [6], [12], [0]]), ['nums'])
    stDevContainer = testRowList.applyFunctionToEachFeature(standardDeviation)
    stDev = stDevContainer.toListOfLists()[0][0]
    assert_almost_equal(stDev, 3.6379, 3)








