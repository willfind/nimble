
import numpy as np
import pytest

import nimble
from nimble._utility import scipy
from nimble.core.logger.data_set_analyzer import featurewiseFunctionGenerator
from nimble.core.logger.data_set_analyzer import produceFeaturewiseInfoTable
from nimble.core.logger.data_set_analyzer import appendColumns
from nimble.core.logger.data_set_analyzer import produceAggregateTable
from tests.helpers import getDataConstructors


def testProduceInfoTable_denseData():
    """
    Test the functionality of calculating statistical/informational
    funcs on objects with dense data using the produceInfoTable.
    """
    data1 = np.array([[1, 2, 3, 1], [3, 3, 1, 5], [1, 1, 5, 2]])
    names1 = ['var1', 'var2', 'var3', 'var4']
    for constructor in getDataConstructors():
        trainObj = constructor(source=data1, featureNames=names1)
        funcs = featurewiseFunctionGenerator()
        rawTable = produceFeaturewiseInfoTable(trainObj, funcs)
        funcNames = rawTable[0]
        for i in range(len(funcNames)):
            funcName = funcNames[i]
            if funcName == "mean":
                assert rawTable[1][i] == pytest.approx(1.6667, abs=1e-4)
                assert rawTable[2][i] == pytest.approx(2.000, abs=1e-4)
                assert rawTable[3][i] == pytest.approx(3.000, abs=1e-4)
                assert rawTable[4][i] == pytest.approx(2.6667, abs=1e-4)
            elif funcName == "minimum":
                assert rawTable[1][i] == 1
                assert rawTable[2][i] == 1
                assert rawTable[3][i] == 1
                assert rawTable[4][i] == 1
            elif funcName == "maximum":
                assert rawTable[1][i] == 3
                assert rawTable[2][i] == 3
                assert rawTable[3][i] == 5
                assert rawTable[4][i] == 5
            elif funcName == "uniqueCount":
                assert rawTable[1][i] == 2
                assert rawTable[2][i] == 3
                assert rawTable[3][i] == 3
                assert rawTable[4][i] == 3
            elif funcName == "standardDeviation":
                assert rawTable[1][i] == pytest.approx(1.1547, abs=1e-4)
                assert rawTable[2][i] == pytest.approx(1.0000, abs=1e-4)
                assert rawTable[3][i] == pytest.approx(2.0000, abs=1e-4)
                assert rawTable[4][i] == pytest.approx(2.0816, abs=1e-4)
            elif funcName == "median":
                assert rawTable[1][i] == 1.0
                assert rawTable[2][i] == 2.0
                assert rawTable[3][i] == 3.0
                assert rawTable[4][i] == 2.0


def testProduceInfoTable_sparseData():
    """
    Test the functionality of calculating statistical/informational
    funcs on objects with sparse data using the produceInfoTable.

    Testing matrix is 6 x 6, with 11 non-zero values, 1 None/Missing value,
    and 24 zero values.
    """
    row = np.array([0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5])
    col = np.array([0, 4, 2, 3, 1, 3, 4, 0, 1, 3, 4, 5])
    vals = np.array([1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1])

    raw = scipy.sparse.coo_matrix((vals, (row, col)))

    for constructor in getDataConstructors():
        testObj = constructor(source=raw)
        funcs = featurewiseFunctionGenerator()
        rawTable = produceFeaturewiseInfoTable(testObj, funcs)
        funcNames = rawTable[0]
        for i in range(len(funcNames)):
            funcName = funcNames[i]
            if funcName == "mean":
                assert rawTable[1][i] == pytest.approx(0.3333, abs=1e-4)
                assert rawTable[2][i] == pytest.approx(0.3333, abs=1e-4)
                assert rawTable[3][i] == pytest.approx(0.1667, abs=1e-4)
                assert rawTable[4][i] == pytest.approx(0.4000, abs=1e-4)
                assert rawTable[5][i] == pytest.approx(0.5000, abs=1e-4)
                assert rawTable[6][i] == pytest.approx(0.1667, abs=1e-4)
            elif funcName == "minimum":
                assert rawTable[1][i] == 0
                assert rawTable[2][i] == 0
                assert rawTable[3][i] == 0
                assert rawTable[4][i] == 0
                assert rawTable[5][i] == 0
                assert rawTable[6][i] == 0
            elif funcName == "maximum":
                assert rawTable[1][i] == 1
                assert rawTable[2][i] == 1
                assert rawTable[3][i] == 1
                assert rawTable[4][i] == 1
                assert rawTable[5][i] == 1
                assert rawTable[6][i] == 1
            elif funcName == "uniqueCount":
                assert rawTable[1][i] == 2
                assert rawTable[2][i] == 2
                assert rawTable[3][i] == 2
                assert rawTable[4][i] == 2
                assert rawTable[5][i] == 2
                assert rawTable[6][i] == 2
            elif funcName == "standardDeviation":
                assert rawTable[1][i] == pytest.approx(0.5164, abs=1e-4)
                assert rawTable[2][i] == pytest.approx(0.5164, abs=1e-4)
                assert rawTable[3][i] == pytest.approx(0.4082, abs=1e-4)
                assert rawTable[4][i] == pytest.approx(0.5477, abs=1e-4)
                assert rawTable[5][i] == pytest.approx(0.5477, abs=1e-4)
                assert rawTable[6][i] == pytest.approx(0.4082, abs=1e-4)
            elif funcName == "median":
                assert rawTable[1][i] == 0
                assert rawTable[2][i] == 0
                assert rawTable[3][i] == 0
                assert rawTable[4][i] == 0
                assert rawTable[5][i] == 0.5
                assert rawTable[6][i] == 0


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

    trainObj = nimble.data('List', source=data1, featureNames=names1)
    rawTable = produceAggregateTable(trainObj)

    for i in range(len(rawTable[0])):
        funcName = rawTable[0][i]
        if funcName == "proportionZero":
            assert rawTable[1][i] == 0.0
        elif funcName == "proportionMissing":
            assert rawTable[1][i] == 0.0
        elif funcName == "Values":
            assert rawTable[1][i] == 12
        elif funcName == "Features":
            assert rawTable[1][i] == 4
        elif funcName == "points":
            assert rawTable[1][i] == 3
