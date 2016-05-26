
# Most of the functions in UML.calculate.statitic are tested not directly
# in this module, but through the functions that call them: featureReport
# in UML.logger.tests.data_set_analyzier_tests and in the data
# hierarchy in UML.data.tests.query_backend

import numpy as np

from nose.tools import assert_almost_equal

from UML import createData
from UML.calculate import standardDeviation
from UML.calculate import quartiles

def testStDev():
    dataArr = np.array([[1], [1], [3], [4], [2], [6], [12], [0]])
    testRowList = createData('List', data=dataArr, featureNames=['nums'])
    stDevContainer = testRowList.calculateForEachFeature(standardDeviation)
    stDev = stDevContainer.copyAs(format="python list")[0][0]
    assert_almost_equal(stDev, 3.6379, 3)

def testQuartilesAPI():
    raw = [5]
    obj = createData("Matrix", raw)
    ret = quartiles(obj.pointView(0))
    assert ret == (5,5,5)

    raw = [1,1,3,3]
    ret = quartiles(raw)
    assert ret == (1,2,3)

    raw = [0,1,2,3,4,5,6,7,8]
    obj = createData("List", raw)
    ret = quartiles(obj)
    assert ret == (2,4,6)
