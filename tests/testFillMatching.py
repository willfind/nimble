
import mock

import nimble
from nimble import match
from nimble.exceptions import ImproperObjectAction, InvalidArgumentValue
from .assertionHelpers import calledException, CalledFunctionException

def test_fillMatching_exception_nansUnmatched():
    raw = [[1, 1, 1, 0], [1, 1, 1, None], [2, 2, 2, 0], [2, 2, 2, 3],
           [2, 2, 2, 4], [2, 2, 2, 4]]
    for t in nimble.data.available:
        data = nimble.createData(t, raw)
        try:
            nimble.fillMatching('skl.IterativeImputer', 1, data)
            assert False # expected ImproperObjectAction
        except ImproperObjectAction:
            pass

@mock.patch('nimble.core.trainAndApply', new=calledException)
def test_fillMatching_trainXUnaffectedByFailure():
    raw = [[2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 2, 0], [2, 2, 2, 3],
           [2, 2, 2, 4], [2, 2, 2, 4]]
    for t in nimble.data.available:
        data = nimble.createData(t, raw)
        dataCopy = data.copy()
        # trying to fill 2 will fail because the training data will be empty
        try:
            nimble.fillMatching('skl.IterativeImputer', 2, data)
            assert False # expected InvalidArgumentValue
        except CalledFunctionException:
            assert data == dataCopy

def backend_fillMatching(matchingElements, raw, expRaw):
    for t in nimble.data.available:
        if t == 'Sparse':
            # at this time, skl.IterativeImputer does not support sparse data
            continue
        data = nimble.createData(t, raw)
        exp = nimble.createData(t, expRaw)
        nimble.fillMatching('skl.IterativeImputer', matchingElements, data,
                            estimator=nimble.Init('KNeighborsClassifier',
                                                  n_neighbors=1))
        assert data == exp

def test_fillMatching_matchingElementsAsSingleValue():
    matchingElements = 0
    raw = [[1, 1, 1, 0], [1, 1, 1, 1], [2, 2, 2, 0], [2, 2, 2, 4]]
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)

def test_fillMatching_matchingElementsAsList():
    matchingElements = [-1, -2]
    raw = [[1, 1, 1, -1], [1, 1, 1, 1], [2, 2, 2, -2], [2, 2, 2, 4]]
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)

def test_fillMatching_matchingElementsAsFunction():
    matchingElements = match.negative
    raw = [[1, 1, 1, -1], [1, 1, 1, 1], [2, 2, 2, -2], [2, 2, 2, 4]]
    expRaw = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 4], [2, 2, 2, 4]]
    backend_fillMatching(matchingElements, raw, expRaw)


