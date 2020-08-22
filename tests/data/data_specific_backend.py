"""
Define data test objects for methods of properties specific to the type

"""

from unittest.mock import patch
from tests.helpers import CalledFunctionException, calledException
from .baseObject import DataTestObject

class SparseSpecific(DataTestObject):
    """ Tests for Sparse (non-view) implementation details """

    def test_sortInternal_avoidsUnecessary(self):
        data = [[1, 0, 3], [0, 5, 0]]
        obj = self.constructor(data)

        # ensure that our mock target is used
        try:
            with patch("numpy.lexsort", calledException):
                obj._sortInternal('point')
            assert False # expected CalledFunctionException
        except CalledFunctionException:
            pass

        obj._sortInternal('point')
        assert obj._sorted['axis'] == 'point'
        assert obj._sorted['indices'] is None

        # call _sortInternal to generate indices on already sorted obj
        with patch("numpy.lexsort", calledException):
            obj._sortInternal('point', setIndices=True)

        # Confirm the desired action actually took place
        assert obj._sorted['indices'] is not None
