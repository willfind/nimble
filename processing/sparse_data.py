"""
Class extending BaseData, defining a generic sparse matrix wrapper class. Specific implementations
of sparse matrices should extend this one. This class defines functions which will either be
the same across all specific formats, or throw warnings of slow operations and use conversions
to non-sparse formats in order to complete their operations.

"""

from scipy.sparse import *
from copy import copy
from copy import deepcopy

from base_data import *
from ..utility.custom_exceptions import ArgumentException


class SparseData(BaseData):


	def __init__(self, data=None, featureNames=None):
		if not isspmatrix(data):
			raise ArgumentException("data must be a sparse matrix")
		self.data = data
		super(SparseData, self).__init__(featureNames)



	def _columns_implementation(self):
		(points, cols) = self.data.get_shape()
		return cols

	def _points_implementation(self):
		(points, cols) = self.data.get_shape()
		return points

	


