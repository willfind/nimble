"""
These out-of-the box learners are all registered during nimble
initialization and are available in the interface named 'nimble'.
"""

from .knn_classification import KNNClassifier
from .mean_constant import MeanConstant
from .multioutput_ridge_regression import MultiOutputRidgeRegression
from .multioutput_linear_regression import MultiOutputLinearRegression
from .ridge_regression import RidgeRegression
from .knn_imputation import KNNImputation

__all__ = ['KNNClassifier', 'KNNImputation', 'MeanConstant',
           'MultiOutputLinearRegression', 'MultiOutputRidgeRegression',
           'RidgeRegression']
