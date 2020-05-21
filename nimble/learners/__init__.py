"""
Contains the base class needed for users to make custom learning
algorithms which run native in nimble and python, along with several
out-of-the box examples of custom learners.

If one wishes to write their own learners in nimble, then they must
inherit from the abstract base class named CustomLearner and fill
in the abstract methods, and then register the class object with
nimble using the nimble.registerCustomLearner function (or
nimble.registerCustomLearnerAsDefault if they want the learner to be
loaded during nimble initialization). Later, during the same or
different nimble session, custom learners can be removed from view
using the nimble.deregisterCustomLearner and
nimble.deregisterCustomLearnerAsDefault functions.

The out-of-the box learners are all registered during nimble
initialization and are available in the interface named 'custom'.
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
