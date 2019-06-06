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
initialization and are avaiable in the interface named 'custom'.
"""

from __future__ import absolute_import

from .custom_learner import CustomLearner
from .knn_classification import KNNClassifier
from .mean_constant import MeanConstant
from .multioutput_ridge_regression import MultiOutputRidgeRegression
from .ridge_regression import RidgeRegression

__all__ = ['CustomLearner', 'KNNClassifier', 'MeanConstant',
           'MultiOutputRidgeRegression', 'RidgeRegression']
