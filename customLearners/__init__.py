"""
Contains the base class needed for users to make custom learning
algorithms which run native in UML and python, along with several
out-of-the box examples of custom learners.

If one wishes to write their own learners in UML, then they must
inherit from the abstract base class named CustomLearner and fill
in the abstract methods, and then register the class object with
UML using the UML.registerCustomLearner function (or
UML.registerCustomLearnerAsDefault if they want the learner to be
loaded during UML initialization). Later, during the same or
different UML session, custom learners can be removed from view
using the UML.deregisterCustomLearner and
UML.deregisterCustomLearnerAsDefault functions.

The out-of-the box learners are all registered during UML
initialization and are avaiable in the interface named 'custom'.

"""

from custom_learner import CustomLearner
from knn_classification import KNNClassifier
from mean_constant import MeanConstant
from multioutput_ridge_regression import MultiOutputRidgeRegression
from ridge_regression import RidgeRegression

__all__ = ['CustomLearner', 'KNNClassifier', 'MeanConstant',
			'MultiOutputRidgeRegression', 'RidgeRegression']
