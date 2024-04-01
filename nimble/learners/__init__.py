
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
These out-of-the box learners are all registered during Nimble initialization.

They are registered under the name 'nimble' to allow access as a string,
e.g. ``nimble.train('nimble.KNNClassifier', ...)``
"""

from .knn_classification import KNNClassifier
from .multioutput_ridge_regression import MultiOutputRidgeRegression
from .multioutput_linear_regression import MultiOutputLinearRegression
from .ridge_regression import RidgeRegression
from .knn_imputation import KNNImputation
from .._utility import _setAll

__all__ = _setAll(vars())
