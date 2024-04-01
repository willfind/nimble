
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
Matrix calculations.
"""

from nimble.core.data import Base
from nimble.exceptions import InvalidArgumentType

def elementwiseMultiply(left, right):
    """
    Perform element wise multiplication of two provided nimble Base objects
    with the result being returned in a separate nimble Base object. Both
    objects must contain only numeric data. The pointCount and featureCount
    of both objects must be equal. The types of the two objects may be
    different. None is always returned.

    """
    # check left is nimble
    if not isinstance(left, Base):
        msg = "'left' must be an instance of a nimble data object"
        raise InvalidArgumentType(msg)

    return left * right

def elementwisePower(left, right):
    """
    Perform an element-wise power operation, with the values in the left object
    as the bases and the values in the right object as exponents. A new object
    will be created, and the input obects will be un-modified.
    """
    # check left is nimble

    if not isinstance(left, Base):
        msg = "'left' must be an instance of a nimble data object"
        raise InvalidArgumentType(msg)

    return left ** right
