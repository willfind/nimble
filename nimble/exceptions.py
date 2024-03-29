
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
Module defining exceptions used in Nimble.
"""

class NimbleException(Exception):
    """
    Override Python's Exception, requiring a message upon instantiation.
    """
    def __init__(self, message):
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.message)})"

class InvalidArgumentType(NimbleException, TypeError):
    """
    Raised when an argument type causes a failure.

    This exception occurs when the type of an argument is not accepted.
    This operation does not accept this type for this argument. This is
    a subclass of Python's TypeError.
    """

class InvalidArgumentValue(NimbleException, ValueError):
    """
    Raised when an argument value causes a failure.

    This exception occurs when the value of an argument is not accepted.
    This operation may only accept a limited set of possible values for
    this argument, a sentinel value may be preventing this operation, or
    the value may be disallowed for this operation. This is a subclass
    of Python's ValueError.
    """

class InvalidArgumentTypeCombination(NimbleException, TypeError):
    """
    Raised when the types of two or more arguments causes a failure.

    This exception occurs when the type of a certain argument is based
    on another argument. The type may be accepted for this argument in
    other instances but not in this specific case. This is a subclass of
    Python's TypeError.
    """

class InvalidArgumentValueCombination(NimbleException, ValueError):
    """
    Raised when the values of two or more arguments causes a failure.

    This exception occurs when the value of a certain argument is based
    on another argument. The value may be accepted for this argument in
    other instances but not in this specific case. This is a subclass of
    Python's ValueError.
    """

class ImproperObjectAction(NimbleException, TypeError):
    """
    Raised when the characteristics of the object prevent the operation.

    This exception occurs when an operation cannot be completed due to
    the object's attribute value or an invalid value in the object's
    data.
    """

class PackageException(NimbleException, ImportError):
    """
    Raised when a package is not installed, but needed.

    This is a subclass of Python's ImportError.
    """

class FileFormatException(NimbleException, ValueError):
    """
    Raised when the formatting of a file is not as expected.

    This is a subclass of Python's ValueError.
    """

# can't use _utility._setAll here due to circular import
__all__ = sorted(var for var in vars() if not var.startswith('_'))
