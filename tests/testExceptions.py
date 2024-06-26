
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
Test the nimble exception hierarchy, output and catching
"""

from nimble.exceptions import NimbleException
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentTypeCombination
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction, PackageException
from nimble.exceptions import FileFormatException
from tests.helpers import raises
from tests.helpers import noLogEntryExpected

@raises(TypeError)
def test_NimbleException_noMessage():
    # TypeError raised because value missing for __init__
    raise NimbleException()

@raises(NimbleException)
def test_NimbleException_withMessage():
    raise NimbleException('message')

@noLogEntryExpected
def test_NimbleException_strOutput():
    try:
        raise NimbleException('message')
    except NimbleException as e:
        assert str(e) == 'message'

def test_NimbleException_reprOutput():
    try:
        raise NimbleException('message')
    except NimbleException as e:
        assert repr(e) == "NimbleException('message')"

def test_NimbleException_customSubClassInheritance():
    class CustomException(NimbleException):
        pass

    try:
        raise CustomException('custom message')
    except CustomException as e:
        assert str(e) == 'custom message'
        assert repr(e) == "CustomException('custom message')"

    try:
        raise CustomException('custom message')
    except NimbleException:
        pass

    try:
        raise CustomException('custom message')
    except Exception:
        pass

    try:
        raise NimbleException('exception')
    except CustomException:
        assert False
    except NimbleException:
        pass

@noLogEntryExpected
def back_NimbleExceptions(exception, subClassOf=None):
    try:
        raise exception('exception')
    except exception:
        pass

    try:
        raise exception('exception')
    except NimbleException:
        pass

    if subClassOf:
        try:
            raise exception('exception')
        except subClassOf:
            pass

def test_ImproperObjectAction():
    back_NimbleExceptions(ImproperObjectAction, TypeError)

def test_InvalidArgumentType():
    back_NimbleExceptions(InvalidArgumentType, TypeError)

def test_InvalidArgumentValue():
    back_NimbleExceptions(InvalidArgumentValue, ValueError)

def test_InvalidArgumentTypeCombination():
    back_NimbleExceptions(InvalidArgumentTypeCombination, TypeError)

def test_InvalidArgumentValueCombination():
    back_NimbleExceptions(InvalidArgumentValueCombination, ValueError)

def test_PackageException():
    back_NimbleExceptions(PackageException, ImportError)

def test_FileFormatException():
    back_NimbleExceptions(FileFormatException, ValueError)
