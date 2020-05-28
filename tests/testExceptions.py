"""
Test the nimble exception hierarchy, output and catching
"""
from nose.tools import raises

from nimble.exceptions import NimbleException
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentTypeCombination
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction, PackageException
from nimble.exceptions import FileFormatException
from .assertionHelpers import noLogEntryExpected

@raises(TypeError)
def test_NimbleException_noMessage():
    # TypeError raised because value missing for __init__
    raise NimbleException()
    assert False

@raises(NimbleException)
def test_NimbleException_withMessage():
    raise NimbleException('message')
    assert False

@noLogEntryExpected
def test_NimbleException_strOutput():
    try:
        raise NimbleException('message')
        assert False
    except NimbleException as e:
        assert str(e) == 'message'

def test_NimbleException_reprOutput():
    try:
        raise NimbleException('message')
        assert False
    except NimbleException as e:
        assert repr(e) == "NimbleException('message')"

def test_NimbleException_customSubClassInheritance():
    class CustomException(NimbleException):
        pass

    try:
        raise CustomException('custom message')
        assert False
    except CustomException as e:
        assert str(e) == 'custom message'
        assert repr(e) == "CustomException('custom message')"

    try:
        raise CustomException('custom message')
        assert False
    except NimbleException:
        pass

    try:
        raise CustomException('custom message')
        assert False
    except Exception:
        pass

    try:
        raise NimbleException('exception')
        assert False
    except CustomException:
        assert False
    except NimbleException:
        pass

@noLogEntryExpected
def back_NimbleExceptions(exception, subClassOf=None):
    try:
        raise exception('exception')
        assert False
    except exception:
        pass

    try:
        raise exception('exception')
        assert False
    except NimbleException:
        pass

    if subClassOf:
        try:
            raise exception('exception')
            assert False
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
