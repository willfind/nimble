"""
Test the nimble exception hierarchy, output and catching
"""
from __future__ import absolute_import

from nose.tools import raises

from UML.exceptions import nimbleException
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue
from UML.exceptions import InvalidArgumentTypeCombination
from UML.exceptions import InvalidArgumentValueCombination
from UML.exceptions import ImproperObjectAction, PackageException
from UML.exceptions import FileFormatException

@raises(TypeError)
def test_nimbleException_noMessage():
    # TypeError raised because value missing for __init__
    raise nimbleException()
    assert False

@raises(nimbleException)
def test_nimbleException_withMessage():
    raise nimbleException('message')
    assert False

def test_nimbleException_strOutput():
    try:
        raise nimbleException('message')
        assert False
    except nimbleException as e:
        assert str(e) == "'message'"

def test_nimbleException_reprOutput():
    try:
        raise nimbleException('message')
        assert False
    except nimbleException as e:
        assert repr(e) == "nimbleException('message')"

def test_nimbleException_customSubClassInheritance():
    class CustomException(nimbleException):
        pass

    try:
        raise CustomException('custom message')
        assert False
    except CustomException as e:
        assert str(e) == "'custom message'"
        assert repr(e) == "CustomException('custom message')"

    try:
        raise CustomException('custom message')
        assert False
    except nimbleException:
        pass

    try:
        raise CustomException('custom message')
        assert False
    except Exception:
        pass

    try:
        raise nimbleException('exception')
        assert False
    except CustomException:
        assert False
    except nimbleException:
        pass

def back_nimbleExceptions(exception, subClassOf=None):
    try:
        raise exception('exception')
        assert False
    except exception:
        pass

    try:
        raise exception('exception')
        assert False
    except nimbleException:
        pass

    if subClassOf:
        try:
            raise exception('exception')
            assert False
        except subClassOf:
            pass

def test_ImproperObjectAction():
    back_nimbleExceptions(ImproperObjectAction, TypeError)

def test_InvalidArgumentType():
    back_nimbleExceptions(InvalidArgumentType, TypeError)

def test_InvalidArgumentValue():
    back_nimbleExceptions(InvalidArgumentValue, ValueError)

def test_InvalidArgumentTypeCombination():
    back_nimbleExceptions(InvalidArgumentTypeCombination, TypeError)

def test_InvalidArgumentValueCombination():
    back_nimbleExceptions(InvalidArgumentValueCombination, ValueError)

def test_PackageException():
    back_nimbleExceptions(PackageException, ImportError)

def test_FileFormatException():
    back_nimbleExceptions(FileFormatException, ValueError)
