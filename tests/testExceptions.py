"""
Test the UML exception hierarchy, output and catching
"""
from __future__ import absolute_import

from nose.tools import raises

from UML.exceptions import UMLException
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue
from UML.exceptions import InvalidTypeCombination, InvalidValueCombination
from UML.exceptions import ImproperActionException, PackageException
from UML.exceptions import FileFormatException

@raises(TypeError)
def test_UMLException_noMessage():
    # TypeError raised because value missing for __init__
    raise UMLException()
    assert False

@raises(UMLException)
def test_UMLException_withMessage():
    raise UMLException('message')
    assert False

def test_UMLException_strOutput():
    try:
        raise UMLException('message')
        assert False
    except UMLException as e:
        assert str(e) == "'message'"

def test_UMLException_reprOutput():
    try:
        raise UMLException('message')
        assert False
    except UMLException as e:
        assert repr(e) == "UMLException('message')"

def test_UMLException_customSubClassInheritance():
    class CustomException(UMLException):
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
    except UMLException:
        pass

    try:
        raise UMLException('exception')
        assert False
    except CustomException:
        assert False
    except UMLException:
        pass

def back_UMLExceptions(exception, subClassOf=None):
    try:
        raise exception('exception')
        assert False
    except exception:
        pass

    try:
        raise exception('exception')
        assert False
    except UMLException:
        pass

    if subClassOf:
        try:
            raise exception('exception')
            assert False
        except subClassOf:
            pass

def test_ImproperActionException():
    back_UMLExceptions(ImproperActionException, TypeError)

def test_InvalidArgumentType():
    back_UMLExceptions(InvalidArgumentType, TypeError)

def test_InvalidArgumentValue():
    back_UMLExceptions(InvalidArgumentValue, ValueError)

def test_InvalidTypeCombination():
    back_UMLExceptions(InvalidTypeCombination, TypeError)

def test_InvalidValueCombination():
    back_UMLExceptions(InvalidValueCombination, ValueError)

def test_PackageException():
    back_UMLExceptions(PackageException, ImportError)

def test_FileFormatException():
    back_UMLExceptions(FileFormatException, ValueError)
