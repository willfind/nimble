"""
Module defining exceptions to be used in UML.

"""
from __future__ import absolute_import
from six.moves import range

class UMLException(Exception):
    """
    Override Python's Exception, requiring a value upon instantiation.
    """
    def __init__(self, value):
        self.value = value
        self.className = self.__class__.__name__

    def __str__(self):
        return repr(self.value)

    def __repr__(self):
        return "{0}({1})".format(self.className, repr(self.value))

class ArgumentException(UMLException):
    pass

class ImproperActionException(UMLException, TypeError):
    pass

class InvalidArgumentType(UMLException, TypeError):
    """
    Raised when an argument type causes a failure.

    This exception occurs when the type of an argument is not accepted.
    This operation does not accept this type for this argument. This is
    a subclass of Python's TypeError.
    """
    pass

class InvalidArgumentValue(UMLException, ValueError):
    """
    Raised when an argument value causes a failure.

    This exception occurs when the value of an argument is not accepted.
    This operation may only accept a limited set of possible values for
    this argument, a sentinel value may be preventing this operation, or
    the value may be disallowed for this operation. This is a subclass
    of Python's ValueError.
    """
    pass

class InvalidTypeCombination(UMLException, TypeError):
    """
    Raised when the types of two or more arguments causes a failure.

    This exception occurs when the type of a certain argument is based
    on another argument. The type may be accepted for this argument in
    other instances but not in this specific case. This is a subclass of
    Python's TypeError.
    """
    pass

class InvalidValueCombination(UMLException, TypeError):
    """
    Raised when the values of two or more arguments causes a failure.

    This exception occurs when the value of a certain argument is based
    on another argument. The value may be accepted for this argument in
    other instances but not in this specific case. This is a subclass of
    Python's ValueError.
    """
    pass

class NewImproperActionException(UMLException, TypeError):
    """
    Raised when an operation on an object is not supported.

    A method or operation is disallowed due to current attributes of the
    object. As an example, calling car.turnOn() when the car object is
    already running. This is a subclass of Python's TypeError.
    """
    pass

# class InvalidObjectValue(UMLException, ValueError):
#     """
#     Raised when a value related to the object causes a failure.
#
#     This exception occurs when the calling object has a characteristic
#     that prevents this operation. These often occur due to a value in
#     the object's data or value describing the object's shape preventing
#     the operation, but any other value describing this object could also
#     cause this exception. This is a subclass of Python's ValueError.
#     """
#     pass

class PackageException(UMLException, ImportError):
    """
    Raised when a package is not installed, but needed.

    This is a subclass of Python's ImportError.
    """
    pass

class MissingEntryException(Exception):
    """
    Exception to be thrown when a dictionary or other data structure is missing
    an entry that is necessary for the proper completion of an operation.
    """

    def __init__(self, entryNames, value):
        self.entryNames = entryNames
        self.value = value

    def __str__(self):
        errMessage = "Missing entry names: " + self.entryNames[0]
        for entryName in self.entryNames[1:]:
            errMessage += ", " + entryName
        errMessage += '. \n' + repr(self.value)

class EmptyFileException(UMLException):
    """
    Raised a file to be read is empty
    """
    pass

class FileFormatException(UMLException):
    """
    Raised when the formatting of a file is not as expected.
    """
    pass


def prettyListString(inList, useAnd=False, numberItems=False, itemStr=str):
    """
    Used in the creation of exception messages to display lists in a more
    appealing way than default

    """
    ret = ""
    number = 0
    for i in range(len(inList)):
        value = inList[i]
        if i > 0:
            ret += ', '
            if useAnd and i == len(inList) - 1:
                ret += 'and '
        if numberItems:
            ret += '(' + str(number) + ') '
        ret += itemStr(value)
    return ret


def prettyDictString(inDict, useAnd=False, numberItems=False, keyStr=str, delim='=',
                     valueStr=str):
    """
    Used in the creation of exception messages to display dicts in a more
    appealing way than default.

    """
    ret = ""
    number = 0
    keyList = list(inDict.keys())
    for i in range(len(keyList)):
        key = keyList[i]
        value = inDict[key]
        if i > 0:
            ret += ', '
            if useAnd and i == len(keyList) - 1:
                ret += 'and '
        if numberItems:
            ret += '(' + str(number) + ') '
        ret += keyStr(key) + delim + valueStr(value)
    return ret
