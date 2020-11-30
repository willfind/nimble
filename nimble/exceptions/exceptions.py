"""
Module defining exceptions used in nimble.
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
        return "{cls}({msg})".format(cls=self.__class__.__name__,
                                     msg=repr(self.message))

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


def _prettyListString(inList, useAnd=False, numberItems=False, itemStr=str):
    """
    Used in the creation of exception messages to display lists in a
    more appealing way than default.
    """
    ret = ""
    for i, value in enumerate(inList):
        if i > 0:
            ret += ', '
            if useAnd and i == len(inList) - 1:
                ret += 'and '
        if numberItems:
            ret += '(' + str(i) + ') '
        ret += itemStr(value)
    return ret


def _prettyDictString(inDict, useAnd=False, numberItems=False, keyStr=str,
                      delim='=', valueStr=str):
    """
    Used in the creation of exception messages to display dicts in a
    more appealing way than default.
    """
    ret = ""
    for i, (key, value) in enumerate(inDict.items()):
        if i > 0:
            ret += ', '
            if useAnd and i == len(inDict) - 1:
                ret += 'and '
        if numberItems:
            ret += '(' + str(i) + ') '
        ret += keyStr(key) + delim + valueStr(value)
    return ret
