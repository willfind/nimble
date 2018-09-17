import UML
import numpy

from UML.match import convertMatchToFunction
from UML.exceptions import ArgumentException

def factory(match, fill, arguments=None):
    if not hasattr(match, '__call__'):
        match = convertMatchToFunction(match)
    if not hasattr(fill, '__call__'):
        fillConstant = fill
        if fillConstant is None:
            fillConstant = numpy.nan
        fill = lambda v, m: [fillConstant if m(val) else val for val in v]
    if arguments is None:
        def fillFunction(vector):
            return fill(vector, match)
    else:
        def fillFunction(vector):
            return fill(vector, match, arguments)
    return fillFunction

def mean(vector, match):
    unmatched = UML.createData('List', [val for val in vector if not match(val)])
    # if len(unmatched) == len(vector):
    #     return vector
    if len(unmatched) == 0:
        msg = 'Cannot calculate mean'
        raise ArgumentException(msg)
    mean = UML.calculate.mean(unmatched)
    return [mean if match(val) else val for val in vector]


def median(vector, match):
    unmatched = UML.createData('List', [val for val in vector if not match(val)])
    # if len(unmatched) == len(vector):
    #     return vector
    if len(unmatched) == 0:
        msg = 'Cannot calculate median'
        raise ArgumentException(msg)
    median = UML.calculate.median(unmatched)
    return [median if match(val) else val for val in vector]

def mode(vector, match):
    unmatched = UML.createData('List', [val for val in vector if not match(val)])
    # if len(unmatched) == len(vector):
    #     return vector
    if len(unmatched) == 0:
        msg = 'Cannot calculate mode'
        raise ArgumentException(msg)
    mode = UML.calculate.mode(unmatched)
    return [mode if match(val) else val for val in vector]

def forward(vector, match):
    vector = vector.copyAs('pythonlist', outputAs1D=True)
    matchIndices = [i for i,v in enumerate(vector) if match(v)]
    if 0 in matchIndices:
        msg = 'Cannot forward fill'
        raise ArgumentException(msg)
    for i in matchIndices:
        if i > 0:
            vector[i] = vector[i - 1]
    return vector

def backward(vector, match):
    vector = list(reversed(vector.copyAs('pythonlist', outputAs1D=True)))
    matchIndices = [i for i,v in enumerate(vector) if match(v)]
    if 0 in matchIndices:
        msg = 'Cannot backward fill'
        raise ArgumentException(msg)
    for i in matchIndices:
        vector[i] = vector[i - 1]
    vector = list(reversed(vector))
    return vector

def interpolate(vector, match, arguments=None):
    x = [i for i,v in enumerate(vector) if match(v)]
    if arguments is not None:
        try:
            tmpArguments = arguments.copy()
            tmpArguments['x'] = x
        except Exception:
            msg = 'for fill.interpolate, arguments must be None or a dict.'
            raise ArgumentException(msg)
    else:
        xp = [i for i,v in enumerate(vector) if not match(v)]
        fp = [v for i,v in enumerate(vector) if not match(v)]
        tmpArguments = {'x': x, 'xp': xp, 'fp': fp}
    tmpV = numpy.interp(**tmpArguments)
    vector = vector.copyAs('pythonlist', outputAs1D=True)
    for j, i in enumerate(x):
        vector[i] = tmpV[j]
    return vector
