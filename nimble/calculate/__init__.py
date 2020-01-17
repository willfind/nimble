"""
This loosely groups together functions which perform calculations on
data objects and other nimble defined objects, including functions which
can be used as performance functions in the nimble testing and cross
validation API. Some of these are also available as methods off of data
objects; the versions here are functions, and take any inputs as
arguments.
"""

from .confidence import confidenceIntervalHelper
from .loss import fractionIncorrect
from .loss import meanAbsoluteError
from .loss import meanFeaturewiseRootMeanSquareError
from .loss import rootMeanSquareError
from .loss import varianceFractionRemaining
from .matrix import elementwiseMultiply
from .matrix import elementwisePower
from .similarity import correlation
from .similarity import cosineSimilarity
from .similarity import covariance
from .similarity import fractionCorrect
from .similarity import rSquared
from .similarity import confusionMatrix
from .statistic import maximum
from .statistic import mean
from .statistic import median
from .statistic import mode
from .statistic import minimum
from .statistic import uniqueCount
from .statistic import proportionMissing
from .statistic import proportionZero
from .statistic import quartiles
from .statistic import residuals
from .statistic import standardDeviation
from .utility import detectBestResult
from .linalg import inverse
from .linalg import pseudoInverse
from .linalg import solve
from .linalg import leastSquaresSolution
from .binary import truePositive
from .binary import trueNegative
from .binary import falsePositive
from .binary import falseNegative
from .binary import recall
from .binary import precision
from .binary import specificity
from .binary import balancedAccuracy
from .binary import f1Score


__all__ = ['balancedAccuracy', 'confidenceIntervalHelper', 'confusionMatrix',
           'correlation', 'cosineSimilarity', 'covariance', 'detectBestResult',
           'elementwiseMultiply', 'elementwisePower', 'f1Score',
           'falseNegative', 'falsePositive', 'fractionCorrect',
           'fractionIncorrect', 'inverse', 'leastSquaresSolution', 'maximum',
           'mean', 'meanAbsoluteError', 'meanFeaturewiseRootMeanSquareError',
           'median', 'minimum', 'mode', 'precision', 'proportionMissing',
           'proportionZero', 'pseudoInverse', 'quartiles', 'rSquared',
           'recall', 'residuals', 'rootMeanSquareError', 'solve',
           'specificity', 'standardDeviation', 'trueNegative', 'truePositive',
           'uniqueCount', 'varianceFractionRemaining']
