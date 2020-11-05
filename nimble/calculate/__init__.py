"""
This module loosely groups together functions which perform calculations
on data objects and other nimble defined objects, including functions
which can be used as performance functions in the nimble testing and
cross-validation API. Some similar functionality may be available as
methods off of data objects; the versions here are functions, and take
any inputs as arguments.

..
   The statement below applies to the online documentation

**Note:** The groups of similar functions created below (and in the
source code) are for organizational purposes only. All of the functions
can be imported and called directly from ``nimble.calculate``.
"""

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
from .statistic import count
from .statistic import maximum
from .statistic import mean
from .statistic import median
from .statistic import medianAbsoluteDeviation
from .statistic import mode
from .statistic import minimum
from .statistic import uniqueCount
from .statistic import proportionMissing
from .statistic import proportionZero
from .statistic import quartiles
from .statistic import residuals
from .statistic import standardDeviation
from .statistic import sum
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
from .normalize import meanNormalize
from .normalize import zScoreNormalize
from .normalize import minMaxNormalize
from .normalize import percentileNormalize


__all__ = ['balancedAccuracy', 'confusionMatrix', 'correlation',
           'cosineSimilarity', 'count', 'covariance', 'detectBestResult',
           'elementwiseMultiply', 'elementwisePower', 'f1Score',
           'falseNegative', 'falsePositive', 'fractionCorrect',
           'fractionIncorrect', 'inverse', 'leastSquaresSolution', 'maximum',
           'mean', 'meanAbsoluteError', 'meanFeaturewiseRootMeanSquareError',
           'meanNormalize', 'median', 'minimum', 'minMaxNormalize', 'mode',
           'percentileNormalize', 'precision', 'proportionMissing',
           'proportionZero', 'pseudoInverse', 'quartiles', 'rSquared',
           'recall', 'residuals', 'rootMeanSquareError', 'solve',
           'specificity', 'standardDeviation', 'sum', 'trueNegative',
           'truePositive', 'uniqueCount', 'varianceFractionRemaining',
           'zScoreNormalize']
