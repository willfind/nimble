"""
This groups together functions which perform calculations on data
objects. Some of these are also availale as methods off of data
objects; the versions here are functions, and take any inputs as
arguments.

"""

from confidence import confidenceIntervalHelper
from loss import fractionIncorrect
from loss import meanAbsoluteError
from loss import meanFeaturewiseRootMeanSquareError
from loss import rootMeanSquareError
from loss import varianceFractionRemaining
from matrix import elementwiseMultiply
from matrix import elementwisePower
from similarity import correlation
from similarity import cosineSimilarity
from similarity import covariance
from similarity import fractionCorrect
from similarity import rSquared
from statistic import maximum
from statistic import mean
from statistic import median
from statistic import minimum
from statistic import uniqueCount
from statistic import proportionMissing
from statistic import proportionZero
from statistic import quartiles
from statistic import standardDeviation
from utility import detectBestResult

__all__ = ['confidenceIntervalHelper', 'correlation', 'cosineSimilarity',
			'covariance', 'detectBestResult', 'elementwiseMultiply',
			'elementwisePower', 'fractionCorrect', 'fractionIncorrect',
			'maximum', 'mean', 'meanAbsoluteError',
			'meanFeaturewiseRootMeanSquareError', 'median', 'minimum',
			'proportionMissing', 'proportionZero', 'quartiles', 'rSquared',
			'rootMeanSquareError', 'standardDeviation', 'uniqueCount',
			'varianceFractionRemaining']
