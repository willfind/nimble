from cfg import defaultSkipSetNonAlpha
from cfg import defaultSkipSetNonAlphaNumeric
from cfg import defaultStopWords
from cfg import numericalChars

from umlExtensible import run
from umlExtensible import data

from umlStatic import normalize
from umlStatic import loadTrainingAndTesting
from umlStatic import listDataRepresentationMethods
from umlStatic import listUMLFunctions
from umlStatic import randomizedData
from umlStatic import listAlgorithms

from combinations.CrossValidate import crossValidate
from combinations.CrossValidate import crossValidateReturnBest
from combinations.Combinations import functionCombinations
from combinations.OrderedCrossValidate import orderedCrossValidate
from combinations.OrderedCrossValidate import orderedCrossValidateReturnBest

from performance.runner import runAndTest
