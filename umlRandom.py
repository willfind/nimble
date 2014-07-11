"""
Contains functions and objects controlling how randomness is used in
UML functions and tests


"""

import random
import numpy

pyRandom = random.Random(42)
npRandom = numpy.random.RandomState(42)


def setRandomSeed(seed):
	"""
	Set the seeds on all sources of randomness in UML. If seed is None, then we use
	os system time

	"""
	pyRandom.seed(seed)
	npRandom.seed(seed)



