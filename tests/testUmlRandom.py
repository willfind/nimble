"""
Tests for the UML.randUML submodule

"""

import UML
import random
import numpy

from UML.umlRandom import pyRandom
from UML.umlRandom import npRandom

def testSetRandomSeedExplicit():
	""" Test UML.setRandomSeed yields uml accessible random objects with the correct random behavior """

	expPy = random.Random(1333)
	expNp = numpy.random.RandomState(1333)
	UML.setRandomSeed(1333)

	for i in xrange(50):
		assert pyRandom.random() == expPy.random()
		assert npRandom.rand() == expNp.rand()


def testSetRandomSeedNone():
	""" Test UML.setRandomSeed operates as expected when passed None (-- use system time as seed) """
	
	UML.setRandomSeed(None)
	pyState = pyRandom.getstate()
	npState = npRandom.get_state()

	origPy = random.Random()
	origPy.setstate(pyState)
	origNp = numpy.random.RandomState()
	origNp.set_state(npState)

	UML.setRandomSeed(None)

	assert origPy.random() != pyRandom.random()
	assert origNp.rand() != npRandom.rand()

