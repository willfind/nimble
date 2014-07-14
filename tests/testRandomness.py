"""
Tests for the UML.randUML submodule

"""

import UML
import random
import numpy

import UML.randomness
from UML.randomness import pythonRandom
from UML.randomness import numpyRandom

#def testResults():
#	print pythonRandom.random()
#	print numpyRandom.rand()
#	assert False

def testSetRandomSeedExplicit():
	""" Test UML.setRandomSeed yields uml accessible random objects with the correct random behavior """
	UML.randomness.startUncontrolledSection()

	expPy = random.Random(1333)
	expNp = numpy.random.RandomState(1333)
	UML.setRandomSeed(1333)

	for i in xrange(50):
		assert pythonRandom.random() == expPy.random()
		assert numpyRandom.rand() == expNp.rand()

	UML.randomness.endUncontrolledSection()


def testSetRandomSeedNone():
	""" Test UML.setRandomSeed operates as expected when passed None (-- use system time as seed) """
	UML.randomness.startUncontrolledSection()	
	
	UML.setRandomSeed(None)
	pyState = pythonRandom.getstate()
	npState = numpyRandom.get_state()

	origPy = random.Random()
	origPy.setstate(pyState)
	origNp = numpy.random.RandomState()
	origNp.set_state(npState)

	UML.setRandomSeed(None)

	assert origPy.random() != pythonRandom.random()
	assert origNp.rand() != numpyRandom.rand()

	UML.randomness.endUncontrolledSection()

