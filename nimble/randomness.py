"""
Contains functions and objects controlling how randomness is used in
nimble functions and tests.
"""

from __future__ import absolute_import
import random

import numpy

from .utility import ImportModule
from .logger import handleLogging

shogun = ImportModule('shogun')
if shogun:
    shogun.Math.init_random(42)

pythonRandom = random.Random(42)
numpyRandom = numpy.random.RandomState(42)

# We use (None, None, None) to signal that we are outside of a section of
# uncontrolled randomness
_saved = (None, None, None)
_stillDefault = True


def setRandomSeed(seed, useLog=None):
    """
    Set the seeds on all sources of randomness in nimble.

    Parameters
    ----------
    seed : int
        Seed for random state. Must be convertible to 32 bit unsigned
        integer for compliance with numpy. If seed is None, then we use
        os system time.
    """
    global _stillDefault
    pythonRandom.seed(seed)
    numpyRandom.seed(seed)
    if shogun:
        if seed is None:
            # use same seed as numpy used
            seed = int(numpyRandom.get_state()[1][0])
        shogun.Math.init_random(seed)
    if _saved != (None, None, None):
        _stillDefault = False

    handleLogging(useLog, 'setRandomSeed', seed=seed)


def generateSubsidiarySeed():
    """
    Randomly generate an integer seed.

    The seed will be used in a call to a subroutine our external system,
    so that even though our internal sources of randomness are not used,
    the results are still dependent on our random state.
    """
    # must range from zero to maxSeed because numpy random wants an
    # unsigned 32 bit int. Negative numbers can cause conversion errors,
    # and larger numbers can cause exceptions. 0 has no effect on randomness
    # control in shogun so start at 1.
    maxSeed = (2 ** 32) - 1
    return pythonRandom.randint(1, maxSeed)


def stillDefaultState():
    """
    Determine if the random state is default or has been modified.

    Will return False if setRandomSeed has been called to modify the
    state of the internal sources of randomness (with the exception of
    calls made within a section designated by startUncontrolledSection
    and endUncontrolledSection). Return True if there has been no
    modification during this 'session' of nimble. To be used in unit
    tests which rely on comparing hard coded results to calls to
    functions that rely on randomness in order to abort if we are in an
    unpredictable state.
    """
    # for now, don't know exactly to to make this correct, so anything that
    # uses it should explode
    assert False

#return _stillDefault

def startAlternateControl(seed=None):
    """
    Begin using a temporary new seed for randomness.

    Called to open a certain section of code that needs to a different
    kind of randomness than the current default, without changing the
    reproducibility of later random calls outside of the section. This
    saves the state of the nimble internal random sources, and calls
    setRandomSeed using the given parameter. The saved state can then be
    restored with a call to endAlternateControl. Meant to be used in
    unit tests, to either protect later calls from the modifications in
    this section, or to ensure consistency regardless of the current
    state of randomness in nimble.

    Parameters
    ----------
    seed : int
        Seed for random state. Must be convertible to 32 bit unsigned
        integer for compliance with numpy. If seed is None, then we use
        os system time.
    """
    global _saved

    if shogun:
        shogunSeed = shogun.Math.get_seed()
    else:
        shogunSeed = None

    _saved = (pythonRandom.getstate(), numpyRandom.get_state(), shogunSeed)
    setRandomSeed(seed, useLog=False)


def endAlternateControl():
    """
    Stop using the temporary seed created by ``startAlternateControl``.

    Called to close a certain section of code that needs to have
    different kind of randomness than the current default without
    changing the reproducibility of later random calls outside of the
    section. This will restore the state saved by
    ``startAlternateControl``.
    """
    global _saved
    if _saved != (None, None, None):
        pythonRandom.setstate(_saved[0])
        numpyRandom.set_state(_saved[1])
        if shogun:
            shogun.Math.init_random(_saved[2])
        _saved = (None, None, None)
