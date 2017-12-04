"""
Contains functions and objects controlling how randomness is used in
UML functions and tests


"""

from __future__ import absolute_import
import random
import numpy
import sys

pythonRandom = random.Random(42)
numpyRandom = numpy.random.RandomState(42)

# We use (None, None) to signal that we are outside of a section of uncontrolled
# randomness 
_saved = (None, None)
_stillDefault = True


def setRandomSeed(seed):
    """
    Set the seeds on all sources of randomness in UML. If seed is None, then we use
    os system time.

    """
    global _stillDefault
    pythonRandom.seed(seed)
    numpyRandom.seed(seed)
    if _saved != (None, None):
        _stillDefault = False


def generateSubsidiarySeed():
    """
    Randomly generate an integer seed to be used in a call to a subroutine our
    external system, so that even though our internal sources of randomness
    are not used, the results are still dependent on our random state.

    """
    # must range from zero to maxSeed because numpy random wants an
    # unsigned 32 bit int. Negative numbers can cause conversion errors,
    # and larger numbers can cause exceptions
    maxSeed = (2 ** 32) - 1
    return pythonRandom.randint(0, maxSeed)


def stillDefaultState():
    """
    Will return False if setRandomSeed has been called to modify the state of the internal
    sources of randomness (with the exception of calls made within a section designated
    by startUncontrolledSection and endUncontrolledSection). Return True if there
    has been no modification during this 'session' of UML. To be used in unit tests
    which rely on comparing hard coded results to calls to functions that rely
    on randomness in order to abort if we are in a unpredictable state.

    """
    # for now, don't know exactly to to make this correct, so anything that
    # uses it should explode
    assert False

#return _stillDefault

def startAlternateControl(seed=None):
    """
    Called to open a certain section of code that needs to a different kind of randomness
    than the current default, without changing the reproducibility of later random
    calls outside of the section. This saves the state of the UML internal random sources,
    and calls setRandomSeed using the given parameter. The saved state can then be restored
    with a call to endAlternateControl. Meant to be used in unit tests, to either protect
    later calls from the modifications in this section, or to ensure consistency regardless
    of the current state of randomness in UML.

    """
    global _saved
    _saved = (pythonRandom.getstate(), numpyRandom.get_state())
    setRandomSeed(seed)


def endAlternateControl():
    """
    Called to close a certain section of code that needs to have different kind of
    randomness than the current default without changing the reproducibility of later
    random calls outside of the section. This will restore the state saved by
    startAlternateControl.

    """
    global _saved
    if _saved is not (None, None):
        pythonRandom.setstate(_saved[0])
        numpyRandom.set_state(_saved[1])
        _saved = (None, None)

