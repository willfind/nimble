"""
Tests a that base class and its subclasses are being documentated as
expected.
"""

from nimble.utility import inheritDocstringsFactory
from nimble.core.data import Base, Matrix, List, Sparse, DataFrame
from nimble.core.interfaces.universal_interface import UniversalInterface
from nimble.core.interfaces.keras_interface import Keras
from nimble.core.interfaces.mlpy_interface import Mlpy
from nimble.core.interfaces.scikit_learn_interface import SciKitLearn
from nimble.core.interfaces.shogun_interface import Shogun
from nimble.core.interfaces import CustomLearnerInterface

############################
# inheritDocstringsFactory #
############################

def test_inheritDocstringsFactory():
    """test docstrings from methods without docstrings are inherited from the passed class"""

    class toInherit(object):
        def __init__(self):
            """toInherit __init__ docstring"""
            pass

        def copy(self):
            """toInherit copy docstring"""
            pass

    @inheritDocstringsFactory(toInherit)
    class InheritDocs(toInherit):
        """InheritDocs class docstring"""

        def __init__(self):
            """inheritDocs __init__ docstring"""
            pass

        def copy(self):
            pass

        # these test there are no issues for methods that are not in inherited class
        def _noDoc(self):
            pass

        def _withDoc(self):
            """_withDoc docstring"""
            pass

    toTest = InheritDocs()

    assert toTest.__doc__ != toInherit.__doc__
    assert toTest.__doc__ == "InheritDocs class docstring"

    assert toTest.__init__.__doc__ != toInherit.__init__.__doc__
    assert toTest.__init__.__doc__ == "inheritDocs __init__ docstring"

    assert toTest.copy.__doc__ is not None
    assert toTest.copy.__doc__ == toInherit.copy.__doc__


    assert toTest._noDoc.__doc__ is None
    assert toTest._withDoc.__doc__ == "_withDoc docstring"


def test_BaseSubclassesInherit():
    """ Test undocumented method in Base subclasses inherits Base documentation """
    baseDoc = Base._getPoints.__doc__
    assert List._getPoints.__doc__ == baseDoc
    assert Matrix._getPoints.__doc__ == baseDoc
    assert Sparse._getPoints.__doc__ == baseDoc
    assert DataFrame._getPoints.__doc__ == baseDoc


def test_UniversalIntefaceSubclassInherit():
    """ Test undocumented method in CustomLearnerInterface inherits UniversalInterface documentation """
    UniversalDoc = UniversalInterface.accessible.__doc__
    assert CustomLearnerInterface.accessible.__doc__ == UniversalDoc
    assert Keras.accessible.__doc__ == UniversalDoc
    assert Mlpy.accessible.__doc__ == UniversalDoc
    assert SciKitLearn.accessible.__doc__ == UniversalDoc
    assert Shogun.accessible.__doc__ == UniversalDoc
