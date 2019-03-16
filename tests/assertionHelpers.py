"""
Common assertions helpers to be used in multiple test files.

Custom assertion types can be helpful if the assertion can be added to
existing tests which are also testing other functionality.
"""

from UML.data import BaseView

class LazyNameGenerationAssertion(AssertionError):
    pass

def assertNoNamesGenerated(obj):
    # By design, BaseView objects will always have names generated
    if isinstance(obj, BaseView):
        return
    if obj.points._namesCreated() or obj.features._namesCreated():
        raise LazyNameGenerationAssertion
