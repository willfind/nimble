# from __future__ import absolute_import
import os
import importlib
import inspect

import UML
from UML.interfaces._collect_completed import collectVisiblePythonModules
from UML.interfaces._collect_completed import collectUnexpectedInterfaces

testsPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
interfacePath = os.path.dirname(testsPath)

_visible = collectVisiblePythonModules(interfacePath)
_visibleTests = collectVisiblePythonModules(testsPath)
_possible = collectUnexpectedInterfaces(_visible)


def _noInstances(superClass, toCheck):
    for obj in toCheck:
        if isinstance(obj, superClass):
            return False
    return True

# for every possible interface implementation (ie everything we could possibly have
# tests for) except for the self contained UML ones (UniversalInterface,
# CustomLearnerInterface, etc.)
for posInt in _possible:
    # check if the interface is actually available
    if _noInstances(posInt, UML.interfaces.available):
        # we want the name of the file in which the non available interface was defined
        toDisableName = posInt.__module__.rsplit('.', 1)[1]
        for posFile in _visibleTests:
            # we want to disable the tests in any file that has toDisableName as a substring
            if toDisableName in posFile:
                toDisable = importlib.import_module('.' + posFile, __package__)
                # since the interface is not available, we mark the tests to be not runnable
                toDisable.__test__ = False
