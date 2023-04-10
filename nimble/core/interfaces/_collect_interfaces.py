"""
Collect the interfaces that will be accessible to the user.
"""

import os
import importlib

import nimble
from .universal_interface import PredefinedInterfaceMixin
from .custom_learner import CustomLearnerInterface, NimbleLearnerInterface


def initInterfaceSetup():
    """
    Setup the list of predefined interfaces during nimble init.
    """
    ##############
    # predefined #
    ##############
    predefined = []
    interfacesPath = os.path.join(nimble.nimblePath, 'core', 'interfaces')
    possibleFiles = os.listdir(interfacesPath)
    pythonModules = []
    for fileName in possibleFiles:
        if '.' not in fileName:
            continue
        # for the case of compiled cython extensions, there might be a platform indicator
        # in a middle segment.
        pieces = fileName.rsplit('.')
        name = pieces[0]
        extension = pieces[-1]
        if extension in ['py', 'so'] and not name.startswith('_'):
            pythonModules.append(name)
    # setup seen with the interfaces we know we don't want to load/try to load
    seen = set(["PredefinedInterfaceMixin"])
    for toImport in pythonModules:
        importedModule = importlib.import_module('.' + toImport, "nimble.core.interfaces")
        contents = dir(importedModule)

        # for each attribute of the module, we will check to see if it is a
        # subclass of the PredefinedInterfaceMixin
        for valueName in contents:
            value = getattr(importedModule, valueName)
            try:
                if (issubclass(value, PredefinedInterfaceMixin)
                        and not valueName in seen
                        and not valueName.startswith('_')):
                    seen.add(valueName)
                    predefined.append(value)
            except TypeError:
                pass

    nimble.core.interfaces.predefined = predefined

    #############
    # available #
    #############
    available = {}
    nimbleInterface = NimbleLearnerInterface()
    for learnerName in nimble.learners.__all__:
        learner = getattr(nimble.learners, learnerName)
        nimbleInterface.registerLearnerClass(learner)
    available['nimble'] = nimbleInterface

    customInterface = CustomLearnerInterface()
    available['custom'] = customInterface

    nimble.core.interfaces.available = available
