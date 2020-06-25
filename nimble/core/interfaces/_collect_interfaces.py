"""
Collect the interfaces that will be accessible to the user.
"""

import os
import importlib
import abc

import nimble
from .universal_interface import PredefinedInterface
from .custom_learner import CustomLearnerInterface


def initInterfaceSetup():
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
        (name, extension) = fileName.rsplit('.', 1)
        if extension == 'py' and not name.startswith('_'):
            pythonModules.append(name)
    # setup seen with the interfaces we know we don't want to load/try to load
    seen = set(["PredefinedInterface"])
    for toImport in pythonModules:
        importedModule = importlib.import_module('.' + toImport, __package__)
        contents = dir(importedModule)

        # for each attribute of the module, we will check to see if it is a
        # subclass of the PredefinedInterface
        for valueName in contents:
            value = getattr(importedModule, valueName)
            if (isinstance(value, abc.ABCMeta)
                    and issubclass(value, PredefinedInterface)):
                if not valueName in seen:
                    seen.add(valueName)
                    predefined.append(value)

    nimble.core.interfaces.predefined = predefined

    #############
    # available #
    #############
    available = {}
    nimbleInterface = CustomLearnerInterface('nimble')
    for learnerName in nimble.learners.__all__:
        learner = getattr(nimble.learners, learnerName)
        nimbleInterface.registerLearnerClass(learner)
    available['nimble'] = nimbleInterface

    customInterface = CustomLearnerInterface('custom')
    available['custom'] = customInterface

    nimble.core.interfaces.available = available
