"""
Collect the interfaces that will be accessible to the user.
"""

import os
import importlib
import abc
from .universal_interface import PredefinedInterface


def collectPredefinedInterfaces(modulePath):
    """
    Import predefined modules and check for predefined interfaces.
    """
    possibleFiles = os.listdir(modulePath)
    pythonModules = []
    for fileName in possibleFiles:
        if '.' not in fileName:
            continue
        (name, extension) = fileName.rsplit('.', 1)
        if extension == 'py' and not name.startswith('_'):
            pythonModules.append(name)
    predefinedInterfaces = []
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
                    predefinedInterfaces.append(value)

    return predefinedInterfaces
