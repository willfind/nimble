"""
Contains code relating to file I/O and manipulation of the optional
configuration of nimble.

During nimble initialization, there is a specific order to tasks
relating to configuration. Before any operation that might rely on being
able to access nimble.settings (for example, interface initialization)
we must load it from file, so that in the normal course of operations,
user set values are available to be used across nimble. Alternatively,
in the case that there is a new source of user set options (for example,
an interface that has been loaded for the first time) we still load from
the config file first, then do initialization of all of the those
modules that might need access to nimble.settings, using hard coded
defaults if needed, then at the end of nimble initialization we will
always perform a syncing helper, which will ensure that the
configuration file reflects all available options.
"""

# Note: .ini format's option names are not case sensitive?

import os
import inspect
import configparser
import pathlib

import nimble
from nimble.exceptions import InvalidArgumentType, InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import ImproperObjectAction, PackageException

# source file is __init__.py, we split to get the directory containing it
nimblePath = os.path.dirname(inspect.getsourcefile(nimble))
configErrors = (configparser.NoSectionError, configparser.NoOptionError)


class SessionConfiguration(object):
    """
    Manage settings during a Session.
    """

    def __init__(self, path):
        """
        Class through which nimble user interacts with the saveable
        configuration options to define behavior dependent on the host
        system.
        The backend is a SortedConfigParser (essentially a
        SafeConfigParser where sections and options are written in
        sorted order) object which deals with the file I/O of the INI
        formated file on disk. This wrapper class allows for temporary
        changes to that configuration set and gives the user control
        over which changes should be saved and when.
        """
        self.parser = configparser.ConfigParser()
        # Needs to be set if you want option names to be case sensitive
        self.parser.optionxform = str
        self.parser.read(path)
        self.path = path

        # dict of section name to dict of option name to value
        self.changes = {}
        self.hooks = {}

    def get(self, section=None, option=None):
        """
        Query the contents of a section, or a specific option.
        """
        if section is None and option is not None:
            msg = "Must specify a section if specifying an option"
            raise InvalidArgumentValueCombination(msg)
        if section is not None:
            if option is not None:
                if section in self.changes and option in self.changes[section]:
                    return self.changes[section][option]
                return self.parser.get(section, option)

            if section in self.changes:
                ret = self.changes[section].copy()
            else:
                ret = {}
            if section in self.parser.sections():
                for optName, optVal in self.parser[section].items():
                    # bypass names already set in self.changes
                    if optName not in ret:
                        ret[optName] = optVal
                return ret
            # section was in changes but not in config file
            if ret:
                return ret
            # section not in changes or config file
            raise configparser.NoSectionError(section)

        ret = self.changes.copy()
        for cpSection in self.parser.sections():
            if cpSection not in ret:
                ret[cpSection] = {}
            for optName, optVal in self.parser[cpSection].items():
                # bypass names already set in self.changes and invalid names
                if optName not in ret[cpSection]:
                    ret[cpSection][optName] = optVal

        return ret

    def hook(self, section, option, toCall):
        """
        Assign a function to be called the next time the value of the
        specified section/option combination is changed. The provided
        function (parameter toCall) may only take one argument: the
        new value of the option. This function is called directly
        after the successful set call.

        None is a sentinal value that may be assigned as a hook.
        It disallows hooking on for this section option combination
        for the remainder of the session.
        """
        key = (section, option)
        if key in self.hooks and self.hooks[key] is None:
            msg = "The hook for (" + str(key) + ") has been previously set as "
            msg += "None, subsequently disabling this feature on that section "
            msg += "/ option combination"
            raise ImproperObjectAction(msg)

        if toCall is not None:
            if not hasattr(toCall, '__call__'):
                msg = 'toCall must be callable (function, method, etc) or None'
                raise InvalidArgumentType(msg)
            if len(nimble._utility.inspectArguments(toCall)[0]) != 1:
                msg = 'toCall may only take one argument'
                raise InvalidArgumentValue(msg)

        self.hooks[key] = toCall

    def setDefault(self, section, option, value):
        """
        Permanently set a value in the configuration file.

        Set a value which will immediately be reflected in the
        configuration file.
        """
        self.set(section, option, value)
        self.saveChanges(section, option)

    def set(self, section, option, value):
        """
        Set an option for this session.

        This change will apply only to this session. saveChanges can be
        called to make this change permanent by saving it to the
        configuration file.
        """
        # if we are setting a value which matches the
        # value in file, we should adjust the changes
        # dict accordingly
        try:
            inFile = self.parser.get(section, option)
            if (inFile == value and section in self.changes
                    and option in self.changes[section]):
                del self.changes[section][option]
                if len(self.changes[section]) == 0:
                    del self.changes[section]
                return
        except configparser.NoSectionError:
            pass
        except configparser.NoOptionError:
            pass

        # check: is this section the name of an interface
        try:
            ignore = True
            # raises InvalidArgumentValue if not an interface name
            interface = nimble.core._learnHelpers.findBestInterface(section)
            ignore = False
            if interface.getCanonicalName() in ['nimble', 'custom']:
                msg = section + " is associated with an interface that does "
                msg += "not support configurable options"
                raise InvalidArgumentValue(msg)
            acceptedNames = interface.optionNames
            if option not in acceptedNames:
                msg = section + " is associated with an interface that only "
                msg += "allows the options: " + str(acceptedNames) + "but "
                msg += option + " was given instead"
                raise InvalidArgumentValue(msg)
        # if ignore is true, this exception comes from the findBestInterface
        # call, and means that the section is not related to an interface.
        except InvalidArgumentValue:
            if not ignore:
                raise
        # a PackageException is the result of a possible interface being
        # unavailable, we will allow setting a location for possible interfaces
        # as this may aid in loading them in the future.
        except PackageException as e:
            if option != 'location':
                msg = section + "is an interface which is not currently "
                msg += "available. Only a 'location' option is permitted "
                msg += "for unavailable interfaces."
                raise InvalidArgumentValue(msg) from e

        if not section in self.changes:
            self.changes[section] = {}
        self.changes[section][option] = value

        # call hook if available
        key = (section, option)
        if key in self.hooks and self.hooks[key] is not None:
            self.hooks[key](value)

    def saveChanges(self, section=None, option=None):
        """
        Permanently set changes made to the configuration file.

        Depending on the values of the inputs, three levels of saving
        are allowed. If both section and option are specified, that
        specific value will be written to file. If only section is
        specified and option is None, then any changes in that section
        will be written to file. And if both section and option are
        None, then all changes will be written to file. If section is
        None and option is not None, an exception is raised.
        """
        # if no changes have been made, nothing needs to be saved.
        if self.changes == {}:
            return

        def changeIndividual(changeSec, changeOpt, changeVal):
            if not self.parser.has_section(changeSec):
                self.parser.add_section(changeSec)
            self.parser.set(changeSec, changeOpt, changeVal)


        if section is None:
            if option is not None:
                msg = "Must specify a section if specifying an option"
                raise InvalidArgumentValueCombination(msg)
            # save all
            for sec, options in self.changes.items():
                for opt in options:
                    optVal = options[opt]
                    changeIndividual(sec, opt, optVal)

            self.changes = {}
        elif option is None and section in self.changes:
            #save section
            for opt in self.changes[section]:
                changeIndividual(section, opt,
                                 self.changes[section][opt])

            del self.changes[section]
        elif (option is not None and section in self.changes
              and option in self.changes[section]):
            # save specific
            optVal = self.changes[section][option]
            changeIndividual(section, option, optVal)
            del self.changes[section][option]
            if len(self.changes[section]) == 0:
                del self.changes[section]

        with open(self.path, 'w') as configFile:
            self.parser.write(configFile)


def loadSettings():
    """
    Function which reads the configuration file and loads the values
    into a SessionConfiguration. The SessionConfiguration object is then
    returned.
    """
    target = os.path.join(nimble.nimblePath, 'configuration.ini')

    if not os.path.exists(target):
        with open(target, 'w'):
            pass

    ret = SessionConfiguration(target)
    return ret


def setInterfaceOptions(interface, save):
    """
    Synchronizes the configuration file, settings object in memory, and
    the the available interfaces, so that all three have the same option
    names and default values. This is called during nimble
    initialization after available interfaces have been detected, but
    before a user could have to rely on accessing options for that
    interface.
    """
    interfaceName = interface.getCanonicalName()
    optionNames = interface.optionNames
    # set new option names
    for opName in optionNames:
        try:
            nimble.settings.get(interfaceName, opName)
        except configparser.Error:
            nimble.settings.set(interfaceName, opName, "")
    if save:
        nimble.settings.saveChanges(interfaceName)

def setFetchPath(settings):
    """
    Set a default path for nimble.fetchFiles, if necessary.
    """
    try:
        _ = settings.get('fetch', 'location')
    except configErrors:
        settings.setDefault('fetch', 'location',
                            str(pathlib.Path.home()))
