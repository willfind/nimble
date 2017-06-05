"""




"""

import regressors_interface_old as regressors
from universal_interface_lookalike import UniversalInterfaceLookalike


class Regressors(UniversalInterfaceLookalike):
    """

    """

    def __init__(self):
        """

        """
        super(Regressors, self).__init__()

    def trainAndApply(self, learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label',
                      sendToLog=True):
        return regressors.regressor(learnerName, trainX, trainY, testX, arguments, output, sendToLog)


    def accessible(self):
        return False

    def _listLearnersBackend(self):
        """
        Return a list of all learners callable through this interface.

        """
        return regressors.listRegressorLearners()


    def _getLearnerParameterNamesBackend(self, name):
        return regressors.getParameters(name)

    def _getLearnerDefaultValuesBackend(self, name):
        return regressors.getDefaultValues(name)

    def _getParameterNamesBackend(self, name):
        """
        Find params for instantiation and function calls
        TAKES string name,
        RETURNS list of list of param names to make the chosen call
        """
        return regressors.getParameters(name)


    def _getDefaultValuesBackend(self, name):
        """
        Find default values
        TAKES string name,
        RETURNS list of dict of param names to default values
        """
        return regressors.getDefaultValues(name)

    def isAlias(self, name):
        """
        Returns true if the name is an accepted alias for this interface

        """
        if name.lower() in ['regressors', 'regressor']:
            return True
        else:
            return False


    def getCanonicalName(self):
        """
        Returns the string name that will uniquely identify this interface

        """
        return "regressors"
