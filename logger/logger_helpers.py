from __future__ import print_function

import UML

def useLogCheck(useLog):
    # if logger is suspended do not log and do not unsuspend
    if UML.logger.active.suspended:
        toLog = False
        unsuspend = False
        return toLog, unsuspend
    # if logger NOT suspended log based on useLog and unsuspend
    if useLog is None:
        useLog = UML.settings.get("logger", "enabledByDefault")
        useLog = True if useLog.lower() == 'true' else False
    toLog = useLog
    unsuspend = True
    UML.logger.active.suspended = True
    return useLog, unsuspend

def _dictToKeywordString(dictionary):
    kvStrings = []
    for key, value in dictionary.items():
        string = "{0}={1}".format(key,value)
        kvStrings.append(string)
    return ", ".join(kvStrings)

def _formatRunLine(*args):
    """ Formats equally spaced values for each column"""
    args = list(map(str, args))
    lineLog = ("{:20s}" * len(args)).format(*args) #TODO works below python2.7?
    lineLog += "\n"
    return lineLog

def _logHeader(timestamp):
    """ Formats the top line of each log entry"""
    lineLog = "\n"
    lineLog += "{0:>80}\n".format(timestamp)
    return lineLog

def _removeItemsWithoutData(columnNames, rowValues):
    """ Prevents the Log from displaying columns that do not have a data"""
    keepIndexes = []
    for index, item in enumerate(rowValues):
        if item !=  "None":
            keepIndexes.append(index)
    keepColumnName = []
    keepRowValue = []
    for index in keepIndexes:
        keepColumnName.append(columnNames[index])
        keepRowValue.append(rowValues[index])
    return keepColumnName, keepRowValue


# def _formatDictLines(columnNames, rowValues):
#     # TODO lines with greater than four columns
#     """ Formats dictionary lines to display key/value pairs """
#     columnNames, rowValues = _removeItemsWithoutData(columnNames, rowValues)
#     if columnNames == []:
#         return ""
#     lineLog = ("{:20s}" * len(columnNames)).format(*columnNames)
#     lineLog += "\n"
#     lineLog += ("{:20s}" * len(rowValues)).format(*rowValues)
#     lineLog += "\n\n"
#     return lineLog



