
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Analyse pylint output to locate code improvements and determine if
code meets the minimum linter requirements for nimble.
Run as main to execute.

By default this script uses the .pylintrc config file in this directory
and pylint's json output to provide a custom output of the results.
--rcfile=<pathtofile> at the command line will override the config file.
--output-format=<format> at the command line will override the custom
format and provide pylint's output.
"""
import os
import sys
import json

from pylint.lint import Run
from pylint import epylint as lint

def getOutputs(commandString):
    """
    Run pylint and catch the output.

    Parameters
    ----------
        commandString : str
            The arguments from the command line as a string.
    """
    pylint_stdout, pylint_stderr = lint.py_run(commandString, return_std=True)
    stderr = pylint_stderr.readlines()
    if stderr:
        print("FAILURE: The following error occurred:")
        print("".join(stderr), file=sys.stderr)
        sys.exit()
    return pylint_stdout

def jsonToDict(jsonFile):
    """
    Parse the json file from pylint and convert to a dict.
    """
    dictionary = {}

    lines = jsonFile.readlines()
    if lines:
        try:
            dictionary = json.loads("".join(lines))
        except Exception:
            # if can't load json format, stdout contains reason this failed
            print("FAILURE: The following error occurred:")
            print("".join(lines), file=sys.stderr)
            sys.exit()

    return dictionary

def analyzeWarnings(commandString):
    """
    Run pylint and convert the output to a dictionary.
    """
    out = getOutputs(commandString)
    outDict = jsonToDict(out)

    return outDict

def notRequired(warning):
    """
    Classify a warning as not required.

    Warnings regarding 'too-many-*' or 'too-few-*' and any warnings in
    the reqIgnore list will return True, meaning they are NOT required.
    """
    reqIgnore = [
        'chained-comparison',
        'duplicate-code',
        'eval-used',
        'exec-used',
        'fixme',
        'invalid-name',
        'no-self-use',
        'anomalous-backslash-in-string'
    ]
    return (warning.startswith('too')
            or warning in reqIgnore)

def sortWarnings(warnings):
    errors = []
    required = []
    consider = []
    for w in warnings:
        if notRequired(w['symbol']):
            consider.append(w)
        elif w['type'] in ['error', 'fatal']:
            errors.append(w)
        else:
            required.append(w)
    return errors, required, consider

def printWarnings(warnings, heading):
    warning = "{module}:{line}: {message} ({symbol})"
    if warnings:
        print(heading)
        print('-' * len(heading))
        for w in warnings:
            print(warning.format(module=w['module'], line=w['line'],
                                 symbol=w['symbol'], message=w['message']))
        print("\n")

def printWarningSummary(warnings):
    """
    Parse and organize warnings then print a summary.
    """
    errors, required, consider = sortWarnings(warnings)
    printWarnings(consider, 'CONSIDER CHANGES')
    printWarnings(required, 'REQUIRED CHANGES')
    printWarnings(errors, 'ERRORS')

    print('SUMMARY')
    print('-------')
    total = len(consider) + len(required) + len(errors)
    print('{} total warnings and errors'.format(total))
    if consider:
        msg = '{} of the warnings should be reviewed; '.format(len(consider))
        msg += 'they may or may not require correction.'
        print(msg)
        print('    See CONSIDER CHANGES section above.')
    if required:
        print('{} of the warnings require correction.'.format(len(required)))
        print('    See REQUIRED CHANGES section above.')
    if errors:
        print('{} are errors which must be corrected.'.format(len(errors)))
        print('    See ERRORS section above.')
    if not required and not errors:
        msg = '* This code satisfies the nimble minimum linter requirements *'
        print('*' * len(msg))
        print(msg)
        print('*' * len(msg))

if __name__ == '__main__':
    hasConfig = False
    # sort and reverse so paths precede other arguments
    # this is necessary for using lint.py_run
    configOptions = list(reversed(sorted(sys.argv[1:])))
    if all(arg.startswith('-') for arg in sys.argv[1:]):
        # default to cwd if no path specified
        configOptions.insert(0, '.')
    for i, arg in enumerate(configOptions):
        if arg.startswith("--output-format"):
            # Use pylint output if --output-format specified
            Run(configOptions)
        elif arg in ['--version', '-h', '--help', '--long-help']:
            # use of Run prints these correctly
            Run([arg])
        elif arg.startswith('--rcfile'):
            hasConfig = True
            absPath = os.path.abspath(arg[len('--rcfile='):])
            configOptions[i] = '--rcfile=' + absPath
        elif not arg.startswith("-"):
            configOptions[i] = os.path.abspath(arg)
    if not hasConfig:
        config = os.path.abspath('.pylintrc')
        configOptions.append('--rcfile={0}'.format(config))
    configOptions.append('--output-format=json')

    command = " ".join(configOptions)

    getWarnings = analyzeWarnings(command)
    printWarningSummary(getWarnings)
