"""
Analyse pylint output to locate code improvements and determine if
code meets the minimum linter requirements for UML.
Run as main to execute.

By default this script uses the .pylintrc config file in this directory
and pylint's json output to provide a custom output of the results.
--rcfile=<pathtofile> at the command line will override the config file.
--output-format=<format> at the command line will override the custom
format and provide pylint's output.
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import json
from pylint.lint import Run
from pylint import epylint as lint

# pylint symbols that can be ignored for minimum linter requirements
def notRequired(warning):
    """
    Classify a warning as not required.

    Warnings regarding 'too-many-*' or 'too-few-*', warnings that state
    'consider-*' and any warnings in the reqIgnore list will return
    True, meaning they are NOT required.
    """
    reqIgnore = [
        'broad-except',
        'chained-comparison',
        'comparison-with-itself',
        'duplicate-code',
        'fixme',
        'invalid-name',
        'len-as-condition',
        'no-else-return',
        'no-self-use',
        'pointless-statement',
        'protected-access',
        'protected-member',
        'unnecessary-pass',
        'unused-argument',
    ]
    return (warning.startswith('too')
            or warning.startswith('consider')
            or warning in reqIgnore)


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
    # python2 is writing the config file location to stderr, so ignore that
    if stderr and not stderr[0].startswith('Using config file'):
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

def printWarningSummary(warnings):
    """
    Parse and organize warnings then print a summary.
    """
    consider = []
    required = []
    for w in warnings:
        if notRequired(w['symbol']):
            consider.append(w)
        else:
            required.append(w)

    warning = "{module}:{line}: {message} ({symbol})"
    if consider:
        print('CONSIDER CHANGES')
        print('----------------')
        for warn in consider:
            print(warning.format(module=warn['module'], line=warn['line'],
                                 symbol=warn['symbol'],
                                 message=warn['message']))
        print("\n")
    reqOnly = []
    errors = []
    if required:
        for req in required:
            if req['type'] in ['error', 'fatal']:
                errors.append(req)
            else:
                reqOnly.append(req)

    if reqOnly:
        print('REQUIRED CHANGES')
        print('----------------')
        for req in reqOnly:
            print(warning.format(module=req['module'], line=req['line'],
                                 symbol=req['symbol'], message=req['message']))
        print("\n")

    if errors:
        print('ERRORS')
        print('------')
        for err in errors:
            print(warning.format(module=err['module'], line=err['line'],
                                 symbol=err['symbol'], message=err['message']))
        print("\n")

    print('SUMMARY')
    print('-------')
    total = len(consider) + len(reqOnly) + len(errors)
    print('{} total warnings and errors'.format(total))
    if consider:
        msg = '{} of the warnings should be reviewed; '.format(len(consider))
        msg += 'they may or may not require correction.'
        print(msg)
        print('    See CONSIDER CHANGES section above.')
    if reqOnly:
        print('{} of the warnings require correction.'.format(len(reqOnly)))
        print('    See REQUIRED CHANGES section above.')
    if errors:
        print('{} are errors which must be corrected.'.format(len(errors)))
        print('    See ERRORS section above.')
    else:
        print('*This code satisfies the UML minimum linter requirements*')

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
