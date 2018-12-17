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
REQ_IGNORE = [
    'invalid-name',
    'too-many-lines',
    'too-many-locals',
    'too-many-branches',
    'too-many-statements',
    'too-many-return-statements',
    'too-many-instance-attributes',
    'comparison-with-itself',
    'protected-access',
    'pointless-statement',
    'broad-except',
    'no-else-return',
    'len-as-condition',
    'consider-using-enumerate',
    'protected-member',
    'unused-argument',
    'duplicate-code'
]

def getOutputs(commandString):
    pylint_stdout, pylint_stderr = lint.py_run(commandString, return_std=True)
    stderr = pylint_stderr.readlines()
    if stderr:
        print("".join(stderr), file=sys.stderr)
        sys.exit()
    return pylint_stdout

def jsonToDict(file):
    dictionary = {}

    lines = file.readlines()
    if lines:
        try:
            dictionary = json.loads("".join(lines))
        except Exception:
            # if can't load json format, stdout contains reason this failed
            print("".join(lines), file=sys.stderr)
            sys.exit()

    return dictionary

def analyzeWarnings(commandString):
    out = getOutputs(commandString)
    outDict = jsonToDict(out)

    return outDict

def printWarningSummary(warnings):
    consider = []
    required = []
    reqCount = 0
    errCount = 0
    for w in warnings:
        if w['type'] in ['error', 'fatal']:
            errCount += 1
        if w['symbol'] in REQ_IGNORE:
            consider.append(w)
        else:
            reqCount += 1
            required.append(w)

    warning = "{module}:{line}: {message} ({symbol})"
    if consider:
        print('CONSIDER CHANGES')
        print('----------------')
        for warn in consider:
            print(warning.format(module=warn['module'], line=warn['line'],
                                 symbol=warn['symbol'], message=warn['message']))
        print("\n")
    errors = []
    if required:
        print('REQUIRED CHANGES')
        print('----------------')
        for req in required:
            if req['type'] in ['error', 'fatal']:
                errors.append(req)
            else:
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
    print('{} total warnings and errors'.format(len(consider) + reqCount))
    if len(consider) > 0:
        msg = '{} of the warnings should be reviewed; '.format(len(consider))
        msg += 'they may or may not require correction.'
        print(msg)
        print('    See CONSIDER CHANGES section above.')
    if reqCount > 0:
        if reqCount - errCount > 0:
            print('{} of the warnings require correction.'.format(reqCount - errCount))
            print('    See REQUIRED CHANGES section above.')
        if errCount > 0:
            print('{} are errors which must be corrected.'.format(errCount))
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
        elif arg in ['--version','-h', '--help', '--long-help']:
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

    warnings = analyzeWarnings(command)
    printWarningSummary(warnings)
