#!/usr/bin/python
"""
Publishes html docs online to github. This is accomplished by using
a specially named branch called gh-pages, which, when pushed to
github, will be automaticlly hosted. The source, configuration, and
make files for documentation are all part of the master branch. So,
since gh-pages is only used to publish the generated docs, it isn't
kept locally. The main contents of this script involves making an
orphaned branch for gh-pages, generating the docs using sphinx's
makefile, adding the wanted outputs to a commit, and pushing that
commit. There is also validation to make sure these tasks are only
done if master is up to date, and some carefull control flow to
protect against failure.

All commands are printed and the standard output and error of those
commands are passed through to the terminal by default. If you
want to supress all output, pass any of ["--suppress", "-q", '--quiet', '-s']
as a command line argument. Be warned, this will obscure output
asking for your github credentials, or any queries for guidance by
a pre-commit script.

Alternate workflows make it occassionally helpful to update gh-pages
using a branch other than master. As such, a branch name can
be passed as an argument when calling this script, and execution will
proceed using that branch instead.

See published docs at:
willfind.github.io/nimble

Further reading for some of the techniques used in this script:
http://stackoverflow.com/questions/3258243/check-if-pull-needed-in-git
https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

"""

from __future__ import absolute_import
from __future__ import print_function
import inspect
import subprocess
import sys
import os

# TODO refactor into class with SUPPRESSOUTPUT as a class attribute
global SUPPRESSOUTPUT
SUPPRESSOUTPUT = False

currFilePath = os.path.abspath(inspect.getfile(inspect.currentframe()))
currDirPath = os.path.dirname(currFilePath)


def printAndCall(cmd):
    if not SUPPRESSOUTPUT:
        print(cmd)
        stdout = None
    else:
        stdout = open(os.devnull, 'w')
    return subprocess.check_call(cmd, shell=True, stdout=stdout, stderr=subprocess.STDOUT)


def checkTargetUpToDate(target):
    printAndCall("git fetch origin")

    # get SHA of current target
    cmd = "git rev-parse --verify refs/heads/"  + target
    if not SUPPRESSOUTPUT:
        print(cmd)
    currP = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    localSHA = currP.stdout.readline().strip()
    if not SUPPRESSOUTPUT:
        print(localSHA)

    # get SHA of origin target
    cmd = "git rev-parse --verify refs/remotes/origin/" + target
    if not SUPPRESSOUTPUT:
        print(cmd)
    currP = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    remoteSHA = currP.stdout.readline().strip()
    if not SUPPRESSOUTPUT:
        print(remoteSHA)

    return localSHA == remoteSHA


def checkTargetScriptUpToDate(target):
    # get diff of HEAD vs target branch's publish script
    cmd = "git diff {} -- publish_gh-pages.py".format(target)
    if not SUPPRESSOUTPUT:
        print(cmd)
    currP = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    diffOutput = currP.stdout.read().decode("utf-8")
    if not SUPPRESSOUTPUT:
        print(diffOutput)
    return len(diffOutput) == 0

def argIsSuppress(toCheck):
    check = toCheck.lower()
    # TODO trim down to two (one abreviation, one long)
    valid = ["--suppress", "-q", '--quiet', '-s']
    return check in valid

def getBranchNames():
    # We omit the potential printing since it's unclear if the user wants the
    # output suppressed at this point
    cmd = "git branch"
    currP = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    stringOutput = currP.stdout.read().decode("utf-8")

    # In case of prefixes, we need a list of exact branch names to check against
    branchNames = []
    for line in stringOutput.split("\n"):
        branchNames.append(line[2:].strip())

    return branchNames

if __name__ == '__main__':
    targetBranch = "master"
    availableBranches = getBranchNames()

    # TODO use argparse to better handle this
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if argIsSuppress(arg):
                SUPPRESSOUTPUT = True
            elif arg in availableBranches:
                targetBranch = arg
            else:
                print("{} is neither a suppress flag nor a branch name".format(arg))
                sys.exit(1)

    # Some commands are reliant on the current working directory being
    # the same as the location of this script.
    if not SUPPRESSOUTPUT:
        print("Changing to: " + currDirPath)
    os.chdir(currDirPath)

    if not checkTargetUpToDate(targetBranch):
        print("We require the target branch to be up to date before publishing docs to gh-pages")
        sys.exit(1)

    if not checkTargetScriptUpToDate(targetBranch):
        msg = "We require the {} branch's publish script to ".format(targetBranch)
        msg += "match that of the calling branch's script."
        print(msg)
        sys.exit(1)

    committed = False
    try:
        printAndCall("git checkout " + targetBranch)
        printAndCall("git checkout --orphan gh-pages")
        printAndCall("git rm -rf --cached ../.")

        printAndCall("make html") # make publish fails to use custom.css
        printAndCall("mv html/* ..")

        printAndCall("git add ../.gitignore")
        printAndCall("git add ../.nojekyll")
        printAndCall("git add ../*.html")
        printAndCall("git add ../*.js")
        printAndCall("git add ../_static/")
        printAndCall("git add ../_sources")
        printAndCall("git add ../_downloads/ -f") # -f to include .ipynb
        printAndCall("git add ../_images")
        printAndCall("git add ../examples")
        printAndCall("git add ../docs -f") # -f to include generated
        printAndCall("git commit -m 'Publish html documentation for Nimble to gh-pages'")
        committed = True
        printAndCall("git push origin HEAD --force")

    finally:
        printAndCall("make clean")
        printAndCall("git checkout {} --force".format(targetBranch))
        if committed:
            printAndCall("git branch -D gh-pages")

    if not checkTargetUpToDate(targetBranch):
        msg = "While publishing, remote {} was updated. ".format(targetBranch)
        msg += "The published docs are therefore no longer current."
        print(msg)
        sys.exit(2)

    print("Successfully published")

# EOF marker
