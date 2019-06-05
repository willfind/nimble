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

See published docs at:
willfind.github.io/Nimble

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


def checkMasterUpToDate():
    printAndCall("git fetch origin")

    # get SHA of current master
    cmd = "git rev-parse --verify refs/heads/master"
    if not SUPPRESSOUTPUT:
        print(cmd)
    currP = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    localSHA = currP.stdout.readline().strip()
    if not SUPPRESSOUTPUT:
        print(localSHA)

    # get SHA of origin master
    cmd = "git rev-parse --verify refs/remotes/origin/master"
    if not SUPPRESSOUTPUT:
        print(cmd)
    currP = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    remoteSHA = currP.stdout.readline().strip()
    if not SUPPRESSOUTPUT:
        print(remoteSHA)

    return localSHA == remoteSHA


if __name__ == '__main__':
    if len(sys.argv) > 1:
        check = sys.argv[1].lower()
        valid = ["--suppress", "-q", '--quiet', '-s']
        if check in valid:
            SUPPRESSOUTPUT = True

    # Some commands are reliant on the current working directory being
    # the same as the location of this script.
    if not SUPPRESSOUTPUT:
        print("Changing to: " + currDirPath)
    os.chdir(currDirPath)

    if not checkMasterUpToDate():
        print("We require the master branch to be up to date before publishing docs to gh-pages")
        sys.exit(1)

    try:
        printAndCall("git checkout --orphan gh-pages")
        printAndCall("git rm -rf --cached ../.")

        printAndCall("make publish")

        printAndCall("git add ../.gitignore")
        printAndCall("git add ../.nojekyll")
        printAndCall("git add ../*.html")
        printAndCall("git add ../*.js")
        printAndCall("git add ../_static/")
        printAndCall("git add ../_sources")
        printAndCall("git commit -m 'Publish html documentation for Nimble to gh-pages'")

        printAndCall("git push origin HEAD --force")

    finally:
        printAndCall("make clean")
        printAndCall("git checkout master --force")
        # add conditional to check if gh-pages exists? (if aborted before commit it won't)
        printAndCall("git branch -D gh-pages")

    if not checkMasterUpToDate():
        msg = "During the publishing process, remote master was updated. The published docs "
        msg += "are therefore no longer current."
        sys.exit(2)

    print("Successfully published")

# EOF marker
