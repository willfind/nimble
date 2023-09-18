"""
Script to add a hidden cell to allow for an example to run via google collab

Takes one arg:
1) the path to the notebook to be modified.
"""

import sys
import nbformat



if __name__ == "__main__":
    target = sys.argv[1]
    toChange = nbformat.read(target, nbformat.NO_CONVERT)

    contentHeader = ["##### Required Colab Setup",]

    installPage = "https://willfind.github.io/nimble/install"
    contentInstall = [
        "!pip install nimble --find-links=" + installPage
        ]

    cellHeader = nbformat.v4.new_markdown_cell(contentHeader)
    cellInstall = nbformat.v4.new_code_cell(contentInstall)

    toChange['cells'] = [cellHeader, cellInstall] + toChange['cells']

    nbformat.write(toChange, target)
