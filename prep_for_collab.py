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

    contentInstall = [
        "!gcloud config set project nimble-302717\n",
        "!gsutil cp gs://nimble/nimble-0.4.2-cp310-cp310-linux_x86_64.whl .\n",
        "!pip install nimble-0.4.2-cp310-cp310-linux_x86_64.whl"
        ]

    cellHeader = nbformat.v4.new_markdown_cell(contentHeader)
    cellInstall = nbformat.v4.new_code_cell(contentInstall)

    toChange['cells'] = [cellHeader, cellInstall] + toChange['cells']

    nbformat.write(toChange, target)
