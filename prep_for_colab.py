
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

    installPage = "www.nimbledata.org/install"
    contentInstall = [
        "!pip install nimble"
        ]

    cellHeader = nbformat.v4.new_markdown_cell(contentHeader)
    cellInstall = nbformat.v4.new_code_cell(contentInstall)

    toChange['cells'] = [cellHeader, cellInstall] + toChange['cells']

    nbformat.write(toChange, target)
