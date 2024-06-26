
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
Tests that the pyproject.toml, conda recipe, and in-nimble dependency
checks are all consistent with each other
"""

import nimble
import os
import inspect
import tomli
import importlib

def test_deps_matchingSections():
    """
    Check that the hard-coded sections of the Dependency objects match the
    structure of the optional dependencies in the pyproject file.
    """

    nimbleSpec = importlib.util.find_spec("nimble")
    if nimbleSpec and 'site-packages' in nimbleSpec.origin:
        checker = importlib.metadata.requires
        infoFull = checker("nimble")
        infoFull = map(lambda x: x.split(";")[0], infoFull)
    else:
        currPath = os.path.abspath(inspect.getfile(inspect.currentframe()))
        repoDir = os.path.dirname(os.path.dirname(currPath))
        pyprojPath = os.path.join(repoDir, "pyproject.toml")

        with open(pyprojPath, "rb") as f:
            tomlDict = tomli.load(f)
            infoFull = tomlDict["project"]

    optional = infoFull['optional-dependencies']
    required = infoFull['dependencies']

    for name, depObj in nimble._dependencies.DEPENDENCIES.items():
        if depObj.section == "required":
            assert depObj.requires in required
        else:
            assert depObj.requires in optional['all']
            assert depObj.requires in optional[depObj.section]


def test_pyproject_matches_recipe():
    """
    Ensure that the requirements specified in the conda recipe match the
    ones in pyproject.toml
    """
    # To be completed later
    pass
