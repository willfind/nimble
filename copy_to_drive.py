
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
Script to overwrite drive files corresponding to generated example notebooks.
"""

import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

if __name__ == "__main__":
    authJSON = os.environ['COLAB_CREDENTIALS']

    # Use a service account as the authentication method, pulling
    # the needed key from the os environment.
    settings = {"client_config_backend": "service",
                "service_config": {"client_json": authJSON}
               }

    gauth = GoogleAuth(settings=settings)
    gauth.ServiceAuth()
    drive = GoogleDrive(gauth)

    # check all files the service account has permissions for
    file_list = drive.ListFile().GetList()
    for currNB in file_list:
        currID = currNB['id']
        currTitle = currNB['title']
        currPath = os.path.join('examples', currTitle)
        # only proceed if there is a corresponding file in the examples folder
        if os.path.exists(currPath):
            # Use the ID for that file, set the status as already uploaded.
            currOnDrive = drive.CreateFile({'id':currID, 'uploaded':True})
            currOnDrive.SetContentFile(currPath)
            currOnDrive.Upload()
