
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

import os.path
import re
import tempfile
import shutil

from weasyprint import HTML, CSS
from jinja2 import Environment, FileSystemLoader, select_autoescape

static = os.path.join('source', '_static')
templates = os.path.join('source', '_templates', 'cheatsheet')

env = Environment(
    loader=FileSystemLoader(templates),
    autoescape=select_autoescape()
)

pdf = env.get_template('cheatsheet-template.html')

web = env.get_template('cheatsheet-web.rst')

cheatsheetPdf = os.path.join(static, 'cheatsheet.html')
with open(cheatsheetPdf, 'w') as f:
    f.write(pdf.render())

cheatsheetDocs = os.path.join('source', 'cheatsheet.rst')
with open(cheatsheetDocs, 'w') as f:
    f.write(web.render())


# sets the pdf page size (96dpi)
pageSize = """
@page {
  size: 816px 1056px;
  margin: 0px;
}

"""

with tempfile.TemporaryDirectory() as tmpDir:
    shutil.copyfile(os.path.join(static, 'nimbleObject.png'),
                    os.path.join(tmpDir, 'nimbleObject.png'))
    with open(os.path.join(static, 'cheatsheet.html'), 'r') as origHTML, \
         open(os.path.join(tmpDir, 'cheatsheet.html'), 'w+') as tmpHTML, \
         open(os.path.join(static, 'cheatsheet.css'), 'r') as origCSS:
        htmlString = origHTML.read()
        # set absolute paths
        htmlString = re.sub('docs/',
                            'https://www.nimbledata.org/docs/',
                            htmlString)
        tmpHTML.write(htmlString)
        tmpHTML.seek(0)
        cssString = pageSize + origCSS.read()
        stylesheets = [CSS(string=cssString)]
        HTML(tmpHTML.name).write_pdf(
            os.path.join(static, 'cheatsheet.pdf'), stylesheets=stylesheets,
            presentational_hints=True)
