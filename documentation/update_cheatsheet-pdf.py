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

web = env.get_template('cheatsheet-web.html')

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
                            'https://willfind.github.io/nimble/docs/',
                            htmlString)
        tmpHTML.write(htmlString)
        tmpHTML.seek(0)
        cssString = pageSize + origCSS.read()
        stylesheets = [CSS(string=cssString)]
        HTML(tmpHTML.name).write_pdf(
            os.path.join(static, 'cheatsheet.pdf'), stylesheets=stylesheets,
            presentational_hints=True)
