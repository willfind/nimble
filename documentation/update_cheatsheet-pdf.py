import os.path
import re
import tempfile
import shutil

from weasyprint import HTML, CSS


# sets the pdf page size
pageSize = """
@page {
  size: 342mm 410mm;
  margin: 1cm;
}
"""

source = os.path.join('source', '_static')
with tempfile.TemporaryDirectory() as tmpDir:
    shutil.copyfile(os.path.join(source, 'nimbleObject.png'),
                    os.path.join(tmpDir, 'nimbleObject.png'))
    with open(os.path.join(source, 'cheatsheet.html'), 'r') as origHTML, \
         open(os.path.join(tmpDir, 'cheatsheet.html'), 'w+') as tmpHTML, \
         open(os.path.join(source, 'cheatsheet.css'), 'r') as origCSS, \
         open(os.path.join(tmpDir, 'cheatsheet.css'), 'w+') as tmpCSS:
        htmlString = origHTML.read()
        # set absolute paths
        htmlString = re.sub('\.\./docs',
                            'https://willfind.github.io/nimble/docs',
                            htmlString)
        tmpHTML.write(htmlString)
        tmpHTML.seek(0)
        cssString = origCSS.read()
        # inline-flex does not render pdf correctly
        cssString = re.sub('inline-flex', 'inline', cssString)
        cssString += pageSize
        tmpCSS.write(cssString)
        tmpCSS.seek(0)
        stylesheets = [CSS(tmpCSS.name),
                       CSS(string=cssString)]
        HTML(tmpHTML.name).write_pdf(
            os.path.join(source, 'cheatsheet.pdf'), stylesheets=stylesheets,
            presentational_hints=True)
