"""
Update the Google Colab files to match the example files.

Local access to the Colab files on Google Drive is required and all of
the colab files must be in the same directory. These are updated from
the .ipynb files in source/examples.
"""

import sys
import os
import json
import argparse

parser = argparse.ArgumentParser(description='Update Colab notebooks')
parser.add_argument('colab_directory',
                    help=("local path to the Google Drive directory "
                          "containing the example Colab notebooks"))

# name of wheel built for the colab notebooks in cloud storage
WHEEL = 'nimble-0.0.0.dev1-cp37-cp37m-linux_x86_64.whl'

# to install nimble, we download the wheel from cloud then pip install the file
INSTALL = ('To use Nimble in Colab, we will need to copy the distribution '
           'file from Google Cloud Storage and then install it using `pip`.')
INSTALL_CODE = ['!gcloud config set project nimble-302717\n'
                f'!gsutil cp gs://nimble/{WHEEL} .\n'
                f'!pip install {WHEEL}']
# all other dependencies for the examples are already installed in colab

def main(colabPath):
    examplesPath = os.path.join('source', 'examples')
    output = '{:.<40} {}'.format
    examples = (f for f in os.listdir(examplesPath) if f.endswith('.ipynb'))
    if not examples:
        raise FileNotFoundError('The .ipynb example files have not been '
                                'generated.')
    for file in examples:
        paddedFile = file + ' '
        try:
            with open(os.path.join(examplesPath, file), 'r') as ex:
                example = json.load(ex)
            with open(os.path.join(colabPath, file), 'r') as cl:
                colab = json.load(cl)
            # need to update the source in each cell in the colab notebook,
            # but all other metadata should remain the same. Colab
            # generates metadata for each cell, so we can only update if
            # the structure is unchanged. If the structure does not align
            # then a manual update is necessary.
            exCells = example['cells']
            clCells = colab['cells']
            newCells = {}
            clIdx = 0
            for exCell in exCells:
                assert clCells[clIdx]['cell_type'] == exCell['cell_type']
                # ignore the download links at bottom of the notebook
                if clIdx == 0:
                    clCells[clIdx]['source'] = exCell['source'][:-10]
                # account for setup for nimble in colab
                elif clIdx == 1:
                    # use heading then add colab nimble install
                    if len(exCell['source']) == 1:
                        # heading only, add newlines for install instructions
                        exCell['source'][0] += '\n'
                        exCell['source'].append('\n')
                    clCells[clIdx]['source'] = exCell['source'][:2]
                    clCells[clIdx]['source'].append(INSTALL)
                    # increment clIdx to account for extra code cell
                    clIdx += 1
                    clCells[clIdx]['source'] = INSTALL_CODE
                    if exCell['source'][2:]:
                        # remaining markdown is in the next cell
                        clIdx += 1
                        clCells[clIdx]['source'] = exCell['source'][2:]
                # all other cells should be exactly the same as example
                else:
                    clCells[clIdx]['source'] = exCell['source']
                clIdx += 1

            # check no additional cells in the colab notebook that we did
            # not update
            assert len(clCells) == clIdx

            with open(os.path.join(colabPath, file), 'w+') as f:
                json.dump(colab, f)

            print(output(paddedFile, 'SUCCESS'))
        except FileNotFoundError:
            print(output(paddedFile, 'ERROR ** NO MATCHING COLAB FILE **'))
        except AssertionError:
            print(output(paddedFile, 'ERROR ** REQUIRES MANUAL UPDATE **'))
        except Exception as e:
            print(output(paddedFile, 'ERROR ** {} **'.format(e)))


if __name__ == '__main__':
    parsed = parser.parse_args()
    main(parsed.colab_directory)
