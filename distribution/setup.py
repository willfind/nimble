import setuptools

setuptools.setup(
    name="UML",
    packages=setuptools.find_packages(exclude=['*.tests',
                                               '*.examples*']),
    install_requires=['six>=1.5.1',
                      'future>=0.11.4',
                      'numpy>=1.10.4',
                     ],
    )

## additional metadata TODO

# with open("README.md", "r") as fh:
#     long_description = fh.read()

# version="0.0.0",
# author="Example Author",
# author_email="author@example.com",
# description="A small example package",
# long_description=long_description,
# long_description_content_type="text/markdown",
# url="https://github.com/pypa/example-project",
# classifiers=(
#     "Programming Language :: Python :: 3",
#     "License :: OSI Approved :: MIT License",
#     "Operating System :: OS Independent",)


## Cython
# from __future__ import absolute_import
# from distutils.core import setup
# from Cython.Build import cythonize
# import os
#
# setup(ext_modules = cythonize(["UML/data/*.py", "UML/helpers.py"],
#                               exclude=["UML/data/dataHelpers.py", "UML/data/__init__.py"],
#                               compiler_directives={'always_allow_keywords': True}))
