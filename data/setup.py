from distutils.core import setup
from Cython.Build import cythonize
import os

setup(ext_modules = cythonize("base.py"))
setup(ext_modules = cythonize("base_view.py"))
setup(ext_modules = cythonize("list.py"))
setup(ext_modules = cythonize("matrix.py"))
setup(ext_modules = cythonize("dataframe.py"))
setup(ext_modules = cythonize("sparse.py"))
setup(ext_modules = cythonize("__init__.py"))
setup(ext_modules = cythonize("dataHelpers.py"))
os.system('mv ./UML/data/*.so ./')
os.system('/bin/rm -rf build')
os.system('/bin/rm -rf UML')
os.system('/bin/rm *.c')