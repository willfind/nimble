"""
Tests functionality that is limited only to View objects.

Methods tested in this file:


"""

from __future__ import absolute_import
import UML
from UML.data.tests.baseObject import DataTestObject
from UML.data import Base, BaseView

class ViewAccess(DataTestObject):

    def test_exception_docstring_decorator(self):
        """ test wrapper prepends view object information to Base docstring
        when for a method that raises an exception"""
        viewDoc = getattr(BaseView, 'extractPoints').__doc__
        baseDoc = getattr(Base, 'extractPoints').__doc__

        viewLines = viewDoc.split('\n')
        baseLines = baseDoc.split('\n')

        assert len(viewLines) == len(baseLines) + 3
        assert viewLines[3:] == baseLines

    def test_inherit_docstring_decorator(self):
        """ test wrapper provides same docstring as Base for a method with a
        separate implementation for view objects"""
        viewDoc = getattr(BaseView, 'view').__doc__
        baseDoc = getattr(Base, 'view').__doc__

        assert viewDoc == baseDoc


# get retType TODO -- still not sure what is correct functionality.
