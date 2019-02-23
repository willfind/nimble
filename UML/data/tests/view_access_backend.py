"""
Tests functionality that is limited only to View objects.

Methods tested in this file:


"""

from __future__ import absolute_import
import UML
from UML.data.tests.baseObject import DataTestObject
from UML.data import Base, BaseView
from UML.data.points import Points
from UML.data.points_view import PointsView
from UML.data.features import Features
from UML.data.features_view import FeaturesView

class ViewAccess(DataTestObject):

    def test_exceptionDocstring_decorator_Base(self):
        """ test wrapper prepends view object information to Base docstring
        when for a method that raises an exception"""
        viewDoc = getattr(BaseView, 'transpose').__doc__
        baseDoc = getattr(Base, 'transpose').__doc__

        viewLines = viewDoc.split('\n')
        baseLines = baseDoc.split('\n')

        assert len(viewLines) == len(baseLines) + 3
        assert viewLines[3:] == baseLines

    def test_exceptionDocstring_decorator_Points(self):
        """ test wrapper prepends view object information to Base docstring
        when for a method that raises an exception"""
        viewDoc = getattr(PointsView, 'splitByCollapsingFeatures').__doc__
        baseDoc = getattr(Points, 'splitByCollapsingFeatures').__doc__

        viewLines = viewDoc.split('\n')
        baseLines = baseDoc.split('\n')

        assert len(viewLines) == len(baseLines) + 3
        assert viewLines[3:] == baseLines

    def test_exceptionDocstring_decorator_Features(self):
        """ test wrapper prepends view object information to Base docstring
        when for a method that raises an exception"""
        viewDoc = getattr(FeaturesView, 'splitByParsing').__doc__
        baseDoc = getattr(Features, 'splitByParsing').__doc__

        viewLines = viewDoc.split('\n')
        baseLines = baseDoc.split('\n')

        assert len(viewLines) == len(baseLines) + 3
        assert viewLines[3:] == baseLines

# get retType TODO -- still not sure what is correct functionality.
