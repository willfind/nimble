"""

"""
from .elements import Elements
from .base_view import readOnlyException

class ElementsView(Elements):
    """

    """
    def __init__(self, source, **kwds):
        kwds['source'] = source
        super(ElementsView, self).__init__(**kwds)

    def multiply(self, other):
        readOnlyException("elementwiseMultiply")

    def power(self, other):
        readOnlyException("elementwisePower")
