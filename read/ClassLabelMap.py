"""
Class to hold a mapping between document Id (key) and class label (or some datum 
resembling class labels).  Contain a map {docId: classLabel}, a name, and a boolean:
isRequired.  If isRequired is true, any data set being created with this label will
drop any rows for which this label is missing.  
"""


class ClassLabelMap():
    """
    Class holding a mapping between documentId (key) and class label. Assumes
    one label per document/key.
    """

    def __init__(self, docIdLabelMap, name='classLabel', isRequired=False):
        self.labelMap = docIdLabelMap
        self.name = name
        self.isRequired = isRequired

