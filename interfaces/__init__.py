import os
import UML
import _collect_completed

available = _collect_completed.collect(os.path.join(UML.UMLPath, 'interfaces'))

