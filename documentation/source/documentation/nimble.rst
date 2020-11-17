nimble
======

All core functionality of Nimble can be accessed through the functions and
classes directly callable from ``nimble``. Additionally, Nimble defined objects
that are returned by some of these functions will be documented here as well.

.. contents::
   :local:

Creating Data Objects
---------------------
Most functions for creating data objects are found at the top-level of
``nimble``. Nimble provides four data object types: List, Matrix, Sparse, and
DataFrame. Each object has the same functionality, but differ on how data is
stored and manipulated on the backend. List uses a Python list, Matrix uses a
numpy array, Sparse uses a scipy COO Matrix and DataFrame uses a pandas
DataFrame. While the functionality is the same, choosing the best type may
provide efficiencies over other types. Note that List and Matrix (because numpy
is a dependency) are always available. However, to use Sparse and DataFrame,
scipy and pandas, respectively, must be installed.

.. autofunction:: nimble.data

.. autofunction:: nimble.ones

.. autofunction:: nimble.zeros

.. autofunction:: nimble.identity

.. autofunction:: nimble.random.data
   :noindex:

.. autofunction:: nimble.loadData

Data Objects
------------

All four data objects have the same functionality because they derive from
Nimble's ``Base`` object. The Base object methods handle operations that apply
to the entire data object or the individual elements of the object. Each
object also has a ``points`` and ``features`` attribute. These attributes
connect the ``Base`` object with the ``Features`` and ``Points`` objects
that provide additional methods that apply specifically to features and
points, respectively.

.. toctree::
   :maxdepth: 1

   data/base
   data/features
   data/points

Machine Learning with Interfaces
--------------------------------

Nimble provides interfaces to use learners defined in popular machine learning
packages, ``nimble.learners``, and user created custom learners. This makes a
wide variety of machine learning algorithms available under the same api. The
functions and classes below allow for querying and utilizing any of these
learners.

.. autofunction:: nimble.learnerType

.. autofunction:: nimble.listLearners

.. autofunction:: nimble.learnerParameters

.. autofunction:: nimble.learnerDefaultValues

.. autofunction:: nimble.train

.. autoclass:: nimble.core.interfaces.TrainedLearner
   :members:

.. autofunction:: nimble.trainAndApply

.. autofunction:: nimble.trainAndTest

.. autofunction:: nimble.trainAndTestOnTrainingData

.. autofunction:: nimble.normalizeData

.. autofunction:: nimble.fillMatching

.. autofunction:: nimble.crossValidate

.. autoclass:: nimble.core.learn.KFoldCrossValidator

.. autoclass:: nimble.CV
   :members:

.. autoclass:: nimble.Init
   :members:

.. autofunction:: nimble.loadTrainedLearner

Custom Learner
--------------

Custom learners can be created by inheriting from ``nimble.CustomLearner``.
These can then be provided as an argument to the functions above to perform
machine learning.

.. autoclass:: nimble.CustomLearner
   :members:

Logging and Configuration
-------------------------

By default, Nimble keeps a running log of the actions taken each session. The
log can be added to and queried using ``nimble.log`` and ``nimble.showLog``,
respectively. ``nimble.settings`` allows for querying and changing configuration
options. These options include the logger settings as well as any options
related to available interfaces. The settings can be changed temporarily during
a session, or permanently by saving them to the configuration file.

.. autofunction:: nimble.log

.. autofunction:: nimble.showLog

.. autodata:: nimble.settings

.. autoclass:: nimble.core.configuration.SessionConfiguration
   :members:
