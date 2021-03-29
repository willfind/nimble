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

.. _configuration:

Configuration
-------------

Nimble has certain settings that can be configured. The default settings load
when the package is imported and can be changed during the session. Changes to
configurable settings are made through `nimble.settings`, a
`SessionConfiguration` object that provides methods for getting and setting
configurable options. Changes to options can apply to the current session or be
saved as the new default settings. Currently, :ref:`logging` and :ref:`fetch`
have sections that can be configured.

.. autodata:: nimble.settings

.. autoclass:: nimble.core.configuration.SessionConfiguration
   :members:

.. _logging:

Logging
-------

By default, Nimble keeps a running log of the actions taken each session. The
log can be added to and queried using ``nimble.log`` and ``nimble.showLog``,
respectively. There are four :ref:`configurable <configuration>` options in the
"logger" section. By default, the "location" is the current directory and the
file "name" is "log-nimble". The "enabledByDefault" option is set to "True" and
"enableCrossValidationDeepLogging" is set to "False".

.. autofunction:: nimble.log

.. autofunction:: nimble.showLog

.. _fetch:

Fetching Files
--------------

Nimble's `fetchFile` and `fetchFiles` provide efficient means for accessing
online data sets. When a fetch function downloads a dataset, it stores it
locally. Once downloaded, future calls to a fetch function for the same data
will identify that the data is already available locally, avoiding a repeated
download. The downloaded files are placed in a directory named "nimbleData" in
a :ref:`configurable <configuration>` local location. The local storage
location is identified by the "location" option in the "fetch" section and is
set to the home directory by default.

.. autofunction:: nimble.fetchFile

.. autofunction:: nimble.fetchFiles
