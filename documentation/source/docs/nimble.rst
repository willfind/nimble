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
provide efficiencies over other types.

.. autosummary::
   :toctree: generated/

   nimble.data
   nimble.ones
   nimble.zeros
   nimble.identity
   nimble.random.data
   nimble.loadData

.. note::
   List and Matrix are always available because ``numpy`` is a dependency.
   To use Sparse, ``scipy`` must be installed.
   To use DataFrame, ``pandas`` must be installed.

Using Data Objects
------------------

All four data objects, List, Matrix, Sparse, and DataFrame, have the same
functionality because they inherit from Nimble's
:class:`Base <nimble.core.data.Base>` object. The ``Base`` object methods
handle operations that apply to the entire data object or the individual
elements of the object. Each object also has ``points`` and ``features``
attributes to connect the :class:`Base <nimble.core.data.Base>` object with the
:class:`Features <nimble.core.data.Features>` and
:class:`Points <nimble.core.data.Points>` objects, respectively. These provide
many additional methods that apply when working specifically with the object's
features and points.

.. autosummary::
   :toctree: generated/

   ~nimble.core.data.Base
   ~nimble.core.data.Features
   ~nimble.core.data.Points

Machine Learning with Interfaces
--------------------------------

Nimble provides interfaces to use learners defined in popular machine learning
packages, ``nimble.learners``, and user created custom learners. This makes a
wide variety of machine learning algorithms available under the same api. The
functions and classes below allow for querying and utilizing any of these
learners.

.. autosummary::
   :toctree: generated/
   :recursive:

   nimble.learnerType
   nimble.listLearners
   nimble.learnerParameters
   nimble.learnerDefaultValues
   nimble.train
   nimble.trainAndApply
   nimble.trainAndTest
   nimble.trainAndTestOnTrainingData
   nimble.normalizeData
   nimble.fillMatching
   nimble.crossValidate
   nimble.loadTrainedLearner
   nimble.core.interfaces.TrainedLearner
   nimble.core.learn.KFoldCrossValidator
   nimble.CV
   nimble.Init

Custom Learner
--------------

Custom learners can be created by inheriting from ``nimble.CustomLearner``.
These can then be provided as an argument to the functions above to perform
machine learning.

.. autosummary::
   :toctree: generated/
   :recursive:

   nimble.CustomLearner

.. _configuration:

Configuration
-------------

Nimble has certain settings that can be configured. The default settings load
when the package is imported and can be changed during the session. Changes to
configurable settings are made through ``nimble.settings``, a
``SessionConfiguration`` object that provides methods for getting and setting
configurable options. Changes to options can apply to the current session or be
saved as the new default settings. Currently, :ref:`logging` and
:ref:`fetching-files` have sections that can be configured.

.. autosummary::
   :toctree: generated/
   :recursive:

   nimble.settings
   ~nimble.core.configuration.SessionConfiguration

.. _logging:

Logging
-------

By default, Nimble keeps a running log of the actions taken each session. The
log can be added to and queried using ``nimble.log`` and ``nimble.showLog``,
respectively. There are four :ref:`configurable <configuration>` options in the
"logger" section. By default, the "location" is the current directory and the
file "name" is "log-nimble". The "enabledByDefault" option is set to "True" and
"enableCrossValidationDeepLogging" is set to "False".

.. autosummary::
   :toctree: generated/
   :recursive:

   nimble.log
   nimble.showLog

.. _fetching-files:

Fetching Files
--------------

Nimble's ``fetchFile`` and ``fetchFiles`` provide efficient means for accessing
online data sets. When a fetch function downloads a dataset, it stores it
locally. Once downloaded, future calls to a fetch function for the same data
will identify that the data is already available locally, avoiding a repeated
download. The downloaded files are placed in a directory named "nimbleData" in
a :ref:`configurable <configuration>` local location. The local storage
location is identified by the "location" option in the "fetch" section and is
set to the home directory by default.

.. autosummary::
   :toctree: generated/
   :recursive:

   nimble.fetchFile
   nimble.fetchFiles
