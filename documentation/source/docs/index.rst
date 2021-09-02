Nimble API Documentation
========================

Welcome to the Nimble project's API documentation for users.

..
   below will create all stubs but commented to not show the autosummary table

   .. autosummary::
      :toctree: generated/
      :recursive:

      nimble

All core functionality of Nimble can be accessed through the functions and
classes directly callable from ``nimble``. Additionally, Nimble defined objects
that are returned by some of these functions are documented here as well.

.. contents::
   :local:

Creating Data Objects
---------------------
Most functions for creating data objects are found at the top-level of
``nimble``. Nimble provides four data object types: List, Matrix, Sparse, and
DataFrame. Each object has the same functionality, but differ on how data are
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

.. note::
   List and Matrix are always available because ``numpy`` is a dependency.
   To use Sparse, ``scipy`` must be installed.
   To use DataFrame, ``pandas`` must be installed.

Using Data Objects
------------------

All four data objects, List, Matrix, Sparse, and DataFrame, have the same
functionality because they inherit from Nimble's ``Base`` object. The ``Base``
object methods handle operations that apply to the entire data object or the
individual elements of the object. Each object also has ``features`` and
``points`` attributes to connect the ``Base`` object with the ``Features``
and ``Points`` objects, respectively. These provide many additional methods
that apply when working specifically with the object's ``features`` and
``points``.

.. autosummary::
   :toctree: generated/

   ~nimble.core.data.Base
   ~nimble.core.data.Features
   ~nimble.core.data.Points

Machine Learning with Interfaces
--------------------------------

In Nimble, all algorithms used for machine learning or deep learning are
referred to as "learners". Nimble provides interfaces to use learners defined
in popular machine learning packages, ``nimble.learners``, and user created
custom learners. This makes a wide variety of algorithms available under the
same api.

**Choosing a learner**

These functions help provide more information about the learners available for
use within Nimble. The functions beginning with "learner" return an object.
Often, a "learner" function has a corresponding function beginning with "show"
that prints a more readable representation of the object to stdout.

.. autosummary::
   :toctree: generated/
   :recursive:

   nimble.learnerNames
   nimble.showLearnerNames
   nimble.learnerParameters
   nimble.showLearnerParameters
   nimble.learnerDefaultValues
   nimble.showLearnerDefaultValues
   nimble.learnerType

**Using a learner**

The following support the learning process. Functions with a ``learnerName``
parameter accept a string in the format "package.learner". This provides access
to learners in Nimble without the need to import them manually or remember
their exact location within the package. For example,
``"nimble.KNNClassifier"``, ``"sklearn.LinearRegression"``, and
``"keras.Sequential"`` are valid ``learnerName`` strings.

.. autosummary::
   :toctree: generated/
   :recursive:

   nimble.train
   ~nimble.core.interfaces.TrainedLearner
   nimble.loadTrainedLearner
   nimble.trainAndApply
   nimble.trainAndTest
   nimble.trainAndTestOnTrainingData
   nimble.normalizeData
   nimble.fillMatching
   nimble.crossValidate
   ~nimble.core.learn.KFoldCrossValidator
   nimble.CV
   nimble.Init

The table below shows the current interfaces built in to Nimble. As an added
convenience, some interfaces have additional aliases that can be used as the
package name in the ``learnerName`` string (i.e. ``"skl.LinearRegression"``
instead of ``"sklearn.LinearRegression"``).

.. table::
   :align: left
   :widths: auto

   +----------------+--------------------------------+
   | Package        | Aliases                        |
   +================+================================+
   | `sklearn`_     | skl, scikitlearn, scikit-learn |
   +----------------+--------------------------------+
   | `keras`_       | tf.keras, tensorflow.keras     |
   +----------------+--------------------------------+
   | `autoimpute`_  |                                |
   +----------------+--------------------------------+
   | `shogun`_      |                                |
   +----------------+--------------------------------+
   | `mlpy`_        |                                |
   +----------------+--------------------------------+

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

Since most packages that support Nimble are optional, ``showAvailablePackages``
is provided to display the available packages in the current environment.
Nimble also has settings that can be configured. The default settings load when
the package is imported and can be changed during the session. Changes to
configurable settings are made through ``nimble.settings``, a
``SessionConfiguration`` instance that provides methods for getting and setting
configurable options. Changes to options can apply to the current session or be
saved as the new default settings. Currently, :ref:`logging` and
:ref:`fetching-files` have sections that can be configured.

.. autosummary::
   :toctree: generated/
   :recursive:

   nimble.showAvailablePackages
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
online datasets. When a new ``source`` is passed to a fetch function, it
downloads and stores the files in a directory named "nimbleData" in a
:ref:`configurable <configuration>` local location. When a repeated ``source``
is passed to a fetch function, no downloading occurs because the data can be
fetched locally. The local storage location is identified by the "location"
option in the "fetch" section and is set to the home directory by default.

.. autosummary::
   :toctree: generated/
   :recursive:

   nimble.fetchFile
   nimble.fetchFiles


Submodules
----------

..
  nimble.calculate is generated but conf.py tells autosummary to ignore it so
  docs/nimble.calculate.rst will be used. While this has the expected result,
  it does result in a warning during make html
.. autosummary::
   :toctree: generated/

   nimble.calculate
   nimble.exceptions
   nimble.fill
   nimble.learners
   nimble.match
   nimble.random

..
  autosummary above does not trigger that nimble.calculate is in a toctree
  this avoids sphinx from issuing a warning
.. toctree::
   :maxdepth: 1
   :hidden:

   nimble.calculate

.. _sklearn: https://scikit-learn.org/stable/install.html
.. _tensorflow: https://www.tensorflow.org/install
.. _autoimpute: https://autoimpute.readthedocs.io/en/latest/user_guide/getting_started.html
.. _shogun: https://www.shogun.ml/install
.. _keras: https://keras.io/getting_started/
.. _mlpy: https://github.com/richardARPANET/mlpy
