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

.. container:: grid-container

   .. container:: row

     .. container:: column-1

        Nimble has 4 data types that share the same API. Each use a different
        backend to optimize the operations based on the type of data in the
        object. **By default, Nimble will assign the type that it detects would
        be best** based on the data and the packages available in the
        environment. The functions also include a parameter to set this
        manually. Selecting the type that best matches the data will make each
        operation as efficient as possible.

     .. container:: column-2

        .. list-table::
           :header-rows: 1
           :name: data-table
           :widths: auto
           :align: right

           * - Type
             - Data
             - Backend
           * - List
             - any data
             - Python list
           * - Matrix
             - all the same type
             - NumPy array
           * - DataFrame
             - each column has 1 type
             - Pandas DataFrame
           * - Sparse
             - mostly missing or 0
             - SciPy coo_matrix


The primary functions for creating data objects are found at the top-level of
``nimble``. The :doc:`../cheatsheet` can also be helpful source of information
about these functions.

.. autosummary::
   :toctree: generated/

   nimble.data
   nimble.ones
   nimble.zeros
   nimble.identity
   nimble.random.data

.. note::
   To use Sparse, `scipy`_ must be installed.
   To use DataFrame, `pandas`_ must be installed.

Using Data Objects
------------------

.. image:: ../_static/nimbleObject.png
 :alt: Nimble data object visualization
 :width: 400
 :height: 200
 :name: floating-img

A **Nimble data object** acts as the container of all individual elements of
your data. But for manipulating that data, Nimble defines an API that abstracts
away from the structure of how it is recorded to emphasize the meaning of how
elements inter-relate.

Instead of operating on rows and columns (as with a spreadsheet or matrix),
Nimble defines methods over ``points`` and ``features``. This aligns with the
goal of machine learning ready data, where each point should be a single
observation of unique variables and each feature should define a single
variable that has been recorded across observations. Nimble's API provides
tools to tidy data towards that goal while behaving in a way that respects the
observational meaning of data.

.. image:: ../_static/nimbleObject.png
 :alt: Nimble data object visualization
 :width: 400
 :height: 200
 :name: data-img

The methods of ``Base`` control operations that apply to the entire object or
each element in the data. The ``Points`` and ``Features`` methods of the object
have additional methods for operations that apply along that axis of the data
object. The :doc:`../cheatsheet` can also be helpful to find data object
methods.

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
same api. The :doc:`../cheatsheet` can also be helpful to understand Nimble's
machine learning API.

**Choosing a learner**

These functions help provide more information about the learners available for
use within Nimble. The functions beginning with "learner" return a Python
container with the relevant information. Often, a "learner" function has a
corresponding function beginning with "show" that prints a more readable
representation of the information to stdout.

.. autosummary::
   :toctree: generated/
   :recursive:

   nimble.learnerNames
   nimble.showLearnerNames
   nimble.learnerParameters
   nimble.showLearnerParameters
   nimble.learnerParameterDefaults
   nimble.showLearnerParameterDefaults
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
   nimble.Tune
   nimble.Tuning
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
"enableDeepLogging" is set to "False".

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

.. _scipy: https://www.scipy.org/install.html
.. _pandas: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
.. _sklearn: https://scikit-learn.org/stable/install.html
.. _tensorflow: https://www.tensorflow.org/install
.. _autoimpute: https://autoimpute.readthedocs.io/en/latest/user_guide/getting_started.html
.. _keras: https://keras.io/getting_started/
