Installation
============

Nimble can be installed in a variety of ways and strives for flexibility
during the install process. To avoid requiring packages that may never be used,
Nimble only has only three required dependencies, `Numpy`_, `packaging`_,
and `tomli`_.
`NumPy`_ enables a baseline portion of the data manipulation API.
The `packaging`_ dependency is used to validate the installed versions of any
optional packages. The `tomli`_ dependency is a fallback for certain dependency
checks of the optional dependencies. All further functionality is accessed by
installing third-party :ref:`optional-packages`.

For convenience, installing optional packages can also be
triggered while installing Nimble. We recommend the :ref:`quickstart-install`
to make almost all of Nimble's functionality available with a single command.

Install Methods
---------------

* :ref:`quickstart-install` (recommended)
* :ref:`specific-install`
* :ref:`basic-install`

.. _quickstart-install:

Nimble Install Quickstart
-------------------------

To install a curated selection of :ref:`optional-packages` while installing Nimble,
both ``pip`` and ``conda`` installs offer a quickstart option. Quickstart
installs all :ref:`operational-support` packages and SciKit-Learn from the
:ref:`ml-interfaces`. These packages were chosen because they are reliably
installable through ``pip`` and ``conda`` and provide access to the
majority of Nimble's functionality.

**With pip:**

Nimble uses extras to provide the ``quickstart`` shortcut

.. code-block::

  pip install nimble[quickstart]

.. note::
  The brackets may need to be escaped in some shells. For example, on a mac,
  which uses z shell as default, the command would be

  .. code-block::

    pip install nimble\[quickstart\]

**With conda:**

The nimble-data channel available through this website provides a package with expanded
dependencies, ``nimble-quickstart``. Once installed, import still occurs with
``import nimble``, the name ``nimble-quickstart`` only serves to install nimble and the
other desired packages.

.. code-block::

  conda install nimble-quickstart -c https://www.nimbledata.org/nimble-data

.. _specific-install:

Install with Specific Packages
------------------------------

:ref:`optional-packages` can always be installed separately with ``pip`` or
``conda``, for example: ``pip install scipy`` or ``conda install scipy``.
However, both package managers offer ways to install optional packages in
the same step as installing nimble.

.. warning:: Installs will fail if any package fails to install.

**With pip:**

``pip`` supports the use of extras included in brackets after the package name.
Nimble provides each optional package as an extra and extras that are shortcuts
to installing multiple packages. The ``[quickstart]`` shortcut was outlined in
:ref:`quickstart-install`, but the ``[data]`` shortcut can also be used to
install both ``scipy`` and ``pandas``, which will make all Nimble data object
types immediately available. Multiple extras can be included during the same
install.

.. code-block::

  # single extra
  pip install nimble[dateutil]
    or
  # multiple extras
  pip install nimble[requests,matplotlib,sklearn]
    or
  # shortcut (scipy and pandas)
  pip install nimble[data]

.. note::
   - The names of the extras match the names displayed in the "Package" columns
     in :ref:`optional-packages`.

   - The brackets may need to be escaped in some shells.


**With conda:**

For ``conda``, :ref:`optional-packages` must be installed manually. However,
``conda`` allows for multiple packages to be installed at the same time so
they can be listed alongside ``nimble``, provided they are available in the
known channels.

.. code-block::

  conda install nimble matplotlib scikit-learn -c https://www.nimbledata.org/nimble-data

.. note::
  - The package names used for the installation do not always match the
    python importable names displayed in the "Package" columns in :ref:`optional-packages`,
    for example, "scikit-learn" is used to install the ``sklearn`` package and
    "python-dateutil" is used to install the ``dateutil`` package.

.. _basic-install:

Basic Installation
------------------

This will install Nimble and its NumPy dependency only. Accessing much
of Nimble's functionality will require manually installing the
:ref:`optional-packages`. Nimble will raise its ``PackageException`` for
operations requiring an optional package that is not installed.

**With pip**::

  pip install nimble


**With conda**::

  conda install nimble -c https://www.nimbledata.org/nimble-data

.. _optional-packages:

Optional Packages
-----------------

Many components of Nimble rely on the following third-party packages.
Most packages are ``pip`` and ``conda`` installable, but install
recommendations vary and some offer further optimizations. **Reading the
linked installation instructions for each package is highly recommended.**

.. |cm| unicode:: U+02713 .. check mark
.. _NumPy: https://numpy.org/
.. _packaging: https://packaging.pypa.io/
.. _tomli: https://github.com/hukkin/tomli
.. _datetime: https://docs.python.org/3/library/datetime.html
.. _scipy: https://www.scipy.org/install.html
.. _pandas: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
.. _matplotlib: https://matplotlib.org/users/installing.html
.. _requests: https://requests.readthedocs.io/en/master/user/install/
.. _cloudpickle: https://github.com/cloudpipe/cloudpickle
.. _dateutil: https://dateutil.readthedocs.io/en/stable/
.. _h5py: https://docs.h5py.org/en/stable/build.html
.. _hyperopt: http://hyperopt.github.io/hyperopt/
.. _storm_tuner: https://github.com/ben-arnao/StoRM

.. _operational-support:

Operational Support
^^^^^^^^^^^^^^^^^^^
.. table::
   :align: left
   :widths: auto

   +----------------+----------------------------------------------+------+-------+
   | Package        | Supports                                     | pip  | conda |
   +================+==============================================+======+=======+
   | `scipy`_       | Nimble's ``Sparse`` object and various       | |cm| | |cm|  |
   |                | scientific calculations.                     |      |       |
   +----------------+----------------------------------------------+------+-------+
   | `pandas`_      | Nimble's ``DataFrame`` object.               | |cm| | |cm|  |
   +----------------+----------------------------------------------+------+-------+
   | `matplotlib`_  | Plotting.                                    | |cm| | |cm|  |
   +----------------+----------------------------------------------+------+-------+
   | `requests`_    | Retrieving data from the web.                | |cm| | |cm|  |
   +----------------+----------------------------------------------+------+-------+
   | `cloudpickle`_ | Saving Nimble data objects.                  | |cm| | |cm|  |
   +----------------+----------------------------------------------+------+-------+
   | `dateutil`_    | Parsing strings to `datetime`_ objects.      | |cm| | |cm|  |
   +----------------+----------------------------------------------+------+-------+
   | `h5py`_        | Loading hdf5 files.                          | |cm| | |cm|  |
   +----------------+----------------------------------------------+------+-------+
   | `hyperopt`_    | Bayesian method for hyperparameter tuning.   | |cm| | |cm|  |
   +----------------+----------------------------------------------+------+-------+
   | `storm_tuner`_ | StochasticRandomMutator for hyperparameter   | |cm| |       |
   |                | tuning.                                      |      |       |
   +----------------+----------------------------------------------+------+-------+

.. _sklearn: https://scikit-learn.org/stable/install.html
.. _tensorflow: https://www.tensorflow.org/install
.. _autoimpute: https://autoimpute.readthedocs.io/en/latest/user_guide/getting_started.html
.. _keras: https://keras.io/getting_started/

.. _ml-interfaces:

Machine-Learning Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. table::
   :align: left
   :widths: auto

   +----------------+--------------------------------------------------+------+----------------------------+
   | Package        | Provides                                         | pip  | conda                      |
   +================+==================================================+======+============================+
   | `sklearn`_     | Machine Learning.                                | |cm| | |cm|                       |
   +----------------+--------------------------------------------------+------+----------------------------+
   | `tensorflow`_/ | Neural Networks.                                 | |cm| | |cm|                       |
   | `keras`_       | See :ref:`install note <tensorflow-note>` below. |      |                            |
   +----------------+--------------------------------------------------+------+----------------------------+
   | `autoimpute`_  | Imputation. Machine Learning with missing data.  | |cm| |                            |
   +----------------+--------------------------------------------------+------+----------------------------+

.. _tensorflow-note:

.. note::
   Tensorflow recommends building from source, but some third parties provide prebuilt
   ``tensorflow`` binaries available for various architectures.

   - Unix: https://github.com/lakshayg/tensorflow-build
   - Windows: https://github.com/fo40225/tensorflow-windows-wheel
