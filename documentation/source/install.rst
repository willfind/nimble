Installation
============

Nimble can be installed in a variety of ways and strives for flexibility
during the install process. The only install dependency is `NumPy`_  but much
of Nimble's functionality is provided through third-party
:ref:`optional-packages`. This ensures that Nimble does not install additional
packages that will not be used, but Nimble's functionality is very limited with
only NumPy installed. For this reason, we offer many ways to install optional
packages while installing Nimble. We recommend the :ref:`quickstart-install`
to make much of Nimble's functionality available with a single install.

Install Methods
---------------

* :ref:`quickstart-install` (recommended)
* :ref:`specific-install`
* :ref:`basic-install`

.. _optional-packages:

Optional Packages
-----------------

Many components of Nimble rely on the following third-party packages.
Most packages are ``pip`` and ``conda`` installable, but install
recommendations vary and some offer further optimizations. **Reading the
linked installation instructions for each package is highly recommended.**

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
   | `cloudpickle`_ | Saving and loading Nimble data objects.      | |cm| | |cm|  |
   +----------------+----------------------------------------------+------+-------+
   | `dateutil`_    | Parsing strings to `datetime`_ objects.      | |cm| | |cm|  |
   +----------------+----------------------------------------------+------+-------+
   | `h5py`_        | Loading hdf5 files.                          | |cm| | |cm|  |
   +----------------+----------------------------------------------+------+-------+

.. _ml-interfaces:

Machine-Learning Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. table::
   :align: left
   :widths: auto

   +------------------------+--------------------------------------------------+------+----------------------------+
   | Package                | Provides                                         | pip  | conda                      |
   +========================+==================================================+======+============================+
   | `scikit-learn`_        | Machine Learning.                                | |cm| | |cm|                       |
   +------------------------+--------------------------------------------------+------+----------------------------+
   | `tensorflow`_/         | Neural Networks.                                 | |cm| | |cm|                       |
   | `keras`_               | See :ref:`install note <tensorflow-note>` below. |      |                            |
   +------------------------+--------------------------------------------------+------+----------------------------+
   | `autoimpute`_          | Imputation. Machine Learning with missing data.  | |cm| | |cm| (nimble-data channel) |
   +------------------------+--------------------------------------------------+------+----------------------------+
   | `shogun`_              | Machine Learning.                                |      | |cm| (conda-forge channel) |
   +------------------------+--------------------------------------------------+------+----------------------------+
   | `machine-learning-py`_ | Machine Learning.                                | |cm| | |cm| (conda-forge channel) |
   +------------------------+--------------------------------------------------+------+----------------------------+

.. _tensorflow-note:

.. note::
   Tensorflow recommends building from source, but some third parties provide prebuilt
   ``tensorflow`` binaries available for various architectures.

   - Unix: https://github.com/lakshayg/tensorflow-build
   - Windows: https://github.com/fo40225/tensorflow-windows-wheel

.. _quickstart-install:

Nimble Install Quickstart
-------------------------

To install a selection of :ref:`optional-packages` while installing Nimble,
both ``pip`` and ``conda`` installs offer a quickstart option. Quickstart
installs all :ref:`operational-support` packages and SciKit-Learn from the
:ref:`ml-interfaces`. These packages were chosen because they are reliably
installable through ``pip`` and ``conda`` and provide access to the majority
of Nimble's functionality.

**With pip:**

Nimble uses extras to provide the ``quickstart`` shortcut

.. code-block::

  pip install nimble[quickstart]

.. note:: The brackets may need to be escaped in some shells.

**With conda:**

The nimble-data channel provides an alternative package, ``nimble-quickstart``.
Once installed, import still occurs with ``import nimble``, the name
``nimble-quickstart`` serves to install nimble and the other included
packages.

.. code-block::

  conda install -c nimble-data nimble-quickstart

.. _specific-install:

Install with Specific Packages
------------------------------

:ref:`optional-packages` can always be installed separately with ``pip`` or
``conda``, for example: ``pip install scipy`` or ``conda install scipy``.
However, both package managers offer ways to install optional packages while
installing Nimble.

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

  pip install nimble[dateutil]                         # single extra
    or
  pip install nimble[requests,matplotlib,scikit-learn] # multiple extras
    or
  pip install nimble[data]                             # shortcut (scipy and pandas)

.. note::
   - The brackets may need to be escaped in some shells.

**With conda:**

For ``conda``, :ref:`optional-packages` must be installed manually. However,
``conda`` allows for multiple packages to be installed at the same time so
they can be listed alongside ``nimble``, provided they are available in the
available channels.

.. code-block::

  conda install -c nimble-data nimble matplotlib scikit-learn

.. _basic-install:

Basic Installation
------------------

This will install Nimble and its NumPy dependency only. Accessing much
of Nimble's functionality will require manually installing the
:ref:`optional-packages`. Nimble will raise its ``PackageException`` for
operations requiring an optional package that is not installed.

**with pip**::

  pip install nimble

**with conda**::

  conda install -c nimble-data nimble

**with setup.py (not recommended)**::

  # from nimble directory
  python setup.py install

.. |cm| unicode:: U+02713 .. check mark

.. _NumPy: https://numpy.org/
.. _datetime: https://docs.python.org/3/library/datetime.html
.. _scipy: https://www.scipy.org/install.html
.. _pandas: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
.. _matplotlib: https://matplotlib.org/users/installing.html
.. _requests: https://requests.readthedocs.io/en/master/user/install/
.. _cloudpickle: https://github.com/cloudpipe/cloudpickle
.. _dateutil: https://dateutil.readthedocs.io/en/stable/
.. _h5py: https://docs.h5py.org/en/stable/build.html
.. _scikit-learn: https://scikit-learn.org/stable/install.html
.. _tensorflow: https://www.tensorflow.org/install
.. _autoimpute: https://autoimpute.readthedocs.io/en/latest/user_guide/getting_started.html
.. _shogun: https://www.shogun.ml/install
.. _keras: https://keras.io/getting_started/
.. _machine-learning-py: https://github.com/richardARPANET/mlpy
