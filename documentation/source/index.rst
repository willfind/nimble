.. Nimble documentation master file, created by
.. sphinx-quickstart on Fri Nov 13 12:34:47 2015.
.. You can adapt this file completely to your liking, but it should at least
.. contain the root `toctree` directive.

Nimble - Python data science package
====================================

Nimble provides a unified framework for data science, data analysis, and machine
learning in Python that can be used as a more elegant alternative to the standard
stack (numpy, pandas, scikit-learn/sklearn, scipy, etc.). Nimble can also be used
alongside these standard tools to make it faster and easier to make predictions and
manipulate, analyze, process and visualize data.

Getting Started
---------------

You can check the :doc:`install` page for detailed options or just started with
this:

.. code-block:: bash

   pip install nimble\[quickstart\]

Once Nimble is installed, you can load data in a script by calling
``nimble.data`` with a URL or local path.

.. code-block:: python

   import nimble
   urlOrPath = "https://storage.googleapis.com/nimble/Metro_Interstate_Traffic_Volume.csv"
   loaded = nimble.data(urlOrPath)

From there, you can explore the examples below or check out the the
:doc:`cheatsheet` and `API Docs <https://www.nimbledata.org/docs/index.html>`_
to see what's possible!

Examples
--------

.. toctree::
   :maxdepth: 1

   examples/cleaning_data
   examples/supervised_learning
   examples/exploring_data
   examples/unsupervised_learning
   examples/neural_networks
   examples/merging_and_tidying_data
   examples/additional_functionality

Resources
---------

.. toctree::
   :maxdepth: 1

   install
   API Documentation <docs/index>
   Cheatsheet <cheatsheet>
   datasets

Contact
-------

Feel free to get in touch with us at:

.. image:: ./_static/sparkwave_email.png
   :width: 334
   :height: 30

or through the `Nimble Github <https://github.com/willfind/nimble>`_ page.
