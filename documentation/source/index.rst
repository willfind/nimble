.. Nimble documentation master file, created by
.. sphinx-quickstart on Fri Nov 13 12:34:47 2015.
.. You can adapt this file completely to your liking, but it should at least
.. contain the root `toctree` directive.

Nimble - python data science package
====================================

Nimble provides a unified framework for data science, data analysis, and machine
learning in Python that can be used as a more elegant alternative to the standard
stack (numpy, pandas, scikit-learn/sklearn, scipy etc.). Nimble can also be used
alongside these standard tools to make it faster and easier to make predictions and
manipulate, analyze, process and visualize data.

Getting Started
---------------

The simplest way to get up and running is to use pip install on a command line, with
the quickstart extra dependencies. You can check :doc:`install` for more detailed options.

.. code-block:: bash

   pip install nimble\[quickstart\]

Then, to get started
in a script, load your data by calling :doc:`nimble.data` with a URL or local path.

.. code-block:: python

   import nimble
   url = "https://storage.googleapis.com/nimble/Metro_Interstate_Traffic_Volume.csv"
   loaded = nimble.data(url)

From there, you can check the links in our :doc:`cheatsheet` or
annotated `API Docs <https://www.nimbledata.org/docs/index.html>`_
to see what's possible.

However, the best way to see what nimble is capable of is to see it in action.
So we also invite you to check out the examples below and explore how Nimble
makes data science easier!

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

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
