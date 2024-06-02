## Nimble - python data science package

Nimble provides a unified framework for data science, data analysis, and machine
learning in Python that can be used as a more elegant alternative to the standard
stack (numpy, pandas, scikit-learn/sklearn, scipy etc.). Nimble can also be used
alongside these standard tools to make it faster and easier to make predictions and
manipulate, analyze, process and visualize data.

### Getting Started

The simplest way to get up and running is to use pip install on a command line, with
the quickstart extra dependencies. You can check [Installation](https://www.nimbledata.org/install.html)
for more detailed options.

```
   pip install nimble\[quickstart\]
```

Then, to get started in a script, load your data by calling
[`nimble.data`](https://www.nimbledata.org/docs/generated/nimble.data.html)
with a URL or local path.

```
   import nimble
   url = "https://storage.googleapis.com/nimble/Metro_Interstate_Traffic_Volume.csv"
   loaded = [nimble.data](https://www.nimbledata.org/docs/generated/nimble.data.html)(url)
```

From there, you can check the links in our [Cheatsheet](https://www.nimbledata.org/cheatsheet.html)
or annotated [API Docs](https://www.nimbledata.org/docs/index.html)
to see what's possible.

However, the best way to see what nimble is capable of is to see it in action.
So we also invite you to check out the examples below and explore how Nimble
makes data science easier!

### Examples

* [Cleaning Data](https://www.nimbledata.org/examples/cleaning_data.html)
* [Supervised Learning](https://www.nimbledata.org/examples/supervised_learning.html)
* [Exploring Data](https://www.nimbledata.org/examples/exploring_data.html)
* [Unsupervised Learning](https://www.nimbledata.org/examples/unsupervised_learning.html)
* [Neural Networks](https://www.nimbledata.org/neural_networks.html)
* [Merging And Tidying Data](https://www.nimbledata.org/examples/merging_and_tidying_data.html)
* [Additional Functionality](https://www.nimbledata.org/examples/additional_functionality.html)

### Resources

* [Installation](https://www.nimbledata.org/install.html)
* [API Documentation](https://www.nimbledata.org/docs/index.html)
* [Cheatsheet](https://www.nimbledata.org/cheatsheet.html)
* [Example Datasets](https://www.nimbledata.org/datasets.html)

