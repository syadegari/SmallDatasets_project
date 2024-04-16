<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Course Small Datasets](#course-small-datasets)
   * [Overview](#overview)
   * [Installation](#installation)
      + [Setting up a Virtual Environment](#setting-up-a-virtual-environment)
      + [Installing Dependencies](#installing-dependencies)
      + [Installing the Package](#installing-the-package)
      + [Uninstalling](#uninstalling)
      + [Rerunning or viewing the results ](#rerunning-or-viewing-the-results)
   * [Problems and Solutions](#problems-and-solutions)
      + [Travel Planner Problem](#travel-planner-problem)
      + [Loan Funding Prediction Problem](#loan-funding-prediction-problem)

<!-- TOC end -->

<!-- TOC --><a name="course-small-datasets"></a>



# Course Small Datasets

## Overview

This is the repository for the Course "Small Datasets in Machine Learning" Course. This project is specifically tailored for working with small datasets. There are two separate subprojects, namely the "Travel Planner Problem" and "Loan Funding Prediction Problem". The formeer is solved using the transfer learning and the latter uses VAEs (Variational Auto Encoders) to tackle the problem of generating synthetic data for an unbalanced tabular dataset. Here, you will find information about how to use this package as well as some design choices that went into the code and specifications of each subproject. 

## Installation

### Setting up a Virtual Environment

First, create and activate a virtual environment to isolate package dependencies:

```bash
python -m venv venv
source venv/bin/activate 
```
### Installing Dependencies

Before installing the package, you might want to install the necessary dependencies which are listed in the `requirements.txt` file. After activating your virtual environment, you can install all required packages using the following command:

```bash
pip install -r requirements.txt
```

where the `requirements.txt` is located at the root of this directory.

### Installing the Package

In the cmdline, navigate to the root of this directory Install the package using pip:

```bash
pip install -e .
```

### Uninstalling

If you need to uninstall the package, use:

```bash
pip uninstall course_small_datasets
```

### Rerunning or viewing the results 

Each subproject's results and methodologies are documented in separate Jupyter notebooks located at the root of this repository: [Transfer Learning Notebook](./transfer_learning.ipynb) for the Travel Planner Problem and [Synthetic Data Notebook](./synthetic_data.ipynb) for the Loan Funding Prediction Problem. 

To make the review process easier and minimize cognitive load, all supporting code, including model definitions, training routines, and data preprocessing, etc., is organized into Python modules within the repository's source code directory. This ensures that the notebooks remain focused on results and methodological explanations. 

## Problems and Solutions
The details of each subproject is thoroughly documented in its respective Jupyter notebook, where you find  discussions on the results and conclusions. Below, we offer a brief overview of the implementation and design choices made for each subproject.

### Travel Planner Problem


This subprojec uses transfer learning via PyTorch Lightning to enhance the training process using pretrained models from torchvision. PyTorch Lightning simplifies the implementation of common routines in supervised learning and offers flexible methods for model adjustments, such as the `.freeze()` method (see this [link](https://lightning.ai/docs/pytorch/latest/_modules/lightning/pytorch/core/module.html#LightningModule.freeze)), which freezes model parameters, a necessary step for transfer learning approach.

In this subproject, rather than settling on a single model, as suggested by the project guidelines, three predefined models were evaluated to determine the best based on prediction accuracy and inference time, critical factors for resource constrained hardware. Further optimizations like pruning and quantization were not explored, but they represent potential future enhancements to improve inference efficiency in this repository.


### Loan Funding Prediction Problem

This subproject uses custom training routines developed in PyTorch, enhancing features from the course tutorial such as restart capabilities, early stopping with adjustable parameters, and the logging and saving of optimal model weights for further evaluation. Unlike other subprojects, this one does not use PyTorch Lightning due to the complexities involved in implementing unsupervised learning within that framework.

Two model architectures and two model size variants are considered, resulting in 4 variants. The primary architecture is based on the course demo/tutorial, constructing the encoder and decoder of the VAE using layers of fully connected blocks, ReLU activation, and batch normalization. An alternative approach uses skip connections within the same layout, similar to the [ResNet](https://arxiv.org/abs/1512.03385) architecture, resulting in four unique model configurations when combined with the two size options. The optimal model, out of four total variants, was selected based on F1-score performance across an expanded dataset.

Potential future enhancements could include experimenting with other VAE types such as beta-VAEs, which modify the traditional VAE loss function to improve disentanglement, or adversarial autoencoders that could potentially refine the quality of synthetic data.
