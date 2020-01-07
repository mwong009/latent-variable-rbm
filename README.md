# Latent Variable RBM
RBM discrete choice model.
The model uses [Theano](https://github.com/Theano/Theano) deep learning library for its backend.
Data is stored as a .csv file with various explanatory variables. 
The dataset is constructed by dividing the data into 3 parts, training, validation and testing sets.
The goal is to develop a choice model using a hybrid latent variable and RBM generative modelling algorithm.
Indicators manage the personal psychometric attributes and are modelled by the hidden units in the CRBM model.
Beta parameters are connected from the input variables to the output, conditional on the hidden units.
This allows the gradient to back prop from the choice likelihood obj. function to the input layer directly to update the beta parameters.

## File description
```data/``` contains the dataset files

Model files: ```crbm.py```, ```rbm.py```, ```mlp.py```, ```logistic_sgd.py```.

```utils.py``` misc. functions to generate output visualizations and analytics.

```optimizers.py```, ```neural_networks.py``` core functional files for DNN optimization.

```build.py``` extracts a .h5 format dataset file from .csv files and automatically segments train, valid and test set. Auto-randomization of observation can be enabled through the ```build_dataset()``` function.

### Dataset
The dataset ```data/santander.csv``` is a panel data structured choice set (13 choices) from product transactions obtained from the Kaggle database. 
The data shows monthly record of each product purchase per customer. 
Time range: Jan 2015 to Jun 2016.
There are 20 explanatory variables listed in the dataset associated with each customer.

## Getting started
To begin, run ```python3 crbm.py``` to estimate the model using the Santander dataset. 
To run other types of models, use ```python3``` command on ```mlp.py```, ```logistic_sgd.py```, or ```rbm.py```.

Note: specify the dataset and hyperparameters in the ```test_crbm(dataset='')``` function

### Prerequisites
Python 3.5+ (with pip3), Numpy, Pandas, Theano

Install requirements through pip:

```
pip3 install --user -r requirements.txt
```

## Versioning
0.1 Inital version

## Authors
Melvin Wong ([Github](https://github.com/mwong009))

## Licence
See [LICENCE](https://github.com/LiTrans/latent-variable-rbm/blob/master/LICENSE) for details

## Bug reporting
Please use the open pull request and describe the problem in detail.
