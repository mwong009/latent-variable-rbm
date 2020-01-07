# Latent Variable RBM
RBM discrete choice model

## File description
```data/``` contains the dataset files

Model files: ```crbm.py```, ```rbm.py```, ```mlp.py```, ```logistic_sgd.py```.

```utils.py``` misc. functions to generate output visualizations and analytics.

```build.py``` extracts a .h5 format dataset file from .csv files and automatically segments train, valid and test set. Auto-randomization of observation can be enabled through the ```build_dataset()``` function.

### Dataset
The dataset ```data/santander.csv``` is a panel data structured choice set (13 choices) from product transactions obtained from the Kaggle database. 
The data shows monthly record of each product purchase per customer. 
Time range: Jan 2015 to Jun 2016.
There are 20 explanatory variables listed in the dataset associated with each customer.
The goal is to develop a choice model using a hybrid latent variable and RBM generative modelling algorithm.

## Getting started
To begin, run ```python3 crbm.py``` to estimate the model using the Santander dataset. 
To run other types of models, use ```python3``` command on ```mlp.py```, ```logistic_sgd.py```, or ```rbm.py```

## Versioning

## Authors

## Licence
