# MAP565 Memoir: A predictive analysis of the Goldman Sachs closing price

Leveraging on the work done in this RGAN repo, I try to build a GAN able to generate realistic financial time-series.

## Architecture

The basic structure is followed ckmarkoh's git https://github.com/ckmarkoh/GAN-tensorflow as in the master branch. 

### Modeling
Each modules, generator and discriminator are designed with LSTMs and one Fully Connected Network.
Generator is designed as one-to-many, and gets one random vector as input, and generates sequential images.
Discriminator is designed as many-to-one model, which get sequential images, and decides what is real or fake.

![alt tag](https://github.com/jaesik817/SequentialData-GAN/blob/master/figures/model.png)

### Results:

Currently it doesn't work that well. :( 

## Code Quickstart

Primary dependencies: `arch`, `scipy`, `numpy`, `statsmodels`

Note: This code is written in Python3!

## Files in this Repository

`MAP565.ipynb`: An iPython notebook contaning the code used to fit ARIMA/GARCH models + the statistical tests.

`/data`: A directory containing the data used or that will be used for this model.

## Data sources (TODO)

Yahoo! Finance
