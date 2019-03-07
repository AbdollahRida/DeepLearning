# RGAN

This repository (will) contain(s) a pytorch implementation of the code from the paper
_[Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633)_, 
by Stephanie L. Hyland* ([@corcra](https://github.com/corcra)), Cristóbal Esteban* 
([@cresteban](https://github.com/cresteban)), and Gunnar Rätsch ([@ratsch](https://github.com/ratsch)), 
from the Ratschlab, also known as the [Biomedical Informatics](http://bmi.inf.ethz.ch/) Group at ETH Zurich.

## Architecture

The basic structure is followed ckmarkoh's git https://github.com/ckmarkoh/GAN-tensorflow

### Modeling
Each modules, generator and discriminator are designed with LSTMs and one Fully Connected Network.
Generator is designed as one-to-many, and gets one random vector as input, and generates sequential images.
Discriminator is designed as many-to-one model, which get sequential images, and decides what is real or fake.

![alt tag](https://github.com/jaesik817/SequentialData-GAN/blob/master/figures/model.png)

## Code Quickstart

Primary dependencies: `torch`, `scipy`, `numpy`

Note: This code is written in Python3!

## Files in this Repository

TODO

## Data sources (TODO)

MNIST
