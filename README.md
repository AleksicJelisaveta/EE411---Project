# EE411-Project
# Reproducibility Challenge: Overcoming catastrophic forgetting in neural networks


<p align="center">
  <a href="#about">About</a> •
  <a href="#data">Data</a> •
  <a href="#installation">Installation</a> •
  <a href="#method">Method</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#code-structure">Code Structure</a> •
  <a href="#results">Results</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
</p>

## Team
The project is accomplished by team:

Jelisaveta Aleksic - @AleksicJelisaveta

Natasa Jovanovic - @natasa-jovanovic

Marija Rakonjac - @marijarakonjac

Strahinja Nikolic - @strajdzsha

Milica Vukasinovic - @milicavukaa

## About

This repository contains the work done for our [Fundamentals of Inference and Learning](https://edu.epfl.ch/coursebook/en/fundamentals-of-inference-and-learning-EE-411) project. The goal of our effort is to reproduce results obtained in (https://arxiv.org/pdf/1612.00796) where using Elastic Weight Consolidation, regularization technique in order to mitigate catastrophic forgetting in neural networks. Results are compared with L2 regularization and no regularization. Datasets are permutated and rotated MNIST dataset. 
## Data
Two datasets are used: PermutedMNIST - a unique fixed random permutation of the pixel values is applied for each task
as well as RotatedMNIST - ten tasks of rotating MNIST, where task i is
produced by a fixed rotation of 10(i−1) degrees). 



## Method


## How To Use


## Code Structure

```
├── metricks and plots:  Folder containing metrics and plots of our models
├── models
    ├── carbonara_embeddings.ipynb: notebok to process carbonara features and extract embeddings
    ├── carbonara_rnn.ipynb: rnn model based on carbonara embeddings
    ├── carbonara_simple.ipyng: MLP model based on carbonara embeddings
    ├── data_exploration.ipynb: notebook for data explotation
    ├── esm.ipynb: model based on the ESM output
    ├── evaluate_models.ipynb: computes the relevant metrics and plots the results
├── predictions: Folder containing model predictions on the validation dataset
├── scripts
    ├── datasets.py: 
    ├── 
    ├── evaluate.py: code to evaluete the performance of the esm model
    ├── run.py: python script to run and evaluate the ESM model
    ├── train.py: code for training the esm model
├── CS_433_Class_Project_2.pdf: a report of the project.
├── README.md
├── requirements.txt
├── test.csv: csv file containg test data
├── train.csv: csv file containing training data
├── train_wildtype_groups.csv: csv file containing grouped trained data
├── train_no_wildtype.csv: csv file containing training data which are not grouped
```

## Results


## Credits
This work was carried out as part of a reproducibility challenge, to validate the findings presented in the study (https://arxiv.org/pdf/1612.00796). We acknowledge the authors of the original paper for their contributions that inspired this reproduction effort. 


This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
