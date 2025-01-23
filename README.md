# EE411-Project
# Reproducibility Challenge:\\Overcoming catastrophic forgetting in neural networks


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

This repository contains the work done for our [Fundamentals of Inference and Learning](https://edu.epfl.ch/coursebook/en/fundamentals-of-inference-and-learning-EE-411)) project. The goal of our effort is to use amino acid sequences to predict the melting temperature (Tm) of proteins, which an important characteristic that indicates a protein's thermal stability.

## Data

The initial approach was to use the [train data](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data?select=train.csv) publicly available for Kaggle's competition [Novozymes Enzyme Stability Prediction](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data). There were updates on this file made by the competition organizers and the final train file used is [train_updated](https://github.com/CS-433/ml-project-2-orlovietf/blob/main/train_updated.csv).

One part of the experiments was conducted by directly using sequences as inputs to our methods and values of tm as outputs. However, based on this [discussion](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/358320), we incorporated the knowledge based data preprocessing to obtain [TO-DO].

## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## Method

We implemented two main approaches for this problem.

- Pretrained ESM 2 Model: Utilize the pretrained [ESM 2 model](https://huggingface.co/docs/transformers/model_doc/esm) from Hugging Face to generate sequence embeddings. These embeddings are further fine-tuned by training a [neural network](https://github.com/CS-433/ml-project-2-orlovietf/blob/main/scripts/baseline_model.py) to predict Tm values.

- Carbonara Architecture: Used the [Carbonara architecture](https://github.com/LBM-EPFL/CARBonAra/tree/main) embeddings from our data (precisely, the output from the penultimate layer) and:
    1. Obtained features used to train a neural network.
    2. Used RNN model on embeddings directly.

## How To Use

To train an esm model, run the following command:

```bash
python3 scripts/run.py
```

Alternatively, you can also run the `models/esm.ipynb` notebook.

To run the Carbonara models,  you need to retieve the carbonara outputs used to train the model from this [link](https://epflch-my.sharepoint.com/:u:/g/personal/igor_pavlovic_epfl_ch/EW0x2__-kNVEib42D4KRgUQBYNWJA5R20PDBnNYSuALEKg?e=LemQ7m), and store it inside the root folder.
Afterward simply run the `models/carbonara_simpl.ipynb` or `models/carbonara_rnn.ipynb` notebooks to reproduce the results.

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
    ├── datasets.py: definition ProteinDataset used by the ESM model
    ├── esm.py: defitinion of the esm model we used
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


The table below shows the results obtained for Model 1 (ESM) and the best Model 2 (Carbonara MLP with pLDDT factor). 

| **Model**    | **PCC**        | **SCC**        | **RMSE**       | **MAE**       |
|--------------|----------------|----------------|----------------|---------------|
| **ESM**      | 0.77 ± 0.01    | 0.56 ± 0.01    | 7.7 ± 0.1      | 5.6 ± 0.1     |
| Carbonara    | 0.49 ± 0.01    | 0.37 ± 0.01    | 11.6 ± 0.2     | 8.7 ± 0.3     |

The estimated training time for 5 epochs of the ESM model is approximately 30 minutes on a workstation with a dedicated GPU. 
In comparison, the estimated training times for 100 epochs of the Carbonara MLP and Carbonara RNN models are roughly 2 minutes and 30 minutes, respectively.

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
