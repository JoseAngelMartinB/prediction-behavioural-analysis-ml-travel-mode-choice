# Code and experiments of the paper "A prediction and behavioural analysis of machine learning methods for modelling travel mode choice"

Authors:
* José Ángel Martín-Baos
* Julio Alberto López-Gomez
* Luis Rodríguez-Benítez
* Tim Hillel
* Ricardo García-Ródenas

This paper has been submitted for publication in *Transportation Research Part C: Emerging Technologies*.

Preprint available at [arXiv](https://arxiv.org/abs/2301.04404).

## Abstract

The emergence of a variety of Machine Learning (ML) approaches for travel mode choice prediction poses
an interesting question for transport modellers: which models should be used for which applications? The
answer to this question goes beyond simple predictive performance, and isinstead a balance of many factors,
including behavioural interpretability and explainability, computational complexity, and data efficiency.
There is a growing body of research which attempts to compare the predictive performance of different ML
classifiers with classical Random Utility Models (RUMs). However, existing studies typically analyse only
the disaggregate predictive performance, ignoring other aspects affecting model choice. Furthermore, many
existing studies are affected by technical limitations, such as the use of inappropriate validation schemes,
incorrect sampling for hierarchical data, a lack of external validation, and the exclusive use of discrete metrics.
In this paper, we address these limitations by conducting a systematic comparison of different modelling
approaches, across multiple modelling problems, in terms of the key factors likely to affect model choice (out-
of-sample predictive performance, accuracy of predicted market shares, extraction of behavioural indicators,
and computational efficiency). The modelling problems combine several real world datasets with synthetic
datasets, where the data generation function is known. The results indicate that the models with the highest
disaggregate predictive performance (namely Extreme Gradient Boosting (XGBoost) and Random Forests
(RF)) provide poorer estimates of behavioural indicators and aggregate mode shares, and are more expensive
to estimate, than other models, including Deep Neural Networks (DNNs) and Multinomial Logit (MNL). It
is further observed that the MNL model performs robustly in a variety of situations, though ML techniques
can improve the estimates of behavioural indices such as Willingness To Pay (WTP).


## Software implementation

All source code used to generate the results and figures in the paper are contained in this repository. The code is written in Python 3.9, and is organised in the following folders:

The data used in this study is provided in the `Data` folder. See the `README.md` file inside the `Data/Datasets` folder for the references of the datasets used in this study. `Data` also contains the adjusted hyperparameters for the models used in this study.

The folder `SimulateDatasets` contains the code used to generate the synthetic datasets used in this study.

The `Models` folder contains a wrapper for the models used in this study. The wrapper is used to train and test the models, and to extract the behavioural indicators. The models used in this study are implemented in Python, using the [scikit-learn](http://scikit-learn.org/stable/) library, the [XGBoost](https://xgboost.readthedocs.io/en/latest/) library, and the [Biogeme](https://biogeme.epfl.ch/) library.

The root folder contains the environment file with the Anaconda dependencies needed to execute the code. The `experiments_functions.py` file contains several functions that are needed during the experiments. Finally, the rest of the python files are used to execute the experiments. The files starting with `0-Preprocess-` are used to preprocess the original datasets. Next, the files starting with `1-` are used to tune the model hyperparameters, which are stored in `Data/adjusted-hyperparameters` folder. Finally, the files starting with `2-Experiment-1-3` and `2-Experiment-4` contains the code used to execute the experiments 1 to 3 and 4, respectively. The calculations and figure generation are all run inside
[Jupyter notebooks](http://jupyter.org/).


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice.git

or [download a zip archive](https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice/archive/master.zip).


## Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `environment.yml`.

We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create


## Reproducing the results

Before running any code you must activate the conda environment:

    source activate MLCompEnv

or, if you're on Windows:

    activate MLCompEnv

This will enable the environment for your current terminal session.
Any subsequent commands will use software that is installed in the environment.

To execute the Jupyter notebooks you must first start the notebook server by going into the
repository top level and running:

    jupyter notebook

This will start the server and open your default web browser to the Jupyter
interface. In the page, select the
notebook that you wish to view/run.

The notebook is divided into cells (some have text while other have code).
Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code
and produces it's output.
To execute the whole notebook, run all cells in order.


## Figures, tables and extra results not included in the paper

The figures and tables included in the paper are generated in the Jupyter notebooks,
and stored in the `Figures/` and `Latex_tables` folders, respectively.
Moreover, those folders also contain some extra figures and tables that are not 
included in the paper. Some of these results include the MNL coefficients table
for each of the real datasets, or the figures showing the SHAP values for each of
the models and datasets.


## License

All source code is made available under a MIT license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in 
*Transportation Research Part C: Emerging Technologies*.