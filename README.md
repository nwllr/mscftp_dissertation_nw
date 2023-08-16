### mscftp_dissertation_nw


# A New Machine Learning Approach to Agnostic Fundamental Valuation


This repository contains all supporting files for the MSc Dissertation *"A New Machine Learning Approach to Agnostic Fundamental Valuation"*.

### Main Experiment

Running the **main.py** file will produce the main results of the study. It makes use of the following custom modules:

- **EvalFunctions.py**: Functions used to calculate mispricing, quintiles and first results
- **EstAlpha.py**: Functions used to estimate alpha returns
- **PrePro.py**: Functions used for preprocessing of the data
- **Visual.py**: Functions to produce visualisations (larger part of visualisations are produced separately in the **Visualisations_jupyter.ipynb** file)
- **DimReduction.py**: Principal Component Analysis implementation used for dimensionality reduction

- **BGLR.py**: Cross-sectional Linear Regression implementation following Bartram and Grinblatt (2018)
- **LR.py**: OLS Linear Regression implementation
- **MultipleValuation.py**: P/E ratio multiples valuation implementation

- **RF.py**: Random Forest implementation
- **ELM.py**: Extreme Learning Machine implementation
- **MLP.py**: Multilayer Perceptron implementation
- **Ensemble.py**: RF+ELM+MLP Ensemble implementation 




### Data Preparation

The **DataPrep.py** file produces the fundamental datasets that are used in this analysis.


### Data

The data folder contains all raw data files and the combined datasets produced by **DataPrep.py**. The main dataset containing all features is *cleaned_dataset_2M_delay.csv*.


### Result Files

The folders *model_evaluations* 1-4 contain different levels of result analyses produced in the experiment.

The folders starting with *alpha* contain the summarised alpha results for each model as well as the detailed regression results for each model as referred to in the main text dissertation. 

The *visualisations* folder contains the visualisations produced by **Visual.py** and **Visualisations_jupyter.ipynb**.










