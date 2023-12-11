# OpenPowerlifting - Predicting TotalKg in Powerlifting Competitions
### Overview
This project focuses on predicting the total lifted weight (TotalKg) in powerlifting competitions using machine learning regression techniques. See below for requirements and usage. Accuracies and Metrics for saved models (see "Models/") are quite good.  

### Project Structure
#### Requirements
Python 3.x (A requirements txt is provided)
Libraries: pandas, scikit-learn, torch, matplotlib, numpy, prettytable

### Files
- main.py: The main script for data loading, preprocessing, model training, and evaluation.
- liftModel.py: Contains the definition of the lift regression model using PyTorch.
- openpowerlifting.csv: Dataset containing powerlifting competition data.

### Sections & Usage
- Data Loading and Preprocessing: loads the dataset, preprocesses it by dropping unnecessary columns, handling missing values, and scaling features.
- Model Definition and Training: regression model is defined and trained on the preprocessed data. Model training can be controlled by the SAVE_MODEL flag, (*Note*: SAVE_MODEL also controls the training - No save == No train).
- Model Save and Load: Trained models can be saved and loaded for future use. The SAVE_MODEL and LOAD_MODEL flags control this functionality.
- Model Evaluation: evaluates the model's performance using metrics such as Average Test Loss, Average R-squared Score, and Average RMSE.
- Results Visualization: visualizations with plots of actual vs. predicted values and residuals plots. Figures can be saved using the SAVE_FIGS flag.
- Table Generation: A table with example inputs and predictions is generated using the PrettyTable library. Tables can be saved as text files if the SAVE_CHARTS flag is set.
