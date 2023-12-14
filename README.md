# OpenPowerlifting - Predicting TotalKg in Powerlifting Competitions
### Overview
This project focuses on predicting the total lifted weight (TotalKg) in [powerlifting competitions](https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database/data) using machine learning regression techniques. The intent here is to determine if a ML model can accurately predict a total weight with minimal information about a competitor and without one of their top lifts, which seems somewhat unintuitive given the discepancies between competitors and their favored lift. The training dataset contains 327,586 competitors and the test dataset contains 140,394 competitors. 

 See below for requirements and usage. Accuracies and Metrics for saved models (see "Models/") are quite good.  

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

### Notes:
- In the persisted pickle models (located in Models/) the model name is the value that we don't have and are predicting for. 
    - i.e. Models/lift_model_Squat.pkl is predicting totalKg with no squat information / feature.
- The "All" Model is trained with all of the lift data provided, it is a proof of concept for the architecture and has a perfect correlation, as it has all the data.
- The "None" Model is trained with none of the top 3 lift data, and is a representation of the predicatibility of a lifters totalKg with only minimal biographical information. 
- The database is fairly large, so it is not included. It can be found [here](https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database/data)

### Side-Notes:
- The "None" Model had an R<sup>2</sup> value of 0.7513. This is very suprising and indicates that Sex, Equipment, Age, and Body Weight alone are good predictors of a competitors final total. 
