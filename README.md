# Titanic Survival Prediction

## Overview

This project analyzes and predicts the survival of passengers aboard the Titanic using machine learning techniques. By leveraging the Titanic dataset, the project demonstrates various aspects of data preprocessing, exploratory data analysis, and model training.

## Table of Contents

- [Project Description](#project-description)
- [Libraries](#libraries)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Description

The Titanic dataset contains information about passengers on the ill-fated ship, including attributes such as age, sex, class, and fare. The primary objective of this project is to predict whether a passenger survived based on these attributes. Key components include:

- Data exploration and visualization
- Handling missing values
- Feature engineering
- Model selection and evaluation

## Libraries

This project utilizes the following libraries:

- `pandas` - For data manipulation and analysis
- `numpy` - For numerical computations
- `matplotlib` - For data visualization
- `seaborn` - For statistical data visualization
- `scikit-learn` - For machine learning and model evaluation

## Dataset

The dataset used in this project is the Titanic dataset, available on [Kaggle](https://www.kaggle.com/c/titanic/data). It includes the following features:

- `PassengerId`: Unique identifier for each passenger
- `Survived`: Survival status (0 = No, 1 = Yes)
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Ticket fare
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

## Data Preprocessing

Data preprocessing steps include:

1. **Handling Missing Values:**
   - Filling missing values in the `Age` and `Embarked` columns.
   - Normalizing the `Fare` column.

2. **Encoding Categorical Variables:**
   - Encoding variables such as `Sex` and `Embarked` to numerical values.

## Feature Engineering

1. **Feature Importance Analysis:**
   - Calculating and visualizing feature importance scores from the initial Random Forest model.

2. **Feature Selection:**
   - Removing the least important feature based on the feature importance analysis.

## Modeling

1. **Initial Model Training:**
   - Using a `RandomForestClassifier` with hyperparameter tuning through `GridSearchCV` to find the best hyperparameters.

2. **Model Evaluation:**
   - Evaluating the model's performance on the test set and reporting the accuracy.

3. **Retraining with Selected Features:**
   - Removing the least important feature and retraining the model with updated hyperparameters.

## Results

- **Best Hyperparameters:** Reported from GridSearchCV
- **Initial Model Accuracy:** Accuracy on the test set before feature removal
- **Retrained Model Accuracy:** Accuracy on the test set after feature removal and hyperparameter tuning

## Conclusion

By analyzing feature importance and removing less significant features, the model's performance improved. The re-tuning of hyperparameters and feature selection contributed to enhanced predictive accuracy and model efficiency.

