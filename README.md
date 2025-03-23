### Machine Learning Regression Using Advertisement Data

### Project Overview:
The goal of this project was to build a machine learning regression model to predict the sales of products based on advertisement data. Multiple regression models were applied and evaluated to determine the best-performing model. After comprehensive analysis and model tuning, the final model was selected as the XGBoost Regressor, achieving a Mean Squared Error (MSE) of 0.708 and an R-squared score of 0.977.

### Steps Performed:

#### Data Collection and Preprocessing:

1. Used advertisement dataset containing features like TV, Radio, and Newspaper budgets and their corresponding Sales values.

2. Handled missing data and checked for inconsistencies.

3. Standardized the features using StandardScaler for consistent model training.

#### Exploratory Data Analysis (EDA) and Visualization:

1. Visualized the relationships between features and sales using scatter plots and pair plots.

2. Checked for correlations using a heatmap.

3. Identified trends and outliers to ensure the data was clean for modeling.

#### Model Building and Training:

Applied multiple regression models for training:

Linear Regression

Ridge Regression

Lasso Regression

Elastic Net Regression

Decision Tree Regressor

Random Forest Regressor

Support Vector Regressor (SVR)

K-Nearest Neighbors (KNN) Regressor

Gradient Boosting Regressor

XGBoost Regressor

#### Model Evaluation and Cross-Validation:

1. Performed cross-validation on all models to ensure robust evaluation.

2. Based on the evaluation metrics, Gradient Boosting Regressor, Random Forest Regressor, and XGBoost Regressor were identified as the top 3 models.

#### Hyperparameter Tuning Using Grid Search CV:

1. Applied Grid Search Cross Validation to optimize hyperparameters of the top 3 models.

2. After fine-tuning, XGBoost Regressor was determined as the best final model.

#### Model Fitting and Prediction:

1. Fit the optimized XGBoost Regressor to the training data.

2. Predicted sales on the test dataset.

#### Evaluation:

Calculated performance metrics for the final model:

1. Mean Squared Error (MSE): 0.708

2. R-squared (R2) Score: 0.977

The model demonstrated exceptional predictive accuracy.

### Results:

1. The XGBoost Regressor outperformed all other models in terms of accuracy and reliability.

2. The final model provided highly accurate predictions with minimal error.

3. The results confirmed that advertisement budgets significantly influence sales, particularly TV and Radio investments.
