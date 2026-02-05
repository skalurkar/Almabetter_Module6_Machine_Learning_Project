# Almabetter_Module6_Machine_Learning_Project

📈 YES Bank Stock Closing Price Prediction
📌 Project Overview

This project focuses on predicting the closing price of YES Bank stock using regression-based machine learning models. Given the major fluctuations in YES Bank’s stock price after 2018, the objective is to analyze historical stock data and identify the model that best predicts the closing price based on available features.

The project applies exploratory data analysis (EDA), feature engineering, and multiple regression models, followed by performance comparison using evaluation metrics.

🎯 Problem Statement

YES Bank, a prominent Indian private sector bank, witnessed significant volatility in its stock prices following financial controversies starting in 2018.

The goal of this project is to:

Perform regression analysis on historical stock data

Build and compare multiple machine learning models

Predict the closing price of the stock

Identify the best-performing model based on evaluation metrics

📂 Dataset

Source: Historical YES Bank stock price data (CSV file)

Features include:

Date

Open Price

High Price

Low Price

Close Price

The dataset is preprocessed and cleaned before model training.

🔍 Exploratory Data Analysis (EDA)

The EDA section includes:

Viewing dataset structure (head, tail, info)

Date formatting and sorting

Visualization of stock price trends

Correlation analysis between numerical features

Multicollinearity checks using Variance Inflation Factor (VIF)

🧠 Models Implemented

The following regression models were trained and evaluated:

Linear Regression

Lasso Regression (with hyperparameter tuning)

Ridge Regression (with hyperparameter tuning)

K-Nearest Neighbors (KNN) Regressor

XGBoost Regressor

Hyperparameter tuning was performed using GridSearchCV wherever applicable.

📏 Evaluation Metrics

Each model was evaluated using the following metrics:

MAE – Mean Absolute Error

MSE – Mean Squared Error

RMSE – Root Mean Squared Error

MAPE – Mean Absolute Percentage Error

A comparison table is created to identify the best-performing model.

🏆 Results & Conclusion

Multiple regression models were compared based on error metrics.

Regularization techniques (Lasso & Ridge) helped in controlling overfitting.

Advanced models like XGBoost showed improved performance over basic linear models.

The model with the lowest RMSE and MAPE was selected as the best predictor.

🛠️ Tech Stack & Libraries

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Statsmodels

XGBoost

👤 Author

Sanket Alurkar
Data Science & Machine Learning Enthusiast
