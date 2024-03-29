
# Sales Forecasting with Machine Learning

This project focuses on forecasting sales using machine learning algorithms. It utilizes various regression models such as Linear Regression, Random Forest Regressor, and XGBoost Regressor to predict future sales based on historical data.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Sales forecasting is crucial for businesses to make informed decisions regarding inventory management, resource allocation, and overall business strategy. This project aims to develop accurate sales forecasting models using machine learning techniques.

## Dataset

The dataset used in this project is `store_sale.csv`, which contains historical sales data for a store. It includes information such as the date of sale and the corresponding sales amount.

## Data Preprocessing

- The dataset is preprocessed to convert the date column to a datetime format.
- Monthly sales data is aggregated from the daily records.
- Stationarity is achieved by differencing the sales data.
- Features are created by lagging the differenced sales data.

## Model Training

- Three machine learning algorithms are used for sales forecasting: Linear Regression, Random Forest Regressor, and XGBoost Regressor.
- The models are trained using the preprocessed data.
- Hyperparameters are tuned to optimize the performance of each model.

## Evaluation

- The performance of each model is evaluated using metrics such as RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R2 Score.
- The evaluation metrics provide insights into the accuracy and predictive power of the models.

## Results

- The sales forecasting results obtained from each model are compared and analyzed.
- Visualizations are created to illustrate the predicted sales versus the actual sales for the forecasted period.

## Usage

1. Clone the repository:

   ```bash
   git clone <repository_URL>
