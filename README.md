#Time Series Sales Forecasting
  
A Python-based project to forecast monthly sales (2011–2014) using AR, ARIMA, and SARIMA models, enabling optimized inventory management and strategic business planning.

#Project Overview

This project allows users to:

Load and clean historical sales data
Aggregate daily sales into monthly sales
Analyze trends, seasonality, and residual patterns
Apply Box-Cox transformation and differencing to stabilize the data
Build AR, ARIMA, and SARIMA models for forecasting
Evaluate model performance using RMSE and MAPE
Visualize predictions and select the best-performing model
It demonstrates practical usage of Python, time series analysis, statsmodels, and data visualization.

#Problem Statement

Companies often face challenges in inventory management due to inaccurate sales forecasts. Overstocking or understocking leads to financial loss or missed sales opportunities.

This project provides a structured approach to forecast future sales, reducing inventory risks and supporting strategic decision-making.

#Objectives

Clean and preprocess historical sales data
Aggregate daily sales into monthly sales
Conduct stationarity checks and apply transformations
Decompose the time series into trend, seasonality, and residuals
Build and tune AR, ARIMA, and SARIMA models
Evaluate models using RMSE and MAPE
Visualize predictions and select the best model

#Technologies Used

Python – Core programming
Pandas & NumPy – Data handling
Matplotlib & Seaborn – Visualization
Statsmodels & Scipy – Time series modeling
Sklearn – Evaluation metrics

#Features

Time series cleaning and preprocessing
Box-Cox transformation and differencing
Stationarity testing (ADF Test)
Seasonal decomposition (trend, seasonality, residuals)
Autocorrelation (ACF) and Partial Autocorrelation (PACF) analysis
AR, ARIMA, and SARIMA forecasting
RMSE & MAPE model evaluation
Train/Test split visualization with predictions

#Graphs

1. Original vs. Box-Cox & Differenced Data
 Shows variance and mean stabilization after transformation.

# Box-Cox transformation
from scipy.stats import boxcox
df_boxcox = pd.Series(boxcox(df_train['Sales'], lmbda = 0), index = df_train.index)
df_boxcox_diff = df_boxcox.diff()

# Plot original vs transformed data
plt.figure(figsize = (14, 6))

plt.subplot(1, 2, 1)
sns.lineplot(data = df_train, x = 'Order Date', y = 'Sales', marker = 'o', color = 'blue')
plt.xticks(rotation = 90)
plt.title('Original Data')

plt.subplot(1, 2, 2)
sns.lineplot(x = df_boxcox_diff.index, y = df_boxcox_diff.values, marker = 'o', color = 'blue')
plt.xticks(rotation = 90)
plt.title('Transformed Data [Box-Cox + Differencing]')

plt.suptitle('Sales Data');
plt.savefig('images/boxcox_diff_plot.png', bbox_inches='tight')  # Save plot for README
plt.show()

2. Seasonal Decomposition
Shows trend, seasonality, and residuals.

from statsmodels.tsa.seasonal import seasonal_decompose

# Seasonal decomposition of original training data
result = seasonal_decompose(df_train['Sales'])
result.plot()
plt.suptitle('Seasonal Decomposition of Sales')
plt.savefig('images/seasonal_decompose_plot.png', bbox_inches='tight')  # Save plot for README
plt.show()

3. Train-Test Split with Predictions (AR, ARIMA, SARIMA)
Compares actual sales and model predictions.

# Using SARIMA predictions as example
plt.figure(figsize = (14, 6))
sns.lineplot(data = df_train, x = df_train.index, y = df_train['Sales'], marker = 'o', color = 'blue', label = 'Train')
sns.lineplot(data = df_test, x = df_test.index, y = df_test['Sales'], marker = 'o', color = 'green', label = 'Test')
sns.lineplot(x = df_preds.index[train_len:], y = df_preds.values[train_len:], marker = 'o', color = 'purple', label = 'Predictions')
plt.title('Sales Forecast: Train, Test, Predictions')
plt.savefig('images/forecast_plot.png', bbox_inches='tight')  # Save plot for README
plt.show()

| Model  | RMSE      | MAPE (%) |
| ------ | --------- | -------- |
| AR     | 14,915.16 | 24.42    |
| ARIMA  | 29,842.05 | 36.74    |
| SARIMA | 11,661.76 | 16.82    |


Observation: SARIMA is the most accurate model, suitable for monthly sales forecasting with seasonality.

Installation / Setup

Clone the repository:

git clone [(https://github.com/khushboo-datasci/Time_Series_Sales_Forecasting/edit/main/README.md)]


Navigate to the project folder:

cd TimeSeriesSalesForecasting


Install required libraries:

pip install pandas numpy matplotlib seaborn statsmodels scipy scikit-learn


Run the notebook or Python script in Google Colab.

Step-by-Step Flow

1.Load and inspect the sales data
2.Clean data, convert Order Date to datetime, sort, aggregate monthly
3.Visualize data and perform exploratory analysis
4.Check stationarity (ADF test), apply Box-Cox and differencing
5.Decompose time series (trend, seasonality, residuals)
6.Plot ACF & PACF to select model parameters
7.Build AR, ARIMA, and SARIMA models
8.Evaluate models using RMSE and MAPE
9.Visualize train, test, and forecast predictions
10.Select best-performing model (SARIMA) and draw insights


Author

Khushboo Kumari
GitHub Profile
