Time Series Sales Forecasting

Overview

Forecasted monthly sales (2011–2014) for a company to optimize inventory management and support strategic business planning.
Accurate forecasting helps reduce overstock/understock risks and enables informed operational decisions.

Problem Statement

Predict future monthly sales based on historical sales data to:

Optimize inventory levels

Support strategic business planning

Minimize operational risks related to stock management

Dataset

Source: Superstore Sales Dataset

Features Used:

Order Date – date of order

Sales – target variable

Data Processing: Aggregated daily sales to monthly sales

Link: [Superstore_Data.csv](/content/drive/MyDrive/Superstore_Data (1).csv)

Folder Structure
Time-Series-Sales-Forecasting/
│
├── data/
│   └── Superstore_Data.csv
├── notebooks/
│   └── sales_forecasting.ipynb
├── README.md
└── requirements.txt

Tools & Libraries

Programming Language: Python

Data Manipulation: pandas, numpy

Visualization: matplotlib, seaborn

Time Series Modeling: statsmodels, scipy

Evaluation Metrics: sklearn.metrics (RMSE, MAPE)

Methodology
1. Data Preparation

Converted Order Date to datetime and set as index

Aggregated daily sales to monthly sales

Visualized monthly sales trends

Sample plot:


2. Stationarity Analysis

Performed Augmented Dickey-Fuller (ADF) test

Applied Box-Cox transformation → stabilizes variance

Applied Differencing → stabilizes mean

ADF Test Plot:


3. Time Series Decomposition

Decomposed data into Trend, Seasonality, Residuals

Visualized before and after transformation

Decomposition Plots:




4. Autocorrelation Analysis

Generated ACF and PACF plots to identify lag orders for AR/ARIMA/SARIMA models

ACF & PACF Plots:




5. Model Building

Built and evaluated the following models:

Model	Description
AR	Autoregressive Model
ARIMA	Autoregressive Integrated Moving Average
SARIMA	Seasonal ARIMA to capture seasonality

Applied inverse transformations (Box-Cox + Differencing) to get predictions in original scale

Model Evaluation
Model	RMSE	MAPE (%)
AR	14915.16	24.42
ARIMA	29842.05	36.74
SARIMA	11661.76	16.82

✅ SARIMA model outperformed AR and ARIMA models, effectively capturing seasonal patterns and providing the most accurate forecasts.

Results & Visualizations

Forecasted vs Actual Sales:


Transformed vs Original Data:


Seasonal Decomposition (Trend, Seasonality, Residuals):


Quick Start

Clone the repository

git clone https://github.com/yourusername/Time-Series-Sales-Forecasting.git


Install dependencies

pip install -r requirements.txt


Run Jupyter Notebook

jupyter notebook notebooks/sales_forecasting.ipynb


Visualize results and analyze forecasts

Conclusion

SARIMA model best captured trend and seasonality

Provides actionable insights for inventory management and strategic planning

Enables optimized monthly sales forecasting for business decision-making
