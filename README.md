# Store Item Demand Forecasting

## Overview
This project was developed as part of the Store Item Demand Forecasting Challenge on Kaggle. The objective was to forecast 3 months of future sales for 50 different products across 10 stores using historical daily sales data.

The project focuses on time series forecasting, feature engineering, and machine learning techniques for demand prediction.

---

## Business Problem
Accurate sales forecasting helps businesses:
- Optimize inventory management
- Reduce stock shortages and overstock
- Improve operational planning
- Support data-driven decision making

Using 5 years of historical sales data, the model predicts future item demand at the store-item level.

---

## Dataset
Source: :contentReference[oaicite:0]{index=0}

### Files
- `train.csv` — Historical sales data
- `test.csv` — Future periods requiring prediction
- `sample_submission.csv` — Submission format

### Features
| Feature | Description |
|---|---|
| date | Date of sales |
| store | Store ID |
| item | Item ID |
| sales | Number of items sold |

---

## Objective
Predict future sales for each store-item combination over the next 3 months.

Evaluation metric:
- **SMAPE (Symmetric Mean Absolute Percentage Error)**
- Lower SMAPE indicates better forecasting performance.

---

## Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost / LightGBM

---

## Project Workflow

### 1. Data Exploration
- Checked missing values
- Analyzed sales trends
- Identified seasonality patterns
- Visualized store and item performance

### 2. Feature Engineering
Created time-series features such as:
- Year
- Month
- Day of week
- Lag features
- Rolling averages
- Moving statistics

### 3. Model Development
The model uses a hybrid forecasting architecture consisting of:

### 1. MLP Branch — Trend Modeling

The sales series is first decomposed into:
- trend component
- seasonal component

using moving-average decomposition.

The trend component is relatively smooth and slowly changing, so a lightweight Multi-Layer Perceptron (MLP) was used instead of heavier recurrent models such as LSTM.

#### Why MLP?
- Trend signals are smoother and lower-frequency
- Sequential memory is less critical for long-term movement
- MLP reduces model complexity and training cost
- Faster training with fewer parameters

The MLP branch processes flattened historical windows to forecast future trend behavior.

---

### 2. TCN Branch — Seasonality Modeling

Seasonal patterns contain repeated temporal structures and local dependencies.

A Temporal Convolutional Network (TCN) was used to model these cyclical behaviors.

#### Why TCN?
- Dilated convolutions efficiently capture temporal patterns
- Convolutional kernels detect repeated seasonal structures
- Parallel processing improves training efficiency
- Strong performance on sequence forecasting tasks

The TCN branch uses:
- causal convolutions
- residual connections
- dilation expansion

to capture seasonal dynamics across multiple temporal ranges.

### 4. Forecasting & Evaluation
Generated predictions for future sales periods and evaluated model performance using SMAPE.

---

## Results
- Metric: SMAPE: 13.42%
- Successfully built an end-to-end time series forecasting pipeline
- Applied feature engineering techniques for temporal data
- Generated sales forecasts for multiple stores and products
- Participated in a Kaggle forecasting competition environment


## Key Learnings
- Time series forecasting techniques
- Feature engineering for temporal datasets
- Model evaluation using SMAPE
- Forecasting workflow in Kaggle competitions
- Machine learning pipeline development

---

## Repository Structure

```text
├── notebooks/
├── data/
├── images/
├── README.md
└── requirements.txt