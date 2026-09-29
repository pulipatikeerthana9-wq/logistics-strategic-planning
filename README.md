# Logistics Strategic Planning and Data Exploration

## Project Overview

This project focuses on strategic planning and data exploration for a logistics and supply chain scenario. The analysis uses the DataCo Supply Chain dataset to study shipment delays, delivery performance, customer/order patterns, and potential logistics optimization opportunities.

The project combines exploratory data analysis, KPI analysis, machine learning, clustering, and an illustrative routing optimization example using Python.

## Objectives

- Analyze logistics and shipment performance.
- Identify important logistics KPIs.
- Explore factors associated with shipment delays.
- Build a classification model to predict late shipments.
- Build a regression model to estimate actual shipping duration.
- Identify customer/order segments using clustering.
- Demonstrate capacity-constrained vehicle routing.
- Translate analytical findings into operational decisions.

## Dataset

The project uses the **DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS** dataset.

The dataset contains supply-chain information related to:

- Orders and order items
- Shipping modes
- Product categories
- Customer segments
- Markets and regions
- Sales and discounts
- Scheduled and actual shipping duration
- Delivery status

Source:

https://data.mendeley.com/datasets/8gx2fvg2k6/5

## Key Performance Indicators

The analysis focuses on the following KPIs:

1. **Late Shipment Rate**  
   Percentage of shipments that exceed the scheduled shipping duration.

2. **Average Actual Shipping Duration**  
   Average number of days taken for shipment.

3. **Average Scheduled Shipping Duration**  
   Average expected shipment duration.

4. **Average Delay**  
   Difference between actual and scheduled shipping duration.

5. **Profit per Order**  
   Used as an additional financial performance indicator.

## Methodology

### 1. Data Loading and Inspection

The dataset is loaded using Pandas and checked for:

- Dataset dimensions
- Missing values
- Data types
- Distinct orders
- Date fields
- Delivery status

### 2. KPI Analysis

Shipment performance is analyzed across different shipping modes and delivery populations.

### 3. Classification

A Random Forest classifier is used to predict whether a non-cancelled shipment will be late.

The model is evaluated using:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC

Orders are grouped during train-test splitting to reduce information leakage between training and testing data.

### 4. Regression

A Random Forest regression model predicts:

`Days for shipping (real)`

The model is compared against a simple scheduled-days baseline using:

- MAE
- RMSE

### 5. Clustering

K-Means clustering is used to identify different order/purchasing profiles based on:

- Order item quantity
- Sales
- Product price
- Discount

Silhouette scores are used to compare different numbers of clusters.

### 6. Routing Optimization

An illustrative capacity-constrained vehicle routing example is implemented using OR-Tools.

The routing example uses sample distances and demands and is **not a routing result generated from the DataCo dataset**.

## Project Structure

```text
Logistics_Project/
│
├── DataCoSupplyChainDataset.csv
├── logistics_analysis.py
├── README.md
└── .gitignore