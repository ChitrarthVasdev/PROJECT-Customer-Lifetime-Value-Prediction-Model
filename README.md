# PROJECT - Customer Lifetime Value (CLV) Prediction Model

This project builds a machine learning model to predict **Customer Lifetime Value (CLV)** using the Online Retail dataset.
It includes full data preprocessing, feature engineering, model training, prediction, customer segmentation, and visualizations.

---

## 📁 Dataset

- **Source**: UCI Machine Learning Repository  
- **File Used**: `Online Retail.xlsx`  
- **Description**: Historical transaction data from a UK-based online retailer, including InvoiceNo, Quantity, UnitPrice, InvoiceDate, CustomerID, etc.

---

## 🎯 Objective

To predict the **Customer Lifetime Value (LTV)** for each customer based on their transaction history and segment customers into 4 groups:
- Low
- Mid-Low
- Mid-High
- High

---

## 🛠️ Tools & Libraries Used

- **Python**  
  - `pandas`, `numpy` – Data manipulation  
  - `matplotlib`, `seaborn` – Visualization  
  - `scikit-learn` – Machine learning (RandomForestRegressor)  
  - `joblib` – Model saving  
  - `gzip`, `zipfile` – Compression  
- Jupyter Notebook / Google Colab (Recommended)

---

## 🧩 Steps Followed

### 1. Data Preprocessing
- Removed canceled orders (InvoiceNo starts with 'C')
- Dropped missing `CustomerID` values
- Created new column: `TotalAmount = Quantity × UnitPrice`

### 2. Feature Engineering
- `Recency`: Days since customer's last purchase
- `Frequency`: Number of unique invoices
- `Monetary`: Total amount spent by the customer
- `AOV`: Average Order Value (`Monetary / Frequency`)
- Target Variable: LTV (approximated using Monetary)

### 3. Model Training
- Features: `Recency`, `Frequency`, `AOV`
- Algorithm: `RandomForestRegressor`
- Evaluation Metrics:  
  - **MAE** (Mean Absolute Error)  
  - **RMSE** (Root Mean Squared Error)

### 4. Prediction & Segmentation
- Predicted LTV for each customer using trained model
- Segmented customers into 4 tiers using `pd.qcut`:
  - `Low`
  - `Mid-Low`
  - `Mid-High`
  - `High`

### 5. Visualizations
- 📊 **LTV Distribution** (`LTV_Prediction_Distribution.png`)
- 📈 **Segment Count** (`Customer_Segment_Counts.png`)

---

## 📦 Project Deliverables

| File | Description |
|------|-------------|
| ✅ `Updated_Online_Retail_Compressed.zip` | Cleaned dataset with engineered features as final csv |
| ✅ `Predicted_LTV_Segments.csv` | Final  LTV predictions with customer segment in CSV |
| ✅ `Online Retail.xlsx` | original Dataset |
| ✅ `LTV_model_compressed.pkl.gz` | Trained RandomForestRegressor model Gzip-compressed model file |
| ✅ `LTV_Prediction_Distribution.png` | Histogram of predicted LTV values |
| ✅ `Customer_Segment_Counts.png` | Bar chart of customers by segment |
| ✅ `Updated_table.csv` |  Dataset after feature engineering |
| ✅ `Online Retail.xlsx` | original Dataset |

---

## 📌 How to Run This Project

1. Clone/download the dataset and scripts.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Load `Updated_Online_Retail.csv` into a DataFrame.
4. Load the model:

```python
import joblib
model = joblib.load("LTV_model.pkl")

# Predict LTV for new customer data using:
    new_data= [add new values]
    predicted_ltv = model.predict(new_data[['Recency', 'Frequency', 'AOV']])

# Segment customers using:

import pandas as pd
df['Segment'] = pd.qcut(df['Predicted_LTV'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
```


# CREATED BY :
# CHITRARTH VASDEV
