# 🛒 Big Mart Sales Intelligence — Streamlit Dashboard

## Setup & Run

### 1. Folder structure
Place these files in the **same folder**:
```
📁 bigmart_dashboard/
├── app.py
├── requirements.txt
└── clean_1_train.csv        ← your dataset file
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```
Your browser will open at **http://localhost:8501**

---

## Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPI cards, outlet sales charts, model summary |
| 🔍 EDA | Data overview, distributions, correlation heatmap |
| 🤖 Models | R² / RMSE comparison, actual vs predicted, feature importance |
| 🎯 Predict | Enter item & outlet details → get predicted sales |

## Models Included
- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor
- XGBoost Regressor *(requires xgboost package)*
