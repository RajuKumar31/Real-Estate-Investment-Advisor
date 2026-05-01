# 🏠 Real Estate Investment Advisor

A full end-to-end data science project that predicts whether a property is a **good investment** and forecasts its **price 5 years into the future** using machine learning.

---

## 🚀 Project Overview

| Item | Detail |
|------|--------|
| Dataset | India Housing Prices (250,000 records) |
| Models | Random Forest (Classification) + XGBoost (Regression) |
| App | Streamlit interactive web app |
| Tracking | MLflow experiment tracking |

---

## 🎯 Objectives

1. **Classify** whether a property is a Good Investment (Yes/No)
2. **Predict** the estimated property price after 5 years
3. **Deploy** an interactive app for real-time property analysis

---

## 📁 Project Structure

```
Real Estate Investment Advisor/
│
├── data_cleaning.ipynb          ← Phase 1: Load, audit and clean raw data
├── eda.ipynb                    ← Phase 2: Exploratory Data Analysis (20 questions)
├── feature_engineering.ipynb   ← Phase 3: Feature creation, encoding, scaling
├── modeling.ipynb               ← Phase 4: Model training, evaluation, MLflow tracking
├── app.py                       ← Phase 5: Streamlit web application
│
├── models/
│   ├── rf_classifier.pkl        ← Trained Random Forest classification model
│   ├── xgb_regressor.pkl        ← Trained XGBoost regression model
│   └── feature_columns.pkl      ← Feature column list for prediction
│
├── india_housing_prices.csv     ← Raw dataset (250,000 records)
├── india_housing_cleaned.csv    ← After Phase 1 cleaning
├── india_housing_featured.csv   ← After Phase 3 feature engineering (unscaled)
├── india_housing_model_ready.csv← After Phase 3 scaling (model input)
│
├── requirements.txt             ← Python dependencies
└── README.md                    ← Project documentation
```

---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Web App | Streamlit |

---

## 📊 Model Performance

### Classification — Good Investment (Random Forest)
| Metric | Score |
|--------|-------|
| Accuracy | 99.73% |
| F1 Score | 99.66% |

### Regression — Future Price 5yr (XGBoost)
| Metric | Score |
|--------|-------|
| RMSE | ₹2.48 Lakhs |
| MAE | ₹1.82 Lakhs |
| R² Score | 0.9999 |

---

## 🏗️ Project Phases

| Phase | Notebook | Description |
|-------|----------|-------------|
| 1 | `data_cleaning.ipynb` | Load raw data, fix outliers, logic errors, save cleaned file |
| 2 | `eda.ipynb` | Answer 20 business questions with charts and insights |
| 3 | `feature_engineering.ipynb` | Build targets, engineer features, encode and scale |
| 4 | `modeling.ipynb` | Train, evaluate and log both models with MLflow |
| 5 | `app.py` | Streamlit app for real-time investment analysis |

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/RajuKumar31/real-estate-investment-advisor.git
cd real-estate-investment-advisor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

**4. View MLflow experiments**
```bash
mlflow ui
```

---

## 💡 Key Insights from EDA

- All features are uniformly distributed — confirming synthetic dataset generation
- `Price_per_SqFt` is the strongest numeric predictor (correlation: 0.64 with price)
- `Year_Built` and `Age_of_Property` are perfectly correlated (-1.0) — one dropped before modeling
- Location features (State, City) show minimal price variation — feature combinations matter more
- Property Type, Facing Direction and Furnished Status have near-zero price impact

---

## 🤖 Investment Logic

A property is classified as **Good Investment** if it meets:
- Price ≤ 75th percentile of its city
- Ready to Move **OR** BHK ≥ 3
- Amenity Count ≥ 2
- Nearby Schools ≥ 5 **OR** Nearby Hospitals ≥ 5

---

## 👤 Author

**Raju Kumar S**
- LinkedIn: [linkedin.com/in/rajukumarsahani](https://linkedin.com/in/rajukumarsahani/)
- GitHub: [github.com/RajuKumar31](https://github.com/RajuKumar31)