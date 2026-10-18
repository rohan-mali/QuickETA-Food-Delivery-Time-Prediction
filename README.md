# Food Delivery Time Prediction

Predict how many **minutes** a food delivery will take, given the order details —
who is delivering it, how far it has to travel, the weather, traffic, time of day,
and more. A supervised **regression** project, end-to-end: raw messy data →
cleaning → feature engineering → model comparison → tuned XGBoost → a live
**Streamlit app** that predicts delivery time for a new order.

## Live app

```bash
pip install -r requirements.txt
python train_model.py      # only needed once — creates the model files
streamlit run app.py
```

In the app you can enter the restaurant and delivery coordinates — the distance
is calculated automatically with the **Haversine formula** — or type the
distance directly, choose weather / traffic / vehicle / order type, and get
the predicted delivery time in minutes.

## Results (measured by running `train_model.py` on this repo)

Dataset: 45,593 raw orders → **45,162 rows** after cleaning and removing
corrupt coordinates (distance ≥ 100 km). 21 features after encoding.
80/20 train-test split, `random_state=42`, scaler fitted on train only.

| Model | Test R² | Test RMSE (min) | Train R² |
|---|---|---|---|
| Linear Regression | 0.57 | 6.21 | 0.58 |
| Decision Tree | 0.66 | 5.49 | 1.00 (overfits) |
| Random Forest | 0.82 | 4.03 | 0.97 |
| **XGBoost (tuned)** | **0.83** | **3.91** | 0.85 |

RMSE is in minutes, so the final model is off by roughly **4 minutes** on average.
Full numbers: `metrics.json`.

## Approach

1. **Load & inspect** the raw data (45,593 orders)
2. **Clean** messy text and missing values (`"conditions Sunny"` → `Sunny`, `"(min) 24"` → `24`, ratings clipped to 5, median/mode imputation)
3. **Feature engineering**
   - `distance_km` from restaurant/customer coordinates via the **Haversine formula**
   - `order_hour` extracted from the order timestamp
4. **Encode** categoricals (ordinal for traffic density — Low < Medium < High < Jam; one-hot for the rest) and **scale** features (scaler fit on train only)
5. **Train & compare** Linear Regression, Decision Tree, Random Forest, XGBoost
6. **Tune** XGBoost (hyperparameters found with RandomizedSearchCV in the notebook: 500 trees, max_depth 8, learning_rate 0.01, subsample 0.9) and **save** model + scaler with joblib

Sanity checks done on the app: the same order predicts ~18 min in Low traffic
vs ~30 min in Jam traffic — the model has learned the traffic effect, not just
memorised numbers.

## Files

```
food-delivery-time-prediction/
├── app.py                                # Streamlit app
├── train_model.py                        # notebook pipeline as a script; writes the files below
├── Food_Delivery_Time_Prediction.ipynb   # full documented notebook (EDA + tuning search)
├── data/Food_delivery.csv                # raw dataset
├── xgb_best.pkl                          # saved tuned XGBoost model (generated)
├── standard_scaler.pkl                   # saved scaler (generated)
├── feature_columns.json                  # exact feature order the model expects (generated)
├── categories.json                       # dropdown values for the app (generated)
├── metrics.json                          # model comparison from the training run (generated)
└── requirements.txt
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · joblib · Streamlit

## Deploy on Streamlit Cloud

1. Push this folder to a GitHub repo (the `.pkl` files are needed — they are
   already generated; if they are missing, run `python train_model.py` first).
2. On share.streamlit.io → New app → select the repo → main file `app.py` → Deploy.

## bullets

- Built an end-to-end regression model to predict food delivery time in minutes using rider details, weather, traffic, vehicle type, festival, city and location data
- Cleaned messy order data from 45,593 orders and engineered new features — delivery distance using the Haversine formula from restaurant and customer coordinates, and order hour from the order time
- Compared Linear Regression, Decision Tree, Random Forest and XGBoost after encoding categorical features and scaling the data, with tuned XGBoost giving the best result (R² 0.83, RMSE 3.91 minutes)
- Deployed the tuned XGBoost model as a Streamlit app where users enter order details or coordinates and get the predicted delivery time instantly
