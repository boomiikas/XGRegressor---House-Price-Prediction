# California Housing Price Prediction using XGBoost

This project uses the [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) to build a regression model with **XGBoost** for predicting house prices.

## 📌 Project Overview

- Dataset: California Housing (20,640 entries, 10 features)
- Goal: Predict **median house value**
- Model: **XGBoost Regressor**
- Validation: **5-Fold Cross-Validation**
- Hyperparameter Tuning: `learning_rate`, `n_estimators`, `max_depth` (+ optional extras)
- Metrics: **RMSE (Root Mean Squared Error)** and **R² Score**

## ⚙️ Preprocessing Steps

1. Handle missing values in `total_bedrooms` with **median imputation**
2. Standardize numeric features (`longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`)
3. Encode categorical feature `ocean_proximity` using **One-Hot Encoding**
4. Build pipeline with preprocessing + XGBoost model

## 🚀 Model Training

- Model: `XGBRegressor(objective="reg:squarederror")`
- Cross-validation: **5 folds**
- Hyperparameters tuned:
  - `learning_rate`: [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
  - `n_estimators`: [50, 100, 200, 400, 800]
  - `max_depth`: [3, 4, 6, 8, 10]
  - Additional: `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`

Hyperparameter search was performed using **RandomizedSearchCV**.

## 📊 Evaluation Metrics

- **RMSE**: Root Mean Squared Error (lower is better)
- **R² Score**: Coefficient of Determination (higher is better)

The model was evaluated using:
- Mean CV RMSE & R² (across folds)
- Test set RMSE & R² (on holdout data)

## 📈 Feature Importance

After training, feature importances from XGBoost were analyzed to see which features contributed most to predictions.

## 🛠️ How to Run

1. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib
   ```

2. Place the dataset (`housing.csv`) in the project folder.

3. Run the script:

   ```bash
   python xgboost_regression_housing.py
   ```

4. Output:
   - Best hyperparameters
   - Cross-validation results
   - Test set RMSE & R²
   - Feature importance plot

## 📌 Results (Example)

- Best Params: `{'model__learning_rate': 0.05, 'model__max_depth': 6, 'model__n_estimators': 400, ...}`
- CV Mean RMSE: ~45,000
- CV Mean R²: ~0.82
- Test RMSE: ~46,000
- Test R²: ~0.81

*(Exact results may vary due to randomness in CV splits)*

## 📂 Project Structure

```
├── housing.csv                   # dataset (download from Kaggle)
├── xgboost_regression_housing.py # main script with preprocessing, CV, tuning, evaluation
├── README.md                     # project documentation
```

## 📜 License

This project is for educational purposes. Dataset provided by [Kaggle Datasets](https://www.kaggle.com/datasets/camnugent/california-housing-prices).
