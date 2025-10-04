# Housing Price Prediction

Machine learning project to predict housing prices using property features, location, and market characteristics.

## Dataset
- **Size**: 269,746 properties with 56 features
- **Target**: Housing prices ($9,000 - $11,000,000)
- **Files**: `housing_data.csv`, `eda_housing_data.csv`, `final_cleaned_housing_data.csv`

## Project Structure
- `explore.ipynb` - Initial data exploration and understanding
- `clean.ipynb` - Data cleaning and preprocessing
- `prepare.ipynb` - Feature engineering and preparation
- `EDA.ipynb` - Exploratory data analysis and visualizations
- `modeling.ipynb` - Model training, evaluation, and final XGBoost model

## Best Model
**XGBoost Regressor**
- Test RMSE: $324,101
- Test R²: 0.8095
- Test MAPE: 17.22%

## Model Files
- `xgb_final_model.pkl` - Trained XGBoost model
- `scaler.pkl` - StandardScaler for preprocessing
- `model_info.pkl` - Model metadata and parameters

## Usage
Load the saved model to make predictions on new housing data:

```python
import pickle
import numpy as np

# Load model and scaler
with open('xgb_final_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions (remember to log-transform predictions)
predictions_log = model.predict(scaler.transform(X_new))
predictions = np.exp(predictions_log)
```