# house-price-prediction-fdm
FDM Mini Project - House Price Prediction

# Data Preprocessing Documentation

## Files Created:
- `house_prices_processed.csv` - Cleaned dataset ready for modeling
- `preprocessor.pkl` - Preprocessing pipeline for consistent transformations

## Preprocessed Data Columns:
- **Target**: `Price (in rupees)`
- **Features**: location, Carpet Area, Status, Transaction, Furnishing, etc.

## Usage for Model Training:
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed data
df = pd.read_csv('house_prices_processed.csv')

# Split features and target
X = df.drop('Price (in rupees)', axis=1)
y = df['Price (in rupees)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)