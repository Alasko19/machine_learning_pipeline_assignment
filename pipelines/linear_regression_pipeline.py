import pandas as pd
import joblib


from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

# file path
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "boston_housing.csv"
MODEL_PATH = BASE_DIR /"saved_models" / "linear_regression_pipeline.pkl"

# loading dataset
data = pd.read_csv(DATA_PATH)

# separating features and target
x = data.drop(columns=["MEDV"])
y = data["MEDV"]

# splitting dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42
)

# Building pipeline
linear_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# training the model
linear_pipeline.fit(x_train, y_train)

# making prediction
predictions = linear_pipeline.predict(x_test)

# evaluating the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nLINEAR REGRESSION PIPELINE RESULTS")

print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# saving pipeline
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(linear_pipeline, MODEL_PATH)

print("\nlinear regression pipeline saved successfully!")