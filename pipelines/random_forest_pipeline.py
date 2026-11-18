import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# file path
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "boston_housing.csv"
MODEL_PATH = BASE_DIR / "saved_models" / "random_forest_pipeline.pkl"


# load dataset
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

# building pipeline
random_forest_pipeline = Pipeline([
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# training the pipeline
random_forest_pipeline.fit(x_train, y_train)

# making prediction
predictions = random_forest_pipeline.predict(x_test)

# evaluating the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nRANDOM FOREST PIPELINE RESULTS")
print("MSE:", round(mse, 4))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))


# saving the pipeline
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(random_forest_pipeline, MODEL_PATH)

print("\nrandom forest pipeline saved successfully!")

