import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# file paths

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "boston_housing.csv"
MODEL_PATH = BASE_DIR / "saved models" / "decision_tree_pipeline.pkl"

# loading dataset
data = pd.read_csv(DATA_PATH)

# features and target
x = data.drop(columns=["MEDV"])
y = data["MEDV"]

# splitting dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42
) 

# building the pipeline
decision_tree_pipeline = Pipeline([
    ("model", DecisionTreeRegressor(random_state=42))
])

# training the mpipeline
decision_tree_pipeline.fit(x_train, y_train)

# making prediction
predictions = decision_tree_pipeline.predict(x_test)

# evaluating the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nDECISION TREE PIPELINE RESULTS")

print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# saving pipeline
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(decision_tree_pipeline, MODEL_PATH)

print("\ndecision  pipeline saved successfully!")