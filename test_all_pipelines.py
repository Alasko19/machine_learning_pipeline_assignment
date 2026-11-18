import pandas as pd
import joblib

from pathlib import Path

# project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
DATA_DIR = BASE_DIR / "datasets"

# loading datasets
spam_data = pd.read_csv(DATA_DIR / "spam_ham_dataset.csv")
titanic_data = pd.read_csv(DATA_DIR / "Titanic.csv")
boston_data = pd.read_csv(DATA_DIR / "boston_housing.csv")

# loading all five pipelines
spam_model = joblib.load(MODEL_DIR / "spam_detection_pipeline.pkl")
titanic_model = joblib.load(MODEL_DIR / "titanic_survival_pipeline.pkl")
linear_model = joblib.load(MODEL_DIR / "linear_regression_pipeline.pkl")
tree_model = joblib.load(MODEL_DIR / "decision_tree-pipeline.pkl")
forest_model = joblib.load(MODEL_DIR / "random_forest_pipeline.pkl")


print("\nALL FIVE PIPELINES LOADED SUCCESSFULLY!")

# spam prediction
spam_sample = [spam_data["text"].iloc[0]]
print("\nSpam prediction:", spam_model.predict(spam_sample))

# titanic prediction
titanic_sample = titanic_data.drop(columns=["survived"]).iloc[[0]]
print("Titanic Survival prediction:", titanic_model.predict(titanic_sample))

# boston housing prediction
boston_sample = boston_data.drop(columns=["MEDV"]).iloc[[0]]

print("Linear Regression prediction:", linear_model.predict(boston_sample)[0])
print("Decision Tree prediction:", tree_model.predict(boston_sample)[0])
print("Random Forest prediction:", forest_model.predict(boston_sample)[0])

print("\nAll pipeline tests completed successfully!")