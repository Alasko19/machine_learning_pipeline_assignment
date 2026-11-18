import pandas as pd
import joblib

from pathlib import Path
from sklearn.datasets import fetch_openml

# projects path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "datasets" / "boston_housing.csv"

# downloading boston housing dataset
boston = fetch_openml(
    name="boston",
    version=1,
    as_frame=True,
    parser="liac-arff"
)

# separating feature and target
x = boston.data
y = pd.to_numeric(boston.target)

# combining features and target
data = x.copy()
data["MEDV"] = y

# saving dataset locally
data.to_csv(DATA_PATH, index=False)

# displaying dataset information
print("BOSTON HOUSING DATASET")
print("Dataset shape:", data.shape)

print("\nFirst Five Rows:")
print(data.head())

print("\nDataset saved successfully!")