import pandas as pd
from pathlib import Path

DATA_DIR = Path("datasets")

# load the spam detection dataset
spam_data = pd.read_csv(DATA_DIR / "spam_ham_dataset.csv")

# loading the titanic dataset
titanic_data = pd.read_csv(DATA_DIR / "titanic.csv")

# inspecting Spam dataset

print("\n====spam dataset====")
print("shape:", spam_data.shape)
print("columns:", spam_data.columns.tolist())
print(spam_data.head())


# inspecting titanic dataset
print("\n====titanic dataset====")
print("shape:", titanic_data.shape)
print("columns:", titanic_data.columns.tolist)
print(titanic_data.head())