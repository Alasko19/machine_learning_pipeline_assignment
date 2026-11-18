import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# file path
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "Titanic.csv"
MODEL_PATH = BASE_DIR / "saved_models" / "titanic_survival_pipeline.pkl"

# loading dataset
data = pd.read_csv(DATA_PATH)

# features and target
x = data.drop(columns=["survived"])
y = data["survived"]

# identifying numerical and categorical columns
numerical_features = ["age", "sibsp", "parch", "fare"]

categorical_features = [
    "sex", "embarked", "class", "who", "alone"
]

# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# combining preprocessing
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# building complete machine learning pipeline
titanic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# splitting dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# training pipeline
titanic_pipeline.fit(x_train, y_train)

# making prediction
predictions = titanic_pipeline.predict(x_test)

# evaluatuing model
print("\nTITANIC SURVIVAL PIPELINE RESULTS")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# saving pipeline
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(titanic_pipeline, MODEL_PATH)

print("\nTitanic pipeline saved successfully")
