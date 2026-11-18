import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# file path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "datasets" / "spam_ham_dataset.csv"
MODEL_PATH = BASE_DIR / "saved_models" / "spam_detection_pipeline.pkl"

# loading dataset
data = pd.read_csv(DATA_PATH)

# removing rows with missing text or labels
data = data.dropna(subset=["text", "label_num"])

# features and target
x = data["text"]
y = data["label_num"]

# split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# building machine learning pipeline
spam_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# training the pipeline
spam_pipeline.fit(x_train, y_train)

# making prediction
predictions = spam_pipeline.predict(x_test)

# evaluating the model
print("SPAM DETECTION PIPELINE RESULTS")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# saving complete pipeline
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(spam_pipeline, MODEL_PATH)

print("\nPipeline saved successfully!")



