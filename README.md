Machine Learning Pipelines Assignment

Project Overview

This project demonstrates the implementation of machine learning pipelines for classification and regression tasks using Python and Scikit-learn.

The objective is to combine data preprocessing and machine learning algorithms into reusable pipelines, train the models, evaluate their performance, and save the trained pipelines for future predictions.

Machine Learning Models

Five machine learning pipelines were developed:

1. Email Spam Detection

* TF-IDF Vectorizer
* Multinomial Naive Bayes
* Task: Classify emails as spam or ham.

2. Titanic Survival Prediction

* SimpleImputer
* StandardScaler
* OneHotEncoder
* Logistic Regression
* Task: Predict passenger survival.

3. Boston Housing Price Prediction — Linear Regression

* StandardScaler
* Linear Regression
* Task: Predict house prices.

4. Boston Housing Price Prediction — Decision Tree

* Decision Tree Regressor
* Task: Predict house prices.

5. Boston Housing Price Prediction — Random Forest

* Random Forest Regressor
* Task: Predict house prices.

Project Structure

machine_learning_pipeline_assignment/
│
├── datasets/
├── pipelines/
├── saved_models/
├── main.py
├── test_all_pipelines.py
├── README.md
├── requirements.txt
└── .gitignore

Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

Installation

Install the required libraries using:

pip install -r requirements.txt

Running the Pipelines

Run each pipeline from the project root:

python pipelines/spam_pipeline.py
python pipelines/titanic_pipeline.py
python pipelines/linear_regression_pipeline.py
python pipelines/decision_tree_pipeline.py
python pipelines/random_forest_pipeline.py

Testing Saved Pipelines

Run:

python test_all_pipelines.py

The test script loads all five saved pipelines and verifies that they can generate predictions.

Saving Models

All trained pipelines are saved in the saved_models/ directory using Joblib.

Conclusion

This assignment demonstrates how Scikit-learn pipelines simplify machine learning workflows by combining preprocessing and model training into reusable objects.

All five pipelines were successfully trained, saved, loaded, and tested.