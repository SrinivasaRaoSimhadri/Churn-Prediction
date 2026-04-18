# Customer Churn Prediction

## Overview

This project predicts whether a customer is likely to churn (leave the service) using machine learning techniques.
It uses customer data such as credit score, age, balance, and activity status to make predictions.

---

## Features

* Data preprocessing (encoding + scaling)
* Exploratory Data Analysis (EDA) using Seaborn
* Multiple ML models tested
* Final model: **Gradient Boosting Classifier**
* Achieved **~87% accuracy**
* Interactive web app using Streamlit

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Seaborn, Matplotlib
* Streamlit
* Joblib

---

## Dataset

The dataset contains customer details like:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary
* Exited (Target Variable)

---

## Model Training

* Data cleaned and preprocessed
* Categorical features encoded
* Numerical features scaled using StandardScaler
* Model trained using GradientBoostingClassifier

---

## Model Performance

* Accuracy: **~87%**
* Evaluated using:

  * Confusion Matrix
