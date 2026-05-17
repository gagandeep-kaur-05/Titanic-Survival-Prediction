import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)

st.title("🚢 Titanic Survival Prediction")
st.markdown("### Logistic Regression Classification Project")

# LOAD DATA
df = pd.read_csv("titanic.csv")

# LOAD MODEL
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# DATA PREVIEW
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# BASIC INFO
col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Survival Rate", f"{df['Survived'].mean()*100:.2f}%")

# EDA
st.subheader("📈 Exploratory Data Analysis")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.countplot(x='Survived', data=df, ax=axes[0])
axes[0].set_title("Survival Count")

sns.histplot(df['Age'].dropna(), bins=30, ax=axes[1])
axes[1].set_title("Age Distribution")

st.pyplot(fig)

# DATA CLEANING
df_clean = df.copy()

df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])

df_clean['Sex'] = df_clean['Sex'].map({
    'male': 0,
    'female': 1
})

df_clean['Embarked'] = df_clean['Embarked'].map({
    'C': 0,
    'Q': 1,
    'S': 2
})

df_clean.drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Cabin'],
    inplace=True
)

# FEATURES
X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived']

# PREDICTIONS
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

# METRICS
st.subheader("📊 Model Performance")

c1, c2 = st.columns(2)

c1.metric(
    "Accuracy",
    f"{accuracy_score(y, pred):.4f}"
)

c2.metric(
    "ROC AUC",
    f"{roc_auc_score(y, prob):.4f}"
)

# CONFUSION MATRIX
fig2, ax2 = plt.subplots(figsize=(5, 4))

sns.heatmap(
    confusion_matrix(y, pred),
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax2
)

ax2.set_title("Confusion Matrix")

st.pyplot(fig2)

# LIVE PREDICTION
st.subheader("🎯 Predict Survival")

col1, col2, col3 = st.columns(3)

pclass = col1.selectbox("Passenger Class", [1, 2, 3])

sex = col2.selectbox(
    "Sex",
    ["male", "female"]
)

age = col3.slider(
    "Age",
    1,
    80,
    30
)

sibsp = col1.number_input(
    "Siblings/Spouses",
    0,
    8,
    0
)

parch = col2.number_input(
    "Parents/Children",
    0,
    6,
    0
)

fare = col3.number_input(
    "Fare",
    0.0,
    600.0,
    32.0
)

embarked = col1.selectbox(
    "Embarked",
    ["C", "Q", "S"]
)

if st.button("Predict"):

    sex = 1 if sex == "female" else 0

    embarked = {
        "C": 0,
        "Q": 1,
        "S": 2
    }[embarked]

    input_data = np.array([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(
            f"Passenger Survived ✅\n\nProbability: {probability:.2%}"
        )
    else:
        st.error(
            f"Passenger Did Not Survive ❌\n\nProbability: {probability:.2%}"
        )