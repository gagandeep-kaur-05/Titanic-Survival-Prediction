import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic - Logistic Regression", layout="wide")
st.title("🚢 Titanic Survival Prediction — Logistic Regression")
st.markdown("**Dataset:** Titanic | **Target:** Survived (0/1) | **Algorithm:** Logistic Regression")
st.divider()

# ── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Titanic-Dataset.csv")

df = load_data()

st.subheader("📊 Raw Dataset")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", df.shape[0])
col2.metric("Total Features", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())
col4.metric("Survival Rate", f"{df['Survived'].mean():.1%}")

# ── EDA ────────────────────────────────────────────────────────────────────────
st.subheader("🔍 Exploratory Data Analysis")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Survival count
df['Survived'].value_counts().plot(kind='bar', ax=axes[0][0], color=['#e74c3c', '#2ecc71'])
axes[0][0].set_title("Survival Count")
axes[0][0].set_xticklabels(['Died', 'Survived'], rotation=0)

# Age distribution
sns.histplot(df['Age'].dropna(), bins=30, ax=axes[0][1], color='#3498db')
axes[0][1].set_title("Age Distribution")

# Survival by class
df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=axes[0][2], color='#9b59b6')
axes[0][2].set_title("Survival Rate by Class")
axes[0][2].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)

# Survival by sex
df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=axes[1][0], color=['#e91e63', '#2196f3'])
axes[1][0].set_title("Survival Rate by Sex")
axes[1][0].set_xticklabels(['Female', 'Male'], rotation=0)

# Fare distribution
sns.histplot(df['Fare'].dropna(), bins=30, ax=axes[1][1], color='#e67e22')
axes[1][1].set_title("Fare Distribution")

# Age vs Fare scatter
survived_mask = df['Survived'] == 1
axes[1][2].scatter(df.loc[~survived_mask, 'Age'], df.loc[~survived_mask, 'Fare'],
                   alpha=0.4, c='#e74c3c', s=15, label='Died')
axes[1][2].scatter(df.loc[survived_mask, 'Age'], df.loc[survived_mask, 'Fare'],
                   alpha=0.4, c='#2ecc71', s=15, label='Survived')
axes[1][2].set_title("Age vs Fare by Survival")
axes[1][2].legend()

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Correlation heatmap
st.subheader("🔥 Correlation Heatmap")
num_df = df.select_dtypes(include=np.number).drop(columns=['PassengerId'])
fig2, ax2 = plt.subplots(figsize=(9, 6))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2, linewidths=0.5)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ── DATA CLEANING ──────────────────────────────────────────────────────────────
st.subheader("🧹 Data Cleaning")
df_clean = df.copy()
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
df_clean.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)
le = LabelEncoder()
df_clean['Sex'] = le.fit_transform(df_clean['Sex'])
df_clean['Embarked'] = le.fit_transform(df_clean['Embarked'])

col1, col2 = st.columns(2)
col1.write("**Cleaned Dataset Shape:**")
col1.write(df_clean.shape)
col2.write("**Missing values after cleaning:**")
col2.write(df_clean.isnull().sum().sum())
st.dataframe(df_clean.head(), use_container_width=True)

# ── SIDEBAR CONTROLS ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Hyperparameters")
C_val = st.sidebar.slider("C (Regularization strength)", 0.01, 10.0, 1.0, 0.01)
max_iter = st.sidebar.slider("Max Iterations", 100, 2000, 1000, 100)
solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

# ── MODEL TRAINING ─────────────────────────────────────────────────────────────
X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y)

model = LogisticRegression(C=C_val, max_iter=max_iter, solver=solver)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ── RESULTS ────────────────────────────────────────────────────────────────────
st.subheader("📈 Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
c2.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
c3.metric("Train Size", len(X_train))
c4.metric("Test Size", len(X_test))

st.text("Classification Report:")
st.code(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            ax=axes3[0], xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
axes3[0].set_title("Confusion Matrix")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes3[1].plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc_score(y_test, y_prob):.3f}')
axes3[1].plot([0, 1], [0, 1], 'k--')
axes3[1].set_title("ROC Curve")
axes3[1].set_xlabel("False Positive Rate")
axes3[1].set_ylabel("True Positive Rate")
axes3[1].legend()

# Feature coefficients
coef_df = pd.Series(model.coef_[0], index=X.columns).sort_values()
coef_df.plot(kind='barh', ax=axes3[2], color=['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df])
axes3[2].set_title("Feature Coefficients")
axes3[2].axvline(0, color='black', linestyle='--', lw=0.8)

plt.tight_layout()
st.pyplot(fig3)
plt.close()

# ── LIVE PREDICTION ────────────────────────────────────────────────────────────
st.subheader("🎯 Predict Survival for a New Passenger")
col_a, col_b, col_c = st.columns(3)
pclass = col_a.selectbox("Passenger Class", [1, 2, 3])
sex = col_b.selectbox("Sex", ["male", "female"])
age = col_c.slider("Age", 1, 80, 30)
sibsp = col_a.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = col_b.number_input("Parents/Children Aboard", 0, 6, 0)
fare = col_c.number_input("Fare Paid ($)", 0.0, 600.0, 32.0)
embarked = col_a.selectbox("Port of Embarkation", ["S", "C", "Q"])

if st.button("🔮 Predict", use_container_width=True):
    sex_enc = 1 if sex == "female" else 0
    emb_enc = {"S": 2, "C": 0, "Q": 1}[embarked]
    inp = np.array([[pclass, sex_enc, age, sibsp, parch, fare, emb_enc]])
    pred = model.predict(inp)[0]
    prob = model.predict_proba(inp)[0][1]
    if pred == 1:
        st.success(f"✅ **Survived** — Probability: {prob:.2%}")
    else:
        st.error(f"❌ **Did Not Survive** — Probability of survival: {prob:.2%}")