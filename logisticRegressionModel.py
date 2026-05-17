import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# LOAD DATA
df = pd.read_csv("Titanic-Dataset.csv")

# CLEANING
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Sex'] = df['Sex'].map({
    'male': 0,
    'female': 1
})

df['Embarked'] = df['Embarked'].map({
    'C': 0,
    'Q': 1,
    'S': 2
})

df.drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Cabin'],
    inplace=True
)

# FEATURES
X = df.drop('Survived', axis=1)
y = df['Survived']

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# SAVE MODEL
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")