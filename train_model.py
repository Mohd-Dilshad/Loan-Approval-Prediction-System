import pandas as pd
from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


# 1. Load 
df = pd.read_csv("dataset/loan_data.csv")
print("Dataset Loaded Successfully!")
print(df.head())


# 2. Preprocess data
df = preprocess_data(df)
# Convert target column
df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0
})
print("\nAfter Preprocessing:")
print(df)


# 3. Separate input and output
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
print("\nInput Features (X):")
print(X)
print("\nTarget (y):")
print(y)


# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# 5. Train Machine Learning Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\nModel Training Completed!")


# 6. Save trained model
with open("model/loan_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\nModel Saved Successfully!")