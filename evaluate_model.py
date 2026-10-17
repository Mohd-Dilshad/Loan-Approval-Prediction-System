import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing import preprocess_data


# 1. Load dataset
df = pd.read_csv("dataset/loan_data.csv")


# 2. Preprocess data
df = preprocess_data(df)


# 3. Convert target column
df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0
})


# 4. Separate input and output
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]


# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Load trained model
with open("model/loan_model.pkl", "rb") as file:
    model = pickle.load(file)


# 7. Make predictions
y_pred = model.predict(X_test)


# 8. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)


# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# 10. Classification Report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)