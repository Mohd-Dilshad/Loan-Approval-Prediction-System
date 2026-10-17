import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# 1. LOAD DATASET

print("==========================================")
print("      LOAN APPROVAL PREDICTION SYSTEM")
print("==========================================")

print("\n1. Loading Dataset...")

data = pd.read_csv("dataset/loan_data.csv")

print("Dataset Loaded Successfully!")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])



# 2. CHECK DATA


print("\n2. Checking Dataset...")

print("Applicant Income:") 
print("Minimum:", data["ApplicantIncome"].min())
print("Maximum:", data["ApplicantIncome"].max())

print("\nLoan Amount:")
print("Minimum:", data["LoanAmount"].min())
print("Maximum:", data["LoanAmount"].max())

print("\nCredit History Values:")
print(data["Credit_History"].unique())


# 3. PREPROCESSING


print("\n3. Preprocessing Dataset...")


data["Gender"] = data["Gender"].map({
    "Male": 1,
    "Female": 0
})


data["Married"] = data["Married"].map({
    "Yes": 1,
    "No": 0
})


data["Education"] = data["Education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})


data["Self_Employed"] = data["Self_Employed"].map({
    "Yes": 1,
    "No": 0
})


data["Loan_Status"] = data["Loan_Status"].map({
    "Y": 1,
    "N": 0
})



# 4. PROPERTY AREA ENCODING


data = pd.get_dummies(
    data,
    columns=["Property_Area"],
    dtype=int
)


print("Preprocessing Completed!")


# 5. FEATURES AND TARGET


print("\n4. Creating Features and Target...")

X = data.drop("Loan_Status", axis=1)

y = data["Loan_Status"]


print("\nFeatures:")
print(list(X.columns))

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)



# 6. TRAIN TEST SPLIT


print("\n5. Splitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)



# 7. CREATE MODEL


print("\n6. Creating Logistic Regression Model...")

model = LogisticRegression(
    max_iter=1000
)

print("Model Created Successfully!")


# 8. TRAIN MODEL

print("\n7. Training Model...")

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")



# 9. PREDICTION


print("\n8. Making Predictions...")

y_pred = model.predict(X_test)

print("Prediction Completed!")


# 10. MODEL EVALUATION


print("\n==========================================")
print("              MODEL RESULTS")
print("==========================================")


accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(accuracy * 100, "%")


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)



# 11. CREATE MODEL FOLDER


print("\n9. Creating Model Folder...")

os.makedirs(
    "model",
    exist_ok=True
)



# 12. SAVE MODEL


print("\n10. Saving Model...")

with open(
    "model/loan_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print("\nModel saved successfully!")

print("Model Location:")
print("model/loan_model.pkl")


# 13. FINAL PROJECT RULES


print("\n==========================================")
print("       PROJECT COMPLETED SUCCESSFULLY")
print("==========================================")


print("\nFINAL LOAN RULES")

print("Applicant Income : Rs. 1,00,000 (FIXED)")
print("Maximum Loan     : Rs. 60,000")
print("Credit History   : 1 = Eligible")
print("Credit History   : 0 = Rejected")


print("\nApproval Rules:")

print("Income = Rs. 1,00,000")
print("Loan Amount = Rs. 0 to Rs. 60,000")
print("Credit History = 1")


print("\nRejection Rule:")

print("Credit History = 0 -> REJECT")
print("Loan Amount > Rs. 60,000 -> REJECT")


print("\nModel saved successfully.")