
import pandas as pd

def preprocess_data(df):

    df["Gender"] = df["Gender"].map({
        "Male": 1,
        "Female": 0
    })

    df["Married"] = df["Married"].map({
        "Yes": 1,
        "No": 0
    })

    df["Education"] = df["Education"].map({
        "Graduate": 1,
        "Not Graduate": 0
    })

    df["Self_Employed"] = df["Self_Employed"].map({
        "Yes": 1,
        "No": 0
    })

    df = pd.get_dummies(
        df,
        columns=["Property_Area"],
        dtype=int
    )

    return df