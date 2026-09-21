"""
Convert a form's raw dict into the exact 47-column DataFrame
the model was trained on, then return a prediction.
"""
import pandas as pd
from .loader import get_model, get_feature_names


def build_feature_row(form_data: dict) -> pd.DataFrame:
    """
    Take the raw form dict (mixed numeric + categorical) and produce
    a one-row DataFrame with exactly the same columns as X_train.
    """
    # Base row: numeric features kept as-is
    row = {k: v for k, v in form_data.items()}

    # Engineered features must be computed the same way as in the notebook
    row["IncomePerYearWorked"] = (
        float(row["MonthlyIncome"]) / (float(row["TotalWorkingYears"]) + 1)
    )
    row["TenureRatio"] = (
        float(row["YearsAtCompany"]) / (float(row["TotalWorkingYears"]) + 1)
    )
    row["ManagerStability"] = (
        float(row["YearsWithCurrManager"]) / (float(row["YearsAtCompany"]) + 1)
    )

    df = pd.DataFrame([row])

    # One-hot encode categoricals with the exact same column names as training
    cat_cols = ["BusinessTravel", "Department", "EducationField",
                "Gender", "JobRole", "MaritalStatus", "OverTime"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Align columns to model's training features (fill missing dummies with 0)
    expected = get_feature_names()
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    df = df[expected]   # reorder + drop any extras

    return df


def predict_one(form_data: dict) -> dict:
    model = get_model()
    X = build_feature_row(form_data)

    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0, 1])

    return {
        "prediction": pred,
        "label": "Likely to Leave" if pred == 1 else "Likely to Stay",
        "probability": round(proba * 100, 2),
    }