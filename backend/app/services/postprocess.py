# backend/app/models/postprocess.py
import pandas as pd

def format_prod_dataframe(transactions, predictions):
    """
    transactions: list[str]
    predictions: list[str] (predicted categories)
    returns DataFrame with 2 columns: transaction, predicted_cat
    """
    df = pd.DataFrame({"transaction": transactions, "predicted_cat": predictions})
    return df[["transaction", "predicted_cat"]]

def format_dev_dataframe(transactions, actuals, predictions):
    """
    transactions: list[str]
    actuals: list[str]
    predictions: list[str]
    returns DataFrame with 3 columns: transaction, actual_cat, predicted_cat
    """
    df = pd.DataFrame({
        "transaction": transactions,
        "actual_cat": actuals,
        "predicted_cat": predictions
    })
    return df[["transaction", "actual_cat", "predicted_cat"]]
