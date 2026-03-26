import numpy as np
import pandas as pd


def load_returns(csv_path="market_data.csv"):
    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    required_columns = ["Stock_Return", "Bond_Return"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    returns = data[required_columns]

    if returns.isna().any().any():
        raise ValueError("Return series contains missing values.")

    stock_returns = returns["Stock_Return"].to_numpy(dtype=np.float32)
    bond_returns = returns["Bond_Return"].to_numpy(dtype=np.float32)

    if len(stock_returns) != len(bond_returns):
        raise ValueError("Stock and bond return series must have the same length.")

    return stock_returns, bond_returns