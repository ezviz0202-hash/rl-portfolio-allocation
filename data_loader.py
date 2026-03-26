import pandas as pd
import numpy as np


def load_returns(csv_path="market_data.csv"):
    """
    读取 market_data.csv，返回股票和债券月收益率 numpy 数组
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    required_cols = ["Stock_Return", "Bond_Return"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    if df[required_cols].isna().any().any():
        raise ValueError("收益率数据中存在缺失值，请先清洗数据。")

    stock_returns = df["Stock_Return"].to_numpy(dtype=np.float32)
    bond_returns = df["Bond_Return"].to_numpy(dtype=np.float32)

    if len(stock_returns) != len(bond_returns):
        raise ValueError("股票和债券收益率长度不一致。")

    return stock_returns, bond_returns


if __name__ == "__main__":
    stock_returns, bond_returns = load_returns("market_data.csv")

    print("✅ 数据读取成功")
    print("股票收益率长度:", len(stock_returns))
    print("债券收益率长度:", len(bond_returns))
    print("股票前5个值:", stock_returns[:5])
    print("债券前5个值:", bond_returns[:5])
    print("股票是否有 NaN:", np.isnan(stock_returns).any())
    print("债券是否有 NaN:", np.isnan(bond_returns).any())