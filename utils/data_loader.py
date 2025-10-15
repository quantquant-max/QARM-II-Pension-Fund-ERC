import pandas as pd

def load_custom_data():
    try:
        df = pd.read_csv("Stock_Returns_With_Names_post2000.csv")
        df.set_index('COMNAM', inplace=True)
        df.columns = pd.to_datetime(df.columns.str.split(' ').str[0])
        df = df.transpose()
        return df
    except Exception as e:
        return pd.DataFrame()

def get_data(tickers, start, end, custom_data):
    try:
        data = custom_data.loc[start:end, tickers]
        data = data.dropna()
        return data  # Already returns from CSV
    except Exception as e:
        return pd.DataFrame()
