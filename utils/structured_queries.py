import pandas as pd

def query_crop_production(df: pd.DataFrame, state: str, crop: str, years: list):
    filtered = df[
        (df['state'].str.lower() == state.lower()) &
        (df['crop'].str.lower() == crop.lower()) &
        (df['year'].isin(years))
    ]
    grouped = filtered.groupby('district')['production_volume'].sum().sort_values(ascending=False)
    return grouped

def query_rainfall_data(df: pd.DataFrame, state: str, years: list):
    filtered = df[
        (df['state'].str.lower() == state.lower()) &
        (df['year'].isin(years))
    ]
    return filtered