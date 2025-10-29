import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def analyze_trend(rainfall_df: pd.DataFrame, crop_df: pd.Series):
    years = sorted(set(rainfall_df['year']).intersection(set(crop_df.index)))
    rainfall_series = [rainfall_df[rainfall_df['year'] == y]['rainfall'].mean() for y in years]
    crop_series = [crop_df[y] for y in years]
    slope = np.polyfit(years, crop_series, 1)[0] if len(years) > 1 else 0.0
    correlation, _ = pearsonr(rainfall_series, crop_series) if len(years) > 2 else 0.0
    return {'years': years, 'slope': slope, 'correlation': correlation}