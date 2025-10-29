import requests
import pandas as pd
import time
import json
from pathlib import Path
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

def fetch_crop_data_paginated(resource_id, filters=None, max_records=10000):
    """
    Fetch crop production data from data.gov.in API with pagination
    """
    print(f"Fetching data from resource ID: {resource_id}")
    
    all_data = []
    offset = 0
    limit = 100
    
    while len(all_data) < max_records:
        params = {
            'api-key': DATAGOVINDIA_API_KEY,
            'format': 'json',
            'offset': offset,
            'limit': limit
        }
        
        if filters:
            for key, value in filters.items():
                params[f'filters[{key}]'] = value
        
        url = f"{BASE_API_URL}/{resource_id}"
        
        try:
            print(f"Fetching records {offset} to {offset + limit}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'records' not in data or len(data['records']) == 0:
                print(f"No more records found. Total fetched: {len(all_data)}")
                break
            
            all_data.extend(data['records'])
            print(f"Total records fetched so far: {len(all_data)}")
            
            if len(data['records']) < limit:
                break
            
            offset += limit
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            if len(all_data) > 0:
                print(f"Returning {len(all_data)} records fetched so far")
                break
            else:
                raise
    
    df = pd.DataFrame(all_data)
    print(f"\nTotal records fetched: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def save_raw_data(df, filename):
    """Save raw data to CSV"""
    filepath = Path(RAW_DATA_DIR) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to: {filepath}")

if __name__ == "__main__":
    print("=" * 60)
    print("FETCHING CROP PRODUCTION DATA FROM DATA.GOV.IN")
    print("=" * 60)
    
    filters = {}
    
    try:
        df_crops = fetch_crop_data_paginated(
            resource_id=CROP_PRODUCTION_RESOURCE_ID,
            filters=filters,
            max_records=50000
        )
        
        save_raw_data(df_crops, 'crop_production_raw.csv')
        
        print("\n" + "=" * 60)
        print("DATA FETCH COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nFirst few rows:")
        print(df_crops.head())
        
    except Exception as e:
        print(f"\nERROR: Failed to fetch data: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if your API key is correct in .env file")
        print("2. Verify the resource ID is valid")
        print("3. Check your internet connection")
