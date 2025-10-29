import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

def load_raw_data(filename):
    """Load raw data from CSV"""
    filepath = Path(RAW_DATA_DIR) / filename
    print(f"Loading data from: {filepath}")
    return pd.read_csv(filepath)

def clean_agricultural_data(df):
    """
    Clean and standardize agricultural dataset
    """
    print("\n" + "=" * 60)
    print("CLEANING AGRICULTURAL DATA")
    print("=" * 60)
    
    print(f"Original shape: {df.shape}")
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    # 2. Handle missing values
    critical_columns = [col for col in df.columns if col.lower() in ['state_name', 'district_name', 'crop', 'production']]
    if critical_columns:
        df = df.dropna(subset=critical_columns)
    print(f"After removing missing values: {df.shape}")
    
    # 3. Standardize text columns
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip().str.title()
    
    # 4. Convert numeric columns
    numeric_columns = [col for col in df.columns if col.lower() in ['production', 'area', 'yield']]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 5. Remove invalid data
    for col in numeric_columns:
        if col in df.columns:
            df = df[df[col] >= 0]
    
    print(f"Final shape after cleaning: {df.shape}")
    
    return df

def create_text_documents(df):
    """
    Convert structured data into text documents for RAG
    """
    print("\n" + "=" * 60)
    print("CREATING TEXT DOCUMENTS FOR RAG")
    print("=" * 60)
    
    documents = []
    
    for idx, row in df.iterrows():
        # Create text from available columns
        text_parts = []
        for col in df.columns:
            value = row.get(col, 'Unknown')
            text_parts.append(f"{col}: {value}")
        
        text = "\n".join(text_parts)
        text += "\n\nSource: District-wise Crop Production Statistics, Ministry of Agriculture & Farmers Welfare, Government of India"
        text += "\nDataset URL: https://data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0"
        
        # Create metadata
        metadata = {
            'source': 'crop_production',
            'dataset_url': 'https://data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0'
        }
        
        # Add common fields if they exist
        for col in df.columns:
            col_lower = col.lower()
            if 'state' in col_lower or 'district' in col_lower or 'crop' in col_lower or 'year' in col_lower:
                metadata[col_lower] = str(row.get(col, 'Unknown'))
        
        documents.append({
            'text': text,
            'metadata': metadata
        })
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} documents...")
    
    print(f"Total documents created: {len(documents)}")
    return documents

def save_processed_data(df, documents, csv_filename, json_filename):
    """Save processed data"""
    
    # Save cleaned CSV
    csv_path = Path(PROCESSED_DATA_DIR) / csv_filename
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nCleaned data saved to: {csv_path}")
    
    # Save documents as JSON
    json_path = Path(PROCESSED_DATA_DIR) / json_filename
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    print(f"Documents saved to: {json_path}")

if __name__ == "__main__":
    try:
        # Load raw data
        df = load_raw_data('crop_production_raw.csv')
        
        # Clean data
        df_clean = clean_agricultural_data(df)
        
        # Create text documents
        documents = create_text_documents(df_clean)
        
        # Save processed data
        save_processed_data(
            df_clean, 
            documents,
            'crop_production_clean.csv',
            'documents.json'
        )
        
        print("\n" + "=" * 60)
        print("DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()