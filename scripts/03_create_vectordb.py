import chromadb
from pathlib import Path
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

def load_documents(json_filename):
    """Load processed documents from JSON"""
    filepath = Path(PROCESSED_DATA_DIR) / json_filename
    print(f"Loading documents from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"Loaded {len(documents)} documents")
    return documents

def create_chroma_collection(documents):
    """
    Create ChromaDB collection and add documents
    """
    print("\n" + "=" * 60)
    print("CREATING CHROMADB VECTOR DATABASE")
    print("=" * 60)
    
    # Create ChromaDB persistent client
    print(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    # Create new collection
    print(f"Creating new collection: {COLLECTION_NAME}")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Agricultural data from data.gov.in"}
    )
    
    # Prepare data for ChromaDB
    print("\nPreparing data for embedding...")
    ids = []
    texts = []
    metadatas = []
    
    for idx, doc in enumerate(documents):
        ids.append(f"doc_{idx}")
        texts.append(doc['text'])
        metadatas.append(doc['metadata'])
    
    # Add documents in batches
    batch_size = 500
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    print(f"\nAdding {len(documents)} documents in {total_batches} batches...")
    
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        batch_num = (i // batch_size) + 1
        
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        collection.add(
            ids=ids[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
    
    print(f"\n✓ Successfully added {len(documents)} documents to ChromaDB")
    
    # Verify collection
    print(f"\nCollection info:")
    print(f"  Name: {collection.name}")
    print(f"  Count: {collection.count()}")
    
    return collection

if __name__ == "__main__":
    try:
        # Load processed documents
        documents = load_documents('documents.json')
        
        # Create ChromaDB collection
        collection = create_chroma_collection(documents)
        
        # Test query
        print("\n" + "=" * 60)
        print("TESTING VECTOR DATABASE")
        print("=" * 60)
        
        test_query = "What is the wheat production?"
        print(f"\nTest query: {test_query}")
        
        results = collection.query(
            query_texts=[test_query],
            n_results=3
        )
        
        print(f"\nTop 3 results:")
        for i, doc in enumerate(results['documents'][0], 1):
            print(f"\n--- Result {i} ---")
            print(doc[:200] + "...")
        
        print("\n" + "=" * 60)
        print("VECTOR DATABASE CREATION COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()