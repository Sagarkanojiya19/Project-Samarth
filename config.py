import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
DATAGOVINDIA_API_KEY = os.getenv('DATAGOVINDIA_API_KEY')
BASE_API_URL = "https://api.data.gov.in/resource"

# Resource ID for crop production statistics
CROP_PRODUCTION_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

# Groq Configuration (for deployment)
GOOGLE_API_KEY =os.getenv('GOOGLE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# ChromaDB Configuration
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')
COLLECTION_NAME = "agricultural_data"

# Data paths
RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"

# Logging
LOG_FILE = "./logs/app.log"