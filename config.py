import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "one-shot-hybrid"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200