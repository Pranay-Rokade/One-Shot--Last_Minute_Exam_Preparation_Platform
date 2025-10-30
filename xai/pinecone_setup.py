# pinecone_setup.py
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, INDEX_NAME

def initialize_pinecone():
    """Initialize Pinecone and create index if it doesn't exist"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    
    index = pc.Index(INDEX_NAME)
    return index

def delete_existing_index():
    """Delete the Pinecone index if it exists"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME in existing_indexes:
        pc.delete_index(INDEX_NAME)
        return f"✅ Index '{INDEX_NAME}' deleted successfully."
    else:
        return f"⚠️ No index found with the name '{INDEX_NAME}'."
    
    