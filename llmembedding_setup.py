from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from config import GROQ_API_KEY, LLM_MODEL, EMBEDDINGS_MODEL

def setup_llm():
    """Initialize the LLM and embeddings"""
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    bm25_encoder = BM25Encoder().default()
    
    return llm, embeddings, bm25_encoder