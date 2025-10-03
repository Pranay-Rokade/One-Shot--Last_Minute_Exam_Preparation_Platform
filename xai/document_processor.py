# document_processor.py
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.retrievers import PineconeHybridSearchRetriever
from config import CHUNK_SIZE, CHUNK_OVERLAP

def process_uploaded_files(uploaded_files, embeddings, sparse_encoder, index):
    """Process uploaded files and add them to Pinecone"""
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(tmp_path)
            else:
                # support pptx and plain text fallback
                if file.name.endswith(".pptx"):
                    # try to load as binary text fallback
                    with open(tmp_path, "rb") as fh:
                        raw = fh.read().decode(errors="ignore")
                    docs.append(type("D", (), {"page_content": raw})())
                else:
                    continue
            if 'loader' in locals():
                docs.extend(loader.load())
                del loader
            os.unlink(tmp_path)
        except Exception as e:
            raise Exception(f"Error processing {file.name}: {str(e)}")
    
    if not docs:
        raise Exception("No valid documents were processed")
    
    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    
    # Fit sparse encoder
    texts = [doc.page_content for doc in chunks]
    sparse_encoder.fit(texts)
    
    # Create retriever
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=sparse_encoder,
        index=index
    )
    retriever.add_texts(texts)
    
    # return retriever and chunks (chunks used by SUMMARIA utilities)
    return retriever, chunks
