# main.py
import streamlit as st
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from pinecone_setup import initialize_pinecone
from llmembedding_setup import setup_llm
from document_processor import process_uploaded_files
from prompts import topics_prompt, future_qs_prompt, get_explanation_prompt, composite_verbalize_prompt
from ui_components import *
from summaria_utils import persist_feedback

# Initialize components
index = initialize_pinecone()
llm, embeddings, bm25_encoder = setup_llm()

# Streamlit UI
st.title("📚 One Shot - Last Minute Exam Preparation Platform")
st.markdown("""
    Upload your study materials (PDFs, PPTs, Word docs) and get:
    - Key topics to focus on
    - Predicted exam questions
    - Transparent explanations of all recommendations
    - SUMMARIA-style composite relations (Evidence, Contrast, Emphasis) and quality metrics
""")

# File upload and processing
uploaded_files = render_file_upload()

def process_files(uploaded_files):
    """Callback function to process uploaded files"""
    with st.spinner("Processing documents..."):
        try:
            retriever, chunks = process_uploaded_files(uploaded_files, embeddings, bm25_encoder, index)
            st.session_state.retriever = retriever
            st.session_state.chunks = chunks
            st.success("✅ Documents processed and embedded successfully!")
            st.balloons()
            return True
        except Exception as e:
            st.error(str(e))
            return False

render_processing_button(uploaded_files, process_files)

# Main interaction section
if "retriever" in st.session_state:
    st.divider()
    st.header("📝 Ask About Your Materials")
    
    query = st.text_input("What would you like to know? (e.g., 'important topics', 'potential questions')")
    
    # Topic analysis
    topics_response = render_topic_analysis(st.session_state.retriever, llm, topics_prompt)
    if topics_response:
        # pass chunks for SUMMARIA metric computations
        render_explanation(topics_response, llm, "topics", get_explanation_prompt, chunks=st.session_state.get("chunks"))
    
    st.divider()
    
    # Question prediction
    questions_response = render_question_prediction(st.session_state.retriever, llm, future_qs_prompt)
    if questions_response:
        # questions explanation (we don't compute SUMMARIA metrics for questions by default)
        render_explanation(questions_response, llm, "questions", get_explanation_prompt, chunks=None)
    
    # System information
    st.divider()
    render_system_info()
else:
    st.info("👆 Please upload and process your study materials first")
