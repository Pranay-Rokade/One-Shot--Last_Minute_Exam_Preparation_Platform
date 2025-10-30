# main.py
import streamlit as st
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from pinecone_setup import initialize_pinecone, delete_existing_index
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
            st.session_state.docs_processed = True  # ✅ Show next-step buttons after success
            st.success("✅ Documents processed and embedded successfully!")
            st.balloons()
            return True
        except Exception as e:
            st.error(str(e))
            st.session_state.docs_processed = False
            return False

render_processing_button(uploaded_files, process_files)

# --- Show Next Steps Buttons after Successful Embedding ---
if st.session_state.get("docs_processed", False):
    st.divider()
    st.subheader("🚀 Next Steps")

    # Custom button styling
    st.markdown("""
        <style>
        .purple-button {
            display: inline-block;
            background-color: #10072b;
            color: white !important;
            text-align: center;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 15px 0;
            width: 100%;
            font-size: 18px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            box-shadow: 0 0 20px #b86bff;
        }
        .purple-button:hover {
            background-color: #4d40ff;
            box-shadow: 0 0 25px #d9aaff;
            transform: scale(1.03);
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<a class="purple-button" href="http://localhost:5173/" target="_blank">🎯 Take Exam</a>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<a class="purple-button" href="http://localhost:9000/" target="_blank">🎥 Video Call with a Friend</a>',
            unsafe_allow_html=True,
        )

# --- Main interaction section (after retriever setup) ---
if "retriever" in st.session_state:
    st.divider()
    st.header("📝 Ask About Your Materials")
    
    query = st.text_input("What would you like to know? (e.g., 'important topics', 'potential questions')")
    
    # Topic analysis
    topics_response = render_topic_analysis(st.session_state.retriever, llm, topics_prompt)
    if topics_response:
        render_explanation(topics_response, llm, "topics", get_explanation_prompt, chunks=st.session_state.get("chunks"))
    
    st.divider()
    
    # Question prediction
    questions_response = render_question_prediction(st.session_state.retriever, llm, future_qs_prompt)
    if questions_response:
        render_explanation(questions_response, llm, "questions", get_explanation_prompt, chunks=None)
    
    st.divider()
    render_system_info()
else:
    st.info("👆 Please upload and process your study materials first")
    
# --- Delete index button (always visible) ---
if st.button("🗑️ Delete Existing Pinecone Index"):
    with st.spinner("Checking and deleting index..."):
        message = delete_existing_index()
        if "deleted successfully" in message:
            st.success(message)
        else:
            st.warning(message)
