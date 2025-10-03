import streamlit as st
import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from summaria_utils import compute_topic_metrics, build_composite_relations, persist_feedback


# -------------------------------
# Helper to clean topics from LLM
# -------------------------------
def _parse_topics_from_answer(answer_text):
    """
    Extract clean topic labels from LLM output.
    Removes 'Topic:', 'Summary:', 'Importance:', 'Sources:', and exam-question lines.
    """
    lines = answer_text.splitlines()
    topics = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.lower().startswith(("summary", "importance", "sources")):
            continue
        if ln.endswith("?"):  # exam-style questions
            continue
        if ln.lower().startswith("topic:"):
            ln = ln.split(":", 1)[1].strip()
        if len(ln) > 120:
            continue
        topics.append(ln)

    # Deduplicate
    seen, clean = set(), []
    for t in topics:
        if t not in seen:
            seen.add(t)
            clean.append(t)
    return clean


# -------------------------------
# UI Components
# -------------------------------
def render_file_upload():
    """Render file upload component"""
    return st.file_uploader(
        "Upload Notes, PPTs, and PYQs",
        type=["pdf", "pptx", "docx"],
        accept_multiple_files=True,
    )


def render_processing_button(uploaded_files, process_callback):
    """Render processing button with callback"""
    if st.button("Process & Embed Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one file")
        else:
            return process_callback(uploaded_files)
    return None


def render_topic_analysis(retriever, llm, topics_prompt):
    """Render topic analysis section"""
    if st.button("Get Important Topics"):
        with st.spinner("Analyzing for key topics..."):
            try:
                document_chain = create_stuff_documents_chain(llm, topics_prompt)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                response = retrieval_chain.invoke({"input": "important topics"})

                st.subheader("📌 Key Topics to Focus On")
                st.write(response["answer"])

                return response
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return None


def render_question_prediction(retriever, llm, future_qs_prompt):
    """Render question prediction section"""
    if st.button("Predict Exam Questions"):
        with st.spinner("Analyzing for potential questions..."):
            try:
                document_chain = create_stuff_documents_chain(llm, future_qs_prompt)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                response = retrieval_chain.invoke({"input": "future questions"})

                st.subheader("🔮 Predicted Exam Questions")
                st.write(response["answer"])

                return response
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return None


def render_explanation(response, llm, answer_type, get_explanation_prompt_func, chunks=None):
    """Render explanation + SUMMARIA metrics + relations + feedback"""
    if response:
        with st.expander(
            "🔍 How these topics were identified"
            if answer_type == "topics"
            else "🔍 How these predictions were made"
        ):
            explanation_prompt = get_explanation_prompt_func(answer_type)

            context_docs = "\n---\n".join(
                [doc.page_content[:500] + "..." for doc in response["context"]]
            )
            explanation_chain = explanation_prompt | llm
            explanation = explanation_chain.invoke(
                {"answer": response["answer"], "context": context_docs}
            )
            st.write(explanation.content)

        # Show source materials
        with st.expander("📚 Relevant Source Materials"):
            for i, doc in enumerate(response["context"], 1):
                st.write(f"**Source {i}:**")
                st.write(doc.page_content[:500] + "...")
                st.write("---")

        # SUMMARIA Metrics & Relations (topics only)
        if answer_type == "topics" and chunks:
            st.subheader("🧾 SUMMARIA Metrics & Composite Relations")

            # Extract topics
            topics = _parse_topics_from_answer(response["answer"])

            # Compute metrics
            topic_info, cooccurrence = compute_topic_metrics(topics, chunks)

            # Table
            st.markdown("**Topic metrics (Truth degree, Coverage, Count)**")
            rows = []
            for t in topics:
                info = topic_info.get(t)
                if not info:
                    continue
                rows.append(
                    {
                        "Topic": t,
                        "Truth": float(info["truth_degree"]),
                        "Coverage": float(round(info["coverage_degree"], 4)),
                        "Count": int(info["count"]),
                    }
                )
            if rows:
                df = pd.DataFrame(rows)
                df = df.sort_values(by="Truth", ascending=False)  # sort by importance
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No valid topic metrics computed yet.")

            # Relations
            relations = build_composite_relations(topic_info, cooccurrence)
            if relations["evidence"]:
                st.markdown("**Evidence relations**")
                for rel in relations["evidence"]:
                    st.write(rel["verbalization"])
            if relations["emphasis"]:
                st.markdown("**Emphasis relations**")
                for rel in relations["emphasis"]:
                    st.write(rel["verbalization"])
            if relations["contrast"]:
                st.markdown("**Contrast relations**")
                for rel in relations["contrast"]:
                    st.write(rel["verbalization"])

            # Feedback
            st.subheader("Feedback — help improve the system")
            for t in topics:
                info = topic_info.get(t)
                if not info:
                    continue
                cols = st.columns([8, 1, 1])
                cols[0].write(
                    f"**{t}** — Truth: {info['truth_degree']}, Coverage: {round(info['coverage_degree'],3)}"
                )
                if cols[1].button(f"👍", key=f"like_{t}"):
                    persist_feedback({"type": "topic", "item": t, "feedback": "useful"})
                    st.success(f"Thanks — saved feedback for {t} (useful).")
                if cols[2].button(f"👎", key=f"dislike_{t}"):
                    persist_feedback(
                        {"type": "topic", "item": t, "feedback": "not_useful"}
                    )
                    st.warning(f"Thanks — saved feedback for {t} (not useful).")

        # Confidence indicators (for question prediction)
        if answer_type == "questions":
            with st.expander("📊 Prediction Confidence Guide"):
                st.markdown(
                    """
                **How to interpret the predictions:**
                
                - 🟢 **High Likelihood**: Strong evidence in materials, appears frequently  
                - 🟡 **Medium Likelihood**: Some supporting evidence, reasonable inference  
                - 🔴 **Low Likelihood**: Possible but less directly supported  
                
                **Factors considered:**
                - Frequency in your materials  
                - Question patterns from past exams  
                - Importance in the subject area  
                """
                )


def render_system_info():
    """Render system information section"""
    with st.expander("ℹ️ About This AI Assistant"):
        st.markdown(
            """
        **How this system works:**
        
        1. **Document Processing**:  
           - Your materials are split into meaningful chunks  
           - Both semantic meaning and keywords are extracted  
        
        2. **Hybrid Search**:  
           - Combines understanding of concepts with keyword matching  
           - Finds the most relevant parts of your materials  
        
        3. **Analysis**:  
           - Identifies patterns and important concepts  
           - Generates predictions based on your specific materials  
        
        **Transparency Features**:  
        - See exactly how recommendations are generated  
        - View the source materials used  
        - Understand confidence levels  
        
        **Remember**: This is an AI study aid, not a substitute for your own preparation.  
        """
        )
