from langchain_core.prompts import ChatPromptTemplate

# Prompt Templates
topics_prompt = ChatPromptTemplate.from_template("""
You are an expert exam analyst. From the given previous year questions and the context, extract the MOST important topics students should prepare for the exam based on frequency of that topic asked in previous year questions. 

For EACH topic you identify:
1. Provide a 2-3 sentence summary
2. Explain WHY this topic is important (based on frequency, weightage, or foundational concepts)
3. List the specific CONTEXT SOURCES that support this topic's importance

Format your response as follows:

Topic: [Topic Name]
Summary: [Brief summary]
Importance: [Explanation why this is important]
Sources: [Document excerpts that support this]

<context>
{context}
</context>
""")

future_qs_prompt = ChatPromptTemplate.from_template("""
You are an experienced exam setter. Based on the given previous year questions and context, predict 5 possible exam questions that could appear in the next exam.

For EACH question:
1. Provide the question text
2. Explain WHY this question might appear (based on trends, importance, or recent focus)
3. Rate the LIKELIHOOD of this question appearing (High/Medium/Low)
4. Suggest the BEST SOURCES from the context to answer this question

Format your response as follows:

Question: [Question text]
Reason: [Why this might appear]
Likelihood: [High/Medium/Low]
Sources: [Relevant document excerpts]

<context>
{context}
</context>
""")

def get_explanation_prompt(answer_type):
    """Get explanation prompt based on answer type"""
    if answer_type == "topics":
        return ChatPromptTemplate.from_template("""
        The following answer about important topics was generated:
        
        {answer}
        
        It was created from these source materials:
        {context}
        
        Explain in simple terms:
        1. How the system determined these were the most important topics
        2. What evidence supports each topic's importance
        3. Any limitations in this analysis
        4. How the student could verify these are indeed key topics
        """)
    else:  # questions
        return ChatPromptTemplate.from_template("""
        The following exam predictions were generated:
        
        {answer}
        
        Based on these source materials:
        {context}
        
        Explain in simple terms:
        1. The patterns that led to these predictions
        2. The confidence level for each prediction
        3. How the student should interpret the likelihood ratings
        4. Any important limitations to consider
        """)