from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import json
import re
from pinecone_utils import get_all_context_from_pinecone
import os

def generate_mcqs_from_context(context: str):
    """Generate 10 MCQs with 4 options and correct answers using LLM"""
    
    prompt_template = PromptTemplate.from_template("""
    You are an expert educator.
    Based on the following context, generate exactly 10 multiple-choice questions.
    Each question must have:
    - question text
    - 4 options (A, B, C, D)
    - one correct answer key

    Output strictly as a valid JSON array:
    [
      {{
        "question": "...",
        "options": ["A", "B", "C", "D"],
        "answer": "A"
      }},
      ...
    ]

    Context:
    {context}
    """)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7,google_api_key=os.getenv("GIMINI_API_KEY"))
    prompt = prompt_template.format(context=context)
    response = llm.invoke([HumanMessage(content=prompt)]).content

    # Attempt to clean and parse JSON from LLM output
    try:
        json_data = json.loads(response)
    except json.JSONDecodeError:
        # fallback cleanup if model returns extra text
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            json_data = json.loads(match.group(0))
        else:
            json_data = {"error": "Invalid JSON format from LLM", "raw_output": response}

    return json.dumps(json_data, indent=2)


if __name__ == "__main__":
    context = get_all_context_from_pinecone()
    mcqs = generate_mcqs_from_context(context)
    print(mcqs)