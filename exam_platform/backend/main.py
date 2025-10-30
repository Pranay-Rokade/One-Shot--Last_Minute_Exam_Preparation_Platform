from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone_utils import get_all_context_from_pinecone
from langchain_pipeline import generate_mcqs_from_context

app = FastAPI(title="Exam Platform ")

class TopicRequest(BaseModel):
    topic: str
    num_contexts: int = 5

@app.post("/generate-questions")
async def generate_questions():
    try:
        # 1. Retrieve relevant context
        context = get_all_context_from_pinecone()

        # 2. Generate MCQs from context
        mcqs = generate_mcqs_from_context(context)

        return {"questions": mcqs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
