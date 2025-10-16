from fastapi import FastAPI
from pydantic import BaseModel
from QA_with_Trans import app as qa_app

# Define the request body format
class QARequest(BaseModel):
    input_question: str
    language: str = "French"  # Default French

api = FastAPI()

# Define the interface: using type annotations
@api.post("/invoke")
def invoke(req: QARequest):
    result = qa_app(req.input_question, req.language)
    return {"answer": result}

# Keep the container running
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
