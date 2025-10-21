from fastapi import FastAPI
from pydantic import BaseModel
from QA_with_Trans import app as qa_app, view_thread_state, view_all_states


api = FastAPI()


########################################################
# Define the request body format
########################################################
# TODO: currently need user to provide thread_id in the request body
class QARequest(BaseModel):
    input_question: str
    thread_id: str  # TODO:user must provide this ID
    language: str = "English"


########################################################
# Invoke the QA system
########################################################
@api.post("/invoke")
def invoke(req: QARequest):
    # pass the thread_id from the request body to langgraph
    result = qa_app(req.input_question, req.thread_id, req.language)
    return {"answer": result}


########################################################
# Get the state of a specific thread
########################################################
@api.get("/state/{thread_id}")
def get_thread_state(thread_id: str):
    try:
        state = view_thread_state(thread_id)
        if state:
            return {
                "thread_id": thread_id,
                "state": state.values,
                "next_node": state.next,
                "config": state.config
            }
        else:
            return {"error": f"Thread {thread_id} not found"}
    except Exception as e:
        return {"error": str(e)}


########################################################
# Get all the states
########################################################
@api.get("/states")
def get_all_states():
    try:
        view_all_states()
        return {"message": "Check console output for all states"}
    except Exception as e:
        return {"error": str(e)}


########################################################
# Main entry point
########################################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)