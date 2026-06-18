from fastapi import FastAPI
from pydantic import BaseModel
from searchTransport import TransportBookingAssistant

class QueryRequest(BaseModel):
    query: str | None = None
    user_answer: str | None = None
    session_id: str

app = FastAPI()
sessions = {}

@app.post("/run")
def run_query(request: QueryRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = TransportBookingAssistant()
    assistant = sessions[request.session_id]

    result = assistant.run(request.query or "", request.user_answer or "")
    return {"result": result.get("result", "No result available")}
