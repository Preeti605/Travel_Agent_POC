from fastapi import FastAPI
from pydantic import BaseModel
from deepagents import create_deep_agent
from ddgs import DDGS
import os

def web_search(query: str) -> str:
    """Search the web for current information on travel, flights, hotels, or attractions."""
    results = DDGS().text(query, max_results=4)
    return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash-lite",
    tools=[web_search],
    system_prompt="You are a helpful travel planning assistant. Help the user research destinations, find things to do, and build itineraries.",
)

app = FastAPI()

class TripRequest(BaseModel):
    message: str

@app.post("/plan-trip")
def plan_trip(req: TripRequest):
    result = agent.invoke({"messages": [{"role": "user", "content": req.message}]})
    return {"reply": result["messages"][-1].content}
