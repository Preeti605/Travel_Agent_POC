from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
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

def extract_text(content):
    """Gemini/Anthropic messages can return content as a plain string
    or as a list of content blocks like [{"type": "text", "text": "..."}].
    This normalizes either shape into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)

@app.post("/plan-trip")
def plan_trip(req: TripRequest):
    result = agent.invoke({"messages": [{"role": "user", "content": req.message}]})
    final_content = result["messages"][-1].content
    return {"reply": extract_text(final_content)}

# Serves everything in the "static" folder, with index.html at "/".
# IMPORTANT: this must be the LAST route registered, or it will swallow
# requests meant for /plan-trip above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
