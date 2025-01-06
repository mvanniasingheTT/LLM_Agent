from custom_llm import CustomLLM
from utils import get_api_key, poll_requests, setup_executer
from code_tool import CodeInterpreterFunctionTool
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
import asyncio
import os 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel



app = FastAPI()

class RequestPayload(BaseModel):
    message: str
    thread_id: str


llm = CustomLLM(server_url="http://model:7000/v1/completions")
memory = ConversationBufferMemory(
    llm=llm,
    memory_key="chat_history", # What dict key to use to parse in the agent
    return_messages=True,
    output_key="output")

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

search = TavilySearchResults(
    max_results=2,
    include_answer=True,
    include_raw_content=True,
    include_images=True)

os.environ["E2B_API_KEY"] = os.getenv("E2B_API_KEY")
code_interpreter = CodeInterpreterFunctionTool()
code_interpreter_tool = code_interpreter.to_langchain_tool()
# tools =[code_interpreter_tool, search]
tools = [search]
agent_executer = setup_executer(llm, memory, tools)
# asyncio.run(poll_requests(agent_executer, config, tools))

@app.post("/poll_requests")
async def handle_requests(payload: RequestPayload):
    config = {"configurable": {"thread_id": payload.thread_id}}
    try:
        # use await to prevent handle_requests from blocking, allow other tasks to execute
        result = await poll_requests(agent_executer, config, tools, payload.message)
        return {"output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# health check
@app.get("/")
def read_root():
    return {"message": "Server is running"}
