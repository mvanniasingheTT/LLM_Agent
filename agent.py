from custom_llm import CustomLLM
from utils import get_api_key, poll_requests, setup_executer
from code_tool import CodeInterpreterFunctionTool
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_community.tools.tavily_search import TavilySearchResults
import asyncio
import os 
# from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
# Define the prompt template
# template = """This is a conversation between a human and a bot:

# {chat_history}

# Write a summary of the conversation for {input}:
# """
# prompt = PromptTemplate(input_variables=["input", "chat_history"], template=template)

# # Create memory objects
# memory = ConversationBufferMemory(memory_key="chat_history")
# readonlymemory = ReadOnlySharedMemory(memory=memory)

# app = FastAPI()

if __name__ == "__main__":

    class RequestPayload(BaseModel):
        message: str
        thread_id: str


    llm = CustomLLM(server_url="http://model:7000/v1/chat/completions")
    # Create the summarization chain
    # summary_chain = LLMChain(
    #     llm=llm, 
    #     prompt=prompt,
    #     verbose=True,
    #     memory=readonlymemory,  # use the read-only memory to prevent the tool from modifying the memory
    # )

    # memory = ConversationBufferMemory(
    #     llm=llm,
    #     memory_key="chat_history", # What dict key to use to parse in the agent
    #     return_messages=True,
    #     output_key="output")

    memory = ConversationBufferMemory(memory_key="chat_history")

    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

    search = TavilySearchResults(
        max_results=2,
        include_answer=True,
        include_raw_content=True,
        include_images=True)

    os.environ["E2B_API_KEY"] = os.getenv("E2B_API_KEY")
    code_interpreter = CodeInterpreterFunctionTool()
    code_interpreter_tool = code_interpreter.to_langchain_tool()
    # summary_tool = Tool(
    #     name="Summary",
    #     func=summary_chain.run,
    #     description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
    # )
    # tools =[code_interpreter_tool, search]
    # tools = [search, summary_tool]
    tools = [search]
    agent_executer = setup_executer(llm, memory, tools)
    config = {"configurable": {"thread_id": "abc-123"}}
    asyncio.run(poll_requests(agent_executer, config, tools, memory))

# @app.post("/poll_requests")
# async def handle_requests(payload: RequestPayload):
#     config = {"configurable": {"thread_id": payload.thread_id}}
#     try:
#         # use await to prevent handle_requests from blocking, allow other tasks to execute
#         result = await poll_requests(agent_executer, config, tools, payload.message)
#         return {"output": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
# # health check
# @app.get("/")
# def read_root():
#     return {"message": "Server is running"}
