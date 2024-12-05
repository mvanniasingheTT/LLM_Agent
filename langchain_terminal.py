import asyncio
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

class CustomLLM(LLM):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """


def get_api_key():
    try:
        api_key = input("Please enter LLM API Key ").strip()
        langchain_key = input("Please enter LangChain API Key ").strip()
        os.environ["GROQ_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = langchain_key
        if not api_key:
            print("LLM and LangChain API key is required to proceed.")
            exit()
    except Exception as e:
        print(f"An error occured: {e}")


async def poll_requests(agent_executor, config):
    try:
        while True:
            message = input("\nSend a message\n ")

            if message.lower() in ["exit", "quit"]:
                print("Exiting the program.")
                break

            async for event in agent_executor.astream_events(
            {"messages": [HumanMessage(content=message)]}, version="v2", config=config
        ):
                kind = event["event"]
                if kind == "on_chain_start":
                    if (
                        event["name"] == "Agent"
                    ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                        # print(
                        #     f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                        # )
                        pass
                elif kind == "on_chain_end":
                    if (
                        event["name"] == "Agent"
                    ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                        # print()
                        # print("--")
                        # print(
                        #     f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                        # )
                        pass
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        # Empty content in the context of OpenAI means
                        # that the model is asking for a tool to be invoked.
                        # So we only print non-empty content
                        print(content, end="|")
                elif kind == "on_tool_start":
                    pass
                    # print("--")
                    # print(
                    #     f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
                    # )
                elif kind == "on_tool_end":
                    # print(f"Done tool: {event['name']}")
                    # print(f"Tool output was: {event['data'].get('output')}")
                    # print("--")
                    pass


    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt.")


def setup_executer(llm, memory, tools):
    return create_react_agent(llm, tools, checkpointer=memory)

if __name__ == "__main__":
    get_api_key()

    memory = MemorySaver()
    llm = ChatGroq(model="llama3-8b-8192")
    search = TavilySearchResults(
    max_results=5,
    include_answer=True,
    include_raw_content=True,
    include_images=True)
    tools =[search]
    config = {"configurable": {"thread_id": "cde123"}}
    agent_executer = setup_executer(llm, memory, tools)
    asyncio.run(poll_requests(agent_executer, config))