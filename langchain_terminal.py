import asyncio
import os
import requests 
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.outputs import GenerationChunk

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Type, Union
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.language_models import LanguageModelInput


from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from typing import Any, Dict, Iterator, List, Optional, Callable, Literal
from langchain_core.tools import BaseTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import (
    AIMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)


from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel, Field

class CustomLLM(BaseChatModel):
    server_url: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        # kwargs["encoded_jwt"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZWFtX2lkIjoidGVuc3RvcnJlbnQiLCJ0b2tlbl9pZCI6ImRlYnVnLXRlc3QifQ.LZeuFMVFzgxD91sDR79qlfDogPHUCG9lLopctYKYvnI"
        # headers = {"Authorization": f"Bearer {kwargs['encoded_jwt']}"}
        # json_data = {
        #     "model": "meta-llama/Meta-Llama-3.1-70B",
        #     "prompt": prompt,
        #     "temperature": 1,
        #     "top_k": 20,
        #     "top_p": 0.9,
        #     "max_tokens": 2048,
        #     "stream": kwargs["stream"],
        #     "stop": ["<|eot_id|>"],
        #     }
        # # response = requests.post(
        # #     self.server_url, json=json_data, headers=kwargs["header"], stream=kwargs["stream"], timeout=600
        # # )
        # res = []
        # with requests.post(
        #     self.server_url, json=json_data, headers=headers, stream=True, timeout=None
        # ) as response:
        #     for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        #         # yield chunk
        #         res += chunk
        pass

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = str(last_message.content)
        # kwargs["encoded_jwt"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZWFtX2lkIjoidGVuc3RvcnJlbnQiLCJ0b2tlbl9pZCI6ImRlYnVnLXRlc3QifQ.LZeuFMVFzgxD91sDR79qlfDogPHUCG9lLopctYKYvnI"
        # headers = {"Authorization": f"Bearer {kwargs['encoded_jwt']}"}
        # json_data = {
        #     "model": "meta-llama/Llama-3.1-70B-Instruct",
        #     "prompt": tokens,
        #     "temperature": 1,
        #     "top_k": 20,
        #     "top_p": 0.9,
        #     "max_tokens": 128,
        #     "stream": True,
        #     "stop": ["<|eot_id|>"],
        #     }
        # # with requests.post(
        #     self.server_url, json=json_data, headers=headers, stream=True, timeout=None
        # ) as response:
        #     for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        #         new_chunk = chunk[len("data: "):]
        #         new_chunk =  new_chunk.strip()
        #         if new_chunk == "[DONE]":
        #                 # Yield [DONE] to signal that streaming is complete
        #                 # new_chunk = Chunk(new_chunk)
        #                 # new_chunk = GenerationChunk(text=new_chunk)
        #                 new_chunk = ChatGenerationChunk(message=AIMessageChunk(content=new_chunk))
        #                 yield new_chunk
        #         else:
        #             new_chunk = json.loads(new_chunk)
        #             new_chunk = new_chunk["choices"][0]
        #             # print(new_chunk["text"])
        #             # new_chunk = Chunk(new_chunk["text"])
        #             # new_chunk = GenerationChunk(text=new_chunk["text"])
        #             new_chunk = ChatGenerationChunk(message=AIMessageChunk(content=new_chunk["text"]))
        #             yield new_chunk
        os.environ["GROQ_API_KEY"] = "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"
        from langchain_groq import ChatGroq
        model = ChatGroq(model="llama3-8b-8192")
        try:
            for chunk in model._stream(messages=messages, stop=stop, **kwargs):
                yield chunk
        except Exception as e:
            print(f"Error during streaming: {e}")


    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                "auto" to automatically determine which function to call
                with the option to not call any function, "any" to enforce that some
                function is called, or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "none", "required")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"

'''
need to look at streaming 
template might be differnt for llama 

streaming token by token was smthg new an in progres feature so need to look into that 
RAG streaming 

f10cs02 - work with anirudh 
'''

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
                        print(
                            f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                        )
                        # pass
                elif kind == "on_chain_end":
                    if (
                        event["name"] == "Agent"
                    ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                        print()
                        print("--")
                        print(
                            f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                        )
                        # pass
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        # Empty content in the context of OpenAI means
                        # that the model is asking for a tool to be invoked.
                        # So we only print non-empty content
                        print(content, end="|")
                elif kind == "on_tool_start":
                    # pass
                    print("--")
                    print(
                        f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
                    )
                elif kind == "on_tool_end":
                    print(f"Done tool: {event['name']}")
                    print(f"Tool output was: {event['data'].get('output')}")
                    print("--")
                    # pass


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


    # search = DuckDuckGoSearchResults()
    tools =[search]
    config = {"configurable": {"thread_id": "cde123"}}
    agent_executer = setup_executer(llm, memory, tools)
    asyncio.run(poll_requests(agent_executer, config))