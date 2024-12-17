from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate

import os

def get_api_key():
    try:
        api_key = input("Please enter LLM API Key ").strip()
        # langchain_key = input("Please enter LangChain API Key ").strip()
        os.environ["GROQ_API_KEY"] = api_key
        # os.environ["LANGCHAIN_API_KEY"] = langchain_key
        if not api_key:
            print("LLM and LangChain API key is required to proceed.")
            exit()
    except Exception as e:
        print(f"An error occured: {e}")


async def poll_requests(agent_executor, config, tools):
    try:
        while True:
            message = input("\nSend a message\n ")

            if message.lower() in ["exit", "quit"]:
                print("Exiting the program.")
                break

            input_data = {
                "input": message,
                "agent_scratchpad": "",
                "tools": "\n".join([tool.name for tool in tools]),  # Add tool descriptions
                "tool_names": ", ".join([tool.name for tool in tools])  # Add tool names
            }

            async for event in agent_executor.astream_events(
            {"input": input_data}, version="v2", config=config
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

            # code_interpreter.close() 


    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt.")


def setup_executer(llm, memory, tools):
    with open("./prompt_template.txt", "r") as f:
        template = f.read()

    prompt = ChatPromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
        )       

    return agent_executor
