from langchain.agents import AgentExecutor, create_react_agent, ZeroShotAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub 
from langchain_core.messages import AIMessage, HumanMessage
import os
from langchain.chains import LLMChain


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


async def poll_requests(agent_executor, config, tools, memory):
    try:
        while True:
            message = input("\nSend a message\n ")

            if message.lower() in ["exit", "quit"]:
                print("Exiting the program.")
                break

            # input_data = {
            #     "input": message,
            #     "agent_scratchpad": "",
            #     "tools": "\n".join([tool.name for tool in tools]),  # Add tool descriptions
            #     "tool_names": ", ".join([tool.name for tool in tools])  # Add tool names
            # }

            # memory.chat_memory.add_message(HumanMessage(content=message))
            complete_output = ""  # Initialize an empty string to accumulate output
            async for event in agent_executor.astream_events(
            {"input": message}, version="v2", config=config
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
                        # complete_output += content
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


            print("\n\nThe complete AI Message: ", complete_output)
            # print(agent_executor.memory.load_memory)

            # memory.chat_memory.add_message(AIMessage(content=complete_output))

            # print("\n\nConversation History:")
            # print(memory.load_memory_variables({}))
            # code_interpreter.close() 


    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt.")


def setup_executer(llm, memory, tools):
    with open("./prompt_template.txt", "r") as f:
        template = f.read()

    prompt = ChatPromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    # prompt = hub.pull("hwchase17/structured-chat-agent")
    # agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

    # prefix = """Have a conversation with a human, answering the following questions as best as you can. You can have access to the following tools:
    #             search"""
    # suffix = """Begin!" 
    
    #         {chat_history}
    #         Question: {input}
    #         {agent_scratchpad}"""
    # prompt = ZeroShotAgent.create_prompt(
    #     tools,
    #     prefix=prefix,
    #     suffix=suffix,
    #     input_variables=["input", "chat_history", "agent_scratchpad"]
    # # )
    llm_chain =  LLMChain(llm=llm, prompt=prompt)
    # agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=10,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        )       

    return agent_executor
