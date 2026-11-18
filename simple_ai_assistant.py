import os
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError
from tool import Tools, execute_tools
from Utilities.utilities import logs
from Utilities.stream_handler import stream_response

load_dotenv()

base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

client = OpenAI(base_url=base_url, api_key=api_key)

MAX_STEP = 5

message = []

message.append(
    {
        "role": "system",
        "content": "You are a helpful assistant."
    }
)

while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the assistant.")
        break

    if not user_input.strip():
        continue

    message.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    try:
        # Agent loop: jab tak AI tools call karta rahega, loop chalega
        for steps in range(MAX_STEP):
            # Modular streaming call (Normal text live print hoga, tool calls assemble honge)
            result = stream_response(client, model, message, Tools)
            
            tool_calls = result["tool_calls"]
            content = result["content"]

            logs(f"Step {steps + 1}: model requested {len(tool_calls or [])} tool calls")

            # Check karo ki kya AI tool call karna chahta hai?
            if tool_calls:
                # 1. AI ka tool_calls wala message history me add karo
                message.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    }
                )

                # Console me dikhao ki kaunsa tool call ho raha hai
                for tool_call in tool_calls:
                    logs(f"🔧 [Tool Call] {tool_call['function']['name']} -> {tool_call['function']['arguments']}")

                # 2. Saare tools execute karo aur unke results history me add karo
                tool_messages = execute_tools(tool_calls)
                message.extend(tool_messages)

                # Loop continue hoga taaki model tool results dekh kar final reply stream kar sake
            else:
                # Jab model koi tool call na kare, text already stream ho chuka hai
                message.append(
                    {
                        "role": "assistant",
                        "content": content
                    }
                )
                break
        else:
            fallback = "Sorry, we ran out of limit. Our LLM could not provide the final answer."    
            message.append(
                {
                    "role": "assistant",
                    "content": fallback
                }
            )
            print(f"Assistant: {fallback}")

    except APIConnectionError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")