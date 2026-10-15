import os
from dotenv import load_dotenv

from openai import OpenAI, APIConnectionError

load_dotenv()

base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

message = []

while True:

    user_input = input("You: ")

    message.append(
            {
                "role" : "user",
                "content": user_input
            }
        )

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the assistant.")
        break

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=message
        )
        message.append(
            {
                "role" : "assistant",
                "content": response.choices[0].message.content
            }
        )



        assistant_reply = response.choices[0].message.content
        print(f"Assistant: {assistant_reply}")

        
    except APIConnectionError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")