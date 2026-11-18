import json
from datetime import datetime
import requests

def get_current_date():
    """
    Returns the current date in the format YYYY-MM-DD.
    """
    return datetime.now().strftime("%Y-%m-%d")

def get_city_weather(city_name):
    """
    Returns the weather of the given city using wttr.in (format=j1).
    """
    try:
        url = f"https://wttr.in/{city_name}?format=j1"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]
        return {
            "city": city_name,
            "temperature_c": current.get("temp_C"),
            "condition": current.get("weatherDesc", [{}])[0].get("value"),
            "humidity": current.get("humidity"),
            "wind_speed_kmph": current.get("windspeedKmph")
        }
    except Exception as e:
        return {"error": f"Could not fetch weather for {city_name}: {str(e)}"}

Tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date in YYYY-MM-DD format",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "Get the weather of the given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "Name of the city"
                    }
                },
                "required": ["city_name"]
            }
        }
    }
]

FUNCTION_MAP = {
    "get_current_date": get_current_date,
    "get_city_weather": get_city_weather
}

def execute_tools(tool_calls):
    """
    Executes a list of tool_calls returned by the model and
    returns a list of tool response messages.
    Supports both OpenAI objects and streaming dictionaries.
    """
    tool_messages = []
    for tool_call in tool_calls:
        # Dictionary format (streaming) ya Object format (non-streaming) dono handle karein
        if isinstance(tool_call, dict):
            function_name = tool_call["function"]["name"]
            arguments_str = tool_call["function"]["arguments"]
            call_id = tool_call["id"]
        else:
            function_name = tool_call.function.name
            arguments_str = tool_call.function.arguments
            call_id = tool_call.id

        if function_name in FUNCTION_MAP:
            function_to_call = FUNCTION_MAP[function_name]
            try:
                args = json.loads(arguments_str) if arguments_str else {}
                result = function_to_call(**args)
            except Exception as e:
                result = f"Error executing tool '{function_name}': {str(e)}"
        else:
            result = f"Error: Tool '{function_name}' not found."

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": function_name,
                "content": str(result)
            }
        )
    return tool_messages