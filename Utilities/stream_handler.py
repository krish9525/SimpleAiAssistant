import json

def stream_response(client, model, messages, tools=None):
    """
    Handles streaming responses for both normal text and tool calls.
    
    Returns:
        dict: {
            "content": str or None,     # Complete text response (if model didn't call tools)
            "tool_calls": list or None  # Assembled tool calls (if model called tools)
        }
    """
    response_stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=True
    )

    collected_content = []
    tool_calls_dict = {}
    is_first_text_chunk = True

    for chunk in response_stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # -------------------------------------------------------------
        # 1. Normal Text Response (Stream live token-by-token to terminal)
        # -------------------------------------------------------------
        if delta.content:
            if is_first_text_chunk:
                print("Assistant: ", end="", flush=True)
                is_first_text_chunk = False

            print(delta.content, end="", flush=True)
            collected_content.append(delta.content)

        # -------------------------------------------------------------
        # 2. Tool Call Streaming (Assemble fragmented chunks into complete tool_calls)
        # -------------------------------------------------------------
        if delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index

                if idx not in tool_calls_dict:
                    tool_calls_dict[idx] = {
                        "id": tc_chunk.id or "",
                        "type": "function",
                        "function": {
                            "name": tc_chunk.function.name or "",
                            "arguments": tc_chunk.function.arguments or ""
                        }
                    }
                else:
                    if tc_chunk.id:
                        tool_calls_dict[idx]["id"] += tc_chunk.id
                    if tc_chunk.function.name:
                        tool_calls_dict[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls_dict[idx]["function"]["arguments"] += tc_chunk.function.arguments

    # Text print hone ke baad terminal me nayi line print karo
    if not is_first_text_chunk:
        print()

    final_content = "".join(collected_content) if collected_content else None
    final_tool_calls = list(tool_calls_dict.values()) if tool_calls_dict else None

    return {
        "content": final_content,
        "tool_calls": final_tool_calls
    }
