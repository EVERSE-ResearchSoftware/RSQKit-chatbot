
import os
import json
from typing import List, Dict, Generator, Optional, Tuple, Any, Union

from openai import OpenAI
from settings import RESOURCES, PROVIDER_TO_RESOURCE_KEY

# Cache clients at module level
_client_cache = {}


def _get_client(provider: str) -> OpenAI:
    """Get or create cached client for provider"""
    if provider not in _client_cache:
        provider_key = PROVIDER_TO_RESOURCE_KEY[provider]
        if provider_key not in RESOURCES:
            raise ValueError(f"Unknown provider: {provider}")

        config = RESOURCES[provider_key]

        # Get API key just-in-time
        api_key = ""
        if api_env_var := config.get("api_env_var"):
            api_key = os.getenv(api_env_var, "")

        # Create and cache client
        _client_cache[provider] = OpenAI(base_url=config["base_url"], api_key=api_key)

    return _client_cache[provider]


Event = Dict[str, Any]


def get_chat_response_stream(
    provider: str,
    messages: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    stream_tools: bool = True,
    **kwargs,
) -> Generator[Union[str, Event], None, None]:
    """
    Universal chat streaming function with client caching and tool support.

    Args:
        stream_tools: If True, yields structured events when tools are present.
                     If False, falls back to non-streaming tool behavior.

    When stream_tools=True and tools are present, yields structured events:
      - {"type": "content.delta", "text": str}
      - {"type": "tool_call.start", "id": str, "name": str}
      - {"type": "tool_call.arguments", "id": str, "arguments_delta": str}
      - {"type": "message.completed", "content": str, "tool_calls": [...]}
      - {"type": "error", "message": str}

    When stream_tools=False or no tools, yields simple strings (backward compatible).
    """
    provider_key = PROVIDER_TO_RESOURCE_KEY[provider]
    if provider_key not in RESOURCES:
        if stream_tools and "tools" in kwargs:
            yield {"type": "error", "message": f"Unknown provider: {provider}"}
            return
        else:
            raise ValueError(f"Unknown provider: {provider}")

    config = RESOURCES[provider_key]
    client = _get_client(provider)
    model = model_name or config["default_llm"]

    # Check if tools are being used
    has_tools = "tools" in kwargs and kwargs["tools"]
    temperature = kwargs.pop("temperature", 0.0)

    if has_tools and stream_tools:
        # Stream with tools - yield structured events
        full_content = []
        tool_calls = {}  # id -> {name, arguments, index}
        current_id = None
        
        try:
            response_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=temperature,
                **kwargs,
            )

            for chunk in response_stream:
                if not chunk.choices or not chunk.choices[0]:
                    continue

                delta = chunk.choices[0].delta

                # Handle content deltas
                if hasattr(delta, 'content') and delta.content:
                    full_content.append(delta.content)
                    yield {"type": "content.delta", "text": delta.content}

                # Handle tool call deltas - improved parsing based on working algorithm
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        # Check for tool call ID
                        if hasattr(tool_call_chunk, 'id') and tool_call_chunk.id:
                            current_id = tool_call_chunk.id
                            if current_id not in tool_calls:
                                tool_calls[current_id] = {
                                    "id": current_id,
                                    "name": None,
                                    "arguments": "",
                                    "index": getattr(tool_call_chunk, 'index', 0)
                                }
                        
                        # Only process function data if we have a current_id
                        if current_id and current_id in tool_calls:
                            if hasattr(tool_call_chunk, 'function') and tool_call_chunk.function:
                                # Handle function name
                                if hasattr(tool_call_chunk.function, 'name') and tool_call_chunk.function.name:
                                    tool_calls[current_id]["name"] = tool_call_chunk.function.name
                                    yield {
                                        "type": "tool_call.start",
                                        "id": current_id,
                                        "name": tool_call_chunk.function.name,
                                    }
                                
                                # Handle function arguments (accumulate)
                                if hasattr(tool_call_chunk.function, 'arguments') and tool_call_chunk.function.arguments:
                                    tool_calls[current_id]["arguments"] += tool_call_chunk.function.arguments
                                    yield {
                                        "type": "tool_call.arguments",
                                        "id": current_id,
                                        "arguments_delta": tool_call_chunk.function.arguments,
                                    }

            # Build final tool calls - only include complete ones
            final_tool_calls = []
            for call_id, call_data in tool_calls.items():
                if call_data["name"] and call_data["arguments"]:  # Only include if we have both name and args
                    final_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call_data["name"],
                            "arguments": call_data["arguments"],
                        },
                    })

            yield {
                "type": "message.completed",
                "content": "".join(full_content),
                "tool_calls": final_tool_calls,
            }

        except Exception as e:
            # More detailed error for debugging
            import traceback
            error_detail = f"Streaming failure: {e}\nTraceback: {traceback.format_exc()}"
            yield {"type": "error", "message": error_detail}

    elif has_tools and not stream_tools:
        # Non-streaming tool behavior (original working approach)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=temperature,
                **kwargs,
            )
            message = response.choices[0].message

            # Check if there are tool calls
            if message.tool_calls:
                # Return the message content first (if any)
                if message.content:
                    yield message.content
                # Indicate that tool calls are present
                yield "\n\n🔧 **Tool calls detected** - processing..."
            else:
                # No tool calls, just return content
                content = getattr(message, "reasoning_content", None) or getattr(
                    message, "content", None
                )
                if content:
                    yield content
        except Exception as e:
            yield f"Error: {e}"

    else:
        # Regular streaming without tools (original working approach)
        try:
            response_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k not in ["tools"]},
            )

            for token_data in response_stream:
                if token_data.choices and len(token_data.choices) > 0:
                    delta = token_data.choices[0].delta
                    if delta:
                        content = getattr(delta, "reasoning_content", None) or getattr(
                            delta, "content", None
                        )
                        if content:
                            yield content
        except Exception as e:
            yield f"Error: {e}"


def get_chat_response_with_tools(
    provider: str, messages: List[Dict], model_name: Optional[str] = None, **kwargs
) -> Tuple[str, Optional[List[Dict]]]:
    """
    Get chat response that properly handles tool calls
    Returns: (content, tool_calls)
    """
    provider_key = PROVIDER_TO_RESOURCE_KEY[provider]
    if provider_key not in RESOURCES:
        raise ValueError(f"Unknown provider: {provider}")

    config = RESOURCES[provider_key]
    client = _get_client(provider)
    model = model_name or config["default_llm"]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=kwargs.get("temperature", 0.0),
        **{k: v for k, v in kwargs.items() if k != "temperature"},
    )

    message = response.choices[0].message
    content = (
        getattr(message, "reasoning_content", None)
        or getattr(message, "content", None)
        or ""
    )

    tool_calls = None
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_calls = []
        for tool_call in message.tool_calls:
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )

    return content, tool_calls


def get_chat_response(
    provider: str, messages: List[Dict], model_name: Optional[str] = None, **kwargs
) -> str:
    """Universal chat function with client caching"""
    provider_key = PROVIDER_TO_RESOURCE_KEY[provider]
    if provider_key not in RESOURCES:
        raise ValueError(f"Unknown provider: {provider}")

    config = RESOURCES[provider_key]
    client = _get_client(provider)
    model = model_name or config["default_llm"]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=kwargs.get("temperature", 0.0),
        **{k: v for k, v in kwargs.items() if k != "temperature"},
    )

    # Handle both reasoning_content and regular content like the stream version
    message = response.choices[0].message
    content = getattr(message, "reasoning_content", None) or getattr(
        message, "content", None
    )

    return content


# Optional: Clear cache function for testing/development
def clear_client_cache():
    """Clear client cache (useful for config reloads)"""
    global _client_cache
    _client_cache.clear()