"""Convert LangChain text messages to the provider-neutral chat format."""

from langchain_core.messages import BaseMessage


def to_provider_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }
    converted = []
    for message in messages:
        role = role_map.get(message.type)
        if role is None:
            continue
        content = message.content
        if not isinstance(content, str):
            text_parts = [
                part["text"] for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ] if isinstance(content, list) else []
            content = "\n".join(text_parts)
        converted.append({"role": role, "content": content})
    return converted
