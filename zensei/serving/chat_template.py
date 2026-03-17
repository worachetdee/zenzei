"""Chat template formatting for Zensei models.

Provides Japanese-aware chat template with system prompt support and
multi-turn conversation formatting.

Format:
    <|system|>
    {system_message}
    <|user|>
    {user_message}
    <|assistant|>
    {assistant_message}

Usage:
    from zensei.serving.chat_template import apply_chat_template

    messages = [
        {"role": "system", "content": "あなたは親切なAIアシスタントです。"},
        {"role": "user", "content": "こんにちは！"},
    ]
    formatted = apply_chat_template(messages)
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLE_TAGS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}

DEFAULT_SYSTEM_PROMPT = (
    "あなたはZensei（禅精）という名前の日本語AIアシスタントです。"
    "丁寧で正確な日本語で回答してください。"
    "ユーザーの質問に対して、分かりやすく役立つ情報を提供します。"
)


# ---------------------------------------------------------------------------
# Template formatting
# ---------------------------------------------------------------------------


def format_message(role: str, content: str) -> str:
    """Format a single message with its role tag.

    Args:
        role: One of "system", "user", or "assistant".
        content: The message content.

    Returns:
        Formatted string with role tag and content.

    Raises:
        ValueError: If role is not recognized.
    """
    if role not in ROLE_TAGS:
        raise ValueError(
            f"Unknown role '{role}'. Expected one of: {list(ROLE_TAGS.keys())}"
        )
    tag = ROLE_TAGS[role]
    return f"{tag}\n{content}"


def apply_chat_template(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
    system_prompt: Optional[str] = None,
) -> str:
    """Apply the Zensei chat template to a list of messages.

    If no system message is present in the messages and ``system_prompt`` is
    provided (or the default is used), a system message is prepended
    automatically.

    Args:
        messages: List of message dicts with "role" and "content" keys.
            Roles can be "system", "user", or "assistant".
        add_generation_prompt: If True, append the assistant tag at the end
            to prompt the model to generate a response.
        system_prompt: Custom system prompt. If None and no system message
            exists in messages, the default Japanese system prompt is used.
            Pass an empty string to suppress the system prompt entirely.

    Returns:
        Formatted conversation string ready for tokenization.

    Examples:
        >>> msgs = [{"role": "user", "content": "こんにちは"}]
        >>> text = apply_chat_template(msgs)
        >>> print(text)
        <|system|>
        あなたはZensei（禅精）という名前の日本語AIアシスタントです。...
        <|user|>
        こんにちは
        <|assistant|>
    """
    parts: list[str] = []

    # Check if a system message already exists
    has_system = any(m["role"] == "system" for m in messages)

    # Prepend system prompt if needed
    if not has_system:
        sys_content = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        if sys_content:  # Skip if explicitly set to empty string
            parts.append(format_message("system", sys_content))

    # Format each message
    for message in messages:
        role = message["role"]
        content = message["content"]
        parts.append(format_message(role, content))

    # Optionally add generation prompt
    if add_generation_prompt:
        parts.append(ROLE_TAGS["assistant"])

    return "\n".join(parts) + "\n"


def apply_chat_template_tokenized(
    messages: list[dict[str, str]],
    tokenizer,
    add_generation_prompt: bool = True,
    system_prompt: Optional[str] = None,
) -> list[int]:
    """Apply the chat template and tokenize the result.

    Args:
        messages: List of message dicts with "role" and "content" keys.
        tokenizer: A HuggingFace tokenizer instance.
        add_generation_prompt: If True, append the assistant tag.
        system_prompt: Custom system prompt override.

    Returns:
        List of token IDs.
    """
    text = apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        system_prompt=system_prompt,
    )
    return tokenizer.encode(text)


def parse_chat_messages(text: str) -> list[dict[str, str]]:
    """Parse a formatted chat string back into a list of messages.

    This is the inverse of ``apply_chat_template``: it splits the text on
    role tags and returns structured messages.

    Args:
        text: Formatted chat text produced by apply_chat_template.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
    messages: list[dict[str, str]] = []

    # Build reverse mapping: tag -> role
    tag_to_role = {tag: role for role, tag in ROLE_TAGS.items()}
    all_tags = list(ROLE_TAGS.values())

    # Split the text by role tags
    current_role: Optional[str] = None
    current_content: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in all_tags:
            # Save previous message
            if current_role is not None:
                content = "\n".join(current_content).strip()
                if content:
                    messages.append({"role": current_role, "content": content})
            # Start new message
            current_role = tag_to_role[stripped]
            current_content = []
        else:
            current_content.append(line)

    # Save final message
    if current_role is not None:
        content = "\n".join(current_content).strip()
        if content:
            messages.append({"role": current_role, "content": content})

    return messages
