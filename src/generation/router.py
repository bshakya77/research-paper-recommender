"""Primary Groq generation with Gemini fallback for retryable service failures."""

import logging
import sys
from dataclasses import dataclass

import groq
from groq import Groq

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    provider: str
    model: str


def _is_retryable_groq_error(error: Exception) -> bool:
    if isinstance(
        error,
        (groq.APIConnectionError, groq.APITimeoutError, groq.RateLimitError),
    ):
        return True
    if isinstance(error, groq.APIStatusError):
        status_code = error.status_code
        if status_code == 404:
            body = getattr(error, "body", None)
            if isinstance(body, dict):
                details = body.get("error", {})
                if isinstance(details, dict) and details.get("code") == "model_not_found":
                    return True
            return "model_not_found" in str(error).lower()
        return status_code in {408, 409, 424, 429, 498} or status_code >= 500
    return False


def _generate_with_groq(messages: list[dict[str, str]]) -> str:
    client = Groq(api_key=settings.groq_api_key)
    completion = client.chat.completions.create(
        model=settings.groq_model_id,
        temperature=0.2,
        max_tokens=1400,
        messages=messages,
    )
    text = completion.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError("Groq returned an empty response")
    return text.strip()


def _generate_with_gemini(messages: list[dict[str, str]]) -> str:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "Groq had a retryable service failure, but Gemini fallback is not configured. "
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            f"Gemini SDK import failed in {sys.executable}: {exc}. "
            "Install it into this same Python environment with "
            "python -m pip install google-genai."
        ) from exc

    system_instruction = "\n\n".join(
        message["content"] for message in messages if message["role"] == "system"
    )
    contents = [
        types.Content(
            role="model" if message["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=message["content"])],
        )
        for message in messages
        if message["role"] != "system"
    ]

    client = genai.Client(api_key=settings.gemini_api_key)
    try:
        response = client.models.generate_content(
            model=settings.gemini_model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction or None,
                temperature=0.2,
                max_output_tokens=1400,
            ),
        )
    finally:
        client.close()

    text = response.text
    if not text or not text.strip():
        raise RuntimeError("Gemini fallback returned an empty response")
    return text.strip()


def generate_messages(messages: list[dict[str, str]]) -> GenerationResponse:
    """Generate from chat messages, preferring Groq with Gemini fallback."""
    if not settings.groq_api_key:
        if not settings.gemini_api_key:
            raise RuntimeError(
                "Neither GROQ_API_KEY nor GEMINI_API_KEY (or GOOGLE_API_KEY) is configured."
            )
        answer = _generate_with_gemini(messages)
        return GenerationResponse(answer, "gemini", settings.gemini_model_id)

    try:
        answer = _generate_with_groq(messages)
        return GenerationResponse(answer, "groq", settings.groq_model_id)
    except Exception as exc:
        if not _is_retryable_groq_error(exc):
            raise
        logger.warning(
            "Groq request failed with %s (status %s); falling back to Gemini",
            type(exc).__name__,
            getattr(exc, "status_code", "unavailable"),
        )
        answer = _generate_with_gemini(messages)
        return GenerationResponse(answer, "gemini", settings.gemini_model_id)


def generate(system_prompt: str, user_prompt: str) -> GenerationResponse:
    """Single-turn convenience wrapper used by paper analysis."""
    return generate_messages([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
