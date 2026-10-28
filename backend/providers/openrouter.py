"""Small OpenRouter client: chat, speech-to-text and text-to-speech."""

from __future__ import annotations

import httpx

from backend.config import settings

BASE = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    """An OpenRouter request failed or answered with an error payload."""

    pass


def configured() -> bool:
    """True when an OpenRouter API key is set."""
    return bool(settings.openrouter_api_key)


def covers_bodhan() -> bool:
    """OpenRouter may take over transcription, translation or voices when Bodhan cannot answer."""
    return configured() and not settings.bodhan_only


def headers() -> dict:
    """Request headers: the API key plus OpenRouter's optional app attribution."""
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dubloom",
        "X-Title": "Dubloom Studio",
    }


def _raise(response: httpx.Response):
    """Raise OpenRouterError carrying the error message of a failed response."""
    try:
        detail = response.json().get("error", {})
        message = detail.get("message") if isinstance(detail, dict) else str(detail)
    except ValueError:
        message = response.text
    raise OpenRouterError(f"OpenRouter ({response.status_code}): {str(message)[:240]}")


def post_json(path: str, body: dict, timeout: float = 180) -> dict:
    """POST a JSON body to an OpenRouter endpoint and return the decoded JSON reply.

    `timeout` bounds the read in seconds. Raises OpenRouterError on an HTTP error or an error
    payload.
    """
    with httpx.Client(timeout=httpx.Timeout(30, read=timeout)) as client:
        response = client.post(BASE + path, headers=headers(), json=body)
    if response.is_error:
        _raise(response)
    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        raise OpenRouterError(f"OpenRouter: {str(data['error'])[:240]}")
    return data


def post_audio(path: str, body: dict, timeout: float = 180) -> tuple[bytes, str]:
    """POST a JSON body and return the raw reply bytes and their content type (audio endpoints)."""
    with httpx.Client(timeout=httpx.Timeout(30, read=timeout)) as client:
        response = client.post(BASE + path, headers=headers(), json=body)
    if response.is_error:
        _raise(response)
    return response.content, response.headers.get("content-type", "")


def chat(model: str, messages: list[dict], usage=None, **options) -> str:
    """Send a chat completion and return the reply text, stripped.

    Google models default to minimal reasoning effort. The reported cost in USD is added to
    `usage` (a Usage) when given; extra `options` go into the request body.
    """
    body = {"model": model, "messages": messages, **options}
    if model.startswith("google/"):
        body.setdefault("reasoning", {"effort": "minimal"})
    data = post_json("/chat/completions", body)
    if usage is not None:
        usage.add("openrouter_cost_usd", (data.get("usage") or {}).get("cost") or 0)
    content = data["choices"][0]["message"].get("content")
    if isinstance(content, list):
        content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return (content or "").strip()
