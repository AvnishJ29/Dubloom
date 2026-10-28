"""Bodhan Indic-Translate.

`POST /translate` is the documented product surface; the OpenAI-style
`/v1/chat/completions` transport returns identical output and is used when the
product route is not deployed. `script` may be `native` (default) or `codemix`,
which keeps the native script but leaves common English words in English.
"""

from __future__ import annotations

from backend.providers.bodhan_api import BodhanAPI
from backend.config import settings

SCRIPTS = {"native", "codemix", "roman"}


def translate(
    api: BodhanAPI,
    text: str,
    language: str,
    script: str = "native",
    source_language: str | None = None,
    usage=None,
) -> str:
    """Translate with Bodhan, using the chat transport when the product route is absent."""
    script = script if script in SCRIPTS else "native"
    use_script = script != "native" and language != "en"
    body = {"text": text, "target_language": language}
    if use_script:
        body["script"] = script
    response = api.post("/translate", settings.bodhan_translate_key, allow_404=True, json=body)
    if response.status_code != 404:
        data = response.json()
        if data.get("truncated"):
            raise RuntimeError("Bodhan translation was truncated")
        if usage is not None:
            usage.add(
                "bodhan_translate_tokens", (data.get("usage") or {}).get("completion_tokens", 0)
            )
        return data["translation"].strip()
    chat = {
        "model": "indic-translate",
        "messages": [{"role": "user", "content": text}],
        "target_language_code": language,
    }
    if source_language:
        chat["source_language_code"] = source_language
    if use_script:
        chat["script"] = script
    try:
        response = api.post("/v1/chat/completions", settings.bodhan_translate_key, json=chat)
    except RuntimeError as exc:
        # The chat transport may not accept every optional field; retry with the minimum.
        if "(400)" not in str(exc) and "(422)" not in str(exc) or len(chat) == 3:
            raise
        response = api.post(
            "/v1/chat/completions",
            settings.bodhan_translate_key,
            json={key: chat[key] for key in ("model", "messages", "target_language_code")},
        )
    data = response.json()
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("Bodhan translation was truncated")
    if usage is not None:
        usage.add("bodhan_translate_tokens", (data.get("usage") or {}).get("completion_tokens", 0))
    return choice["message"]["content"].strip()
