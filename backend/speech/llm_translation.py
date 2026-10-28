"""Dubbing translation through an OpenRouter chat model, with spoken-length budgets.

Whole scenes are translated together so pronouns, honorifics and gendered verb
forms stay consistent per speaker. Every line carries its time slot and a word
budget. `condense_lines` shortens lines that still run long after voicing.
"""

from __future__ import annotations

import json
import logging
import re

from backend.providers import openrouter
from backend.config import settings
from backend.models import LANGUAGES

logger = logging.getLogger(__name__)
BATCH_LINES = 40
WORDS_PER_SECOND = {
    "hi": 2.5,
    "ur": 2.5,
    "pa": 2.5,
    "en": 2.6,
    "bn": 2.3,
    "gu": 2.3,
    "ne": 2.3,
    "as": 2.2,
    "or": 2.2,
    "mr": 2.1,
    "te": 1.9,
    "ta": 1.8,
    "kn": 1.8,
    "ml": 1.7,
}

SYSTEM_PROMPT = """You are a senior dubbing translator and adaptation writer.
Translate every line into {language} for a voice dub of the same video.
Rules:
- Each line has a time slot and a WORD BUDGET. The translation must be speakable within the slot at a natural pace, so stay within the budget (never more than 10 % over). Dubbing adaptation means saying the same thing more economically: drop filler and repetition, prefer short idiomatic spoken phrasing, keep every fact.
- Preserve meaning, tone, humour and register. Use natural spoken {language} in its native script; keep names, brands, acronyms and numbers.{style}
- Speaker labels include a likely gender; use it for gendered verb and adjective agreement and consistent honorifics. Keep the same person's style consistent across lines.
- Never merge, split, reorder or skip lines. Never add commentary, transliteration, or notes.
Respond with JSON only: {{"lines": [{{"id": <number>, "text": "<translation>"}}, ...]}}"""

CONDENSE_PROMPT = """You are a dubbing adaptation editor working in {language}.
Each line below is too long to be spoken in its time slot. Rewrite each line in {language} (native script) so that it is about the given percentage SHORTER when spoken, keeping the meaning, the key facts, names and numbers, and the speaker's tone. Cut filler, merge clauses, choose shorter words. Never translate into another language, never drop a line.
Respond with JSON only: {{"lines": [{{"id": <number>, "text": "<shorter line>"}}, ...]}}"""


def word_budget(seconds: float, language: str) -> int:
    return max(2, round(max(0.6, seconds) * WORDS_PER_SECOND.get(language, 2.1)))


def _label(speaker: str, profiles: dict) -> str:
    match = re.search(r"(\d+)$", speaker or "")
    name = f"Speaker {int(match.group(1)) + 1}" if match else (speaker or "Speaker")
    gender = (profiles or {}).get(speaker, {}).get("gender")
    return f"{name} ({gender})" if gender in {"male", "female"} else name


def _parse(content: str, expected: set[int]) -> dict[int, str]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start : end + 1])
    lines = data.get("lines") if isinstance(data, dict) else data
    result = {}
    for item in lines or []:
        try:
            identifier, value = int(item["id"]), str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if identifier in expected and value:
            result[identifier] = value
    missing = expected - result.keys()
    if missing:
        raise RuntimeError(f"The model skipped {len(missing)} lines")
    return result


def _models() -> list[str]:
    return [settings.openrouter_model] + [
        m for m in settings.openrouter_fallback_models if m != settings.openrouter_model
    ]


def _ask(
    messages: list[dict], expected: set[int], usage=None, temperature: float = 0.2
) -> tuple[dict[int, str], str]:
    last_error: Exception | None = None
    for model in _models():
        for attempt in range(2):
            try:
                content = openrouter.chat(
                    model,
                    messages,
                    usage=usage,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                return _parse(content, expected), model
            except (RuntimeError, ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
                last_error = exc
                logger.warning("LLM request attempt %s with %s failed: %s", attempt + 1, model, exc)
    raise RuntimeError(f"OpenRouter translation failed: {last_error}")


def translate_lines(
    lines: list[dict],
    language: str,
    profiles: dict | None = None,
    progress=None,
    script: str = "native",
    usage=None,
) -> tuple[dict[int, str], str]:
    """Translate [{id, speaker, text, seconds}] -> ({id: translation}, model used)."""
    if language not in LANGUAGES:
        raise ValueError("Unsupported target language")
    style = (
        " Write natural code-mixed speech: keep common English technical words, brand names and everyday "
        "English terms in English (Latin script) the way educated speakers actually talk."
        if script == "codemix"
        else ""
    )
    system = SYSTEM_PROMPT.format(language=LANGUAGES[language], style=style)
    translations: dict[int, str] = {}
    used = settings.openrouter_model
    batches = [lines[i : i + BATCH_LINES] for i in range(0, len(lines), BATCH_LINES)]
    for index, batch in enumerate(batches):
        script_text = "\n".join(
            f'{item["id"]} | {_label(item["speaker"], profiles)} | {item["seconds"]:.1f}s | max {word_budget(item["seconds"], language)} words | {item["text"]}'
            for item in batch
        )
        context = ""
        if index:
            recent = [item for item in batches[index - 1][-6:] if item["id"] in translations]
            context = (
                "Previous lines (already translated, for continuity):\n"
                + "\n".join(
                    f'{item["id"]} | {_label(item["speaker"], profiles)} | {item["text"]} => {translations[item["id"]]}'
                    for item in recent
                )
                + "\n\n"
            )
        user = f"{context}Translate these {len(batch)} lines (format: id | speaker | slot | word budget | text):\n{script_text}"
        result, used = _ask(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            {item["id"] for item in batch},
            usage,
        )
        translations.update(result)
        if progress:
            progress(index + 1, len(batches))
    return translations, used


def condense_lines(lines: list[dict], language: str, usage=None) -> tuple[dict[int, str], str]:
    """Shorten [{id, text, percent}] where percent is how much shorter the spoken line must be."""
    if not lines:
        return {}, settings.openrouter_model
    system = CONDENSE_PROMPT.format(language=LANGUAGES.get(language, language))
    result: dict[int, str] = {}
    used = settings.openrouter_model
    for start in range(0, len(lines), BATCH_LINES):
        batch = lines[start : start + BATCH_LINES]
        user = "Shorten these lines (format: id | percent shorter | text):\n" + "\n".join(
            f'{item["id"]} | {int(item["percent"])}% | {item["text"]}' for item in batch
        )
        got, used = _ask(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            {item["id"] for item in batch},
            usage,
            temperature=0.3,
        )
        result.update(got)
    return result, used
