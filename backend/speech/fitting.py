"""Fit every dubbed line to the duration of the original line.

For each line the natural TTS duration d is compared with the original slot s.
A line whose ratio d / s falls outside [FIT_MIN_SPEED, FIT_MAX_SPEED] is
rewritten to a target length (fewer or more characters at the measured
speaking rate of its voice) and voiced again; the version closest to the slot
is kept. The timeline then stretches each line by exactly d / s, so it starts
and ends with the original speaker.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from backend.providers import openrouter
from backend.media.audio_merge import natural_durations
from backend.config import settings
from backend.models import LANGUAGES
from backend.media.timeline import SHORT_TAIL_SECONDS
from backend.voices.library import speaking_rate

# Characters per second of the Bodhan Indic-Speak voices (silence trimmed), measured on studio projects.
# Used to size lines before the first (paid) voicing; the fitting loop then works from measured rates.
SPEAKING_RATE = {"hi": 12.4}
DEFAULT_SPEAKING_RATE = 12.5
PREFIT_TOLERANCE = 0.06  # predicted pace within 1 +- this is voiced as is

logger = logging.getLogger(__name__)

RETIME_PROMPT = """You adapt a {language} dubbing script so that each line fits the time its original speaker took.
For each line you get: the original {source} line (the meaning to keep), its current {language} translation, the current length in characters, the TARGET length, and the neighbouring lines for context.
Rewrite the {language} line to come as close to the target length as natural speech allows:
- the meaning of the original line must stay exactly the same: never drop what the line says, never add facts, names, adjectives or emphasis that are not in the original;
- never pad with filler words (such as वास्तव में, दरअसल, बिल्कुल, सच में, यहाँ) and never cut a line down to a fragment that no longer says what the original says;
- to shorten: choose shorter words and constructions, drop only true redundancy; numbers, symbols and formulas stay in the spoken-word form of the current line (W one transpose, lambda one, sum over i from one to n): never shorten a line into bare symbols and never glue a letter to a digit;
- to lengthen: only fuller natural forms of the same words (complete verb forms, वह instead of वो) or a more complete phrasing of the same meaning;
- when a sentence runs across consecutive lines of one speaker, keep the pieces in order so that together they still form one correct sentence;
- if the line cannot reach the target without breaking these rules, get as close as the rules allow, or return it unchanged.
Write {language} in the same script as the current translation; keep the commas and sentence punctuation that mark the voice's pauses (add or drop one only when the phrasing changes), and keep every word that is in Roman letters in Roman letters. Never merge, split or skip lines.
Respond with JSON only: {{"lines": [{{"id": <number>, "text": "<rewritten line>"}}, ...]}}"""

VERIFY_PROMPT = """You are the quality check of a {language} dubbing script. Each line was rewritten to fit its timing.
For each item you get the original {source} line, the translation before the rewrite, the proposed rewrite, and the neighbouring lines (with their translations).
First read the whole sentence the line belongs to: the previous lines, the proposed line and the next lines, in order (a sentence often runs over several short lines, and {language} word order differs from {source}).
Accept a rewrite only if that sentence, with the proposed line in it, says exactly what the original lines say: nothing lost that matters (who does what to whom, the verb, negation, tense, names), nothing added, no filler words, natural {language}. Watch for a changed grammatical role: a dropped verb can turn a noun phrase into a different verb with a different meaning.
English words written in Roman letters stay English: reject a rewrite that translates or transliterates such a term into {language} script. Dropping a redundant English word (right, so, a set of) is fine when the sentence keeps its meaning.
Respond with JSON only: {{"lines": [{{"id": <number>, "ok": true or false}}, ...]}}"""

ABSOLUTE_TOLERANCE = (
    0.1  # seconds: a line within this of its slot fits (a one-word line cannot be rephrased)
)
SHORT_LINE_WORDS = (
    3  # lines this short are never lengthened (only padding could do it): they play slower
)


def may_lengthen(segment) -> bool:
    return len((segment.text or segment.translation).split()) > SHORT_LINE_WORDS


def may_shorten(segment) -> bool:
    """A single word (a name, an exclamation) has nothing to cut: it is stretched instead."""
    return len((segment.text or segment.translation).split()) > 1


@dataclass
class FitReport:
    """Summarize fitting rounds, accepted rewrites, and lines still outside the band."""

    rounds: int = 0
    rewritten: int = 0
    in_band: int = 0
    total: int = 0
    outside: list = field(default_factory=list)


def _parse(content: str, expected: set[int]) -> dict[int, str]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S)
    data = json.loads(text[text.find("{") : text.rfind("}") + 1])
    out = {}
    for item in data.get("lines", []):
        try:
            identifier, value = int(item["id"]), str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if identifier in expected and value:
            out[identifier] = value
    return out


def _request(index: int, segment, target: int, ordered: list) -> dict:
    """One rewrite request with the original line and its neighbours for meaning and sentence context."""
    position = next((k for k, other in enumerate(ordered) if other is segment), None)

    def neighbours(offsets):
        lines = []
        for offset in offsets:
            if position is not None and 0 <= position + offset < len(ordered):
                other = ordered[position + offset]
                lines.append(f"[{other.speaker}: {other.text} => {other.translation}]")
        return " ".join(lines)

    # two lines each side: a sentence often runs over several short lines of one speaker
    return {
        "id": index,
        "text": segment.translation,
        "source": segment.text or "",
        "target_chars": target,
        "before": neighbours((-2, -1)),
        "after": neighbours((1, 2)),
    }


def retime_lines(
    requests: list[dict], language: str, usage=None, source_language: str | None = None
) -> dict[int, str]:
    """[{id, text, source, target_chars, before, after}] -> {id: rewritten text}. Skipped lines are absent."""
    if not requests or not openrouter.configured():
        return {}
    system = RETIME_PROMPT.format(
        language=LANGUAGES.get(language, language),
        source=LANGUAGES.get(source_language or "", "source"),
    )
    user = "Lines:\n" + "\n\n".join(
        f'id {r["id"]}: original "{r.get("source", "")}"\n  current ({len(r["text"])} chars -> TARGET {r["target_chars"]}): '
        f'{r["text"]}\n  before: {r.get("before") or "-"}\n  after: {r.get("after") or "-"}'
        for r in requests
    )
    models = [settings.openrouter_model] + [
        m for m in settings.openrouter_fallback_models if m != settings.openrouter_model
    ]
    for model in models:
        try:
            content = openrouter.chat(
                model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                usage=usage,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            return _parse(content, {r["id"] for r in requests})
        except (RuntimeError, ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Retiming with %s failed: %s", model, exc)
    return {}


def verify_rewrites(
    items: list[dict], language: str, usage=None, source_language: str | None = None
) -> set[int]:
    """[{id, source, before, after, context_before, context_after}] -> ids whose rewrite keeps the meaning.
    Rejected rewrites are never voiced. If the check cannot run, every rewrite is accepted."""
    if not items or not openrouter.configured():
        return {item["id"] for item in items}
    system = VERIFY_PROMPT.format(
        language=LANGUAGES.get(language, language),
        source=LANGUAGES.get(source_language or "", "source"),
    )
    user = "Items:\n" + "\n\n".join(
        f'id {i["id"]}: original "{i["source"]}"\n  before rewrite: {i["before"]}\n  proposed: {i["after"]}\n'
        f'  previous lines: {i.get("context_before") or "-"}\n  next lines: {i.get("context_after") or "-"}'
        for i in items
    )
    models = [settings.openrouter_model] + [
        m for m in settings.openrouter_fallback_models if m != settings.openrouter_model
    ]
    for model in models:
        try:
            content = openrouter.chat(
                model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                usage=usage,
                temperature=0,
                response_format={"type": "json_object"},
            )
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S)
            data = json.loads(text[text.find("{") : text.rfind("}") + 1])
            verdicts = {
                int(item["id"]): bool(item.get("ok"))
                for item in data.get("lines", [])
                if "id" in item
            }
            return {item["id"] for item in items if verdicts.get(item["id"], False)}
        except Exception as exc:
            logger.warning("Rewrite check with %s failed: %s", model, exc)
    return {item["id"] for item in items}


ROMAN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def drops_roman_terms(before: str, after: str) -> bool:
    """True when a rewrite lost what the code-mixed script must keep in Roman letters: a name, acronym or figure
    (a capital letter or a digit), or most of the line's English words (the line was translated out of its style).
    Condensing may drop a redundant ordinary English word ("right", "set of"); the meaning check judges those.
    """
    kept = {t.lower() for t in ROMAN_TOKEN.findall(after)}
    terms = [
        t for t in ROMAN_TOKEN.findall(before) if len(t) > 1 and len(set(t.lower())) > 1
    ]  # words, not fixtures like "yyyy"
    dropped = [t for t in terms if t.lower() not in kept]
    if any(t != t.lower() or any(c.isdigit() for c in t) for t in dropped):
        return True
    return 2 * len(dropped) > len(terms)


def _checked(
    rewritten: dict[int, str], requests: list[dict], language: str, usage, source_language
) -> dict[int, str]:
    """Keep only rewrites that change the line, keep its Roman-letter terms, and pass the meaning check."""
    by_id = {r["id"]: r for r in requests}
    rewritten = {
        i: text
        for i, text in rewritten.items()
        if i in by_id and not drops_roman_terms(by_id[i]["text"], text)
    }
    items = [
        {
            "id": i,
            "source": by_id[i]["source"],
            "before": by_id[i]["text"],
            "after": text,
            "context_before": by_id[i].get("before"),
            "context_after": by_id[i].get("after"),
        }
        for i, text in rewritten.items()
        if i in by_id and text != by_id[i]["text"]
    ]
    accepted = verify_rewrites(items, language, usage, source_language)
    return {i: text for i, text in rewritten.items() if i in accepted}


def acceptable(ratio: float, slot: float, low: float, high: float) -> bool:
    """In the band, within 0.1 s of the slot, or slightly short so that the 0.9x floor leaves only a brief
    pause (see timeline)."""
    if low <= ratio <= high or abs(ratio - 1.0) * slot <= ABSOLUTE_TOLERANCE:
        return True
    return ratio < low and slot - ratio * slot / low <= max(SHORT_TAIL_SECONDS, 0.05 * slot)


def prefit_lines(
    lines, language: str, usage=None, source_language: str | None = None, context=None
) -> int:
    """Before the first voicing, rewrite lines whose predicted spoken length misses their slot.

    Rewriting text (OpenRouter) is far cheaper than voicing it again (Bodhan bills TTS per character),
    so lines are sized from the voice's speaking rate first. Lines the user edited keep their words.
    """
    requests, targets = [], {}
    for index, segment in enumerate(lines):
        rate = (
            speaking_rate(segment.voice, SPEAKING_RATE.get(language, DEFAULT_SPEAKING_RATE))
            if segment.voice
            else SPEAKING_RATE.get(language, DEFAULT_SPEAKING_RATE)
        )
        slot = max(0.05, segment.end - segment.start)
        predicted = len(segment.translation) / rate / slot
        if (
            segment.edited
            or not segment.translation
            or abs(predicted - 1.0) <= PREFIT_TOLERANCE
            or abs(predicted - 1.0) * slot <= ABSOLUTE_TOLERANCE
            or not (may_lengthen(segment) if predicted < 1.0 else may_shorten(segment))
        ):
            continue
        targets[index] = max(4, round(rate * slot))
        requests.append(
            _request(
                index, segment, targets[index], sorted(context or lines, key=lambda s: s.start)
            )
        )
    rewritten = _checked(
        retime_lines(requests, language, usage, source_language),
        requests,
        language,
        usage,
        source_language,
    )
    changed = 0
    for index, text in rewritten.items():
        segment = lines[index]
        # keep the rewrite only if it moved the length towards the target
        if text != segment.translation and abs(len(text) - targets[index]) < abs(
            len(segment.translation) - targets[index]
        ):
            segment.translation = text
            segment.condensed = True
            changed += 1
    return changed


def fit_lines(
    lines,
    directory: Path,
    language: str,
    render,
    workers: int,
    run_stage,
    control=None,
    progress=None,
    usage=None,
    source_language: str | None = None,
) -> FitReport:
    """`lines`: segments with audio_path/translation; `render(segment)` voices segment.translation into audio_path."""
    low, high = settings.fit_min_speed, settings.fit_max_speed
    report = FitReport(total=len(lines))
    ordered = sorted(lines, key=lambda s: s.start)
    best: dict[int, tuple[float, str]] = {}  # id(segment) -> (distance, text)

    def measure() -> dict[int, float]:
        durations = natural_durations(ordered, directory)
        return {
            id(ordered[i]): d / max(0.05, ordered[i].end - ordered[i].start)
            for i, d in durations.items()
        }

    slot_of = {id(s): max(0.05, s.end - s.start) for s in ordered}

    def distance(ratio: float, segment) -> float:
        if acceptable(ratio, slot_of[id(segment)], low, high):
            return 0.0
        return low - ratio if ratio < low else ratio - high

    ratios = measure()
    for segment in ordered:
        if id(segment) in ratios:
            best[id(segment)] = (distance(ratios[id(segment)], segment), segment.translation)
    for round_index in range(settings.fit_rounds):
        # Lines the user wrote in the script editor keep their words: they are only stretched (and reported).
        outside = [
            s
            for s in ordered
            if id(s) in ratios
            and distance(ratios[id(s)], s) > 0
            and not s.edited
            and (may_shorten(s) if ratios[id(s)] > high else may_lengthen(s))
        ]
        if not outside:
            break
        if control:
            control.check()
        if progress:
            progress(round_index, len(outside))
        # Characters per second of each voice, from all of its lines: steadier than one short line alone.
        rates: dict[str, list[float]] = {}
        for segment in ordered:
            if id(segment) in ratios:
                seconds = ratios[id(segment)] * max(0.05, segment.end - segment.start)
                rates.setdefault(segment.speaker, []).append(
                    len(segment.translation) / max(0.1, seconds)
                )
        requests = []
        for index, segment in enumerate(outside):
            slot = max(0.05, segment.end - segment.start)
            own_rate = len(segment.translation) / max(0.1, ratios[id(segment)] * slot)
            voice_rate = sorted(rates.get(segment.speaker, [own_rate]))[
                len(rates.get(segment.speaker, [own_rate])) // 2
            ]
            weight = min(1.0, slot / 8.0)  # long lines trust their own rate, short ones the voice's
            rate = own_rate * weight + voice_rate * (1 - weight)
            # Aim for the middle of the band.
            target = max(4, round(rate * slot * (low + high) / 2))
            requests.append(_request(index, segment, target, ordered))
        proposals = retime_lines(requests, language, usage, source_language)
        if not proposals:
            break
        rewritten = _checked(proposals, requests, language, usage, source_language)
        report.rounds += 1
        changed = []
        for index, segment in enumerate(outside):
            text = rewritten.get(index)
            if text and text != segment.translation:
                shutil.copyfile(segment.audio_path, segment.audio_path + ".best")
                segment.translation = text
                changed.append(segment)
        run_stage(changed, lambda _, segment: render(segment), workers, lambda count: None, control)
        new_ratios = measure()
        for segment in changed:
            previous_distance, previous_text = best[id(segment)]
            new_distance = distance(new_ratios[id(segment)], segment)
            if new_distance <= previous_distance:
                best[id(segment)] = (new_distance, segment.translation)
                segment.condensed = True
                report.rewritten += 1
                Path(segment.audio_path + ".best").unlink(missing_ok=True)
            else:  # the rewrite moved away from the slot: keep the better take
                shutil.move(segment.audio_path + ".best", segment.audio_path)
                Path(segment.audio_path).touch()  # newer than any cached trim of the rejected take
                segment.translation = previous_text
                segment.voiced_text = previous_text
                new_ratios[id(segment)] = ratios[id(segment)]
        ratios = new_ratios
    for segment in ordered:
        ratio = ratios.get(id(segment))
        if ratio is None:
            continue
        if acceptable(ratio, slot_of[id(segment)], low, high):
            report.in_band += 1
        else:
            report.outside.append((segment, ratio))
    return report
