"""Script supervision around Bodhan ASR and translation, as a dubbing script editor would do it.

1. Transcript check: Bodhan's transcript of each line is compared with an
   independent transcript (OpenRouter, made during the speaker review) and the
   whole dialogue; misheard names and similar-sounding words are corrected.
2. Translation post-edit: Bodhan's line-by-line translation is read as a scene;
   only real errors are fixed (meaning, names kept as names, sentences that run
   across several lines, overly literal phrasing).

Both run before any voicing, so the paid voices are generated once, on the
final text. Lines the user edited are never touched.
"""

from __future__ import annotations

import difflib
import json
import logging
import re

from backend.providers import openrouter
from backend.config import settings
from backend.models import LANGUAGES

logger = logging.getLogger(__name__)

TRANSCRIPT_PROMPT = """You check the transcript of a video's dialogue before it is translated for dubbing.
Each line has two automatic transcripts of the same audio, A and B (identical when only one was made). Either can mishear names and similar-sounding words.
Using both and the context of the whole dialogue, write what was most likely actually said, in {source}. Choose words that make sense in context and keep names consistent across lines.
Do not add words neither transcript supports, do not rephrase, do not complete a sentence that the line itself leaves unfinished.
Punctuate as it was spoken: a comma at each natural pause, a question mark for questions, a full stop at sentence ends; normal capitalisation, with names, brands and acronyms spelled correctly and consistently.
Write symbols, variables and formulas the way they were said aloud (u sub t, beta times u t minus one, x squared), never as notation.
Respond with JSON only: {{"lines": [{{"id": 0, "text": "..."}}]}}"""

POSTEDIT_PROMPT = """You are the script editor of a dub from {source} into {target}.
You get every line in order with its speaker, the source text exactly as recognised, a machine translation and, where one exists, a second
independent recognition of the same audio ("also heard as"). The source text is never yours to change: when the two recognitions
differ or a word is clearly misheard, decide from context what was actually said and translate that. Fix only real problems in the translation:
- a wrong meaning or word sense (use the context of the whole scene; idioms and figurative language must keep their intended meaning);
- names: a name stays a name, never translated as an ordinary word (its spelling follows the script rule below); a name, product, version or figure is kept exactly as recognised (Sarvam M stays Sarvam M), never "corrected" from your own knowledge;
- punctuation is the voice's breathing: the text-to-speech voice pauses on commas and stops at sentence ends, so put a comma wherever the speaker would pause briefly (between clauses, after a discourse marker or a name being addressed, before a contrast), a full stop or danda at every sentence end and a question mark for questions; no comma where speech would not pause;
- a sentence spoken across several consecutive lines of one speaker: translate those lines so that, heard in order, they form one natural {target} sentence while each line keeps its own part;
- stiff, overly literal or unnatural phrasing for spoken dialogue;
- hesitation sounds and false starts (uh, um, er, a word repeated while thinking) are dropped: the voice must not read them. Deliberate discourse words (right, okay, so) stay;
- symbols, variables and formulas are written the way they are said aloud (u sub t, beta times u t minus one), never as notation.
Keep the speaker's tone, stance and facts: this is a dub, not a rewrite. Change as little as the rules above require, never add or drop information, and when the machine translation is already right and fits, return it unchanged. {conservative} {length} {content} {script}
Respond with JSON only: {{"lines": [{{"id": 0, "text": "..."}}]}}"""


def _parse(content: str, expected: set[int]) -> dict[int, str]:
    """{id: text} for the expected ids found in a JSON reply; other items are ignored."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S)
    data = json.loads(text[text.find("{") : text.rfind("}") + 1])
    result = {}
    for item in data.get("lines", []):
        try:
            identifier, value = int(item["id"]), str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if identifier in expected and value:
            result[identifier] = value
    return result


def _ask(system: str, user: str, expected: set[int], usage) -> dict[int, str]:
    """{id: text} from the first model (main, then fallbacks) that answers; {} when all fail."""
    models = [settings.openrouter_model] + [
        m for m in settings.openrouter_fallback_models if m != settings.openrouter_model
    ]
    for model in models:
        try:
            content = openrouter.chat(
                model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                usage=usage,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return _parse(content, expected)
        except Exception as exc:
            logger.warning("Script supervision with %s failed: %s", model, exc)
    return {}


def check_transcripts(segments, source_language: str | None, usage=None, only=None) -> int:
    """Correct Bodhan transcripts against the independent review transcript. Returns lines changed.
    `only`: the segments that may change (new in this run); all lines are given as context."""
    lines = [
        (i, s)
        for i, s in enumerate(segments)
        if s.text and (only is None or any(s is o for o in only))
    ]
    if not lines or not openrouter.configured():
        return 0
    source = LANGUAGES.get(source_language or "", "the spoken language")
    user = "Lines (id | speaker | A | B):\n" + "\n".join(
        f"{i} | {s.speaker} | {s.text} | {s.reference_text or s.text}"
        for i, s in enumerate(segments)
        if s.text
    )
    corrected = _ask(TRANSCRIPT_PROMPT.format(source=source), user, {i for i, _ in lines}, usage)
    changed = 0
    for index, segment in lines:
        text = corrected.get(index)
        if text and text != segment.text and not segment.edited:
            segment.asr_text = segment.text
            segment.text = text
            changed += 1
    return changed


def script_rule(target: str, script: str) -> str:
    """How the target language is written: the one sentence every script prompt gets."""
    if script == "codemix":
        return (
            f"Write natural code-mixed {target} as educated urban speakers talk: {target} grammar and everyday vocabulary "
            f"in {target} script. Use English only for what such speakers actually say in English: technical terms, "
            "names of people, places, brands and products, acronyms, and loanwords with no everyday {target} equivalent; "
            f"never replace an ordinary {target} word with its English synonym. Every English word keeps its English "
            f"spelling in Roman letters and is never transliterated into {target} script (write 'diffusion model', not a "
            f"{target}-script spelling of it)."
        )
    return f"Write {target} in its native script; transliterate names and English terms into {target} script."


PROFILE_PROMPT = """You classify a video from its metadata and the opening of its dialogue, for a dubbing studio that must
match the register of the dub to the content. Respond with JSON only:
{{"genre": one of "lecture", "news", "film", "advert", "documentary", "story", "interview", "service", "other",
 "label": "<3-6 words a person would say, e.g. 'University lecture on optimisation'>",
 "register": "<one sentence on how the {target} should sound: formality, pace, address form, vocabulary; for a lecture
 a formal teaching register with आप, for a film the characters' own register, for news a composed anchor register, etc.>"}}"""


def profile_content(
    segments, meta: dict, title: str, source_language: str | None, usage=None, target: str = "Hindi"
) -> dict:
    """Genre and register of the content, from yt-dlp metadata and the first lines of the transcript."""
    if not openrouter.configured() or not (
        segments or meta.get("categories") or meta.get("tags") or title
    ):
        return {}
    opening = (
        "\n".join(f"{s.speaker}: {s.text}" for s in segments[:40] if s.text)
        or "(not transcribed yet: judge from the metadata)"
    )
    user = (
        f"Title: {title}\nChannel: {meta.get('channel', '')}\nCategories: {', '.join(meta.get('categories') or [])}\n"
        f"Tags: {', '.join(meta.get('tags') or [])}\nDescription: {(meta.get('description') or '')[:400]}\n\nOpening dialogue:\n{opening}"
    )
    for model in [settings.openrouter_model] + [
        m for m in settings.openrouter_fallback_models if m != settings.openrouter_model
    ]:
        try:
            content = openrouter.chat(
                model,
                [
                    {"role": "system", "content": PROFILE_PROMPT.format(target=target)},
                    {"role": "user", "content": user},
                ],
                usage=usage,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            data = json.loads(content)
            if data.get("genre") and data.get("register"):
                return {
                    "genre": str(data["genre"]).lower(),
                    "label": str(data.get("label") or data["genre"]),
                    "register": str(data["register"]),
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Content profiling with %s failed: %s", model, exc)
    return {}


LENGTH_RULE = (
    "Each line is spoken in the fixed time its original speaker took: a target length in characters comes with every "
    "line. Only when a line is more than 20% off its target, bring it within range with the smallest change of wording "
    "(shorter rather than longer), never by dropping or adding information and never with filler. A line within 20% of "
    "its target is not changed for length."
)
MIN_KEEP_RATIO = (
    0.4  # character similarity below which an edit counts as a rewrite rather than an enhancement
)

CONSERVATIVE_RULE = (
    "Bodhan's translation is the dub. Return it unchanged unless one of the rules above requires a change, and then "
    "make the smallest edit that satisfies the rule (a word, a name, a comma), never a rephrase of the whole line. "
    "Changing a correct line is a mistake."
)

SPOKEN_RULE = (
    "Everything must be writable as speech: numbers, units, dates, abbreviations and symbols are written the way a "
    "{target} speaker says them aloud (24B -> 24 billion, 3.5% -> three point five percent, km -> kilometre), "
    "never as notation the voice would have to guess."
)

TECHNICAL_RULE = (
    "This is teaching or technical content. Every variable, index, operator and formula is written exactly as the "
    "lecturer would say it in {target}, in words, with the technical English terms kept in Roman letters: "
    "W1 transpose C W1 -> W one transpose C W one, x_i -> x i, lambda_1 -> lambda one, sum_{{i=1}}^{{n}} -> sum over i "
    "from one to n, u_t-1 -> u t minus one, a^2 -> a squared. Never glue a letter to a digit (W1, x2) and never leave a "
    "line that is only symbols: keep the connecting words so the sentence stays speakable."
)

TECHNICAL_GENRES = {
    "lecture",
    "documentary",
    "tutorial",
    "education",
    "science",
    "maths",
    "math",
    "course",
    "class",
}


def content_rule(content: dict | None, target: str = "the target language") -> str:
    """The genre's own instruction for the editor: what it is, how it is spoken, and for teaching content how maths is read."""
    rules = [SPOKEN_RULE.format(target=target)]
    if not content:
        return " ".join(rules)
    genre = str(content.get("genre") or "").lower()
    label = str(content.get("label") or genre)
    rules.insert(0, f"Content: {label}. Register: {content.get('register')}.")
    if genre in TECHNICAL_GENRES or any(
        word in label.lower()
        for word in (
            "lecture",
            "lesson",
            "tutorial",
            "course",
            "math",
            "physics",
            "science",
            "engineering",
        )
    ):
        rules.append(TECHNICAL_RULE.format(target=target))
    return " ".join(rules)


PAUSE_RULE = (
    "Keep the punctuation that marks the voice's pauses (commas, sentence ends, question marks) and keep every word "
    "that is in Roman letters in Roman letters."
)

REWRITE_PROMPT = """You are the script editor of a {target} dub (dubbed from {source}). The person who owns the dub asks:
"{instruction}"
Apply that request to the lines you are allowed to change, and only that request. Keep everything the request does not
mention: the meaning of the original line, names, numbers, negation, who does what to whom. Each line is spoken in a fixed
time slot. Where a line comes with a target length in characters (what its voice can say in that slot), the new line must
not run longer than the target, whatever the request: a longer line is sped up and its last words are cut. It may be
shorter; never pad a line to reach the target. Without a target, keep within about 15 % of the current line's length
unless the request is about length.
When a sentence runs across consecutive lines of one speaker, keep the pieces in order so they still form one sentence.
{content} {script} {pause} Never merge, split or skip lines; return a line unchanged if the request does not apply to it.
Respond with JSON only: {{"lines": [{{"id": <number>, "text": "<line>"}}, ...]}}"""


SOURCE_REWRITE_PROMPT = """You correct the original {source} transcript of a video as its owner asks:
"{instruction}"
Apply only that request. Keep everything it does not mention: what was said, names, numbers, negation, who speaks.
Never merge, split or skip lines; return a line unchanged if the request does not apply to it.
Respond with JSON only: {{"lines": [{{"id": <number>, "text": "<line>"}}, ...]}}"""


def slot_chars(line: dict) -> int | None:
    """Characters the line's voice says in its slot, from the clip it last voiced: len / (clip seconds / slot)."""
    ratio, voiced = line.get("fit_ratio"), line.get("voiced_text") or ""
    if not voiced or not ratio or ratio <= 0:
        return None
    return max(4, round(len(voiced) / ratio))


def rewrite_lines(
    segments,
    language: str,
    source_language: str | None,
    instruction: str,
    script: str = "native",
    only=None,
    usage=None,
    content: dict | None = None,
    field: str = "translation",
) -> dict[int, str]:
    """Rewrite lines the way the user asks (tone, wording, length...) in scene context: the dub's lines
    (`field="translation"`) or the original transcript (`field="text"`). Returns {index: new text} for the
    lines that actually changed; nothing is applied here."""
    if field == "text":
        lines = [(i, s) for i, s in enumerate(segments) if s.get("text")]
        editable = {i for i, _ in lines if only is None or i in only}
        if not editable or not openrouter.configured():
            return {}
        source = LANGUAGES.get(source_language or "", "the source language")
        user = "Lines (id | speaker | text):\n" + "\n".join(
            f"{i} | {s['speaker']} | {s['text']}" for i, s in lines
        )
        if len(editable) < len(lines):
            user += (
                "\nOnly lines "
                + ", ".join(map(str, sorted(editable)))
                + " may change; the others are given for context."
            )
        proposed = _ask(
            SOURCE_REWRITE_PROMPT.format(
                source=source, instruction=instruction.strip().replace('"', "'")
            ),
            user,
            editable,
            usage,
        )
        return {
            i: t.strip()
            for i, t in proposed.items()
            if i in editable and t and t.strip() != (segments[i].get("text") or "").strip()
        }
    lines = [(i, s) for i, s in enumerate(segments) if s.get("text") and s.get("translation")]
    editable = {i for i, _ in lines if only is None or i in only}
    if not editable or not openrouter.configured():
        return {}
    target = LANGUAGES.get(language, language)
    source = LANGUAGES.get(source_language or "", "the source language")
    rule = script_rule(target, script)
    budgets = {i: slot_chars(s) for i, s in lines}
    user = "Lines (id | speaker | source | current translation | target length):\n" + "\n".join(
        f"{i} | {s['speaker']} | {s['text']} | {s['translation']} | "
        + (f"{budgets[i]} chars (now {len(s['translation'])})" if budgets[i] else "-")
        for i, s in lines
    )
    if len(editable) < len(lines):
        user += (
            "\nOnly lines "
            + ", ".join(map(str, sorted(editable)))
            + " may change; the others are given for context."
        )
    system = REWRITE_PROMPT.format(
        target=target,
        source=source,
        instruction=instruction.strip().replace('"', "'"),
        script=rule,
        pause=PAUSE_RULE,
        content=content_rule(content, target),
    )
    proposed = _ask(system, user, editable, usage)
    return {
        i: text.strip()
        for i, text in proposed.items()
        if i in editable and text and text.strip() != (segments[i].get("translation") or "").strip()
    }


def post_edit(
    segments,
    language: str,
    source_language: str | None,
    script: str = "native",
    usage=None,
    only=None,
    content: dict | None = None,
    targets: dict | None = None,
    verify: bool = True,
) -> int:
    """Scene-level correction of the machine translation. Returns lines changed.
    `only`: the segments that may change (new in this run, so voiced lines are never re-voiced for it).
    `targets`: {index: characters} the line should have to fill its slot at its voice's pace, so the enhanced line fits
    first time. Every changed line is checked against the original for meaning before it is kept."""
    lines = [(i, s) for i, s in enumerate(segments) if s.text and s.translation]
    editable = {i for i, s in lines if only is None or any(s is o for o in only)}
    if not editable or not openrouter.configured():
        return 0
    target = LANGUAGES.get(language, language)
    source = LANGUAGES.get(source_language or "", "the source language")
    rule = script_rule(target, script)
    targets = targets or {}
    user = (
        "Lines (id | speaker | source | machine translation | also heard as"
        + (" | target length" if targets else "")
        + "):\n"
        + "\n".join(
            f"{i} | {s.speaker} | {s.text} | {s.translation} | {s.reference_text if s.reference_text and s.reference_text != s.text else ''}"
            + (f" | {targets[i]} chars (now {len(s.translation)})" if i in targets else "")
            for i, s in lines
        )
    )
    if len(editable) < len(lines):
        user += (
            "\nOnly lines "
            + ", ".join(map(str, sorted(editable)))
            + " may change; the others are final and given for context."
        )
    system = POSTEDIT_PROMPT.format(
        source=source,
        target=target,
        script=rule,
        content=content_rule(content, target),
        length=LENGTH_RULE if targets else "",
        conservative=CONSERVATIVE_RULE,
    )
    edited = _ask(system, user, editable, usage)
    proposals = {
        i: text
        for i, text in edited.items()
        if i in editable and text and text != segments[i].translation and not segments[i].edited
    }
    # a near-complete rewrite of a line that was not far off its length is not an enhancement: Bodhan's words stay
    for i in list(proposals):
        before, after = segments[i].translation, proposals[i]
        far_off = i in targets and abs(len(before) - targets[i]) > 0.2 * max(1, targets[i])
        if (
            targets
            and difflib.SequenceMatcher(None, before, after).ratio() < MIN_KEEP_RATIO
            and not far_off
        ):
            logger.info(
                "Post-edit: line %d rewritten too far (%.2f similar); keeping Bodhan's words",
                i,
                difflib.SequenceMatcher(None, before, after).ratio(),
            )
            del proposals[i]
    if proposals and verify:
        # the same meaning check the timing fit uses: a line that lost or gained meaning goes back to Bodhan's words
        from backend.speech.fitting import _checked, _request

        ordered = sorted((s for _, s in lines), key=lambda s: s.start)
        requests = [
            _request(i, segments[i], targets.get(i, len(segments[i].translation)), ordered)
            for i in proposals
        ]
        kept = _checked(dict(proposals), requests, language, usage, source_language)
        if len(kept) < len(proposals):
            logger.info(
                "Post-edit: %d of %d changed lines rejected by the meaning check",
                len(proposals) - len(kept),
                len(proposals),
            )
        proposals = kept
    changed = 0
    for index, text in proposals.items():
        segment = segments[index]
        segment.machine_translation = segment.translation
        segment.translation = text
        changed += 1
    return changed
