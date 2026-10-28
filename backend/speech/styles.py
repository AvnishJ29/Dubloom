"""Delivery styles: how a line is spoken.

Bodhan Indic-Speak accepts a fixed set of styles in its `instructions` JSON.
The same style ids map to a natural-language direction for prompt-driven
engines (Gemini TTS) and to an emotion tag for the voice-cloning engine.
"""

DELIVERY_STYLES = {
    "neutral": {"label": "Neutral", "bodhan": None, "prompt": None, "tag": None},
    "encouraging": {
        "label": "Encouraging",
        "bodhan": "happy",
        "prompt": "warm, encouraging and uplifting",
        "tag": "(encouraging)",
    },
    "happy": {
        "label": "Happy",
        "bodhan": "happy",
        "prompt": "happy and cheerful",
        "tag": "(happy)",
    },
    "angry": {
        "label": "Angry",
        "bodhan": "anger",
        "prompt": "angry, sharp and forceful",
        "tag": "(angry)",
    },
    "sad": {"label": "Sad", "bodhan": "sad", "prompt": "sad and subdued", "tag": "(sad)"},
    "surprised": {
        "label": "Surprised",
        "bodhan": "surprise",
        "prompt": "surprised and astonished",
        "tag": "(surprised)",
    },
    "fearful": {
        "label": "Fearful",
        "bodhan": "fear",
        "prompt": "fearful and tense",
        "tag": "(scared)",
    },
    "disgusted": {
        "label": "Disgusted",
        "bodhan": "disgust",
        "prompt": "disgusted and contemptuous",
        "tag": "(disgusted)",
    },
    "lecture": {
        "label": "Educational lecture",
        "bodhan": "educational lecture",
        "prompt": "clear, measured and explanatory, like a good lecturer",
        "tag": None,
    },
    "news": {
        "label": "News",
        "bodhan": "news",
        "prompt": "crisp and authoritative, like a news anchor",
        "tag": None,
    },
    "tv_news": {
        "label": "TV news",
        "bodhan": "TV style news",
        "prompt": "energetic television news delivery",
        "tag": None,
    },
    "air_news": {
        "label": "Radio news (AIR)",
        "bodhan": "AIR style news",
        "prompt": "formal public-radio news delivery",
        "tag": None,
    },
    "audiobook": {
        "label": "Audiobook narration",
        "bodhan": "single person narration audiobook",
        "prompt": "calm, intimate audiobook narration",
        "tag": None,
    },
    "story": {
        "label": "Children's story",
        "bodhan": "children's stories",
        "prompt": "playful storytelling for children",
        "tag": "(excited)",
    },
    "advert": {
        "label": "Advertisement",
        "bodhan": "advertisements",
        "prompt": "upbeat, persuasive advertisement voice-over",
        "tag": "(excited)",
    },
    "customer_care": {
        "label": "Customer care",
        "bodhan": "Customer Care",
        "prompt": "polite, patient customer-care agent",
        "tag": None,
    },
}
BODHAN_STYLES = sorted({item["bodhan"] for item in DELIVERY_STYLES.values() if item["bodhan"]})

# Script-assistant tones -> the delivery style that carries the same feeling in the voice.
TONE_TO_STYLE = {
    "neutral": "neutral",
    "encouraging": "encouraging",
    "angry": "angry",
    "calm": "audiobook",
    "excited": "happy",
    "formal": "news",
    "casual": "neutral",
    "humorous": "happy",
    "serious": "lecture",
    "sad": "sad",
}


def normalise_style(style: str | None) -> str:
    return style if style in DELIVERY_STYLES else "neutral"


def bodhan_style(style: str | None) -> str | None:
    return DELIVERY_STYLES[normalise_style(style)]["bodhan"]


def style_prompt(style: str | None) -> str | None:
    return DELIVERY_STYLES[normalise_style(style)]["prompt"]


def style_tag(style: str | None) -> str | None:
    return DELIVERY_STYLES[normalise_style(style)]["tag"]


def style_options() -> list[dict]:
    return [
        {"id": key, "label": value["label"], "bodhan": value["bodhan"]}
        for key, value in DELIVERY_STYLES.items()
    ]
