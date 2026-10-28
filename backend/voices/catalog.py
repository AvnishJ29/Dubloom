"""Load provider voice metadata independently of speaker-assignment code."""

import json
import os
from pathlib import Path
from backend.config import ROOT


def load_catalog():
    """Load and validate configured voice, language-group, and gender metadata."""
    configured = os.getenv('BODHAN_VOICE_CATALOG_PATH', '')
    path = (
        ROOT / configured if configured else Path(__file__).with_name('data') / 'voice_catalog.json'
    )
    catalog = json.loads(path.read_text(encoding='utf-8-sig'))
    voices = catalog.get('voices', {})
    if not isinstance(voices, dict) or not voices:
        raise ValueError('Voice catalogue must contain a non-empty voices mapping')
    for language, names in voices.items():
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name.strip() for name in names)
        ):
            raise ValueError(f'Invalid voice list for {language}')
    for group in catalog.get('priority_groups', []):
        if not isinstance(group, list) or not all(code in voices for code in group):
            raise ValueError('Voice priority groups must reference configured languages')
    genders = catalog.get('genders', {})
    if not isinstance(genders, dict) or any(
        value not in {'male', 'female', 'unknown'} for value in genders.values()
    ):
        raise ValueError('Voice genders must map voice names to male, female or unknown')
    return catalog
