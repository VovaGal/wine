"""``search_key`` is the exact key stored in catalog aliases. OCR lines must use
the same function before comparison. ``search_forms`` adds an optional Latin
accent-insensitive key for candidate retrieval, while retaining the original
key for scoring.

Examples::

    search_key("FROZEN—ROSÉ")      # "frozen rosé"
    search_forms("Château")        # ("château", "chateau")
    search_key("Розовое Ёж")       # "розовое еж"
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


SPACE = re.compile(r"\s+")


def display_text(value: Any) -> str:
    """Keep original spelling while normalizing whitespace and controls."""
    if value is None or isinstance(value, (dict, list, tuple)):
        return ""
    value = unicodedata.normalize("NFC", str(value))
    value = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in value
    )
    return SPACE.sub(" ", value).strip()


def search_key(value: Any) -> str:
    """Index and query key, shared byte-for-byte with the catalog builder."""
    text = unicodedata.normalize("NFKC", display_text(value)).casefold()
    text = text.replace("ё", "е").replace("’", "'")
    text = "".join(char if char.isalnum() else " " for char in text)
    return SPACE.sub(" ", text).strip()


def accent_key(value: Any) -> str:
    """Fold Latin accents for lookup only; preserve Cyrillic and other scripts."""
    primary = search_key(value)
    if not primary:
        return ""
    # Some letters are not decomposed by NFKD (e.g. ø and ł).
    special = str.maketrans({"æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d", "ð": "d", "þ": "th"})
    decomposed = unicodedata.normalize("NFKD", primary.translate(special))
    result: list[str] = []
    last_base_is_latin = False
    for char in decomposed:
        if unicodedata.category(char).startswith("M"):
            if not last_base_is_latin:
                result.append(char)
            continue
        last_base_is_latin = "LATIN" in unicodedata.name(char, "")
        result.append(char)
    return search_key(unicodedata.normalize("NFC", "".join(result)))


def search_forms(value: Any) -> tuple[str, ...]:
    """Deduplicated exact + accent-folded keys in preference order."""
    primary = search_key(value)
    secondary = accent_key(value)
    if not primary:
        return ()
    return (primary, secondary) if secondary and secondary != primary else (primary,)


def tokens(value: Any) -> tuple[str, ...]:
    """Normalized tokens for later field-aware fuzzy matching."""
    key = search_key(value)
    return tuple(key.split()) if key else ()


def script_of(value: Any) -> str:
    """Return latin, cyrillic, mixed, numeric or other."""
    text = display_text(value)
    scripts = {
        script
        for char in text
        if char.isalpha()
        for script in (unicodedata.name(char, "").split(" ", 1)[0],)
        if script in {"LATIN", "CYRILLIC"}
    }
    if len(scripts) == 2:
        return "mixed"
    if "LATIN" in scripts:
        return "latin"
    if "CYRILLIC" in scripts:
        return "cyrillic"
    if any(char.isdigit() for char in text):
        return "numeric"
    return "other"
