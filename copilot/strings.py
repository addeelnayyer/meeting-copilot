"""UI strings: one table per language, keyed by stable ids. UI_LANGUAGE selects the table.

The window's strings are sent to the page on connect (`window_strings`); the page holds only the
ids. Placeholders are plain "{name}" fields only (no braces, format specs or attributes),
because the page fills them with its own small replace.
"""

from . import config

TABLES = {
    "en": {
        "meeting.start": "Start meeting",
        "meeting.stop": "End meeting",
        "meeting.saved": "Meeting saved: {file}",
        "meeting.waiting": "Waiting for the other party …",
        "meeting.idle_hint": "Click “{button}” as soon as your call begins.",
        "open_folder": "Open folder",
    },
    "de": {
        "meeting.start": "Meeting starten",
        "meeting.stop": "Meeting beenden",
        "meeting.saved": "Meeting gespeichert: {file}",
        "meeting.waiting": "Warte auf den Gesprächspartner …",
        "meeting.idle_hint": "Klicke auf „{button}“, sobald dein Gespräch beginnt.",
        "open_folder": "Ordner öffnen",
    },
}
FALLBACK = "en"
_hinted = set()


def language():
    """The configured UI language; an unknown value falls back to English, with a console hint."""
    code = config.UI_LANGUAGE.strip().lower()
    if code in TABLES:
        return code
    if code not in _hinted:
        _hinted.add(code)
        print(f"UI_LANGUAGE={config.UI_LANGUAGE!r} is unknown ({', '.join(TABLES)}); using {FALLBACK}")
    return FALLBACK


def t(key, **values):
    """The text for `key` in the configured language, with placeholders filled."""
    return TABLES[language()][key].format(**values)


def window_strings():
    """The event that gives the page its strings, sent ahead of any other event."""
    lang = language()
    return {"type": "strings", "lang": lang, "strings": TABLES[lang]}
