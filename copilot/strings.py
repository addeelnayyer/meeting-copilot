"""UI text: one table per language, keyed by stable ids. UI_LANGUAGE selects the table.

The page holds only ids; `event()` is the event that sends it the table on connect.
Placeholders are written as {name} and filled by `t()` (server) or the page's own `t()`.
"""

from . import config

TABLES = {
    "en": {
        "meeting.start": "Start meeting",
        "meeting.stop": "End meeting",
        "saved.note": "Meeting saved: {name}",
        "folder.open": "Open folder",
        "provider.title": "AI provider (fallback always: local Ollama)",
        "provider.no_key": "{label} (no key)",
        "engine.title": "Answered last",
        "stt.loading": "STT loading",
        "stt.ready": "STT ready",
        "stt.disconnected": "disconnected",
        "meter.user": "You (microphone)",
        "meter.participant": "Participants (loopback)",
        "proactive.label": "Proactive",
        "proactive.title": "Ctrl+Alt+P",
        "translate.label": "Translate",
        "translate.title": "Ctrl+Alt+T",
        "on_top.title": "Always on top (Ctrl+Alt+O)",
        "opacity.title": "Opacity",
        "open_meetings.title": "Open saved meetings",
        "open_notes.title": "Open the knowledge base (changes apply at once)",
        "open_settings.title": "Open the settings (.env) (apply after a restart)",
        "mic.title": "Microphone on/off (Ctrl+Alt+M). The participants are still heard.",
        "mic.on": "Mic on",
        "mic.off": "Mic off",
        "pending.title": "Show now",
        "pending.note": "New suggestion waiting: {trigger} · show",
        "card.title": "Click to pin",
        "card.pinned": "pinned",
        "card.question": "Question:",
        "asking": "Asking: {query}",
        "empty.waiting": "Waiting for the other side …",
        "empty.idle": "Click “Start meeting” as soon as your call begins.",
        "audio.error": "Audio: {error}",
        "kb.empty": "Your knowledge base is still empty. Put Markdown files (.md) in the folder, "
                    "e.g. products, prices, FAQ or your CV. They are searched right away.",
        "who.user": "You",
        "who.participant": "Participant",
        "facts.title": "Facts",
        "memory.title": "Memory (every 5 min)",
        "none_yet": "(none yet)",
        "ask.placeholder": "Ask a question (Ctrl+Alt+Space) …",
        "ask.button": "Ask",
        "key.ctrl": "Ctrl",
    },
    "de": {
        "meeting.start": "Meeting starten",
        "meeting.stop": "Meeting beenden",
        "saved.note": "Meeting gespeichert: {name}",
        "folder.open": "Ordner öffnen",
        "provider.title": "KI-Anbieter (Fallback immer: Ollama lokal)",
        "provider.no_key": "{label} (kein Key)",
        "engine.title": "Hat zuletzt geantwortet",
        "stt.loading": "STT lädt",
        "stt.ready": "STT bereit",
        "stt.disconnected": "getrennt",
        "meter.user": "Du (Mikrofon)",
        "meter.participant": "Teilnehmer (Loopback)",
        "proactive.label": "Proaktiv",
        "proactive.title": "Strg+Alt+P",
        "translate.label": "Übersetzen",
        "translate.title": "Strg+Alt+T",
        "on_top.title": "Immer im Vordergrund (Strg+Alt+O)",
        "opacity.title": "Deckkraft",
        "open_meetings.title": "Gespeicherte Meetings öffnen",
        "open_notes.title": "Wissensbasis öffnen (Änderungen gelten sofort)",
        "open_settings.title": "Einstellungen (.env) öffnen (gelten nach Neustart)",
        "mic.title": "Mikrofon an/aus (Strg+Alt+M). Die Teilnehmer werden weiter gehört.",
        "mic.on": "Mikro an",
        "mic.off": "Mikro aus",
        "pending.title": "Sofort anzeigen",
        "pending.note": "Neuer Vorschlag wartet: {trigger} · anzeigen",
        "card.title": "Klicken zum Anheften",
        "card.pinned": "angeheftet",
        "card.question": "Frage:",
        "asking": "Frage läuft: {query}",
        "empty.waiting": "Warte auf den Gesprächspartner …",
        "empty.idle": "Klicke auf „Meeting starten“, sobald dein Gespräch beginnt.",
        "audio.error": "Audio: {error}",
        "kb.empty": "Deine Wissensbasis ist noch leer. Lege Markdown-Dateien (.md) in den Ordner, "
                    "z. B. Produkte, Preise, FAQ oder deinen Lebenslauf. Sie werden sofort durchsucht.",
        "who.user": "Du",
        "who.participant": "Teilnehmer",
        "facts.title": "Fakten",
        "memory.title": "Gedächtnis (alle 5 min)",
        "none_yet": "(noch keine)",
        "ask.placeholder": "Frage stellen (Strg+Alt+Leertaste) …",
        "ask.button": "Fragen",
        "key.ctrl": "Strg",
    },
}

_hinted = set()


def language():
    """The configured UI language; an unknown value falls back to English (with a hint, once)."""
    code = config.UI_LANGUAGE.strip().lower()
    if code in TABLES:
        return code
    if code not in _hinted:
        _hinted.add(code)
        print(f"[ui] UI_LANGUAGE={config.UI_LANGUAGE} is unknown ({' | '.join(TABLES)}); using English")
    return "en"


def t(key, **values):
    return TABLES[language()][key].format(**values)


def event():
    """The event that gives the page its strings; sent ahead of every other event on connect."""
    lang = language()
    return {"type": "strings", "lang": lang, "strings": TABLES[lang]}
