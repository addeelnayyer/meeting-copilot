"""UI text: one table per language, keyed by stable ids. UI_LANGUAGE selects the table.

The page holds only ids; `event()` is the event that sends it the table on connect.
Placeholders are written as {name} and filled by `t()` (server) or the page's own `t()`.
Server notes use `t()` too; an error shown in the window is raised as `UiError`, whose str() stays
English for the console. Console log lines meant for developers do not use these tables.
"""

from . import config

TABLES = {
    "en": {
        "meeting.start": "Start meeting",
        "meeting.stop": "End meeting",
        "saved.note": "Meeting saved: {name}",
        "folder.open": "Open folder",
        "provider.title": "AI provider",
        "provider.no_key": "{label} (no key)",
        "engine.title": "Answered last",
        "stt.loading": "STT loading",
        "stt.ready": "STT ready",
        "stt.idle": "STT off",
        "stt.disconnected": "disconnected",
        "meter.user": "You (microphone)",
        "meter.participant": "Participants (loopback)",
        "proactive.label": "Proactive",
        "proactive.title": "{hotkey}",
        "translate.label": "Translate",
        "translate.title": "{hotkey}",
        "on_top.title": "Always on top ({hotkey})",
        "opacity.title": "Opacity",
        "open_meetings.title": "Open saved meetings",
        "open_notes.title": "Open the knowledge base (changes apply at once)",
        "open_settings.title": "Open the settings (.env) (apply after a restart)",
        "mic.title": "Microphone on/off ({hotkey}). The participants are still heard.",
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
        "ask.placeholder": "Ask a question ({hotkey}) …",
        "ask.button": "Ask",
        "key.ctrl": "Ctrl",
        "key.space": "Space",
        # Meeting record
        "record.summary": "Summary",
        "record.no_summary": "(none, the conversation was too short)",
        "record.cards": "Suggestions and answers",
        "record.transcript": "Transcript",
        "record.none": "(none)",
        # Server notes
        "llm.no_key": "{provider}: no API key/URL",
        "llm.quota": "{provider}: rate limit/quota reached, back in {minutes} min",
        "llm.failed": "{provider} failed: {error}",
        "llm.none": "No LLM configured: add an API key to .env and pick a provider",
        "llm.not_json": "The answer wasn't valid JSON.",
        "stt.restarting": "{engine} failed ({reason}), restarting it",
        "stt.switched": "{failed} failed ({error}), STT now on {fallback}",
        "stt.fallback_failed": "{failed} failed, {fallback} too: {error}",
        "audio.no_wasapi_mic": "No WASAPI microphone matching '{name}'",
        "audio.no_loopback": "No loopback device matching '{name}'",
        "audio.no_pulse_source": "No PulseAudio/PipeWire source matching '{name}'",
        "audio.no_input": "No input device matching '{name}'",
        "audio.no_mac_loopback": "{error}. macOS has no built-in loopback: install BlackHole and route "
                                 "the meeting audio to it (see README), or set LOOPBACK_DEVICE.",
        "audio.no_parec": "'parec' not found. Install it (Debian/Ubuntu/Fedora: pulseaudio-utils, Arch: libpulse).",
    },
    "de": {
        "meeting.start": "Meeting starten",
        "meeting.stop": "Meeting beenden",
        "saved.note": "Meeting gespeichert: {name}",
        "folder.open": "Ordner öffnen",
        "provider.title": "KI-Anbieter",
        "provider.no_key": "{label} (kein Key)",
        "engine.title": "Hat zuletzt geantwortet",
        "stt.loading": "STT lädt",
        "stt.ready": "STT bereit",
        "stt.idle": "STT aus",
        "stt.disconnected": "getrennt",
        "meter.user": "Du (Mikrofon)",
        "meter.participant": "Teilnehmer (Loopback)",
        "proactive.label": "Proaktiv",
        "proactive.title": "{hotkey}",
        "translate.label": "Übersetzen",
        "translate.title": "{hotkey}",
        "on_top.title": "Immer im Vordergrund ({hotkey})",
        "opacity.title": "Deckkraft",
        "open_meetings.title": "Gespeicherte Meetings öffnen",
        "open_notes.title": "Wissensbasis öffnen (Änderungen gelten sofort)",
        "open_settings.title": "Einstellungen (.env) öffnen (gelten nach Neustart)",
        "mic.title": "Mikrofon an/aus ({hotkey}). Die Teilnehmer werden weiter gehört.",
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
        "ask.placeholder": "Frage stellen ({hotkey}) …",
        "ask.button": "Fragen",
        "key.ctrl": "Strg",
        "key.space": "Leertaste",
        # Meeting record
        "record.summary": "Zusammenfassung",
        "record.no_summary": "(keine, das Gespräch war zu kurz)",
        "record.cards": "Vorschläge und Antworten",
        "record.transcript": "Transkript",
        "record.none": "(keine)",
        # Server notes
        "llm.no_key": "{provider}: kein API-Key/URL",
        "llm.quota": "{provider}: Limit/Kontingent erreicht, wieder in {minutes} min",
        "llm.failed": "{provider} fehlgeschlagen: {error}",
        "llm.none": "Kein LLM konfiguriert: API-Key in .env eintragen und Anbieter wählen",
        "llm.not_json": "Antwort war kein gültiges JSON.",
        "stt.restarting": "{engine} ausgefallen ({reason}), wird neu gestartet",
        "stt.switched": "{failed} ausgefallen ({error}), STT jetzt mit {fallback}",
        "stt.fallback_failed": "{failed} ausgefallen, {fallback} auch: {error}",
        "audio.no_wasapi_mic": "Kein WASAPI-Mikrofon passend zu '{name}'",
        "audio.no_loopback": "Kein Loopback-Gerät passend zu '{name}'",
        "audio.no_pulse_source": "Keine PulseAudio/PipeWire-Quelle passend zu '{name}'",
        "audio.no_input": "Kein Eingabegerät passend zu '{name}'",
        "audio.no_mac_loopback": "{error}. macOS hat kein eingebautes Loopback: BlackHole installieren und den "
                                 "Meeting-Ton dorthin leiten (siehe README), oder LOOPBACK_DEVICE setzen.",
        "audio.no_parec": "'parec' nicht gefunden. Bitte installieren (Debian/Ubuntu/Fedora: pulseaudio-utils, "
                          "Arch: libpulse).",
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
    """The text for `key` in the UI language, with the placeholders filled."""
    return _render(language(), key, values)


def _render(lang, key, values):
    values = {k: _error_text(v, lang) if isinstance(v, BaseException) else v for k, v in values.items()}
    return TABLES[lang][key].format(**values)


def _error_text(error, lang):
    return error.render(lang) if isinstance(error, UiError) else str(error) or type(error).__name__


class UiError(RuntimeError):
    """An error shown in the window: str() is English for the console, ui_text() follows UI_LANGUAGE."""

    def __init__(self, key, **values):
        self.key, self.values = key, values
        super().__init__(self.render("en"))

    def render(self, lang):
        return _render(lang, self.key, self.values)


def ui_text(error):
    """How `error` reads in the window: translated for a UiError, else its own message."""
    return _error_text(error, language())


def hotkey(combo):
    """"ctrl+alt+space" -> "Strg+Alt+Leertaste" in the UI language."""
    table = TABLES[language()]
    return "+".join(table.get(f"key.{k}", k.capitalize()) for k in combo.split("+"))


# Strings that show a hotkey; their {hotkey} comes from config, so a changed hotkey shows everywhere.
_HOTKEY_STRINGS = {
    "proactive.title": "HOTKEY_PROACTIVE",
    "translate.title": "HOTKEY_TRANSLATE",
    "on_top.title": "HOTKEY_ON_TOP",
    "mic.title": "HOTKEY_MIC",
    "ask.placeholder": "HOTKEY_FOCUS",
}


def event():
    """The event that gives the page its strings; sent ahead of every other event on connect."""
    lang = language()
    filled = {key: t(key, hotkey=hotkey(getattr(config, name))) for key, name in _HOTKEY_STRINGS.items()}
    return {"type": "strings", "lang": lang, "strings": {**TABLES[lang], **filled}}
