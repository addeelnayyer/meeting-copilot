"""Text the copilot writes for the window, one table per UI_LANGUAGE, keyed by stable ids.

Console log lines meant for developers stay English and do not use these tables.
"""

from . import config

TABLES = {
    "en": {
        # LLM and Ollama
        "llm.no_key": "{provider}: no API key/URL",
        "llm.quota": "{provider}: rate limit/quota reached, back in {minutes} min",
        "llm.errors": "{provider} errors, paused for {seconds} s",
        "llm.not_json": "The answer wasn't valid JSON.",
        "ollama.label": "Ollama (local)",
        "ollama.not_running": "Ollama isn't running (ollama.com): no local fallback",
        "ollama.model_missing": "Ollama model {model} is missing: ollama pull {model}, or pick an installed one above",
        "ollama.no_model": "No Ollama model installed: ollama pull {model}",
        # STT engine
        "stt.restarting": "{engine} failed ({reason}), restarting it",
        "stt.switched": "{failed} failed ({error}), STT now on {fallback}",
        "stt.fallback_failed": "{failed} failed, {fallback} too: {error}",
        # Audio devices
        "audio.no_wasapi_mic": "No WASAPI microphone matching '{name}'",
        "audio.no_loopback": "No loopback device matching '{name}'",
        "audio.no_pulse_source": "No PulseAudio/PipeWire source matching '{name}'",
        "audio.no_input": "No input device matching '{name}'",
        "audio.no_mac_loopback": "{error}. macOS has no built-in loopback: install BlackHole and route "
                                 "the meeting audio to it (see README), or set LOOPBACK_DEVICE.",
        "audio.no_parec": "'parec' not found. Install it (Debian/Ubuntu/Fedora: pulseaudio-utils, Arch: libpulse).",
    },
    "de": {
        "llm.no_key": "{provider}: kein API-Key/URL",
        "llm.quota": "{provider}: Limit/Kontingent erreicht, wieder in {minutes} min",
        "llm.errors": "{provider}-Fehler, Pause {seconds} s",
        "llm.not_json": "Antwort war kein gültiges JSON.",
        "ollama.label": "Ollama (lokal)",
        "ollama.not_running": "Ollama läuft nicht (ollama.com): kein lokaler Fallback",
        "ollama.model_missing": "Ollama-Modell {model} fehlt: ollama pull {model}, oder oben ein installiertes wählen",
        "ollama.no_model": "Kein Ollama-Modell installiert: ollama pull {model}",
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
FALLBACK = "en"
_hinted = set()


def language():
    """The configured UI_LANGUAGE; an unknown value falls back to English with a console hint (once)."""
    lang = config.UI_LANGUAGE.strip().lower()
    if lang in TABLES:
        return lang
    if lang not in _hinted:
        _hinted.add(lang)
        print(f"UI_LANGUAGE={config.UI_LANGUAGE!r} is unknown ({', '.join(TABLES)}); using {FALLBACK}")
    return FALLBACK


def t(key, **params):
    """The text for `key` in the UI language, with the placeholders filled."""
    return _render(language(), key, params)


def _render(lang, key, params):
    params = {k: v.render(lang) if isinstance(v, UiError) else v for k, v in params.items()}
    return TABLES[lang][key].format(**params)


class UiError(RuntimeError):
    """An error shown in the window: str() is English for the console, ui_text() follows UI_LANGUAGE."""

    def __init__(self, key, **params):
        self.key, self.params = key, params
        super().__init__(self.render(FALLBACK))

    def render(self, lang):
        return _render(lang, self.key, self.params)


def ui_text(error):
    """How `error` reads in the window: translated for a UiError, else its own message."""
    return error.render(language()) if isinstance(error, UiError) else str(error) or type(error).__name__
