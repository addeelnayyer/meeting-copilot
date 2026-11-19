import re

from copilot import config, strings


def test_every_language_has_the_same_keys_and_placeholders():
    def shape(table):
        return {key: sorted(re.findall(r"\{(\w+)\}", text)) for key, text in table.items()}

    en = shape(strings.TABLES["en"])
    for lang, table in strings.TABLES.items():
        assert shape(table) == en, lang


def test_window_strings_follow_ui_language(monkeypatch):
    monkeypatch.setattr(config, "UI_LANGUAGE", "de")
    assert strings.event()["strings"]["meeting.start"] == "Meeting starten"
    monkeypatch.setattr(config, "UI_LANGUAGE", "en")
    assert strings.event()["strings"]["meeting.start"] == "Start meeting"


def test_unknown_ui_language_falls_back_to_english_with_a_hint(monkeypatch, capsys):
    monkeypatch.setattr(config, "UI_LANGUAGE", "xx")
    event = strings.event()
    assert event["lang"] == "en" and event["strings"]["meeting.start"] == "Start meeting"
    assert "UI_LANGUAGE=xx" in capsys.readouterr().out


def test_lookup_fills_placeholders(monkeypatch):
    monkeypatch.setattr(config, "UI_LANGUAGE", "en")
    assert strings.t("saved.note", name="Meeting 1.md") == "Meeting saved: Meeting 1.md"


def test_hotkey_tooltips_show_the_configured_hotkey_in_the_ui_language(monkeypatch):
    monkeypatch.setattr(config, "HOTKEY_FOCUS", "ctrl+alt+space")
    monkeypatch.setattr(config, "UI_LANGUAGE", "de")
    assert strings.event()["strings"]["ask.placeholder"] == "Frage stellen (Strg+Alt+Leertaste) …"
    monkeypatch.setattr(config, "UI_LANGUAGE", "en")
    assert strings.event()["strings"]["ask.placeholder"] == "Ask a question (Ctrl+Alt+Space) …"
