import asyncio
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from copilot import config
from copilot.facts import Facts
import httpx

from copilot.llm import LLM, SUGGESTION_SCHEMA, parse_json
from copilot.providers import GeminiProvider, Provider
from copilot.rag import split_text
from copilot.hotkeys import to_pynput, to_win32
from copilot import stt
from copilot.stt import Segment, build_stt_engine, is_hallucination, stt_engine, whisper_settings
from copilot.transcript import Transcript
from copilot.vad import Chunker

F = config.VAD_FRAME
FPS = config.SAMPLE_RATE / F


def feed(chunker, pattern):
    """pattern: list of (seconds, is_speech)."""
    out = []
    for secs, speech in pattern:
        for _ in range(int(secs * FPS)):
            frame = np.full(F, 0.1 if speech else 0.0, dtype=np.float32)
            out += chunker.feed(frame, 0.9 if speech else 0.0)
    return out


# --- VAD chunker ---------------------------------------------------------------
def test_long_speech_is_force_split_at_max():
    chunks = feed(Chunker("P"), [(6.0, True), (1.0, False)])
    assert len(chunks) >= 3
    assert all(c.duration <= config.CHUNK_MAX_S + 0.05 for c in chunks)


def test_short_utterance_flushed_after_longer_pause():
    chunks = feed(Chunker("P"), [(0.6, True), (1.0, False)])
    assert len(chunks) == 1 and chunks[0].duration < config.CHUNK_MIN_S


def test_brief_pause_in_short_chunk_does_not_split():
    # 0.8 s speech, 0.35 s pause (> SILENCE_END but chunk < CHUNK_MIN), more speech.
    chunks = feed(Chunker("P"), [(0.8, True), (0.35, False), (0.5, True), (1.0, False)])
    assert len(chunks) == 1


def test_pause_after_min_length_ends_chunk():
    chunks = feed(Chunker("P"), [(1.7, True), (0.4, False), (1.7, True), (1.0, False)])
    assert len(chunks) == 2
    assert all(config.CHUNK_MIN_S <= c.duration <= config.CHUNK_MAX_S for c in chunks)


def test_clicks_are_dropped():
    assert feed(Chunker("P"), [(0.1, True), (1.0, False)]) == []


# --- transcript ------------------------------------------------------------------
def seg(source, text, t_end, duration=2.0, lang="de"):
    return Segment(source, text, lang, duration, t_end)


def test_turn_merging_and_speaker_switch():
    tr = Transcript()
    _, new1, _ = tr.add(seg("Participant", "Hallo zusammen,", 10.0))
    _, new2, _ = tr.add(seg("Participant", "wie geht es?", 12.5))  # gap 0.5 s
    _, new3, _ = tr.add(seg("User", "Gut, danke.", 15.0))
    _, new4, _ = tr.add(seg("Participant", "Schön.", 30.0))
    assert (new1, new2, new3, new4) == (True, False, True, True)
    assert tr.render().splitlines()[0] == "[Participant] Hallo zusammen, wie geht es?"


def test_echo_mic_after_loopback_is_dropped():
    tr = Transcript()
    tr.add(seg("Participant", "Das Budget liegt bei fünfzigtausend Euro", 10.0))
    turn, _, _ = tr.add(seg("User", "das Budget liegt bei fünfzigtausend Euro", 10.4))
    assert turn is None and len(tr.turns) == 1


def test_echo_mic_before_loopback_is_retracted():
    tr = Transcript()
    tr.add(seg("User", "das Budget liegt bei fünfzigtausend Euro", 10.0))
    _, _, retracted = tr.add(seg("Participant", "Das Budget liegt bei fünfzigtausend Euro", 10.3))
    assert len(retracted) == 1
    assert [t.source for t in tr.turns] == ["Participant"]
    assert [s.source for s in tr.segments] == ["Participant"]


def test_short_user_reply_is_not_echo():
    tr = Transcript()
    tr.add(seg("Participant", "Passt das für Sie so, ja?", 10.0))
    turn, _, _ = tr.add(seg("User", "Ja.", 10.5))
    assert turn is not None


# --- facts -------------------------------------------------------------------------
def test_facts_union_and_replace(tmp_path):
    f = Facts(tmp_path / "facts.json")
    f.data = {"names": ["Sarah"], "Budget": "10k"}
    f.merge([{"key": "names", "value": "Tom"}, {"key": "budget", "value": "50,000 EUR"},
             {"key": "Deadline", "value": ""}])
    assert f.data == {"names": ["Sarah", "Tom"], "Budget": "50,000 EUR"}
    assert Facts(tmp_path / "facts.json").data == f.data


# --- llm -----------------------------------------------------------------------------
def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("kein json") is None


class FakeProvider(Provider):
    def __init__(self, id, delay=0.0, error=None, configured=True):
        self.id, self.label, self.model = id, id.title(), f"{id}-model"
        self.delay, self.error, self._configured = delay, error, configured
        self.calls = []  # "warmup" and "complete", in the order they reached the provider
        self.warmup_error = None  # raised by warmup()

    @property
    def configured(self):
        return self._configured

    async def warmup(self):
        self.calls.append("warmup")
        if self.warmup_error:
            raise self.warmup_error

    async def complete(self, system, prompt, schema=None):
        self.calls.append("complete")
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return '{"trigger": "%s", "suggestions": [], "new_facts": []}' % self.id


def _llm(monkeypatch, cloud_delay, ollama_delay, cloud_error=None, cloud=None):
    monkeypatch.setattr(config, "CLOUD_TIMEOUT_S", 0.05)
    monkeypatch.setattr(config, "LLM_PROVIDER", "claude")
    cloud = cloud or GeminiProvider()  # real retry_after parsing for Gemini quota errors
    fake = FakeProvider("claude", cloud_delay, cloud_error)
    fake.retry_after = cloud.retry_after
    return LLM([fake, FakeProvider("grok", configured=False), FakeProvider("ollama", ollama_delay)])


def test_slow_cloud_still_wins_race_against_slower_ollama(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=0.15, ollama_delay=0.5)
    res, engine, _ = asyncio.run(client.generate_json("s", "p"))
    assert engine == "claude" and res["trigger"] == "claude"
    assert client.primary_available


def test_ollama_wins_when_cloud_is_much_slower(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=1.0, ollama_delay=0.05)
    res, engine, _ = asyncio.run(client.generate_json("s", "p"))
    assert engine == "ollama" and res["trigger"] == "ollama"
    assert client.primary_available  # slowness alone does not trip the breaker


def test_cloud_errors_fall_back_and_open_breaker(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=0.0, ollama_delay=0.0, cloud_error=RuntimeError("503"))

    async def run():
        await client.generate_json("s", "p")
        return await client.generate_json("s", "p")

    res, engine, _ = asyncio.run(run())
    assert engine == "ollama" and res["trigger"] == "ollama"
    assert not client.primary_available


def test_gemini_quota_429_skips_primary_for_retry_delay(monkeypatch):
    err = RuntimeError("429 RESOURCE_EXHAUSTED {'retryDelay': '62535s'}")
    err.code = 429
    client = _llm(monkeypatch, cloud_delay=0.0, ollama_delay=0.0, cloud_error=err)
    _, engine, _ = asyncio.run(client.generate_json("s", "p"))
    assert engine == "ollama" and not client.primary_available
    assert "1042 min" in client.note


def test_cloud_primary_does_not_warm_the_local_llm(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=0.0, ollama_delay=0.0)
    asyncio.run(client.warmup())
    assert client.fallback.calls == []


def test_local_primary_is_warmed_at_startup(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=0.0, ollama_delay=0.0)
    client.set_primary("ollama")
    asyncio.run(client.warmup())
    assert client.fallback.calls == ["warmup"]


def test_cloud_failure_warms_the_fallback_once_before_it_answers(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=0.0, ollama_delay=0.0, cloud_error=RuntimeError("503"))

    async def run():
        first = await client.generate_json("s", "p")  # the cloud fails, the fallback answers
        await client.generate_json("s", "p")
        await client.generate_json("s", "p")  # breaker open: straight to the fallback
        return first

    _, engine, _ = asyncio.run(run())
    assert engine == "ollama"
    assert client.fallback.calls == ["warmup", "complete", "complete", "complete"]


def test_hedge_warms_the_fallback_once_for_concurrent_requests(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=1.0, ollama_delay=0.0)

    async def run():
        return await asyncio.gather(client.generate_json("s", "p"), client.generate_json("s", "p"))

    assert [engine for _, engine, _ in asyncio.run(run())] == ["ollama", "ollama"]
    assert client.fallback.calls == ["warmup", "complete", "complete"]


def test_a_failed_warm_up_is_not_kept_and_the_fallback_still_answers(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=0.0, ollama_delay=0.0, cloud_error=RuntimeError("503"))
    client.fallback.warmup_error = ValueError("unexpected reply")  # not an httpx.HTTPError

    async def run():
        engines = [(await client.generate_json("s", "p"))[1] for _ in range(2)]
        client.fallback.warmup_error = None
        engines += [(await client.generate_json("s", "p"))[1] for _ in range(2)]
        return engines

    assert asyncio.run(run()) == ["ollama"] * 4
    # Tried again on every request until one warm-up succeeds, then kept.
    assert client.fallback.calls == ["warmup", "complete", "warmup", "complete", "warmup", "complete", "complete"]


def test_a_failed_warm_up_of_the_local_primary_is_retried(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=0.0, ollama_delay=0.0)
    client.set_primary("ollama")
    client.fallback.warmup_error = ValueError("unexpected reply")

    async def run():
        await client.warmup()  # reported, not raised
        client.fallback.warmup_error = None
        await client.warmup()
        await client.warmup()

    asyncio.run(run())
    assert client.fallback.calls == ["warmup", "warmup"]


def test_http_429_uses_retry_after_header():
    class Resp:
        headers = {"retry-after": "30"}

    err = RuntimeError("rate limited")
    err.status_code, err.response = 429, Resp()
    assert Provider().retry_after(err) == 30
    assert Provider().retry_after(RuntimeError("boom")) is None


def test_switching_provider(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=0.0, ollama_delay=0.0)
    assert client.primary.id == "claude"
    client.set_primary("grok")  # no key -> stays usable via Ollama, with a note
    assert client.primary.id == "ollama" and not client.primary_available and "grok" in client.note
    client.set_primary("claude")
    assert client.primary.id == "claude" and client.note == ""
    listed = client.describe()
    assert listed["active"] == "claude"
    assert [p["configured"] for p in listed["list"]] == [True, False, True]


def test_openai_compatible_downgrades_json_mode():
    import openai

    from copilot.providers import OpenAICompatProvider

    p = OpenAICompatProvider("custom", "OmniRoute", "http://localhost:9/v1", "", "gh/gpt-x", needs_key=False)
    calls = []

    class Choice:
        finish_reason = "stop"

        class message:
            content = '{"trigger": "ok", "suggestions": [], "new_facts": []}'

    class Completions:
        async def create(self, **kw):
            calls.append(kw.get("response_format", {}).get("type"))
            if kw.get("response_format"):
                raise openai.BadRequestError("response_format json_schema not supported",
                                             response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                                             body=None)
            return type("R", (), {"choices": [Choice]})()

    p._client = type("C", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    assert parse_json(asyncio.run(p.complete("s", "p", SUGGESTION_SCHEMA)))["trigger"] == "ok"
    assert calls == ["json_schema", "json_object", None] and p._json_mode == "prompt"


def _ollama_recording(reply):
    """An OllamaProvider whose HTTP client records each request body and answers with `reply`."""
    from copilot.providers import OllamaProvider

    p, sent = OllamaProvider(), []

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": reply}})

    p._http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return p, sent


@pytest.mark.parametrize("schema, reply", [
    (SUGGESTION_SCHEMA, '{"trigger": "ok", "suggestions": [], "new_facts": []}'),
    (None, "Hallo Welt"),
])
def test_ollama_chat_turns_thinking_off(schema, reply):
    p, sent = _ollama_recording(reply)
    assert asyncio.run(p.complete("s", "p", schema)) == reply
    assert sent[0]["think"] is False and sent[0].get("format") == schema


def test_ollama_warmup_turns_thinking_off():
    p, sent = _ollama_recording("")
    asyncio.run(p.warmup())
    assert sent[0]["think"] is False and sent[0]["messages"] == []


def test_schema_is_closed_for_strict_providers():
    assert SUGGESTION_SCHEMA["additionalProperties"] is False
    assert SUGGESTION_SCHEMA["properties"]["suggestions"]["items"]["additionalProperties"] is False


# --- memory --------------------------------------------------------------------------
def _memory_assistant(tmp_path, reply=None):
    """A fake assistant whose LLM records every memory call as (system, prompt)."""
    from copilot.memory import Memory

    class FakeLLM:
        calls = []

        async def generate(self, system, prompt):
            self.calls.append((system, prompt))
            return reply or f"- Zusammenfassung {len(self.calls)}", "gemini", 0.5

    class FakeAssistant:
        transcript = Transcript()
        memory = Memory(tmp_path / "s.memory.md")
        llm = FakeLLM()
        events = []
        emit = events.append

    return FakeAssistant()


def _summarize_once(tmp_path, reply):
    from copilot import memory

    a = _memory_assistant(tmp_path, reply)
    a.transcript.add(seg("Participant", "We need the website by the end of March " * 6, 5.0))
    asyncio.run(memory.summarize(a))
    return a.llm.calls[0][0], a


def test_memory_summarizes_only_new_segments(tmp_path):
    from copilot import memory

    a = _memory_assistant(tmp_path)
    prompts = lambda: [prompt for _, prompt in a.llm.calls]
    a.transcript.add(seg("Participant", "Kurz.", 1.0))
    asyncio.run(memory.summarize(a))
    assert prompts() == []  # below MEMORY_MIN_WORDS

    a.transcript.add(seg("Participant", "Wir brauchen die Website bis Ende März " * 6, 5.0))
    asyncio.run(memory.summarize(a))
    assert "Ende März" in prompts()[0] and a.memory.text == "- Zusammenfassung 1"
    assert (tmp_path / "s.memory.md").read_text(encoding="utf-8") == "- Zusammenfassung 1"

    a.transcript.add(seg("User", "Das bekommen wir hin, versprochen " * 8, 30.0))
    asyncio.run(memory.summarize(a))
    assert "Ende März" not in prompts()[1].split("[NEW TRANSCRIPT]")[1]  # only the new part
    assert "- Zusammenfassung 1" in prompts()[1]  # previous memory is carried forward
    assert a.events[-1]["type"] == "memory"


def test_memory_follows_reply_language(tmp_path, monkeypatch):
    from copilot import export

    monkeypatch.setattr(config, "REPLY_LANGUAGE", "en")
    reply = "## Discussion\n- Website by the end of March"
    system, a = _summarize_once(tmp_path, reply)
    assert "'en'" in system and "German" not in system
    assert "## Discussion" in system and "## Decisions & commitments" in system and "## Verlauf" not in system
    # the panel and the meeting record show the memory as written
    assert a.events[-1] == {"type": "memory", "text": reply, "engine": "gemini", "ms": 500}
    record = export.render(a.transcript.segments, a.memory, Facts(tmp_path / "facts.json"), [], 0.0, 60.0)
    assert reply in record


def test_reply_language_overrides_the_prompt_and_is_repeated(monkeypatch):
    from copilot.assistant import proactive_instruction, reply_reminder, suggestion_system

    monkeypatch.setattr(config, "REPLY_LANGUAGE", "en")
    assert "overrides every other language instruction" in suggestion_system().split("# Language")[-1]
    assert proactive_instruction().endswith("code 'en'.") and "'en'" in reply_reminder()
    monkeypatch.setattr(config, "REPLY_LANGUAGE", "")
    assert reply_reminder() == "" and "# Language" not in suggestion_system()


def test_memory_stays_german_without_reply_language(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPLY_LANGUAGE", "")
    system, _ = _summarize_once(tmp_path, "- Zusammenfassung")
    assert "in German" in system
    for heading in ("## Verlauf", "## Entscheidungen & Zusagen", "## Offene Punkte & Einwände",
                    "## Stimmung & Prioritäten der Teilnehmer"):
        assert heading in system


def test_kb_sources_include_extra_files_and_folders(tmp_path, monkeypatch):
    from copilot import rag

    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "a.md").write_text("x")
    (tmp_path / "kb" / "skip.txt").write_text("x")
    (tmp_path / "prep.md").write_text("x")
    (tmp_path / "vault" / "sub").mkdir(parents=True)
    (tmp_path / "vault" / "sub" / "b.md").write_text("x")
    monkeypatch.setattr(config, "KB_DIR", tmp_path / "kb")
    monkeypatch.setattr(config, "KB_EXTRA_PATHS", [tmp_path / "prep.md", tmp_path / "vault"])
    assert [p.name for p in rag._sources()] == ["a.md", "prep.md", "b.md"]


# --- misc ----------------------------------------------------------------------------
def test_hallucination_filter():
    assert is_hallucination("Untertitel im Auftrag des ZDF", 0.5, -0.5)
    assert is_hallucination("Untertitel im Auftrag des ZDF für funk, 2017", 0.0, -0.2)
    assert is_hallucination("  ...  ", 0.0, 0.0)
    assert not is_hallucination("Was kostet das?", 0.1, -0.3)


def test_split_text_respects_size():
    text = "\n\n".join(f"Absatz {i}. " + "Wort " * 60 for i in range(10))
    chunks = split_text(text)
    assert len(chunks) > 1 and all(len(c) <= 500 for c in chunks)


# --- platform ------------------------------------------------------------------------
def test_whisper_auto_picks_gpu_or_cpu_defaults(monkeypatch):
    for name in ("WHISPER_DEVICE", "WHISPER_MODEL", "WHISPER_COMPUTE"):
        monkeypatch.setattr(config, name, "auto")
    assert whisper_settings(1) == ("large-v3-turbo", "cuda", "float16")
    assert whisper_settings(0) == ("small", "cpu", "int8")
    monkeypatch.setattr(config, "WHISPER_MODEL", "base")
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cpu")
    assert whisper_settings(1) == ("base", "cpu", "int8")


KIT = "/opt/homebrew/bin/whisperkit-cli"


def test_stt_engine_auto_is_whisperkit_only_on_apple_silicon():
    assert stt_engine("auto", "darwin", "arm64", KIT) == ("whisperkit", None)
    for system, machine in [("darwin", "x86_64"), ("win32", "AMD64"), ("win32", "ARM64"),
                            ("linux", "x86_64"), ("linux", "aarch64")]:
        assert stt_engine("auto", system, machine, KIT) == ("faster-whisper", None)


def test_stt_engine_forced_and_unknown():
    assert stt_engine("faster-whisper", "darwin", "arm64", KIT) == ("faster-whisper", None)
    assert stt_engine("whisperkit", "darwin", "arm64", KIT) == ("whisperkit", None)
    with pytest.raises(ValueError, match="STT_ENGINE"):
        stt_engine("vosk", "darwin", "arm64", KIT)


@pytest.mark.parametrize("settings, system, binary, why", [
    (("auto", "whisperkit"), "darwin", None, "brew install whisperkit-cli"),
    (("whisperkit",), "linux", KIT, "needs macOS"),
])
def test_stt_engine_falls_back_with_a_reason(settings, system, binary, why):
    # A model found nowhere is no reason: the engine downloads it (a failed download falls back).
    for setting in settings:
        engine, reason = stt_engine(setting, system, "arm64", binary)
        assert engine == "faster-whisper" and why in reason
        assert ("requested" in reason) == (setting == "whisperkit")


class FakeEngine:
    def __init__(self, fail=None):
        self.fail, self.warm, self.closed = fail, False, False
        if fail == "start":  # the engine launches the server and waits on /health while it is built
            raise RuntimeError("whisperkit-cli not healthy after 300 s")

    def warm_up(self):
        if self.fail == "warm_up":
            raise RuntimeError("warm-up request failed")
        self.warm = True

    def close(self):
        self.closed = True


@pytest.mark.parametrize("setting", ["auto", "whisperkit"])
@pytest.mark.parametrize("fail, why", [("start", "not healthy"), ("warm_up", "warm-up request failed")])
def test_build_stt_engine_falls_back_when_whisperkit_fails_to_start(setting, fail, why, monkeypatch, capsys):
    built = []

    def kit(on_status, max_growth_gb):
        built.append(FakeEngine(fail))  # only reached when the start succeeds
        return built[-1]

    monkeypatch.setattr(config, "STT_ENGINE", setting)
    monkeypatch.setattr(stt, "chosen_stt_engine", lambda s: ("whisperkit", None))
    monkeypatch.setattr(stt, "ENGINES", {"whisperkit": kit, "faster-whisper": FakeEngine})
    engine = build_stt_engine()
    assert type(engine) is FakeEngine and engine.fail is None and engine.warm
    if fail == "warm_up":
        assert built[0].closed  # a started server that fails the warm-up is stopped
    out = capsys.readouterr().out
    assert why in out and ("requested" in out) == (setting == "whisperkit")


def test_build_stt_engine_prints_the_choice_reason(monkeypatch, capsys):
    monkeypatch.setattr(stt, "chosen_stt_engine", lambda s: ("faster-whisper", "WhisperKit is unavailable: x"))
    monkeypatch.setattr(stt, "ENGINES", {"faster-whisper": FakeEngine})
    assert build_stt_engine().warm
    assert "WhisperKit is unavailable: x" in capsys.readouterr().out


def test_faster_whisper_close_releases_its_stt_model(monkeypatch):
    import gc
    import weakref

    import faster_whisper

    class Model:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(faster_whisper, "WhisperModel", Model)
    engine = stt.FasterWhisperEngine()
    model = weakref.ref(engine.model)
    engine.close()
    gc.collect()
    assert model() is None


def test_hotkeys_convert_to_pynput_format():
    assert to_pynput("ctrl+alt+1") == "<ctrl>+<alt>+1"
    assert to_pynput("ctrl+alt+space") == "<ctrl>+<alt>+<space>"
    assert all(to_pynput(p["hotkey"]) for p in config.PRESETS)


def test_hotkeys_convert_to_win32_modifiers_and_keys():
    assert to_win32("ctrl+alt+t") == (0x2 | 0x1, ord("T"))
    assert to_win32("ctrl+alt+space") == (0x3, 0x20)
    assert to_win32("ctrl+alt+1") == (0x3, ord("1"))


# --- ollama models -------------------------------------------------------------------
def test_ollama_model_switch_and_status_note(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=0, ollama_delay=0)
    ollama = client.fallback
    installed = None

    async def fake_installed():
        return installed

    ollama.installed = fake_installed
    assert asyncio.run(client.check_ollama()) and "isn't running" in client.ollama_note

    installed = ["llama3.2:3b", "qwen3:4b"]
    assert asyncio.run(client.check_ollama()) and "ollama-model is missing" in client.ollama_note
    assert not asyncio.run(client.check_ollama())  # unchanged: no UI update

    client.set_primary("ollama:qwen3:4b")
    assert client.primary is ollama and ollama.model == "qwen3:4b" and client.ollama_note == ""
    assert client.describe()["ollama_models"] == installed


# --- knowledge base sync -------------------------------------------------------------
def test_kb_sync_reindexes_only_on_change_and_drops_stale_index(tmp_path, monkeypatch):
    import lancedb

    from copilot import rag

    monkeypatch.setattr(config, "KB_DIR", tmp_path / "kb")
    monkeypatch.setattr(config, "KB_EXTRA_PATHS", [])
    monkeypatch.setattr(config, "DB_DIR", tmp_path / "db")
    monkeypatch.setattr(rag, "_embed", lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "preise.md").write_text("Eine Website kostet ab 999 Euro.", encoding="utf-8")
    calls = []
    real_index = rag.index
    monkeypatch.setattr(rag, "index", lambda: (calls.append(1), real_index()))

    assert rag.sync() == 1 and len(calls) == 1
    assert rag.sync() == 1 and len(calls) == 1  # nothing changed
    (tmp_path / "kb" / "preise.md").unlink()
    assert rag.sync() == 0 and len(calls) == 2
    assert config.DB_TABLE not in lancedb.connect(str(tmp_path / "db")).list_tables().tables


# --- meeting -------------------------------------------------------------------------
class FakeSource:
    def __init__(self):
        self.open = False

    def start(self):
        self.open = True

    def stop(self):
        self.open = False


def _meeting_assistant(tmp_path, monkeypatch, open_capture, open_stt=None, close_stt=None):
    from copilot.assistant import Assistant
    from copilot.memory import Memory

    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path)
    events = []
    a = Assistant(Transcript(), Facts(tmp_path / "facts.json"), None, None, events.append, Memory(), open_capture,
                  open_stt, close_stt)
    return a, events


@pytest.mark.parametrize("choice, warmups", [
    ("claude", []),  # a cloud primary leaves the local LLM unloaded
    ("ollama", ["warmup"]),
    ("ollama:qwen3:4b", ["warmup"]),
])
def test_switching_the_primary_to_the_local_llm_warms_it(choice, warmups, tmp_path, monkeypatch):
    a, events = _meeting_assistant(tmp_path, monkeypatch, None)
    a.llm = _llm(monkeypatch, cloud_delay=0.0, ollama_delay=0.0)

    async def run():
        a.set_provider(choice)
        await asyncio.sleep(0.01)

    asyncio.run(run())
    assert a.llm.fallback.calls == warmups
    assert events[-1]["type"] == "providers"


def test_a_new_local_model_is_warmed_again(monkeypatch):
    client = _llm(monkeypatch, cloud_delay=0.0, ollama_delay=0.0)

    async def run():
        client.set_primary("ollama:qwen3:4b")
        await client.warmup()
        client.set_primary("ollama:qwen3:4b")  # same model: already loaded
        await client.warmup()
        client.set_primary("ollama:llama3.2:3b")
        await client.warmup()

    asyncio.run(run())
    assert client.fallback.calls == ["warmup", "warmup"]


def test_meeting_opens_devices_only_while_running_and_mute_keeps_participants(tmp_path, monkeypatch):
    class FakeCapture:
        mic, loopback = FakeSource(), FakeSource()
        sources = [mic, loopback]

    cap = FakeCapture()
    a, events = _meeting_assistant(tmp_path, monkeypatch, lambda: cap)

    async def run():
        assert a.capture is None  # idle: no device touched
        a.start_meeting()
        assert cap.mic.open and cap.loopback.open and a.transcript.session_path.parent == tmp_path
        assert {"type": "reset"} in events
        a.set_mic(False)
        assert not cap.mic.open and cap.loopback.open
        a.stop_meeting()
        assert not cap.mic.open and not cap.loopback.open
        a.start_meeting()  # mute is remembered
        assert not cap.mic.open and cap.loopback.open
        await asyncio.sleep(0)

    asyncio.run(run())
    assert events[-1] == {"type": "meeting", "active": True, "mic": False, "error": ""}


class FakeSttWorker:
    """Records whether the STT worker was told to hold an engine."""

    def __init__(self):
        self.calls = []

    def open(self):
        self.calls.append("open")

    def close(self):
        self.calls.append("close")


def test_meeting_opens_the_stt_worker_only_while_running(tmp_path, monkeypatch):
    worker = FakeSttWorker()
    a, _ = _recording(*_meeting_assistant(tmp_path, monkeypatch, None, worker.open, worker.close), tmp_path,
                      monkeypatch)

    async def run():
        assert worker.calls == []  # idle: no STT engine
        a.start_meeting()
        assert worker.calls[-1] == "open"
        a.set_mic(False)  # muting keeps the participant transcribed
        assert worker.calls[-1] == "open"
        await a.stop_meeting()
        assert worker.calls[-1] == "close"
        a.start_meeting()
        assert worker.calls[-1] == "open"
        await a.stop_meeting()

    asyncio.run(run())


def test_a_segment_of_an_ended_meeting_never_lands_in_the_next_one(tmp_path, monkeypatch):
    a, events = _recording(*_meeting_assistant(tmp_path, monkeypatch, None), tmp_path, monkeypatch)
    a.proactive = False

    async def run():
        a.start_meeting()
        ended = a.transcript
        a.on_segment(seg("User", "Erstes Meeting.", time.monotonic()))
        a.stop_meeting()
        # The VAD ends the utterance that was open at the stop: still the ended meeting's.
        a.on_segment(seg("User", "Noch im ersten.", time.monotonic()))
        # Cut before the next start, transcribed after it (the STT engine was already closed).
        # Clearly earlier: Windows' monotonic clock ticks every ~15.6 ms, and a chunk cut in
        # the same tick as the start is kept on purpose (real speech is never dropped).
        stale = seg("User", "Zu spät.", time.monotonic() - 1.0)
        a.start_meeting()
        a.on_segment(stale)
        a.on_segment(seg("User", "Zweites Meeting.", time.monotonic()))
        await a.stop_meeting()
        return ended

    ended = asyncio.run(run())
    assert [s.text for s in ended.segments] == ["Erstes Meeting.", "Noch im ersten."]
    assert [s.text for s in a.transcript.segments] == ["Zweites Meeting."]
    assert "Zu spät." not in [e.get("text") for e in events]


def test_an_audio_error_leaves_the_stt_worker_closed(tmp_path, monkeypatch):
    def broken():
        raise RuntimeError("'parec' not found")

    worker = FakeSttWorker()
    a, _ = _meeting_assistant(tmp_path, monkeypatch, broken, worker.open, worker.close)
    a.start_meeting()
    assert "open" not in worker.calls


def test_meeting_audio_error_is_reported_not_fatal(tmp_path, monkeypatch):
    def broken():
        raise RuntimeError("'parec' not found")

    a, events = _meeting_assistant(tmp_path, monkeypatch, broken)
    a.start_meeting()
    assert events[-1] == {"type": "meeting", "active": False, "mic": True, "error": "'parec' not found"}


def _recording(a, events, tmp_path, monkeypatch):
    """The assistant with fake devices, saving its meeting records to a temporary folder."""
    cap = SimpleNamespace(mic=FakeSource(), loopback=FakeSource())
    cap.sources = [cap.mic, cap.loopback]
    a.open_capture = lambda: cap
    monkeypatch.setattr(config, "MEETINGS_DIR", tmp_path / "meetings")
    return a, events


@pytest.mark.parametrize("lang, lines", [
    ("en", ["## Summary", "(none, the conversation was too short)", "## Facts", "(none yet)",
            "## Suggestions and answers", "Question: What now?", "## Transcript",
            "**Participant:** Was kostet die Website?", "**You:** Ab 999 Euro."]),
    ("de", ["## Zusammenfassung", "(keine, das Gespräch war zu kurz)", "## Fakten", "(noch keine)",
            "## Vorschläge und Antworten", "Frage: What now?", "## Transkript",
            "**Teilnehmer:** Was kostet die Website?", "**Du:** Ab 999 Euro."]),
])
def test_meeting_record_is_saved_on_stop_with_headings_in_the_ui_language(tmp_path, monkeypatch, lang, lines):
    monkeypatch.setattr(config, "UI_LANGUAGE", lang)
    a, events = _recording(*_meeting_assistant(tmp_path, monkeypatch, None), tmp_path, monkeypatch)

    async def run():
        a.start_meeting()
        t0 = time.monotonic()  # Segment.t_end: cut during the meeting
        a.on_segment(seg("Participant", "Was kostet die Website?", t0 + 1.0))
        a.on_segment(seg("User", "Ab 999 Euro.", t0 + 3.0))
        a._card({"type": "suggestion", "trigger": "Preisfrage", "ts": 0,
                 "suggestions": [{"text": "Das hängt vom Umfang ab."}]})
        a._card({"type": "answer", "query": "What now?", "ts": 0, "suggestions": [{"text": "Wait."}]})
        await a.stop_meeting()

    asyncio.run(run())
    [record] = (tmp_path / "meetings").iterdir()
    text = record.read_text(encoding="utf-8")
    for line in lines + ["Preisfrage", "Das hängt vom Umfang ab."]:  # LLM-written text stays as written
        assert line in text
    assert events[-1] == {"type": "saved", "path": str(record)}


def test_empty_meeting_writes_no_record(tmp_path, monkeypatch):
    a, _ = _recording(*_meeting_assistant(tmp_path, monkeypatch, None), tmp_path, monkeypatch)

    async def run():
        a.start_meeting()
        await a.stop_meeting()

    asyncio.run(run())
    assert not (tmp_path / "meetings").exists()


# --- languages -----------------------------------------------------------------------
class RecordingLLM:
    def __init__(self, reply=lambda prompt: "Roman " + prompt, delay=0.0, error=None):
        self.calls, self.reply, self.delay, self.error = [], reply, delay, error

    async def generate(self, system, prompt):
        self.calls.append((system, prompt))
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.reply(prompt), "ollama", self.delay


def test_translation_goes_into_your_language_and_from_it_into_the_second(tmp_path, monkeypatch):
    a, _ = _meeting_assistant(tmp_path, monkeypatch, None)
    a.llm = RecordingLLM()
    monkeypatch.setattr(config, "LANGUAGES", ["ar", "de", "en"])
    for lang in ("tr", "ar"):
        asyncio.run(a._translate(seg("Participant", "Merhaba", 1.0, lang=lang)))
    assert [system.split("'")[-2] for system, _ in a.llm.calls] == ["ar", "de"]


def test_translations_run_one_at_a_time_and_stale_lines_are_skipped(tmp_path, monkeypatch):
    a, events = _meeting_assistant(tmp_path, monkeypatch, None)
    a.llm = RecordingLLM(reply=lambda prompt: prompt, delay=0.2)
    monkeypatch.setattr(config, "PLAIN_MAX_WAIT_S", 0.3)

    async def run():
        await asyncio.gather(*(a._translate(seg("Participant", f"line {i}", 1.0)) for i in range(4)))

    asyncio.run(run())
    # line 0 runs at once, line 1 waits 0.2 s, lines 2 and 3 have waited 0.4 s > 0.3 s
    assert [prompt for _, prompt in a.llm.calls] == ["line 0", "line 1"]
    assert [e["text"] for e in events if e["type"] == "translation"] == ["line 0", "line 1"]


# --- window --------------------------------------------------------------------------
def test_window_settings_are_clamped_and_persisted(tmp_path, monkeypatch):
    from copilot import window

    monkeypatch.setattr(window, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(window, "supported", lambda: False)
    w = window.Window()
    assert w.state() == {"type": "window", "supported": False, "on_top": True, "opacity": 1.0}
    w.set(on_top=False, opacity=0.1)
    assert window.Window().settings == {"on_top": False, "opacity": 0.4}


# --- proactive pacing ----------------------------------------------------------------
def _proactive_assistant(tmp_path, monkeypatch, triggers):
    a, events = _meeting_assistant(tmp_path, monkeypatch, None)
    answers = iter(triggers)

    class FakeLLM:
        async def generate_json(self, system, prompt):
            t = next(answers)
            return {"trigger": t, "suggestions": [{"text": f"{t} 1"}, {"text": f"{t} 2"}], "new_facts": []}, "ollama", 0.1

    a.llm = FakeLLM()
    a.kb = SimpleNamespace(context=lambda q: "")
    a.transcript.add(seg("Participant", "Das ist uns zu teuer.", 1.0))
    return a, events


def test_proactive_skips_the_same_trigger_but_shows_a_new_one(tmp_path, monkeypatch):
    a, events = _proactive_assistant(tmp_path, monkeypatch, ["Preiseinwand", "preiseinwand", "Terminfrage"])
    for _ in range(3):
        asyncio.run(a._run_proactive())
    assert [e["trigger"] for e in events if e["type"] == "suggestion"] == ["Preiseinwand", "Terminfrage"]


def test_single_suggestion_option(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SINGLE_SUGGESTION", True)
    a, events = _proactive_assistant(tmp_path, monkeypatch, ["Preiseinwand"])
    asyncio.run(a._run_proactive())
    [card] = [e for e in events if e["type"] == "suggestion"]
    assert [s["text"] for s in card["suggestions"]] == ["Preiseinwand 1"]


def test_proactive_waits_for_the_end_of_the_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROACTIVE_DEBOUNCE_S", 0.2)
    monkeypatch.setattr(config, "PROACTIVE_MAX_PENDING_S", 5)
    a, _ = _meeting_assistant(tmp_path, monkeypatch, None)
    fired = []
    a._fire = lambda: fired.append(True)

    async def run():
        for _ in range(5):  # the participant keeps talking: one segment every 0.1 s
            a._schedule()
            await asyncio.sleep(0.1)
        assert fired == []
        await asyncio.sleep(0.25)  # pause after the turn
        assert fired == [True]

    asyncio.run(run())


# --- UI language ---------------------------------------------------------------------
@pytest.mark.parametrize("lang, note", [
    ("en", "Ollama model ollama-model is missing: ollama pull ollama-model, or pick an installed one above"),
    ("de", "Ollama-Modell ollama-model fehlt: ollama pull ollama-model, oder oben ein installiertes wählen"),
])
def test_missing_ollama_model_note_follows_ui_language(monkeypatch, lang, note):
    monkeypatch.setattr(config, "UI_LANGUAGE", lang)
    client = _llm(monkeypatch, cloud_delay=0, ollama_delay=0)

    async def fake_installed():
        return ["llama3.2:3b"]

    client.fallback.installed = fake_installed
    asyncio.run(client.check_ollama())
    assert client.ollama_note == note


@pytest.mark.parametrize("lang, error", [
    ("en", "'parec' not found. Install it (Debian/Ubuntu/Fedora: pulseaudio-utils, Arch: libpulse)."),
    ("de", "'parec' nicht gefunden. Bitte installieren (Debian/Ubuntu/Fedora: pulseaudio-utils, Arch: libpulse)."),
])
def test_audio_device_error_follows_ui_language_and_console_stays_english(tmp_path, monkeypatch, capsys,
                                                                           lang, error):
    from copilot import audio

    monkeypatch.setattr(config, "UI_LANGUAGE", lang)
    monkeypatch.setattr(audio.sys, "platform", "linux")
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    a, events = _meeting_assistant(tmp_path, monkeypatch, audio.Capture)
    a.start_meeting()
    assert events[-1] == {"type": "meeting", "active": False, "mic": True, "error": error}
    assert "Audio:    'parec' not found. Install it" in capsys.readouterr().out


def test_answer_that_is_not_json_is_reported_in_the_ui_language(tmp_path, monkeypatch):
    class FakeLLM:
        async def generate_json(self, system, prompt):
            return None, "ollama", 0.1

    class NoKnowledge:
        def context(self, query):
            return ""

    monkeypatch.setattr(config, "UI_LANGUAGE", "en")
    a, events = _meeting_assistant(tmp_path, monkeypatch, None)
    a.llm, a.kb = FakeLLM(), NoKnowledge()
    asyncio.run(a.ask("What now?"))
    assert events[-1] == {"type": "error", "text": "The answer wasn't valid JSON."}


@pytest.mark.parametrize("lang, label", [("en", "Ollama (local)"), ("de", "Ollama (lokal)")])
def test_ollama_label_follows_ui_language(monkeypatch, lang, label):
    from copilot.providers import OllamaProvider

    monkeypatch.setattr(config, "UI_LANGUAGE", lang)
    client = LLM([FakeProvider("claude"), OllamaProvider()])
    assert {p["id"]: p["label"] for p in client.describe()["list"]}["ollama"] == label


# --- presets per UI language ---------------------------------------------------------
@pytest.mark.parametrize("lang, labels, first_query", [
    ("en", ["What should I answer?", "Weak points", "Summary"], "What should I answer now?"),
    ("de", ["Was antworten?", "Schwächen", "Zusammenfassung"], "Was soll ich jetzt antworten?"),
])
def test_presets_carry_the_ui_language_labels_and_questions(lang, labels, first_query, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UI_LANGUAGE", lang)
    a, _ = _meeting_assistant(tmp_path, monkeypatch, None)
    presets = a.presets()
    assert [p["label"] for p in presets] == labels
    assert presets[0]["query"] == first_query
    assert [p["hotkey"] for p in presets] == ["ctrl+alt+1", "ctrl+alt+2", "ctrl+alt+3"]


@pytest.mark.parametrize("lang, question", [("en", "What should I answer now?"),
                                            ("de", "Was soll ich jetzt antworten?")])
def test_preset_hotkey_sends_its_question_in_the_ui_language(lang, question, tmp_path, monkeypatch):
    from copilot import hotkeys

    monkeypatch.setattr(config, "UI_LANGUAGE", lang)
    a, events = _meeting_assistant(tmp_path, monkeypatch, None)
    prompts = []

    class FakeLLM:
        async def generate_json(self, system, prompt):
            prompts.append(prompt)
            return {"trigger": "Answer", "suggestions": [{"text": "Yes."}], "new_facts": []}, "ollama", 0.1

    a.llm = FakeLLM()
    a.kb = SimpleNamespace(context=lambda q: "")
    bound = {}
    monkeypatch.setattr(hotkeys, "_bind", bound.update)

    async def press():
        hotkeys.register(asyncio.get_running_loop(), a, events.append, SimpleNamespace(supported=False))
        bound["ctrl+alt+1"]()  # on the hotkey thread in the app
        for _ in range(20):
            await asyncio.sleep(0.01)

    asyncio.run(press())
    assert prompts[0].endswith(f"[USER QUESTION]\n{question}")
    [card] = [e for e in events if e["type"] == "answer"]
    assert card["query"] == question


# --- transliteration -------------------------------------------------------------------
def _speak(a, *segments):
    """Feed segments into the assistant on a loop and wait for the LLM tasks it starts."""
    async def run():
        for s in segments:
            a.on_segment(s)
        await _drain()

    asyncio.run(run())


async def _drain():
    """Wait for the LLM tasks the assistant started."""
    while tasks := asyncio.all_tasks() - {asyncio.current_task()}:
        await asyncio.gather(*tasks)


def _urdu(source="Participant", text="آپ کیسے ہیں", t_end=1.0, lang="ur"):
    return seg(source, text, t_end, lang=lang)


def _transliterating(tmp_path, monkeypatch, llm=None, langs=("ur",)):
    a, events = _meeting_assistant(tmp_path, monkeypatch, None)
    a.llm = llm or RecordingLLM()
    a.proactive = False
    monkeypatch.setattr(config, "TRANSLITERATE", list(langs))
    return a, events


def test_a_transliterate_language_segment_gets_a_transliteration_event(tmp_path, monkeypatch):
    a, events = _transliterating(tmp_path, monkeypatch)
    s = _urdu()
    _speak(a, s)
    assert [e for e in events if e["type"] == "transliteration"] == [
        {"type": "transliteration", "id": s.id, "source": "Participant", "text": "Roman آپ کیسے ہیں",
         "engine": "ollama", "ms": 0}]


@pytest.mark.parametrize("langs, lang", [(("ur",), "en"), ((), "ur")])
def test_other_languages_or_an_empty_setting_make_no_transliteration_call(langs, lang, tmp_path, monkeypatch):
    a, events = _transliterating(tmp_path, monkeypatch, langs=langs)
    _speak(a, _urdu(lang=lang), _urdu(source="User", lang=lang, t_end=20.0))
    assert a.llm.calls == [] and not [e for e in events if e["type"] == "transliteration"]


def test_transliteration_prompt_keeps_the_language_latin_words_names_and_numbers(tmp_path, monkeypatch):
    a, _ = _transliterating(tmp_path, monkeypatch)
    _speak(a, _urdu(text="میرا meeting کل 3 بجے Ahmed کے ساتھ ہے"))
    [(system, prompt)] = a.llm.calls
    assert prompt == "میرا meeting کل 3 بجے Ahmed کے ساتھ ہے"
    assert "'ur'" in system and "Latin script" in system and "Do not translate" in system
    assert "already in Latin script unchanged" in system and "names and numbers intact" in system


def test_the_transcript_context_keeps_the_original_script(tmp_path, monkeypatch):
    a, events = _transliterating(tmp_path, monkeypatch)
    prompts = []

    async def generate_json(system, prompt):
        prompts.append(prompt)
        return {"trigger": "None", "suggestions": []}, "ollama", 0.1

    a.llm.generate_json = generate_json
    a.kb = SimpleNamespace(context=lambda q: "")
    _speak(a, _urdu())
    assert any(e["type"] == "transliteration" for e in events)
    asyncio.run(a._run_proactive())
    assert "[Participant] آپ کیسے ہیں" in prompts[0] and "Roman" not in prompts[0]


def test_transliteration_shares_the_translation_queue_and_skips_stale_lines(tmp_path, monkeypatch):
    a, events = _transliterating(tmp_path, monkeypatch, RecordingLLM(delay=0.2))
    a.translate = True
    monkeypatch.setattr(config, "PLAIN_MAX_WAIT_S", 0.3)
    _speak(a, _urdu(text="ایک"), _urdu(text="دو", t_end=10.0))
    # one queue: transliteration "ایک" runs at once, its translation waits 0.2 s, both of "دو" 0.4 s+
    assert [prompt for _, prompt in a.llm.calls] == ["ایک", "ایک"]
    assert [(e["type"], e["text"]) for e in events if e["type"] in ("translation", "transliteration")] == [
        ("transliteration", "Roman ایک"), ("translation", "Roman ایک")]


def test_a_segment_retracted_as_echo_gets_no_transliteration(tmp_path, monkeypatch):
    a, events = _transliterating(tmp_path, monkeypatch, RecordingLLM(delay=0.1))
    mic = _urdu(source="User", text="آپ کیسے ہیں میرے دوست", t_end=10.0)
    loopback = _urdu(text="آپ کیسے ہیں میرے دوست", t_end=10.3)
    _speak(a, mic, loopback)
    assert {"type": "retract", "id": mic.id} in events
    assert [e["id"] for e in events if e["type"] == "transliteration"] == [loopback.id]


def test_a_failed_transliteration_emits_one_error_and_no_event(tmp_path, monkeypatch):
    a, events = _transliterating(tmp_path, monkeypatch, RecordingLLM(error=RuntimeError("quota")))
    _speak(a, _urdu())
    assert [e["type"] for e in events] == ["segment", "error"]
    assert events[1] == {"type": "error", "text": "Transliteration: quota"}


def test_a_reloaded_window_replays_transliterations():
    from copilot.server import Hub

    hub = Hub()
    hub.publish({"type": "segment", "id": 1, "source": "Participant", "text": "آپ کیسے ہیں"})
    hub.publish({"type": "transliteration", "id": 1, "source": "Participant", "text": "Aap kaise hain"})
    assert [e["type"] for e in hub.history] == ["segment", "transliteration"]


def _transliterated_meeting(tmp_path, monkeypatch):
    """A meeting with an Urdu line (transliterated), a German line (not in TRANSLITERATE) and an
    Urdu line whose transliteration failed, started and stopped through the assistant."""
    def reply(prompt):
        if prompt != "آپ کیسے ہیں":
            raise RuntimeError("quota")
        return "Aap kaise hain"

    a, _ = _recording(*_transliterating(tmp_path, monkeypatch, RecordingLLM(reply=reply)), tmp_path, monkeypatch)
    monkeypatch.setattr(config, "UI_LANGUAGE", "de")  # the record's speaker labels

    async def run():
        a.start_meeting()
        t0 = time.monotonic()  # Segment.t_end: cut during the meeting
        for s in (_urdu(t_end=t0 + 1.0), seg("User", "Guten Tag.", t0 + 3.0),
                  _urdu(text="ٹھیک ہوں", t_end=t0 + 6.0)):
            a.on_segment(s)
        await _drain()
        await a.stop_meeting()  # too few words for a memory call

    asyncio.run(run())
    return a


def test_the_meeting_record_shows_the_transliteration_and_the_original_elsewhere(tmp_path, monkeypatch):
    _transliterated_meeting(tmp_path, monkeypatch)
    [record] = (tmp_path / "meetings").iterdir()
    text = record.read_text(encoding="utf-8")
    assert "**Teilnehmer:** Aap kaise hain" in text and "آپ کیسے ہیں" not in text
    assert "**Du:** Guten Tag." in text and "**Teilnehmer:** ٹھیک ہوں" in text


def test_the_session_log_keeps_the_original_and_adds_the_transliteration(tmp_path, monkeypatch):
    a = _transliterated_meeting(tmp_path, monkeypatch)
    lines = a.transcript.session_path.read_text(encoding="utf-8").splitlines()
    entries = [(e["text"], e.get("transliteration")) for e in map(json.loads, lines)]
    assert entries == [("آپ کیسے ہیں", "Aap kaise hain"), ("Guten Tag.", None), ("ٹھیک ہوں", None)]


def test_a_cut_off_session_log_line_is_kept_and_the_transliteration_still_arrives(tmp_path, monkeypatch):
    a, events = _transliterating(tmp_path, monkeypatch)
    a.start_meeting()
    s = _urdu()
    a.transcript.add(s)
    with open(a.transcript.session_path, "a", encoding="utf-8") as f:
        f.write('{"source": "Partic\n')
    asyncio.run(a._transliterate(s))
    lines = a.transcript.session_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["transliteration"] == "Roman آپ کیسے ہیں" and lines[1] == '{"source": "Partic'
    assert events[-1]["type"] == "transliteration"


def test_a_transliteration_in_flight_at_stop_still_reaches_the_record(tmp_path, monkeypatch):
    a, _ = _recording(*_transliterating(tmp_path, monkeypatch, RecordingLLM(delay=0.1)), tmp_path, monkeypatch)
    monkeypatch.setattr(config, "UI_LANGUAGE", "de")  # the record's speaker labels

    async def run():
        a.start_meeting()
        a.on_segment(_urdu(t_end=time.monotonic()))
        await asyncio.sleep(0)  # the transliteration call has started
        await a.stop_meeting()

    asyncio.run(run())
    [record] = (tmp_path / "meetings").iterdir()
    assert "**Teilnehmer:** Roman آپ کیسے ہیں" in record.read_text(encoding="utf-8")
