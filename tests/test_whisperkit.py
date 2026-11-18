"""WhisperKit STT engine against a fake whisperkit-cli server: no Mac or WhisperKit needed."""

import json
import threading
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from copilot import config, stt, whisperkit
from copilot.vad import Chunk

MODEL = "openai_whisper-large-v3-v20240930"


def seg(text, no_speech_prob=0.0, avg_logprob=-0.2, compression_ratio=1.2):
    return {"text": text, "no_speech_prob": no_speech_prob, "avg_logprob": avg_logprob,
            "compression_ratio": compression_ratio}


class FakeServer:
    """/health and /v1/audio/transcriptions; reply(fields) gives the verbose_json body."""

    def __init__(self):
        self.engines = []  # closed by the fixture
        self.requests = []  # form fields of each transcription request (file left out)
        self.reply = lambda fields: {"language": fields.get("language", "en"), "segments": [seg(" Hello there.")]}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self._send(200, {"status": "ok"} if self.path == "/health" else {})

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                head = f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode()
                msg = BytesParser(policy=default).parsebytes(head + body)
                fields = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                          for p in msg.iter_parts()}
                assert fields.pop("file").startswith(b"RIFF")
                fields = {k: v.decode() for k, v in fields.items()}
                fake.requests.append(fields)
                self._send(200, fake.reply(fields))

            def _send(self, status, data):
                out = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, args=(0.05,), daemon=True).start()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(config, "WHISPER_LANGUAGE", "auto")
    monkeypatch.setattr(config, "LANGUAGES", ["de", "en"])
    fake = FakeServer()
    yield fake
    for engine in fake.engines:
        engine.close()
    fake.httpd.shutdown()
    fake.httpd.server_close()


def engine_for(server):
    engine = whisperkit.WhisperKitEngine(url=server.url)
    server.engines.append(engine)
    engine.warm_up()
    server.requests.clear()
    return engine


def chunk(source="Participant", seconds=2.0):
    return Chunk(source, np.zeros(int(seconds * config.SAMPLE_RATE), dtype=np.float32), seconds, 42.0)


def make_model(folder):
    (folder / "AudioEncoder.mlmodelc").mkdir(parents=True)
    return folder


# --- Finding the STT model -----------------------------------------------------
class FakeHub:
    """huggingface_hub.snapshot_download over a fake repo: the cache is a folder, the
    network a list of downloads (allow_patterns of each); network=False makes downloads fail."""

    def __init__(self, cache):
        self.cache, self.downloads, self.network = cache, [], True

    def snapshot_download(self, repo_id, allow_patterns, local_files_only=False):
        from huggingface_hub.errors import LocalEntryNotFoundError

        assert repo_id == "argmaxinc/whisperkit-coreml"
        model = allow_patterns.removesuffix("/*")
        if not (self.cache / model).is_dir():
            if local_files_only or not self.network:  # what the real hub raises for both
                raise LocalEntryNotFoundError(f"{model} is not in the cache")
            self.downloads.append(allow_patterns)
            make_model(self.cache / model)
        return str(self.cache)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import huggingface_hub

    fake = FakeHub(tmp_path / "hf-cache")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    monkeypatch.setattr(whisperkit, "MACWHISPER_MODELS", tmp_path / "MacWhisper")
    monkeypatch.setattr(config, "WHISPERKIT_MODEL", MODEL)
    monkeypatch.setattr(config, "WHISPERKIT_MODEL_PATH", "")
    return fake


def test_model_search_order(tmp_path, hub, monkeypatch, capsys):
    # Nowhere yet: only the configured model's subfolder is downloaded, into the HF cache.
    assert whisperkit.model_path() == hub.cache / MODEL
    assert hub.downloads == [f"{MODEL}/*"]

    # A second run finds it in the cache without the network.
    hub.network = False
    assert whisperkit.model_path() == hub.cache / MODEL
    assert len(hub.downloads) == 1

    # MacWhisper's own pick is another model; the configured one is used, ahead of the cache.
    make_model(whisperkit.MACWHISPER_MODELS / "openai_whisper-base")
    assert whisperkit.model_path() == hub.cache / MODEL
    mac = make_model(whisperkit.MACWHISPER_MODELS / MODEL)
    assert whisperkit.model_path() == mac

    own = tmp_path / "mine"
    monkeypatch.setattr(config, "WHISPERKIT_MODEL_PATH", str(own))
    assert whisperkit.model_path() == mac  # an override without a model is skipped, and said so
    assert "WHISPERKIT_MODEL_PATH" in capsys.readouterr().out
    make_model(own)
    assert whisperkit.model_path() == own
    assert len(hub.downloads) == 1


def test_a_failed_download_falls_back_to_faster_whisper_with_the_reason(hub, monkeypatch, capsys):
    class FasterWhisper:
        def warm_up(self):
            pass

        def close(self):
            pass

    hub.network = False
    monkeypatch.setattr(stt, "chosen_stt_engine", lambda s: ("whisperkit", None))
    monkeypatch.setattr(stt, "ENGINES", {"whisperkit": whisperkit.WhisperKitEngine, "faster-whisper": FasterWhisper})
    assert type(stt.build_stt_engine("auto")) is FasterWhisper
    out = capsys.readouterr().out
    assert f"could not download model {MODEL}" in out and "using faster-whisper" in out


# --- Launching the server ------------------------------------------------------
def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_server_launches_with_compute_units_and_no_temperature_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WHISPERKIT_COMPUTE_UNITS", "cpuAndNeuralEngine")
    cmd = whisperkit.server_command("/opt/homebrew/bin/whisperkit-cli", tmp_path, 50123)
    assert cmd[:2] == ["/opt/homebrew/bin/whisperkit-cli", "serve"]
    assert flag(cmd, "--model-path") == str(tmp_path)
    assert flag(cmd, "--host") == "127.0.0.1" and flag(cmd, "--port") == "50123"
    assert flag(cmd, "--temperature-fallback-count") == "0"
    assert flag(cmd, "--audio-encoder-compute-units") == "cpuAndNeuralEngine"
    assert flag(cmd, "--text-decoder-compute-units") == "cpuAndNeuralEngine"
    assert "--prompt" not in cmd and "--model" not in cmd  # never downloads by itself


# --- Transcribing a chunk ------------------------------------------------------
def test_chunk_becomes_one_segment_without_a_prompt(server):
    server.reply = lambda f: {"language": "en", "segments": [seg(" Hello there."), seg(" How are you?")]}
    engine = engine_for(server)
    s = engine.transcribe(chunk())
    assert (s.source, s.text, s.lang, s.duration, s.t_end) == ("Participant", "Hello there. How are you?", "en", 2.0, 42.0)
    req = server.requests[0]
    assert req["temperature"] == "0" and req["response_format"] == "verbose_json"
    assert "prompt" not in req
    engine.transcribe(chunk())
    assert all("prompt" not in r for r in server.requests)


def test_hallucinations_loops_and_special_tokens_are_dropped(server):
    loop = " Das sind 400.000.000.000.000.000.000.000.000.000."
    server.reply = lambda f: {"language": "de", "segments": [
        seg("<|de|><|transcribe|> Guten Morgen.<|endoftext|>"),
        seg(" Untertitel im Auftrag des ZDF, 2021"),
        seg(loop, compression_ratio=3.1),
        seg(" Vielen Dank.", no_speech_prob=0.5),
        seg("<|nocaptions|>"),
    ]}
    engine = engine_for(server)
    assert engine.transcribe(chunk()).text == "Guten Morgen."

    server.reply = lambda f: {"language": "de", "segments": [seg(loop, compression_ratio=3.1), seg(" ...")]}
    assert engine.transcribe(chunk()) is None
    server.reply = lambda f: {"language": "de", "segments": []}
    assert engine.transcribe(chunk()) is None


# --- Language ------------------------------------------------------------------
def sent_languages(server):
    langs = [r.get("language") for r in server.requests]
    server.requests.clear()
    return langs


def test_language_is_sticky_per_source_and_redetected_on_schedule(server, monkeypatch):
    monkeypatch.setattr(config, "LANG_DETECT_EVERY", 3)
    server.reply = lambda f: {"language": f.get("language", "en"), "segments": [seg(" Hi.")]}
    engine = engine_for(server)
    assert engine.transcribe(chunk()).lang == "en"  # first chunk: detected
    engine.transcribe(chunk())
    engine.transcribe(chunk("User"))  # its own sticky language
    assert sent_languages(server) == [None, "en", None]
    engine.transcribe(chunk(seconds=1.0))  # 3rd chunk is due, but too short to detect reliably
    engine.transcribe(chunk())
    engine.transcribe(chunk())
    engine.transcribe(chunk())  # 6th chunk: due again
    assert sent_languages(server) == ["en", "en", "en", None]


def test_language_outside_languages_is_thrown_away(server):
    detected = {"lang": "cy"}  # Welsh, a typical misdetection
    server.reply = lambda f: {"language": f.get("language", detected["lang"]), "segments": [seg(" Hallo.")]}
    engine = engine_for(server)
    # First chunk: nothing sticky yet, so it is sent again in your language.
    assert engine.transcribe(chunk()).lang == "de"
    assert sent_languages(server) == [None, "de"]
    detected["lang"] = "en"
    for _ in range(config.LANG_DETECT_EVERY - 1):
        engine.transcribe(chunk())
    assert sent_languages(server)[-1] is None  # re-detected: en is allowed
    detected["lang"] = "cy"
    for _ in range(config.LANG_DETECT_EVERY):
        s = engine.transcribe(chunk())
    assert s.lang == "en"  # cy dropped, the sticky en kept
    assert sent_languages(server)[-2:] == [None, "en"]


def test_fixed_language_is_always_sent(server, monkeypatch):
    monkeypatch.setattr(config, "WHISPER_LANGUAGE", "fr")
    engine = engine_for(server)
    for _ in range(config.LANG_DETECT_EVERY + 1):
        assert engine.transcribe(chunk()).lang == "fr"
    assert sent_languages(server) == ["fr"] * (config.LANG_DETECT_EVERY + 1)
