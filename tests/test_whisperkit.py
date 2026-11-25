"""WhisperKit STT engine against a fake whisperkit-cli server: no Mac or WhisperKit needed."""

import json
import queue
import sys
import threading
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from copilot import config, stt, whisperkit
from copilot.segment import Segment
from copilot.vad import Chunk

MODEL = "openai_whisper-large-v3-v20240930"
GB = 1024 ** 3


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
                self._send(fake.status, fake.reply(fields) if fake.status == 200 else {"error": "boom"})

            def _send(self, status, data):
                out = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        self.closed = self.dead = False
        self.status = 200  # of transcription replies
        self.rss_bytes = 2 * GB  # what rss() reports; tests grow it
        threading.Thread(target=self.httpd.serve_forever, args=(0.05,), daemon=True).start()

    # The engine's view of a launched whisperkit-cli server.
    def check(self):
        if self.dead:
            raise RuntimeError("whisperkit-cli exited with code -9")

    def rss(self):
        return self.rss_bytes

    def kill(self):
        """A crash: connections are refused from now on."""
        self.dead = True
        self.httpd.shutdown()
        self.httpd.server_close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.kill()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(config, "WHISPER_LANGUAGE", "auto")
    monkeypatch.setattr(config, "LANGUAGES", ["de", "en"])
    fake = FakeServer()
    yield fake
    for engine in fake.engines:
        engine.close()
    fake.close()


def engine_for(server):
    engine = whisperkit.WhisperKitEngine(launch=lambda: server)
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


# --- Recovering from a crash during a meeting ------------------------------------
class Launcher:
    """Starts a fresh fake server on every launch, as a restart of whisperkit-cli would."""

    def __init__(self):
        self.servers = []
        self.fail = False
        self.launches = 0  # calls, failed ones included
        self.hold = threading.Event()  # cleared: a launch waits until it is set
        self.hold.set()

    def __call__(self):
        self.launches += 1
        assert self.hold.wait(10), "a launch was held for too long"
        if self.fail:
            raise RuntimeError("whisperkit-cli is not on the PATH")
        self.servers.append(FakeServer())
        return self.servers[-1]


class FakeFasterWhisper:
    name = "faster-whisper"

    def __init__(self):
        self.closed = False

    def warm_up(self):
        pass

    def transcribe(self, chunk):
        return Segment(chunk.source, "from faster-whisper", "de", chunk.duration, chunk.t_end)

    def close(self):
        self.closed = True


@pytest.fixture
def worker(monkeypatch):
    """An SttWorker on WhisperKit (fake servers) with a fake faster-whisper to fall back to.
    transcribe(chunk) feeds it one chunk and waits for the segment."""
    monkeypatch.setattr(config, "WHISPER_LANGUAGE", "en")
    monkeypatch.setattr(config, "STT_ENGINE", "whisperkit")
    launcher = Launcher()
    fallbacks = []

    def fallback():
        fallbacks.append(FakeFasterWhisper())
        return fallbacks[-1]

    monkeypatch.setattr(stt, "chosen_stt_engine", lambda setting: (setting, None))  # no Mac needed
    monkeypatch.setattr(stt, "ENGINES", {"whisperkit": lambda on_status: whisperkit.WhisperKitEngine(launch=launcher,
                                                                                         on_status=on_status),
                                         "faster-whisper": fallback})
    chunks, segments, statuses, stop, ready = queue.Queue(), queue.Queue(), [], threading.Event(), threading.Event()

    def on_status(status):
        statuses.append(status)
        ready.set()

    w = stt.SttWorker(chunks, segments.put, stop, on_status=on_status)
    w.start()
    assert ready.wait(10), "the worker never became ready"

    def transcribe(c=None):
        chunks.put(c or chunk())
        return segments.get(timeout=10).text

    yield SimpleNamespace(transcribe=transcribe, launcher=launcher, fallbacks=fallbacks, statuses=statuses)
    stop.set()
    w.join(timeout=10)
    assert not w.is_alive()
    assert all(s.closed for s in launcher.servers), "a whisperkit-cli server outlived the worker"
    assert all(f.closed for f in fallbacks)


def test_one_crash_is_recovered_by_one_restart(worker):
    assert worker.transcribe() == "Hello there."
    worker.launcher.servers[-1].kill()
    # The chunk in flight is retried on the restarted server, and later chunks stay on WhisperKit.
    assert worker.transcribe() == "Hello there."
    assert worker.transcribe() == "Hello there."
    assert len(worker.launcher.servers) == 2
    assert worker.launcher.servers[0].closed
    assert worker.fallbacks == []


@pytest.mark.parametrize("lang, restarting_note, switched_note", [
    ("en", "whisperkit failed ({reason}), restarting it", "whisperkit failed ({error}), STT now on faster-whisper"),
    ("de", "whisperkit ausgefallen ({reason}), wird neu gestartet",
     "whisperkit ausgefallen ({error}), STT jetzt mit faster-whisper"),
])
def test_second_failure_switches_to_faster_whisper_with_a_note(worker, monkeypatch, lang, restarting_note,
                                                               switched_note):
    monkeypatch.setattr(config, "UI_LANGUAGE", lang)
    assert worker.transcribe() == "Hello there."
    worker.launcher.servers[-1].kill()
    assert worker.transcribe() == "Hello there."  # restarted
    worker.launcher.servers[-1].kill()
    # The chunk in flight is not lost: faster-whisper transcribes it.
    assert worker.transcribe() == "from faster-whisper"
    assert worker.transcribe() == "from faster-whisper"
    assert len(worker.launcher.servers) == 2  # no second restart
    assert all(s.closed for s in worker.launcher.servers)
    assert len(worker.fallbacks) == 1
    first, restarting, restarted, switching, switched = worker.statuses
    reason = "whisperkit-cli exited with code -9"  # why, in both notes
    assert first == {"ready": True, "engine": "whisperkit"}
    assert restarting == {"ready": False, "engine": "whisperkit", "note": restarting_note.format(reason=reason)}
    assert restarted == {"ready": True, "engine": "whisperkit", "note": ""}
    assert not switching["ready"] and switched["ready"]  # "STT loading" while faster-whisper loads
    assert switching["note"] == switched["note"]
    assert switched["engine"] == "faster-whisper"
    assert switched["note"] == switched_note.format(error=f"whisperkit-cli failed again: {reason}")


def test_failed_restart_switches_to_faster_whisper(worker):
    assert worker.transcribe() == "Hello there."
    worker.launcher.fail = True
    worker.launcher.servers[-1].kill()
    assert worker.transcribe() == "from faster-whisper"
    assert "not on the PATH" in worker.statuses[-1]["note"]


def test_server_errors_count_as_failures(worker):
    worker.launcher.servers[-1].status = 500
    assert worker.transcribe() == "Hello there."  # restarted
    worker.launcher.servers[-1].status = 500
    assert worker.transcribe() == "from faster-whisper"
    assert "500" in worker.statuses[-1]["note"]


# --- Replacing a server that outgrew its RAM limit (ADR 0001) ----------------------
def chunk_requests(server):
    """Transcription requests the server got after its warm-up."""
    return len(server.requests) - 1


def until(condition, transcribe, limit=200):
    """Feeds chunks until condition() holds; the texts of all of them."""
    texts = []
    while not condition():
        assert len(texts) < limit, "condition never held"
        texts.append(transcribe())
    return texts


def test_server_past_its_growth_limit_is_replaced_without_losing_a_chunk(worker, capsys):
    old = worker.launcher.servers[0]
    texts = [worker.transcribe()]
    old.rss_bytes += int(1.6 * GB)  # past the default 1.5 GB over its warm-up size
    texts += until(lambda: len(worker.launcher.servers) == 2 and chunk_requests(worker.launcher.servers[1]) > 0,
                   worker.transcribe)
    new = worker.launcher.servers[1]
    assert old.closed and not new.closed
    texts.append(worker.transcribe())
    assert chunk_requests(new) == 2  # the next chunk is on the new server too
    # Every chunk is transcribed exactly once, on one of the two servers.
    assert texts == ["Hello there."] * len(texts)
    assert chunk_requests(old) + chunk_requests(new) == len(texts)
    assert worker.launcher.launches == 2
    lines = [line for line in capsys.readouterr().out.splitlines() if "replaced" in line]
    assert len(lines) == 1 and "3.6 GB" in lines[0] and "2.0 GB" in lines[0]


def test_failed_replacement_keeps_the_old_server_and_is_retried_without_using_the_crash_restart(worker):
    old = worker.launcher.servers[0]
    worker.launcher.fail = True
    old.rss_bytes += 2 * GB
    texts = until(lambda: worker.launcher.launches >= 3, worker.transcribe)  # tried, failed, tried again
    assert texts == ["Hello there."] * len(texts)
    assert chunk_requests(old) == len(texts) and not old.closed
    worker.launcher.fail = False
    until(lambda: old.closed, worker.transcribe)
    new = worker.launcher.servers[-1]
    assert len(worker.launcher.servers) == 2
    # A real crash afterwards still gets its one restart.
    new.kill()
    assert worker.transcribe() == "Hello there."
    assert len(worker.launcher.servers) == 3 and worker.fallbacks == []


def test_only_one_replacement_is_in_progress_and_none_outlives_the_worker(worker):
    old = worker.launcher.servers[0]
    worker.launcher.hold.clear()  # the replacement's launch waits
    old.rss_bytes += 2 * GB
    for _ in range(5):
        assert worker.transcribe() == "Hello there."
    assert worker.launcher.launches == 2  # the first server and one replacement, however many chunks are over the limit
    assert chunk_requests(old) == 5
    worker.launcher.hold.set()
    # The worker stops while the warm replacement waits for the next request: both servers are
    # closed by the fixture's check.


def test_a_replacement_needs_no_window_status(worker):
    old = worker.launcher.servers[0]
    statuses = list(worker.statuses)
    old.rss_bytes += 2 * GB
    until(lambda: old.closed, worker.transcribe)
    assert worker.statuses == statuses


def test_log_is_appended_with_a_marker_line_per_server_start(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(whisperkit, "binary", lambda: sys.executable)
    monkeypatch.setattr(whisperkit, "server_command",
                        lambda cli, model, port: [cli, "-c", f"print('serving on {port}')"])
    servers = []
    for _ in range(2):  # one after the other, as a replacement or restart starts them
        servers.append(whisperkit.Server(tmp_path / MODEL))
        servers[-1].proc.wait(timeout=10)
        servers[-1].close()
    lines = (tmp_path / "whisperkit.log").read_text(encoding="utf-8").splitlines()
    markers = [line for line in lines if line.startswith("---")]
    assert len(markers) == 2
    for s, marker in zip(servers, markers):
        port = s.url.rsplit(":", 1)[1]
        assert f"port {port}" in marker and MODEL in marker
        assert f"serving on {port}" in lines
    assert lines.index(markers[0]) < lines.index(f"serving on {servers[0].url.rsplit(':', 1)[1]}") < lines.index(markers[1])


@pytest.mark.skipif(sys.platform == "win32", reason="ps is POSIX")
def test_server_rss_is_read_from_the_child(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(whisperkit, "binary", lambda: sys.executable)
    monkeypatch.setattr(whisperkit, "server_command",
                        lambda cli, model, port: [cli, "-c", "import time; time.sleep(30)"])
    server = whisperkit.Server(tmp_path / MODEL)
    try:
        assert server.rss() > 1024 * 1024  # a Python interpreter holds more than 1 MB
    finally:
        server.close()
    assert server.rss() is None


@pytest.mark.skipif(sys.platform == "win32", reason="Homebrew paths are POSIX")
def test_binary_is_found_in_homebrew_without_it_on_the_path(tmp_path, monkeypatch):
    """Apps started from Finder get no Homebrew PATH."""
    cli = tmp_path / "whisperkit-cli"
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(whisperkit, "HOMEBREW_BINS", (str(tmp_path),))
    assert whisperkit.binary() == str(cli)
    monkeypatch.setattr(whisperkit, "HOMEBREW_BINS", ())
    assert whisperkit.binary() is None
