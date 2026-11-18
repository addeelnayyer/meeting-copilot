"""WhisperKit STT engine: Homebrew's whisperkit-cli serve as a child process on localhost."""

import atexit
import io
import re
import shutil
import socket
import subprocess
import time
import wave
from pathlib import Path

import httpx
import numpy as np

from . import config
from .segment import Segment, StickyLanguage, is_hallucination

HOST = "127.0.0.1"

SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")

# Used read-only: nothing is ever written under MacWhisper's folder.
MACWHISPER_MODELS = (Path.home() / "Library/Application Support/MacWhisper/models/whisperkit/models"
                     / "argmaxinc/whisperkit-coreml")


NO_BINARY = "whisperkit-cli is not on the PATH (brew install whisperkit-cli)"

HF_REPO = "argmaxinc/whisperkit-coreml"


def model_path():
    """The folder of the WHISPERKIT_MODEL STT model: WHISPERKIT_MODEL_PATH, then MacWhisper's
    folder, then the Hugging Face cache. When it is in none of them, only that model's subfolder
    is downloaded into the Hugging Face cache; RuntimeError when the download fails."""
    override = config.WHISPERKIT_MODEL_PATH
    if override:
        if _is_model(Path(override).expanduser()):
            return Path(override).expanduser()
        print(f"STT: WHISPERKIT_MODEL_PATH={override} holds no WhisperKit model; looking elsewhere")
    mac = MACWHISPER_MODELS / config.WHISPERKIT_MODEL
    if _is_model(mac):
        return mac
    return _cached_model() or _download_model()


def _is_model(folder):
    return (folder / "AudioEncoder.mlmodelc").is_dir()


def _snapshot(**kwargs):
    import huggingface_hub

    return Path(huggingface_hub.snapshot_download(HF_REPO, allow_patterns=f"{config.WHISPERKIT_MODEL}/*",
                                                  **kwargs)) / config.WHISPERKIT_MODEL


def _cached_model():
    """The model folder in the Hugging Face cache, without the network; None when it is not
    (completely) there."""
    from huggingface_hub.errors import LocalEntryNotFoundError  # also an incomplete snapshot

    try:
        folder = _snapshot(local_files_only=True)
    except LocalEntryNotFoundError:
        return None
    return folder if _is_model(folder) else None


def _download_model():
    model = config.WHISPERKIT_MODEL
    print(f"STT: downloading WhisperKit model {model} from {HF_REPO} into the Hugging Face cache (once)")
    try:
        folder = _snapshot()
    except Exception as e:  # also without a network: then it is a LocalEntryNotFoundError
        raise RuntimeError(f"could not download model {model} from {HF_REPO}: {e}") from e
    if not _is_model(folder):
        raise RuntimeError(f"{HF_REPO} has no model {model} (WHISPERKIT_MODEL)")
    return folder


def server_command(binary, model, port):
    """whisperkit-cli serve on localhost; --model-path, so it never downloads by itself."""
    units = config.WHISPERKIT_COMPUTE_UNITS
    return [str(binary), "serve", "--model-path", str(model), "--host", HOST, "--port", str(port),
            "--temperature-fallback-count", "0",  # fallback re-decodes double latency on short chunks
            "--audio-encoder-compute-units", units, "--text-decoder-compute-units", units]


def free_port():
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def wav_bytes(audio):
    """16 kHz mono 16-bit WAV, as the server reads it."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(config.SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


class Server:
    """The whisperkit-cli serve child process; stopped at exit at the latest."""

    def __init__(self, model):
        binary = shutil.which("whisperkit-cli")
        if not binary:
            raise RuntimeError(NO_BINARY)
        port = free_port()
        self.url = f"http://{HOST}:{port}"
        config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.log = config.SESSIONS_DIR / "whisperkit.log"
        with open(self.log, "w", encoding="utf-8") as log:
            self.proc = subprocess.Popen(server_command(binary, model, port), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
        atexit.register(self.close)

    def check(self):
        if self.proc.poll() is not None:
            raise RuntimeError(f"whisperkit-cli exited with code {self.proc.returncode}, see {self.log}")

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class WhisperKitEngine:
    """STT engine on a whisperkit-cli server; url points at a running one instead of launching it."""

    name = "whisperkit"

    def __init__(self, url=None):
        self.server = None
        if url is None:
            model = model_path()
            print(f"STT: {self.name} {config.WHISPERKIT_MODEL} ({config.WHISPERKIT_COMPUTE_UNITS}) from {model}")
            self.server = Server(model)
            url = self.server.url
        self.http = httpx.Client(base_url=url, timeout=30)
        try:
            self._wait_healthy()
        except BaseException:
            self.close()
            raise
        self._language = StickyLanguage()

    def _wait_healthy(self):
        """The first start specializes the Core ML model for this chip, which can take minutes."""
        deadline = time.monotonic() + config.WHISPERKIT_START_TIMEOUT_S
        while True:
            if self.server:
                self.server.check()
            try:
                if self.http.get("/health", timeout=2).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"whisperkit-cli not healthy after {config.WHISPERKIT_START_TIMEOUT_S} s")
            time.sleep(0.2)

    def close(self):
        self.http.close()
        if self.server:
            self.server.close()

    def warm_up(self):
        """About 1 s of silence, so the first meeting chunk does not pay for the first decode."""
        self._post(wav_bytes(np.zeros(config.SAMPLE_RATE, dtype=np.float32)), config.LANGUAGES[0])

    def _post(self, wav, lang):
        # No prompt: WhisperKit pre-fills it through the decoder, which doubled latency and,
        # with temperature fallback off, set off repetition loops (step 0 in #1).
        data = {"model": config.WHISPERKIT_MODEL, "temperature": "0", "response_format": "verbose_json"}
        if lang:
            data["language"] = lang
        r = self.http.post("/v1/audio/transcriptions", data=data,
                           files={"file": ("chunk.wav", wav, "audio/wav")})
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _text(reply):
        """Kept segment texts joined, without hallucinations, repetition loops and special tokens."""
        kept = []
        for s in reply.get("segments", []):
            text = SPECIAL_TOKEN.sub("", s.get("text", "")).strip()
            if s.get("compression_ratio", 0.0) > config.WHISPERKIT_MAX_COMPRESSION:
                continue
            if not is_hallucination(text, s.get("no_speech_prob", 0.0), s.get("avg_logprob", 0.0)):
                kept.append(text)
        return " ".join(kept)

    def transcribe(self, chunk):
        wav = wav_bytes(chunk.audio)
        if config.WHISPER_LANGUAGE != "auto":
            lang = config.WHISPER_LANGUAGE
        elif self._language.due(chunk):
            # The server gives no language probabilities, so detection can't be limited to LANGUAGES.
            lang = None
        else:
            lang = self._language.lang[chunk.source]
        reply = self._post(wav, lang)
        text = self._text(reply)
        if lang is None:
            lang = reply.get("language", "")
            if lang in config.LANGUAGES:
                self._language.lang[chunk.source] = lang
            elif text:
                # A language outside LANGUAGES is a misdetection: keep the sticky one (your
                # own on a first chunk) and decode the chunk again in it.
                lang = self._language.lang.setdefault(chunk.source, config.LANGUAGES[0])
                text = self._text(self._post(wav, lang))
        if not text:
            return None
        return Segment(chunk.source, text, lang, chunk.duration, chunk.t_end)
