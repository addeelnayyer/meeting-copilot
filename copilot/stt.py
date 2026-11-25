"""STT worker: one STT engine, chunks from both sources, labelled segments out."""

import functools
import platform
import queue
import sys
import threading
import time

from . import cuda  # noqa: F401  (registers NVIDIA DLL dirs before ctranslate2 loads)
from . import config
from .segment import EngineFailed, Segment, StickyLanguage, is_hallucination
from .strings import t
from . import whisperkit


def whisper_settings(cuda_devices):
    """(model, device, compute_type), resolving "auto" by whether a CUDA device is usable."""
    device = config.WHISPER_DEVICE
    if device == "auto":
        device = "cuda" if cuda_devices > 0 else "cpu"
    model, compute = config.WHISPER_DEFAULTS.get(device, config.WHISPER_DEFAULTS["cpu"])
    if config.WHISPER_MODEL != "auto":
        model = config.WHISPER_MODEL
    if config.WHISPER_COMPUTE != "auto":
        compute = config.WHISPER_COMPUTE
    return model, device, compute


def stt_engine(setting, system, machine, binary):
    """(engine, reason) for STT_ENGINE `setting` on this platform (sys.platform, platform.machine()),
    given the whisperkit-cli path (None when missing). `auto` is WhisperKit only on macOS arm64;
    reason says why WhisperKit was wanted but is not used. The model is found or downloaded when
    the engine is built."""
    if setting not in ("auto", *STT_ENGINES):
        raise ValueError(f"STT_ENGINE={setting!r}: use auto, {' or '.join(STT_ENGINES)}")
    wanted = setting == "whisperkit" or (setting == "auto" and system == "darwin" and machine == "arm64")
    if not wanted:
        return FasterWhisperEngine.name, None
    if system != "darwin":
        why = "it needs macOS"
    elif not binary:
        why = whisperkit.NO_BINARY
    else:
        return whisperkit.WhisperKitEngine.name, None
    return FasterWhisperEngine.name, _fallback_reason(setting, why)


def _fallback_reason(setting, why):
    if setting == "whisperkit":
        return f"WhisperKit was requested but is unavailable: {why}"
    return f"WhisperKit is unavailable: {why}"


def chosen_stt_engine(setting):
    """stt_engine() for this machine."""
    return stt_engine(setting, sys.platform, platform.machine(), whisperkit.binary())


def build_stt_engine(setting=None, on_status=None):
    """Build and warm up the chosen STT engine (`setting` overrides STT_ENGINE) once; it then
    serves transcribe(chunk) and close(). WhisperKit that fails to start or become healthy falls
    back to faster-whisper, like a missing binary or a failed model download; the reason is printed.
    WhisperKit reports a server restart through `on_status`."""
    setting = setting or config.STT_ENGINE
    name, reason = chosen_stt_engine(setting)
    if name == whisperkit.WhisperKitEngine.name:
        try:
            return _build_warm(functools.partial(ENGINES[name], on_status=on_status))
        except Exception as e:
            name, reason = FasterWhisperEngine.name, _fallback_reason(setting, e)
    if reason:
        print(f"STT: {reason}; using {FasterWhisperEngine.name}")
    return _build_warm(ENGINES[name])


def _build_warm(engine_class):
    engine = engine_class()
    try:
        engine.warm_up()
    except BaseException:
        engine.close()
        raise
    return engine


class FasterWhisperEngine:
    name = "faster-whisper"

    def __init__(self):
        import ctranslate2
        from faster_whisper import WhisperModel

        model, device, compute = whisper_settings(ctranslate2.get_cuda_device_count())
        print(f"STT: {self.name} {model} ({device}, {compute})")
        args = (model,)
        kwargs = {"device": device, "compute_type": compute}
        try:
            # The Hugging Face freshness check can take minutes on a slow link; the
            # cached model loads in ~4 s.
            self.model = WhisperModel(*args, local_files_only=True, **kwargs)
        except Exception:
            self.model = WhisperModel(*args, **kwargs)  # first run: download
        self._last_text = {}
        self._language = StickyLanguage()

    def warm_up(self):
        """The first call is slow (CUDA compiles kernels, ~5 s); do it before the meeting audio arrives."""
        import numpy as np

        list(self.model.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32), language="de")[0])

    def _pick_language(self, chunk):
        if config.WHISPER_LANGUAGE != "auto":
            return config.WHISPER_LANGUAGE
        if self._language.due(chunk):
            _, _, probs = self.model.detect_language(chunk.audio)
            probs = dict(probs)
            # Restricting the choice to the expected languages avoids rare misdetections.
            self._language.lang[chunk.source] = max(config.LANGUAGES, key=lambda l: probs.get(l, 0.0))
        return self._language.lang[chunk.source]

    def transcribe(self, chunk):
        prompt = self._last_text.get(chunk.source, "")[-config.PROMPT_CONTEXT_CHARS:]
        lang = self._pick_language(chunk)
        segs, _ = self.model.transcribe(
            chunk.audio,
            language=lang,
            beam_size=1,
            temperature=0.0,  # no fallback re-decodes; they double latency on short chunks
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=prompt or None,
        )

        text = " ".join(
            s.text.strip() for s in segs if not is_hallucination(s.text, s.no_speech_prob, s.avg_logprob)
        ).strip()
        if not text:
            return None
        self._last_text[chunk.source] = (self._last_text.get(chunk.source, "") + " " + text)[-1000:]
        return Segment(chunk.source, text, lang, chunk.duration, chunk.t_end)

    def close(self):
        self.model = None  # the last reference: CTranslate2 frees the STT model's RAM (and VRAM)


ENGINES = {e.name: e for e in (FasterWhisperEngine, whisperkit.WhisperKitEngine)}
STT_ENGINES = tuple(ENGINES)


# After close(), the VAD still ends an utterance that was open when the devices stopped (a few
# 0.2 s silence feeds); the engine waits this long without a chunk before it closes.
CLOSE_GRACE_S = 1.5


class SttWorker(threading.Thread):
    """Holds an STT engine only between open() and close() (a meeting): open() builds and warms
    one, close() closes it once the queued chunks are transcribed. Chunks wait in the queue while
    no engine is ready. Both only signal this thread, which owns the engine."""

    def __init__(self, in_queue, on_segment, stop_event, on_status=None):
        super().__init__(daemon=True, name="stt")
        self.in_queue = in_queue
        self.on_segment = on_segment
        self.stop_event = stop_event
        self.on_status = on_status or (lambda status: None)
        self.engine = None
        self._wanted = threading.Event()  # a meeting runs
        self._opened = threading.Event()  # open() was called: build, even if close() came first
        self._setting = None  # STT_ENGINE, until a switch fixes faster-whisper for the run

    def open(self):
        self._wanted.set()
        self._opened.set()

    def close(self):
        self._wanted.clear()

    def run(self):
        try:
            while not self.stop_event.is_set():
                if self.engine is None:
                    if not self._opened.wait(0.2):
                        continue
                    self._opened.clear()
                    self.on_status({"ready": False})
                    self.engine = build_stt_engine(self._setting, on_status=self.on_status)  # built warm
                    self.on_status({"ready": True, "engine": self.engine.name})
                self._opened.clear()  # an open() while the engine exists needs no second one
                if self._wanted.is_set():
                    self._serve(0.2)
                elif not self._serve(CLOSE_GRACE_S) and not self._wanted.is_set():  # drained
                    self._close_engine()
                    self.on_status({"ready": False, "idle": True})
        finally:
            self._close_engine()

    def _serve(self, timeout):
        """Transcribe one chunk; False when none came within `timeout`."""
        try:
            chunk = self.in_queue.get(timeout=timeout)
        except queue.Empty:
            return False
        seg = self._transcribe(chunk)
        if seg:
            seg.latency = time.monotonic() - chunk.t_end
            self.on_segment(seg)
        return True

    def _close_engine(self):
        if self.engine is not None:
            engine, self.engine = self.engine, None
            engine.close()

    def _transcribe(self, chunk):
        try:
            return self.engine.transcribe(chunk)
        except EngineFailed as e:
            self._switch(e)
            return self.engine.transcribe(chunk)  # the chunk in flight is not lost

    def _switch(self, error):
        """faster-whisper for the rest of the run, and a note in the UI saying why."""
        failed, fallback = self.engine.name, FasterWhisperEngine.name
        print(f"STT: {failed} failed ({error}), switching to {fallback} for the rest of the run")
        self._close_engine()
        self._setting = fallback  # later meetings of this run too
        status = {"ready": False, "engine": fallback, "note": t("stt.switched", failed=failed, error=error, fallback=fallback)}
        self.on_status(status)
        try:
            self.engine = build_stt_engine(fallback)  # built warm
        except Exception as e:
            self.on_status({**status, "note": t("stt.fallback_failed", failed=failed, fallback=fallback, error=e)})
            raise
        self.on_status({**status, "ready": True})


if __name__ == "__main__":
    # The installers prepare the STT model: download it and, for WhisperKit, the first Core ML load.
    build_stt_engine().close()
