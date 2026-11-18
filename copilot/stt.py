"""STT worker: one STT engine, chunks from both sources, labelled segments out."""

import queue
import threading
import time

from . import cuda  # noqa: F401  (registers NVIDIA DLL dirs before ctranslate2 loads)
from . import config
from .segment import EngineFailed, Segment, StickyLanguage, is_hallucination
from .whisperkit import WhisperKitEngine


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


def stt_engine():
    """The STT engine named by STT_ENGINE; "auto" is faster-whisper on every platform for now."""
    engine = config.STT_ENGINE
    if engine == "auto":
        return FasterWhisperEngine.name
    if engine not in STT_ENGINES:
        raise ValueError(f"STT_ENGINE={engine!r}: use auto, {' or '.join(STT_ENGINES)}")
    return engine


def build_stt_engine(engine=None):
    """Build the chosen STT engine (or `engine`) once: it loads its STT model and then serves
    warm_up(), transcribe(chunk) and close()."""
    return ENGINES[engine or stt_engine()]()


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
        pass


ENGINES = {e.name: e for e in (FasterWhisperEngine, WhisperKitEngine)}
STT_ENGINES = tuple(ENGINES)


class SttWorker(threading.Thread):
    def __init__(self, in_queue, on_segment, stop_event, on_status=None):
        super().__init__(daemon=True, name="stt")
        self.in_queue = in_queue
        self.on_segment = on_segment
        self.stop_event = stop_event
        self.on_status = on_status or (lambda status: None)
        self.engine = None

    def run(self):
        self.engine = build_stt_engine()
        try:
            self._warm_up()
            while not self.stop_event.is_set():
                try:
                    chunk = self.in_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                seg = self._transcribe(chunk)
                if seg:
                    seg.latency = time.monotonic() - chunk.t_end
                    self.on_segment(seg)
        finally:
            self.engine.close()

    def _warm_up(self):
        try:
            self.engine.warm_up()
            self.on_status({"ready": True, "engine": self.engine.name})
        except EngineFailed as e:
            self._switch(e)

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
        self.engine.close()
        status = {"ready": False, "engine": fallback, "note": f"{failed} ausgefallen ({error}), STT jetzt mit {fallback}"}
        self.on_status(status)
        try:
            self.engine = build_stt_engine(fallback)
            self.engine.warm_up()
        except Exception as e:
            self.on_status({**status, "note": f"{failed} ausgefallen, {fallback} auch: {e}"})
            raise
        self.on_status({**status, "ready": True})
