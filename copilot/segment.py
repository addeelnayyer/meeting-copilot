"""What an STT engine produces, and the hallucination filter every engine shares."""

import time
from dataclasses import dataclass, field

from . import config

# Subtitle credits from Whisper's training data, with varying suffixes; never real speech.
CREDITS = ("untertitel im auftrag des zdf", "untertitel der amara.org-community")

# Classic Whisper hallucinations on silence/noise, dropped when the model is unsure.
FILLER = {
    "vielen dank.",
    "vielen dank fürs zuschauen.",
    "thank you.",
    "thanks for watching!",
    "bis zum nächsten mal.",
    "...",
}


@dataclass
class Segment:
    source: str
    text: str
    lang: str
    duration: float
    t_end: float
    latency: float = 0.0
    ts: float = field(default_factory=time.time)
    retracted: bool = False  # pulled back out of the transcript as an echo
    id: int = 0


class EngineFailed(RuntimeError):
    """The STT engine can no longer transcribe; the worker switches to faster-whisper."""


def is_hallucination(text, no_speech_prob, avg_logprob):
    norm = text.strip().lower()
    if not norm or all(not c.isalnum() for c in norm) or norm.startswith(CREDITS):
        return True
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        return True
    return norm in FILLER and no_speech_prob > 0.2


class StickyLanguage:
    """Language per source, detected on its first chunk and then on every Nth chunk long enough
    to detect reliably: detection costs an extra encoder pass or decode."""

    def __init__(self):
        self.lang = {}
        self._chunks = {}

    def due(self, chunk):
        n = self._chunks[chunk.source] = self._chunks.get(chunk.source, 0) + 1
        return chunk.source not in self.lang or (
            n % config.LANG_DETECT_EVERY == 0 and chunk.duration >= config.CHUNK_MIN_S)
