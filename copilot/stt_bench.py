"""Run a recording through the real VAD chunker + the STT engine and print per-chunk latency.

    python -m copilot.stt_bench path/to/audio.wav [Participant|User] [--engine faster-whisper|whisperkit]
"""

import argparse
import time

import numpy as np

from . import config
from .stt import STT_ENGINES, build_stt_engine
from .vad import Chunker, SileroStream


def read_wav(path):
    import wave

    import soxr

    with wave.open(path, "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return soxr.resample(pcm, rate, config.SAMPLE_RATE).astype(np.float32)


def main(path, source="Participant", engine=None):
    audio = read_wav(path)
    vad, chunker = SileroStream(), Chunker(source)
    F = config.VAD_FRAME
    chunks = []
    for i in range(0, len(audio) - F + 1, F):
        frame = audio[i:i + F]
        chunks += chunker.feed(frame, vad(frame))
    chunks += chunker.flush()
    print(f"{len(chunks)} chunks, durations {[round(c.duration, 2) for c in chunks]}")

    t = time.perf_counter()
    stt = build_stt_engine(engine)
    print(f"model load and warm-up {time.perf_counter() - t:.1f}s")
    try:
        lat = []
        for c in chunks:
            t = time.perf_counter()
            seg = stt.transcribe(c)
            lat.append(time.perf_counter() - t)
            print(f"{c.duration:4.2f}s audio  {lat[-1] * 1000:5.0f} ms  [{seg.lang if seg else '--'}] {seg.text if seg else ''}")
        print(f"\nmedian {np.median(lat) * 1000:.0f} ms, p95 {np.percentile(lat, 95) * 1000:.0f} ms")
    finally:
        stt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="python -m copilot.stt_bench")
    p.add_argument("path")
    p.add_argument("source", nargs="?", default="Participant", choices=("Participant", "User"))
    p.add_argument("--engine", choices=STT_ENGINES, help="default: STT_ENGINE from .env")
    a = p.parse_args()
    main(a.path, a.source, a.engine)
