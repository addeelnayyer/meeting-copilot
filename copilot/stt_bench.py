"""Run a recording through the real VAD chunker + the STT engine and print per-chunk latency.

    python -m copilot.stt_bench path/to/audio.wav [Participant|User] [--engine faster-whisper|whisperkit]
                                [--reference transcript.txt] [--requests N [--keep-replacement]]

--reference adds the word error rate against a reference transcript of the whole WAV.
--requests N is the request loop: the WAV's chunks are sent round-robin until N requests have
been made, and the RSS after warm-up and its growth per 1,000 requests are printed next to the
latency (and the word error rate of the first pass). The RSS is that of the whisperkit-cli server
(WhisperKit) or of this process (faster-whisper). The server replacement (WHISPERKIT_MAX_GROWTH_GB)
is off during the loop, so the raw growth shows; --keep-replacement leaves it on.
"""

import argparse
import re
import time

import numpy as np

from . import config
from .ram import GB, gb
from .stt import STT_ENGINES, build_stt_engine
from .vad import Chunker, SileroStream

SAMPLE_EVERY = 100  # requests between two RSS samples in the loop

WORD = re.compile(r"[\w']+")


def _words(text):
    return WORD.findall(text.lower())


def word_error_rate(reference, hypothesis):
    """(substitutions + insertions + deletions) / reference words, ignoring case and punctuation."""
    ref, hyp = _words(reference), _words(hypothesis)
    row = list(range(len(hyp) + 1))  # edit distance from an empty reference
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / max(len(ref), 1)


def growth_per_1000(samples):
    """RSS growth in bytes per 1,000 requests: the least-squares slope through (requests, rss)
    samples; None with fewer than two."""
    if len(samples) < 2:
        return None
    x, y = np.array(samples, dtype=float).T
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0] * 1000)


def _latency(lat):
    return f"median {np.median(lat) * 1000:.0f} ms, p95 {np.percentile(lat, 95) * 1000:.0f} ms"


def read_wav(path):
    import wave

    import soxr

    with wave.open(path, "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return soxr.resample(pcm, rate, config.SAMPLE_RATE).astype(np.float32)


def chunks_of(path, source):
    audio = read_wav(path)
    vad, chunker = SileroStream(), Chunker(source)
    F = config.VAD_FRAME
    chunks = []
    for i in range(0, len(audio) - F + 1, F):
        frame = audio[i:i + F]
        chunks += chunker.feed(frame, vad(frame))
    return chunks + chunker.flush()


def main(path, source="Participant", engine=None, reference=None, requests=None, keep_replacement=False):
    chunks = chunks_of(path, source)
    print(f"{len(chunks)} chunks, durations {[round(c.duration, 2) for c in chunks]}")
    if not chunks:
        return
    ref = open(reference, encoding="utf-8").read() if reference else None

    t = time.perf_counter()
    replacement_off = requests and not keep_replacement
    stt = build_stt_engine(engine, max_growth_gb=float("inf") if replacement_off else None)
    print(f"{stt.name}: model load and warm-up {time.perf_counter() - t:.1f}s")
    try:
        if requests:
            _loop(stt, chunks, requests, ref, keep_replacement)
        else:
            _once(stt, chunks, ref)
    finally:
        stt.close()


def _timed(stt, c, lat, texts):
    """Transcribes chunk c: its latency goes into lat, its text (if any) into texts unless None."""
    t = time.perf_counter()
    seg = stt.transcribe(c)
    lat.append(time.perf_counter() - t)
    if seg and texts is not None:
        texts.append(seg.text)
    return seg


def _print_wer(ref, texts, note=""):
    if ref is not None:
        print(f"word error rate {word_error_rate(ref, ' '.join(texts)) * 100:.1f} %{note}")


def _once(stt, chunks, ref):
    lat, texts = [], []
    for c in chunks:
        seg = _timed(stt, c, lat, texts)
        print(f"{c.duration:4.2f}s audio  {lat[-1] * 1000:5.0f} ms  [{seg.lang if seg else '--'}] {seg.text if seg else ''}")
    print(f"\n{_latency(lat)}")
    _print_wer(ref, texts)


def _servers(stt):
    """WhisperKit servers started so far; faster-whisper never changes process."""
    return getattr(stt, "servers_started", 1)


def _loop(stt, chunks, requests, ref, keep_replacement):
    servers = _servers(stt)  # changes when another whisperkit-cli server takes over
    warm = stt.rss()
    print(f"RSS of the STT engine after warm-up {gb(warm)}; server replacement "
          f"{'on' if keep_replacement else 'off'}; {requests} requests")
    samples = [(0, warm)] if warm is not None else []
    lat, texts, switches = [], [], 0
    for n in range(1, requests + 1):
        _timed(stt, chunks[(n - 1) % len(chunks)], lat, texts if n <= len(chunks) else None)
        if _servers(stt) != servers:  # a replacement or a crash restart
            servers, switches = _servers(stt), switches + 1
            samples = []  # from here on: the growth of the server now serving
            print(f"request {n}: another whisperkit-cli server serves from here")
        if n % SAMPLE_EVERY == 0 or n == requests:
            rss = stt.rss()
            if rss is not None:
                samples.append((n, rss))
            print(f"request {n}/{requests}  RSS {gb(rss)}  {_latency(lat[-SAMPLE_EVERY:])}", flush=True)

    growth = growth_per_1000(samples)
    print(f"\n{requests} requests through {stt.name}")
    print(f"RSS after warm-up {gb(warm)}, at the end {gb(samples[-1][1] if samples else None)}")
    if growth is None:
        print("RSS growth per 1,000 requests: unknown (too few RSS samples)")
    else:
        since = " (of the server serving since the last switch)" if switches else ""
        print(f"RSS growth per 1,000 requests {growth / GB:+.3f} GB{since}")
    if switches:
        print(f"servers switched {switches} times (replacement or crash restart)")
    print(_latency(lat))
    if requests >= len(chunks):
        _print_wer(ref, texts, " (first pass)")
    elif ref is not None:
        print(f"word error rate: needs at least one full pass over the WAV ({len(chunks)} requests)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="python -m copilot.stt_bench")
    p.add_argument("path")
    p.add_argument("source", nargs="?", default="Participant", choices=("Participant", "User"))
    p.add_argument("--engine", choices=STT_ENGINES, help="default: STT_ENGINE from .env")
    p.add_argument("--reference", help="reference transcript of the whole WAV (text file): adds the word error rate")
    p.add_argument("--requests", type=int, metavar="N",
                   help="request loop: N requests, with the RSS after warm-up and its growth per 1,000 requests")
    p.add_argument("--keep-replacement", action="store_true",
                   help="leave the WhisperKit server replacement on during the request loop")
    a = p.parse_args()
    if a.keep_replacement and not a.requests:
        p.error("--keep-replacement needs --requests")
    main(a.path, a.source, a.engine, a.reference, a.requests, a.keep_replacement)
