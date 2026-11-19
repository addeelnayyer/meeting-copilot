"""Run a fixed, synthetic meeting workload through the real Ollama provider and compare local LLMs.

    python -m copilot.llm_bench [model ...]      (default: OLLAMA_MODEL from .env)

Call types: suggestion (English, Urdu script and code-switched lines), translation (every
participant line) and german (suggestions on a German meeting, for the shipped default).
Suggestion calls use the role prompt, context layout and schema of a meeting. Each model is
warmed up first. Needs a running Ollama with the models pulled, so it is not part of CI.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from . import config
from .assistant import (
    PROACTIVE_INSTRUCTION,
    TRANSLATE_SYSTEM,
    meeting_context,
    reply_language_rule,
    translation_target,
)
from .audio import PARTICIPANT
from .facts import Facts
from .llm import SUGGESTION_SCHEMA, parse_json
from .memory import Memory
from .providers import OllamaProvider
from .segment import Segment
from .transcript import Transcript

DATA = Path(__file__).with_name("llm_bench.json")
CALL_TYPES = ("suggestion", "translation", "german")
JSON_CALLS = ("suggestion", "german")
SAMPLES = 5
WARMUP_TRIES = 5


def matches(schema, value):
    """value fits the (closed) JSON schema: object/array/string types and required keys."""
    kind = schema.get("type")
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list) and all(matches(schema["items"], v) for v in value)
    if kind == "object":
        props = schema.get("properties", {})
        return (isinstance(value, dict) and all(k in value for k in schema.get("required", []))
                and all(k in props and matches(props[k], v) for k, v in value.items()))
    return True


def workload():
    """[(call type, label, system, prompt)] in a fixed order."""
    system = config.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8") + reply_language_rule()
    calls = []
    for meeting in json.loads(DATA.read_text(encoding="utf-8"))["meetings"]:
        facts = Facts(fresh=True)  # never saved: merge() is not called
        facts.data = dict(meeting["facts"])
        memory = Memory()
        memory.text = meeting["memory"]
        transcript = Transcript()
        for i, line in enumerate(meeting["lines"]):
            # 5 s apart, so every line is its own turn as in a real transcript
            transcript.add(Segment(line["source"], line["text"], line["lang"], duration=3.0, t_end=5.0 * (i + 1)))
            if line["source"] != PARTICIPANT:
                continue
            label = f"[{line['lang']}] {line['text']}"
            prompt = meeting_context(facts, memory, meeting["knowledge"], transcript) + PROACTIVE_INSTRUCTION
            calls.append((meeting["call"], label, system, prompt))
            target = translation_target(line["lang"])
            calls.append(("translation", f"{label} -> {target}", TRANSLATE_SYSTEM.format(target=target), line["text"]))
    return calls


def _short(text, n=240):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3] + "..."


def _output(kind, text):
    """(valid, sample text) of one reply."""
    if kind not in JSON_CALLS:
        return True, _short(text)
    result = parse_json(text)
    if not isinstance(result, dict) or not matches(SUGGESTION_SCHEMA, result):
        return False, "INVALID JSON: " + _short(text)
    said = " / ".join(s["text"] for s in result["suggestions"]) or "(no suggestion)"
    return True, _short(f"{result['trigger']}: {said}")


class OllamaDown(Exception):
    pass


async def run_model(provider, model, calls):
    """{call type: [(label, seconds or None, valid, sample)]}; None seconds marks a failed call."""
    provider.model = model
    t = time.perf_counter()
    await provider.warmup()
    for attempt in range(WARMUP_TRIES):  # a large model can take longer than one request timeout to load
        try:
            await provider.complete(calls[0][2], calls[0][3], SUGGESTION_SCHEMA)
            break
        except httpx.ConnectError as e:
            raise OllamaDown(str(e)) from e
        except httpx.TimeoutException:
            if attempt == WARMUP_TRIES - 1:
                raise
    print(f"{model}: warm-up {time.perf_counter() - t:.1f} s")
    results = {kind: [] for kind in CALL_TYPES}
    for kind, label, system, prompt in calls:
        t = time.perf_counter()
        try:
            text = await provider.complete(system, prompt, SUGGESTION_SCHEMA if kind in JSON_CALLS else None)
        except httpx.ConnectError as e:
            raise OllamaDown(str(e)) from e
        except httpx.HTTPError as e:
            results[kind].append((label, None, False, f"ERROR {type(e).__name__}: {_short(e)}"))
            continue
        secs = time.perf_counter() - t
        valid, sample = _output(kind, text)
        results[kind].append((label, secs, valid, sample))
        print(f"  {kind:<11} {secs * 1000:6.0f} ms  {'' if valid else 'invalid  '}{_short(label, 60)}")
    return results


def _picks(n):
    """SAMPLES indices spread over n calls, so the samples cover every language of the transcript."""
    return sorted({round(i * (n - 1) / (SAMPLES - 1)) for i in range(SAMPLES)}) if n > 1 else list(range(n))


def report(results):
    """results: {model: {call type: [...]}} -> summary table, then the samples of all models side by side."""
    models = list(results)
    print("\n" + "=" * 78)
    print(f"{'model':<28} {'call type':<12} {'n':>3} {'p50 ms':>7} {'p95 ms':>7}  {'valid JSON':>10}  errors")
    for model in models:
        for kind in CALL_TYPES:
            rows = results[model][kind]
            lat = [s for _, s, _, _ in rows if s is not None]
            p50, p95 = (f"{np.percentile(lat, q) * 1000:.0f}" for q in (50, 95)) if lat else ("-", "-")
            valid = f"{sum(v for _, _, v, _ in rows)}/{len(rows)}" if kind in JSON_CALLS else ""
            errors = sum(s is None for _, s, _, _ in rows)
            print(f"{model:<28} {kind:<12} {len(rows):>3} {p50:>7} {p95:>7}  {valid:>10}  {errors or ''}")
    for kind in CALL_TYPES:
        rows = results[models[0]][kind]
        for n, i in enumerate(_picks(len(rows)), 1):
            print(f"\n--- {kind} sample {n}: {_short(rows[i][0], 200)}")
            for model in models:
                _, secs, _, sample = results[model][kind][i]
                ms = f"{secs * 1000:.0f} ms" if secs is not None else "failed"
                print(f"  {model} ({ms}): {sample}")


async def main(models):
    provider = OllamaProvider()
    installed = await provider.installed()
    if installed is None:
        print(f"Ollama is not running at {config.OLLAMA_URL}. Start it (ollama serve, or the Ollama app) and retry.")
        return 1
    missing = [m for m in models if m not in installed and f"{m}:latest" not in installed]
    for m in missing:
        print(f"Model {m} is not installed: ollama pull {m}")
    if missing:
        return 1
    calls = workload()
    print(f"Ollama {config.OLLAMA_URL}, prompt {config.SYSTEM_PROMPT_FILE.name}, LANGUAGES {','.join(config.LANGUAGES)}, "
          f"{len(calls)} calls per model")
    results = {}
    try:
        for model in models:
            results[model] = await run_model(provider, model, calls)
    except OllamaDown as e:
        print(f"Lost the connection to Ollama at {config.OLLAMA_URL} ({e}). Is it still running?")
        return 1
    except httpx.HTTPStatusError as e:
        print(f"Ollama could not load {model}: {e.response.status_code} {_short(e.response.text)}")
        return 1
    except httpx.TimeoutException:
        print(f"Ollama did not answer {model} within {WARMUP_TRIES} x {config.OLLAMA_TIMEOUT_S} s while warming up.")
        return 1
    report(results)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="python -m copilot.llm_bench")
    p.add_argument("models", nargs="*", default=[config.OLLAMA_MODEL], metavar="model",
                   help=f"Ollama model tags (default: OLLAMA_MODEL, {config.OLLAMA_MODEL})")
    try:
        sys.exit(asyncio.run(main(list(dict.fromkeys(p.parse_args().models)))))
    except KeyboardInterrupt:
        sys.exit(130)
