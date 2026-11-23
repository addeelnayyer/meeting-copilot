"""Run a fixed, synthetic meeting workload through the real Ollama provider and compare local LLMs.

    python -m copilot.llm_bench [model ...]      (default: OLLAMA_MODEL from .env)

Call types: suggestion (English, Urdu script and code-switched lines), translation (every
participant line), transliteration (every Urdu line, pure and code-switched, into Roman Urdu)
and german (suggestions on a German meeting, for the shipped default). Every call uses the
prompt of a meeting; suggestion calls also its context layout and schema. Each model is
warmed up first. Needs a running Ollama with the models pulled, so it is not part of CI.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import httpx
import numpy as np

from . import config
from .assistant import (
    TRANSLATE_SYSTEM,
    TRANSLITERATE_SYSTEM,
    meeting_context,
    proactive_instruction,
    suggestion_system,
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
CALL_TYPES = ("suggestion", "translation", "transliteration", "german")
KIND_W = max(map(len, CALL_TYPES))  # column width of the call type
TRANSLITERATED = ("ur",)  # languages whose lines get a transliteration call, as with TRANSLITERATE=ur
JSON_CALLS = ("suggestion", "german")
SAMPLES = 5
WARMUP_TRIES = 5


class Call(NamedTuple):
    kind: str  # one of CALL_TYPES
    label: str  # the input line, shown with the samples
    system: str
    prompt: str

    @property
    def schema(self):
        return SUGGESTION_SCHEMA if self.kind in JSON_CALLS else None


# Not part of the workload, so no timed call starts on a prompt Ollama has just cached.
WARMUP = Call("suggestion", "warm-up", "Reply with the JSON object you are given a schema for.", "Hello.")


class Result(NamedTuple):
    label: str
    secs: float | None  # None: the call failed
    valid: bool
    sample: str


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
    """[Call] in a fixed order."""
    system = suggestion_system()
    calls = []
    for meeting in json.loads(DATA.read_text(encoding="utf-8"))["meetings"]:
        facts = Facts(path=None, fresh=True)  # only rendered, never saved
        facts.data = dict(meeting["facts"])
        memory = Memory()
        memory.text = meeting["memory"]
        transcript = Transcript()
        for i, line in enumerate(meeting["lines"]):
            # 5 s apart, so every line is its own turn as in a real transcript
            transcript.add(Segment(line["source"], line["text"], line["lang"], duration=3.0, t_end=5.0 * (i + 1)))
            label = f"[{line['lang']}] {line['text']}"
            if line["lang"] in TRANSLITERATED:  # either source, as in a meeting
                calls.append(Call("transliteration", label, TRANSLITERATE_SYSTEM.format(lang=line["lang"]), line["text"]))
            if line["source"] != PARTICIPANT:
                continue
            prompt = meeting_context(facts, memory, meeting["knowledge"], transcript) + proactive_instruction()
            calls.append(Call(meeting["call"], label, system, prompt))
            target = translation_target(line["lang"])
            calls.append(Call("translation", f"{label} -> {target}", TRANSLATE_SYSTEM.format(target=target), line["text"]))
    return calls


def _short(text, n=240):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3] + "..."


def _output(call, text):
    """(valid, sample text) of one reply."""
    if not call.schema:
        return True, _short(text)
    result = parse_json(text)
    if not isinstance(result, dict) or not matches(SUGGESTION_SCHEMA, result):
        return False, "INVALID JSON: " + _short(text)
    said = " / ".join(s["text"] for s in result["suggestions"]) or "(no suggestion)"
    return True, _short(f"{result['trigger']}: {said}")


async def run_model(provider, model, calls):
    """{call type: [Result]}"""
    provider.model = model
    t = time.perf_counter()
    for attempt in range(WARMUP_TRIES):  # a large model can take longer than one request timeout to load
        try:
            await provider.complete(WARMUP.system, WARMUP.prompt, WARMUP.schema)
            break
        except httpx.TimeoutException:
            if attempt == WARMUP_TRIES - 1:
                raise
    print(f"{model}: warm-up {time.perf_counter() - t:.1f} s")
    results = {kind: [] for kind in CALL_TYPES}
    for call in calls:
        t = time.perf_counter()
        try:
            text = await provider.complete(call.system, call.prompt, call.schema)
        except httpx.ConnectError:
            raise  # Ollama is gone: main() stops the run
        except (httpx.HTTPError, KeyError, ValueError) as e:  # ValueError: a reply that isn't JSON
            results[call.kind].append(Result(call.label, None, False, f"ERROR {type(e).__name__}: {_short(e)}"))
            continue
        secs = time.perf_counter() - t
        valid, sample = _output(call, text)
        results[call.kind].append(Result(call.label, secs, valid, sample))
        print(f"  {call.kind:<{KIND_W}} {secs * 1000:6.0f} ms  {'' if valid else 'invalid  '}{_short(call.label, 60)}")
    return results


def _picks(n):
    """SAMPLES indices spread over n calls, so the samples cover every language of the transcript."""
    return sorted({round(i * (n - 1) / (SAMPLES - 1)) for i in range(SAMPLES)}) if n > 1 else list(range(n))


def report(results):
    """results: {model: {call type: [Result]}} -> summary table, then the samples of all models side by side."""
    models = list(results)
    print("\n" + "=" * 78)
    print(f"{'model':<28} {'call type':<{KIND_W + 1}} {'n':>3} {'p50 ms':>7} {'p95 ms':>7}  {'valid JSON':>10}  errors")
    for model in models:
        for kind in CALL_TYPES:
            rows = results[model][kind]
            lat = [r.secs for r in rows if r.secs is not None]
            p50, p95 = (f"{np.percentile(lat, q) * 1000:.0f}" for q in (50, 95)) if lat else ("-", "-")
            valid = f"{sum(r.valid for r in rows)}/{len(rows)}" if kind in JSON_CALLS else ""
            errors = sum(r.secs is None for r in rows)
            print(f"{model:<28} {kind:<{KIND_W + 1}} {len(rows):>3} {p50:>7} {p95:>7}  {valid:>10}  {errors or ''}")
    for kind in CALL_TYPES:
        rows = results[models[0]][kind]
        for n, i in enumerate(_picks(len(rows)), 1):
            print(f"\n--- {kind} sample {n}: {_short(rows[i].label, 200)}")
            for model in models:
                r = results[model][kind][i]
                ms = f"{r.secs * 1000:.0f} ms" if r.secs is not None else "failed"
                print(f"  {model} ({ms}): {r.sample}")


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
    version = await provider.version() or "?"
    print(f"Ollama {version} at {config.OLLAMA_URL}, prompt {config.SYSTEM_PROMPT_FILE.name}, LANGUAGES {','.join(config.LANGUAGES)}, "
          f"{len(calls)} calls per model")
    results = {}
    try:
        for model in models:
            results[model] = await run_model(provider, model, calls)
    except httpx.ConnectError as e:
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
