"""Meeting memory: the segments after the checkpoint are folded into it after each request the
user makes and once when the meeting stops, so the LLM still knows what happened before the
segments it is sent."""

import asyncio

from . import config
from .transcript import speaker_lines

MEMORY_SYSTEM = """You maintain the running memory of a live meeting for an AI meeting advisor.
You get the current memory and the new transcript lines since the last update.
Return the updated memory {language} as short bullet points, grouped under these headings:
{headings}
Keep concrete numbers, names, dates and commitments. Drop small talk. Max ~250 words.
Output only the memory, no preamble."""

GERMAN_HEADINGS = """## Verlauf (what was discussed, in order)
## Entscheidungen & Zusagen
## Offene Punkte & Einwände
## Stimmung & Prioritäten der Teilnehmer"""

ENGLISH_HEADINGS = """## Discussion (what was discussed, in order)
## Decisions & commitments
## Open points & objections
## Mood & priorities of the participants"""


def system_prompt(lang):
    """The memory follows REPLY_LANGUAGE (an ISO 639-1 code), else German."""
    if not lang:
        return MEMORY_SYSTEM.format(language="in German", headings=GERMAN_HEADINGS)
    return MEMORY_SYSTEM.format(
        language=f"in the language with the ISO 639-1 code '{lang}'",
        headings=f"{ENGLISH_HEADINGS}\n(Write the headings in that language as well.)")


class Memory:
    def __init__(self, path=None):
        self.text = ""
        self.path = path  # written next to the session log
        self.checkpoint = 0  # id of the last segment already in the memory
        self.until = None  # time.time() of that segment, shown in the window
        self._task = None  # the running fold: at most one at a time

    def pending(self, segments):
        return [s for s in segments if s.id > self.checkpoint]

    def prompt(self, segments):
        lines = speaker_lines((s.source, s.text) for s in segments)
        return f"[CURRENT MEMORY]\n{self.text or '(leer)'}\n\n[NEW TRANSCRIPT]\n{lines}"

    async def update(self, text, segments):
        self.text = text.strip()
        self.checkpoint, self.until = segments[-1].id, segments[-1].ts
        if self.path:  # off the loop
            await asyncio.to_thread(self.path.write_text, self.text, encoding="utf-8")

    def render(self):
        return self.text or "(noch keine)"

    def event(self, **extra):
        return {"type": "memory", "text": self.text, "until": self.until, **extra}

    # --- folding -------------------------------------------------------------------
    def folding(self):
        """True while a fold runs."""
        return self._task is not None and not self._task.done()

    async def fold(self, llm, segments, min_words=config.MEMORY_MIN_WORDS, wanted=lambda: True):
        """Fold the segments after the checkpoint into the memory and move the checkpoint to the
        last of them. One fold at a time: a running fold is waited for first, then nothing is done
        unless `wanted()` still holds. Returns (engine, secs), or None when skipped (not wanted, or
        fewer than `min_words` new words). An LLM failure is raised and leaves the checkpoint where
        it was."""
        while self.folding():
            await asyncio.wait({self._task})
        if not wanted():
            return None
        self._task = asyncio.ensure_future(self._fold(llm, segments, min_words))
        return await self._task

    def cancel(self):
        """Cancel the running fold, if any."""
        if self._task:
            self._task.cancel()

    async def _fold(self, llm, segments, min_words):
        new = self.pending(segments)
        if sum(len(s.text.split()) for s in new) < min_words:
            return None
        text, engine, secs = await llm.generate(system_prompt(config.REPLY_LANGUAGE), self.prompt(new))
        await self.update(text, new)
        return engine, secs
