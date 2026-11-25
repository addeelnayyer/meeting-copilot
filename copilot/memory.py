"""Meeting memory: the segments after the checkpoint are folded into a compact summary after
each request the user makes and once when the meeting stops, so the LLM still knows what happened
before the segments it is sent."""

from . import config
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

    def pending(self, segments):
        return [s for s in segments if s.id > self.checkpoint]

    def prompt(self, segments):
        lines = "\n".join(f"[{s.source}] {s.text}" for s in segments)
        return f"[CURRENT MEMORY]\n{self.text or '(leer)'}\n\n[NEW TRANSCRIPT]\n{lines}"

    def update(self, text, segments):
        self.text = text.strip()
        self.checkpoint, self.until = segments[-1].id, segments[-1].ts
        if self.path:
            self.path.write_text(self.text, encoding="utf-8")

    def render(self):
        return self.text or "(noch keine)"

    def event(self, **extra):
        return {"type": "memory", "text": self.text, "until": self.until, **extra}


async def fold(llm, memory, segments):
    """Fold the segments after the checkpoint into the memory and move the checkpoint to the last
    of them. Returns (engine, secs), or None when fewer than MEMORY_MIN_WORDS new words were said.
    An LLM failure is raised and leaves the checkpoint where it was."""
    new = memory.pending(segments)
    if sum(len(s.text.split()) for s in new) < config.MEMORY_MIN_WORDS:
        return None
    text, engine, secs = await llm.generate(system_prompt(config.REPLY_LANGUAGE), memory.prompt(new))
    memory.update(text, new)
    return engine, secs
