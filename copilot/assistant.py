"""Orchestration: transcript in, proactive suggestions / push-to-ask answers / translations and
transliterations out.

All methods run on the asyncio loop. Segments arrive from the STT thread via
loop.call_soon_threadsafe(assistant.on_segment, seg).
"""

import asyncio
import time
from types import SimpleNamespace

from . import config, export, strings
from .audio import PARTICIPANT
from .memory import Memory, summarize
from .transcript import Transcript, new_session_path

ASK_ADDENDUM = """
# Push-to-Ask
The user has asked you an explicit question (see [USER QUESTION]). Answer exactly that question,
grounded in the transcript, facts and knowledge. Keep the same JSON format: put your answer(s) in
"suggestions" (1-2 items, each 1-2 sentences, in German unless the question is in English) and set
"trigger" to a short label for the question.
"""

TRANSLATE_SYSTEM = (
    "You are a live interpreter. Translate the user's text into the language with the ISO 639-1 code "
    "'{target}'. Output only the translation, no quotes, no comments."
)
TRANSLITERATE_SYSTEM = (
    "Rewrite the user's text, spoken in the language with the ISO 639-1 code '{lang}', in Latin script "
    "(e.g. Urdu as Roman Urdu). Do not translate it: keep the same words in the same language. Keep "
    "English and other words already in Latin script unchanged, and keep names and numbers intact. "
    "Output only the rewritten text, no quotes, no comments."
)
REPLY_LANGUAGE_RULE = "\n# Language\nWrite every suggestion in the language with the ISO 639-1 code '{lang}'.\n"
PROACTIVE_INSTRUCTION = "\n\nReagiere auf den letzten [Participant]-Beitrag."


def suggestion_system(addendum=""):
    """The role prompt of suggestions and answers, with the reply-language rule when REPLY_LANGUAGE is set."""
    rule = REPLY_LANGUAGE_RULE.format(lang=config.REPLY_LANGUAGE) if config.REPLY_LANGUAGE else ""
    return config.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8") + addendum + rule


def meeting_context(facts, memory, knowledge, transcript):
    """The context layout every suggestion and answer is built on."""
    return (f"[FACTS]\n{facts.render()}\n\n[MEMORY]\n{memory.render()}\n\n"
            f"[KNOWLEDGE]\n{knowledge}\n\n[TRANSCRIPT]\n{transcript.render()}")


def translation_target(lang):
    """Lines in another language go into the user's (LANGUAGES[0]), the user's into the second."""
    mine = config.LANGUAGES[0]
    return mine if lang != mine else (config.LANGUAGES[1:] or ["en"])[0]


class Assistant:
    def __init__(self, transcript, facts, kb, llm, emit, memory, open_capture=None):
        self.transcript = transcript
        self.memory = memory
        self.facts = facts
        self.kb = kb
        self.llm = llm
        self.emit = emit  # callable(dict), must be called on the loop
        self.system = suggestion_system()
        self.ask_system = suggestion_system(ASK_ADDENDUM)
        self.proactive = True
        self.translate = False
        self._one_at_a_time = asyncio.Lock()  # translations and transliterations, one at a time
        self.engine = "-"
        self._timer = None
        self._pending_since = None
        self._running = False
        self._again = False
        self._last_trigger = ("", 0.0)  # (trigger, time) of the last proactive card
        self.open_capture = open_capture  # () -> Capture, opened on the first meeting start
        self.capture = None
        self.active = False  # a meeting is running: the participants are being listened to
        self.mic_on = True
        self.audio_error = ""
        self.cards = []  # suggestions and answers of this meeting, for the meeting record
        self.started = None

    # --- meeting ---------------------------------------------------------------
    def start_meeting(self):
        """New session with a fresh transcript and memory (the facts sheet is kept)."""
        if self.active:
            self.stop_meeting()
        session = new_session_path()
        self.transcript = Transcript(session)
        self.memory = Memory(session.with_suffix(".memory.md"))
        self.cards = []
        self.started = time.time()
        self.emit({"type": "reset"})
        self.emit({"type": "memory", "text": self.memory.render()})
        self.active = True
        self._apply_audio()
        if self.active:
            print(f"Session:  {session}")

    def stop_meeting(self):
        """Returns the task that saves the meeting record (awaited on shutdown)."""
        self.active = False
        self._apply_audio()
        return asyncio.ensure_future(self._save(self.transcript, self.memory, self.cards, self.started))

    async def _save(self, transcript, memory, cards, started):
        """Fold the last minutes into the memory, then write the meeting record. A new meeting
        may already be running, so this works on the old meeting's objects."""
        current = lambda: self.transcript is transcript  # noqa: E731
        ended = time.time()
        meeting = SimpleNamespace(transcript=transcript, memory=memory, llm=self.llm,
                                  emit=lambda e: current() and self.emit(e))
        try:
            await asyncio.wait_for(summarize(meeting), 30)
        except asyncio.TimeoutError:
            pass  # save with the last memory
        path = export.write(transcript, memory, self.facts, cards, started, ended)
        if path:
            print(f"Saved:    {path}")
            if current():
                self.emit({"type": "saved", "path": str(path)})

    def set_mic(self, on):
        self.mic_on = on
        self._apply_audio()

    def meeting_state(self):
        return {"type": "meeting", "active": self.active, "mic": self.mic_on, "error": self.audio_error}

    def _apply_audio(self):
        """Open only what is needed: the devices are closed while stopped or muted."""
        self.audio_error = ""
        try:
            if self.active and self.capture is None:
                self.capture = self.open_capture()
            if self.capture:
                for source, on in ((self.capture.mic, self.active and self.mic_on),
                                   (self.capture.loopback, self.active)):
                    source.start() if on else source.stop()
        except Exception as e:
            print(f"Audio:    {e}")
            self.audio_error = str(e)
            self.active = False
            for source in self.capture.sources if self.capture else []:
                source.stop()
        self.emit(self.meeting_state())

    # --- input ---------------------------------------------------------------
    def on_segment(self, seg):
        turn, new_turn, retracted = self.transcript.add(seg)
        for rid in retracted:
            self.emit({"type": "retract", "id": rid})
        if turn is None:
            return
        self.emit({"type": "segment", "id": seg.id, "source": seg.source, "text": seg.text,
                   "lang": seg.lang, "new_turn": new_turn, "stt_ms": round(seg.latency * 1000)})
        if seg.lang in config.TRANSLITERATE:
            asyncio.ensure_future(self._transliterate(seg))
        if seg.source != PARTICIPANT:
            return
        if self.translate:
            asyncio.ensure_future(self._translate(seg))
        if self.proactive:
            self._schedule()

    def _schedule(self):
        """Wait for the end of the participant's turn, but never hold text longer than PROACTIVE_MAX_PENDING_S."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        self._pending_since = self._pending_since or now
        delay = min(config.PROACTIVE_DEBOUNCE_S,
                    max(0.0, self._pending_since + config.PROACTIVE_MAX_PENDING_S - now))
        if self._timer:
            self._timer.cancel()
        self._timer = loop.call_later(delay, self._fire)

    def _fire(self):
        self._timer = None
        self._pending_since = None
        if self._running:
            self._again = True  # run once more with the newest transcript when this one ends
            return
        asyncio.ensure_future(self._proactive_loop())

    # --- proactive -----------------------------------------------------------
    async def _proactive_loop(self):
        self._running = True
        try:
            while True:
                self._again = False
                await self._run_proactive()
                if not self._again or not self.proactive:
                    break
        finally:
            self._running = False

    def _last_participant_text(self, max_chars=300):
        """Tail of the latest participant turn (a monologue becomes one long turn)."""
        for t in reversed(self.transcript.turns):
            if t.source == PARTICIPANT:
                return t.text[-max_chars:]
        return ""

    async def _context(self, rag_query):
        knowledge = await asyncio.to_thread(self.kb.context, rag_query)
        return meeting_context(self.facts, self.memory, knowledge, self.transcript)

    async def _run_proactive(self):
        latest = self._last_participant_text()
        if not latest:
            return
        prompt = await self._context(latest) + PROACTIVE_INSTRUCTION
        try:
            result, engine, secs = await self.llm.generate_json(self.system, prompt)
        except Exception as e:  # a timeout's message is empty
            self.emit({"type": "error", "text": f"LLM: {str(e) or type(e).__name__}"})
            return
        self._engine(engine, secs)
        if not isinstance(result, dict):
            return
        self._merge_facts(result)
        suggestions = [s for s in result.get("suggestions") or [] if s.get("text")]
        trigger = (result.get("trigger") or "None").strip()
        if not suggestions or trigger.lower() == "none":
            return
        last, at = self._last_trigger
        if trigger.lower() == last and time.time() - at < config.PROACTIVE_REPEAT_S:
            return  # same point as the card on screen: don't replace it
        self._last_trigger = (trigger.lower(), time.time())
        self._card({"type": "suggestion", "trigger": trigger, "suggestions": self._limit(suggestions),
                    "engine": engine, "ms": round(secs * 1000), "ts": time.time()})

    # --- push-to-ask -----------------------------------------------------------
    async def ask(self, query):
        query = query.strip()
        if not query:
            return
        self.emit({"type": "asking", "query": query})
        prompt = (await self._context(f"{query} {self._last_participant_text()}")
                  + f"\n\n[USER QUESTION]\n{query}")
        try:
            result, engine, secs = await self.llm.generate_json(self.ask_system,
                                                                prompt)
        except Exception as e:
            self.emit({"type": "error", "text": f"LLM: {str(e) or type(e).__name__}"})
            return
        self._engine(engine, secs)
        if not isinstance(result, dict):
            self.emit({"type": "error", "text": "Antwort war kein gültiges JSON."})
            return
        self._merge_facts(result)
        self._card({"type": "answer", "query": query, "trigger": result.get("trigger", ""),
                    "suggestions": self._limit(result.get("suggestions") or []),
                    "engine": engine, "ms": round(secs * 1000), "ts": time.time()})

    # --- translation / transliteration ------------------------------------------
    async def _translate(self, seg):
        target = translation_target(seg.lang)
        result = await self._plain(seg, TRANSLATE_SYSTEM.format(target=target), "Translation")
        if result:
            self.emit({"type": "translation", "id": seg.id, "source": seg.source, "target": target.upper(),
                       **result})

    async def _transliterate(self, seg):
        result = await self._plain(seg, TRANSLITERATE_SYSTEM.format(lang=seg.lang), "Transliteration")
        if result:
            self.emit({"type": "transliteration", "id": seg.id, "source": seg.source, **result})

    async def _plain(self, seg, system, label):
        """Plain-text LLM call for one line: {text, engine, ms}, or None when skipped or failed.
        A line retracted as an echo, or from a meeting that has ended, is skipped."""
        transcript, queued = self.transcript, time.monotonic()
        gone = lambda: seg.retracted or self.transcript is not transcript  # noqa: E731
        # One at a time: a local model can't keep up with every line in parallel, and a pile-up
        # made all of them time out. Lines that waited too long are skipped to catch up.
        async with self._one_at_a_time:
            if gone() or time.monotonic() - queued > config.PLAIN_MAX_WAIT_S:
                return None
            try:
                text, engine, secs = await self.llm.generate(system, seg.text)
            except Exception as e:
                self.emit({"type": "error", "text": f"{label}: {str(e) or type(e).__name__}"})
                return None
        return None if gone() else {"text": text.strip(), "engine": engine, "ms": round(secs * 1000)}

    # --- toggles / helpers ---------------------------------------------------
    def set_mode(self, proactive=None, translate=None):
        if proactive is not None:
            self.proactive = proactive
        if translate is not None:
            self.translate = translate
        self.emit({"type": "mode", "proactive": self.proactive, "translate": self.translate})

    @staticmethod
    def presets():
        """The preset buttons in the UI language (English when the language has no presets)."""
        def localized(texts):
            return texts.get(strings.language()) or texts["en"]
        return [{"hotkey": p["hotkey"], "label": localized(p["label"]), "query": localized(p["query"])}
                for p in config.PRESETS]

    def set_provider(self, provider_id):
        self.llm.set_primary(provider_id)
        self.emit(self.llm.describe())
        if provider_id.startswith("ollama:"):
            asyncio.ensure_future(self.llm.fallback.warmup())

    @staticmethod
    def _limit(suggestions):
        return suggestions[:1 if config.SINGLE_SUGGESTION else 2]

    def _card(self, event):
        if self.active:
            self.cards.append(event)
        self.emit(event)

    def _engine(self, engine, secs):
        self.engine = engine
        self.emit({"type": "engine", "engine": engine, "ms": round(secs * 1000)})

    def _merge_facts(self, result):
        new = result.get("new_facts")
        if isinstance(new, list) and self.facts.merge(new):
            self.emit({"type": "facts", "text": self.facts.render()})
