"""Orchestration: transcript in, push-to-ask answers / proactive suggestions / translations and
transliterations out.

The LLM is called only on demand: a request sends the meeting memory plus the segments after the
checkpoint, and the meeting memory is folded after it (and once when the meeting stops).

All methods run on the asyncio loop. Segments arrive from the STT thread via
loop.call_soon_threadsafe(assistant.on_segment, seg).
"""

import asyncio
import time
from dataclasses import dataclass, field

from . import config, export, strings
from .audio import PARTICIPANT
from .memory import Memory
from .strings import t, ui_text
from .transcript import Transcript, new_session_path

ASK_ADDENDUM = """
# Push-to-Ask
The user has asked you an explicit question (see [USER QUESTION]). Answer exactly that question,
grounded in the transcript, facts and knowledge. Keep the same JSON format: put your answer(s) in
"suggestions" (1-2 items, each 1-2 sentences, in the language the rules above ask for) and set
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
REPLY_LANGUAGE_RULE = (
    "\n# Language\nWrite every suggestion, answer, trigger and fact in the language with the ISO 639-1 "
    "code '{lang}'. This overrides every other language instruction above (e.g. \"German\"), whatever "
    "language the participants speak.\n"
)
REPLY_LANGUAGE_REMINDER = "\n\nReply in the language with the ISO 639-1 code '{lang}'."
PROACTIVE_INSTRUCTION = "\n\nRespond to the latest [Participant] turn."
STOP_LIMIT_S = 30  # the last fold and the transliterations at a meeting stop, then the record is written


def reply_reminder():
    """The reply-language rule repeated at the end of the user prompt, where small local models heed it."""
    return REPLY_LANGUAGE_REMINDER.format(lang=config.REPLY_LANGUAGE) if config.REPLY_LANGUAGE else ""


def proactive_instruction():
    return PROACTIVE_INSTRUCTION + reply_reminder()


def suggestion_system(addendum=""):
    """The role prompt of suggestions and answers, with the reply-language rule when REPLY_LANGUAGE is set."""
    rule = REPLY_LANGUAGE_RULE.format(lang=config.REPLY_LANGUAGE) if config.REPLY_LANGUAGE else ""
    return config.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8") + addendum + rule


def meeting_context(facts, memory, knowledge, transcript):
    """The context layout every suggestion and answer is built on (transcript: its rendered lines)."""
    return (f"[FACTS]\n{facts.render()}\n\n[MEMORY]\n{memory.render()}\n\n"
            f"[KNOWLEDGE]\n{knowledge}\n\n[TRANSCRIPT]\n{transcript}")


def translation_target(lang):
    """Lines in another language go into the user's (LANGUAGES[0]), the user's into the second."""
    mine = config.LANGUAGES[0]
    return mine if lang != mine else (config.LANGUAGES[1:] or ["en"])[0]


@dataclass(eq=False)
class Meeting:
    """What belongs to one meeting. A stopped meeting's objects are still saved while the next
    one may already run."""
    transcript: Transcript
    memory: Memory
    cards: list = field(default_factory=list)  # suggestions and answers, for the meeting record
    started: float = None  # time.time() at the start
    # time.monotonic() at the start, the clock of Segment.t_end: an earlier segment was cut in an
    # ended meeting.
    since: float = float("-inf")
    transliterations: set = field(default_factory=set)  # the running ones, awaited by the record


class Assistant:
    def __init__(self, transcript, facts, kb, llm, emit, memory, open_capture=None, open_stt=None, close_stt=None):
        self.meeting = Meeting(transcript, memory)
        self.facts = facts
        self.kb = kb
        self.llm = llm
        self.emit = emit  # callable(dict), must be called on the loop
        self.system = suggestion_system()
        self.ask_system = suggestion_system(ASK_ADDENDUM)
        self.proactive = False  # off at every meeting start
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
        # The STT worker holds an engine only during a meeting: () -> None each, only signalling it.
        self.open_stt = open_stt or (lambda: None)
        self.close_stt = close_stt or (lambda: None)
        self.active = False  # a meeting is running: the participants are being listened to
        self.mic_on = True
        self.audio_error = ""

    # --- meeting ---------------------------------------------------------------
    @property
    def transcript(self):
        return self.meeting.transcript

    @property
    def memory(self):
        return self.meeting.memory

    def _current(self, meeting):
        """True while `meeting` is the latest one (running, or stopped with no new one started):
        the one the window shows."""
        return self.meeting is meeting

    def _ongoing(self, meeting):
        """True while `meeting` runs."""
        return self.active and self._current(meeting)

    def start_meeting(self):
        """New session with a fresh transcript and memory (the facts sheet is kept)."""
        if self.active:
            self.stop_meeting()
        session = new_session_path()
        self.meeting = Meeting(Transcript(session), Memory(session.with_suffix(".memory.md")),
                               started=time.time(), since=time.monotonic())
        self.emit({"type": "reset"})
        self.emit(self.memory.event())
        self.set_mode(proactive=False)  # no LLM call unless asked, until turned on for this meeting
        self.active = True
        self._apply_audio()
        if self.active:
            print(f"Session:  {session}")

    def stop_meeting(self):
        """Returns the task that saves the meeting record (awaited on shutdown)."""
        self.active = False
        self._apply_audio()
        return asyncio.ensure_future(self._save(self.meeting))

    async def _save(self, meeting):
        """Fold the segments after the checkpoint into the meeting memory once (after a fold still
        running), however few, and let the queued transliterations land, all within STOP_LIMIT_S,
        then write the meeting record. A new meeting may already be running, so this works on the
        old meeting's objects."""
        ended = time.time()
        fold = asyncio.ensure_future(self._stop_fold(meeting))
        await asyncio.wait({fold, *meeting.transliterations}, timeout=STOP_LIMIT_S)
        fold.cancel()  # still running: save with the meeting memory so far
        meeting.memory.cancel()  # nothing writes this meeting's memory after its record
        path = export.write(meeting.transcript, meeting.memory, self.facts, meeting.cards, meeting.started, ended)
        if path:
            print(f"Saved:    {path}")
            if self._current(meeting):
                self.emit({"type": "saved", "path": str(path)})

    def set_mic(self, on):
        self.mic_on = on
        self._apply_audio()

    def meeting_state(self):
        return {"type": "meeting", "active": self.active, "mic": self.mic_on, "error": self.audio_error}

    def _apply_audio(self):
        """Open only what is needed: the devices are closed while stopped or muted, the STT engine
        while stopped."""
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
            self.audio_error = ui_text(e)
            self.active = False
            for source in self.capture.sources if self.capture else []:
                source.stop()
        (self.open_stt if self.active else self.close_stt)()  # closed once the queued chunks are done
        self.emit(self.meeting_state())

    # --- input ---------------------------------------------------------------
    def on_segment(self, seg):
        if seg.t_end < self.meeting.since:
            # Cut before this meeting started, transcribed after it: the ended meeting's last
            # words (its STT engine closed before the VAD sent them). Never this meeting's.
            print("STT:      a segment of the ended meeting arrived after the next start; dropped")
            return
        turn, new_turn, retracted = self.transcript.add(seg)
        for rid in retracted:
            self.emit({"type": "retract", "id": rid})
        if turn is None:
            return
        self.emit({"type": "segment", "id": seg.id, "source": seg.source, "text": seg.text,
                   "lang": seg.lang, "new_turn": new_turn, "stt_ms": round(seg.latency * 1000)})
        if seg.lang in config.TRANSLITERATE:
            task = asyncio.ensure_future(self._transliterate(seg))
            self.meeting.transliterations.add(task)
            task.add_done_callback(self.meeting.transliterations.discard)
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
        for turn in reversed(self.transcript.turns):
            if turn.source == PARTICIPANT:
                return turn.text[-max_chars:]
        return ""

    async def _context(self, rag_query, transcript):
        knowledge = await asyncio.to_thread(self.kb.context, rag_query)
        return meeting_context(self.facts, self.memory, knowledge, transcript)

    async def _run_proactive(self):
        latest = self._last_participant_text()
        if not latest:
            return
        asked = time.time()
        prompt = await self._context(latest, self.transcript.render()) + proactive_instruction()
        try:
            result, engine, secs = await self.llm.generate_json(self.system, prompt)
        except Exception as e:  # a timeout's message is empty
            self.emit({"type": "error", "text": f"LLM: {ui_text(e)}"})
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
                    **self._timing(engine, secs, asked)})

    # --- push-to-ask -----------------------------------------------------------
    async def ask(self, query):
        query = query.strip()
        if not query:
            return
        asked = time.time()
        self.emit({"type": "asking", "query": query})
        recent = self.transcript.render_since(self.memory.checkpoint, config.ASK_MIN_TURNS)
        prompt = (await self._context(f"{query} {self._last_participant_text()}", recent) + reply_reminder()
                  + f"\n\n[USER QUESTION]\n{query}")
        try:
            result, engine, secs = await self.llm.generate_json(self.ask_system,
                                                                prompt)
        except Exception as e:
            self.emit({"type": "error", "text": f"LLM: {ui_text(e)}"})
            return
        self._engine(engine, secs)
        if not isinstance(result, dict):
            self.emit({"type": "error", "text": t("llm.not_json")})
            return
        self._merge_facts(result)
        self._card({"type": "answer", "query": query, "trigger": result.get("trigger", ""),
                    "suggestions": self._limit(result.get("suggestions") or []),
                    **self._timing(engine, secs, asked)})
        self._fold_in_background()

    # --- meeting memory ----------------------------------------------------------
    async def summary(self, query):
        """Fold now and show the meeting memory as the card (no call when nothing new was said)."""
        asked, meeting = time.time(), self.meeting
        self.emit({"type": "asking", "query": query})
        try:
            done = await self._fold(meeting)
        except Exception as e:
            self._fold_failed(meeting, e, asked=True)  # ends this request
            return
        engine, secs = done or ("", 0.0)
        # Not recorded as a card: the meeting record holds the meeting memory itself.
        self.emit({"type": "answer", "summary": True, "query": query, "trigger": "",
                   "suggestions": [{"text": meeting.memory.text or t("record.no_summary")}],
                   **self._timing(engine, secs, asked)})

    def _fold_in_background(self):
        """After an answer: fold the segments after the checkpoint, unless a fold already runs
        (the next one takes what this one leaves)."""
        async def fold(meeting):
            if not meeting.memory.folding():
                await self._fold_quietly(meeting)
        asyncio.ensure_future(fold(self.meeting))

    async def _stop_fold(self, meeting):
        """At the stop: fold the ended meeting's rest once (after a running fold), however few words."""
        await self._fold_quietly(meeting, min_words=1, wanted=lambda: True)

    async def _fold_quietly(self, meeting, **kwargs):
        try:
            await self._fold(meeting, **kwargs)
        except Exception as e:
            self._fold_failed(meeting, e)

    async def _fold(self, meeting, min_words=config.MEMORY_MIN_WORDS, wanted=None):
        """The one fold routine, so at most one fold runs per meeting memory: (engine, secs), or
        None when skipped; an LLM failure is raised. Unless `wanted` says otherwise, nothing is
        folded once the meeting has stopped. The memory event reaches the window while this is
        the current meeting."""
        done = await meeting.memory.fold(self.llm, meeting.transcript.segments, min_words,
                                         wanted or (lambda: self._ongoing(meeting)))
        if done and self._current(meeting):
            self.emit(meeting.memory.event(**self._timing(*done)))
        return done

    def _fold_failed(self, meeting, error, asked=False):
        """The error of a fold the user asked for (Summary) ends that request; any other carries
        source "memory", so the window does not end a request in flight on it."""
        print(f"Memory:   {error}")
        if self._current(meeting):
            self.emit({"type": "error", "text": t("memory.failed", error=ui_text(error)),
                       **({} if asked else {"source": "memory"})})

    # --- translation / transliteration ------------------------------------------
    async def _translate(self, seg):
        target = translation_target(seg.lang)
        result = await self._plain(seg, TRANSLATE_SYSTEM.format(target=target), "Translation")
        if result:
            self.emit({"type": "translation", "id": seg.id, "source": seg.source, "target": target.upper(),
                       **result})

    async def _transliterate(self, seg):
        transcript = self.transcript
        result = await self._plain(seg, TRANSLITERATE_SYSTEM.format(lang=seg.lang), "Transliteration")
        if result:
            transcript.transliterate(seg, result["text"])
            self.emit({"type": "transliteration", "id": seg.id, "source": seg.source, **result})

    async def _plain(self, seg, system, label):
        """Plain-text LLM call for one line: {text, engine, ms}, or None when skipped or failed.
        A line retracted as an echo, or from a meeting that has ended, is skipped."""
        meeting, queued = self.meeting, time.monotonic()
        gone = lambda: seg.retracted or not self._current(meeting)  # noqa: E731
        # One at a time: a provider may not keep up with every line in parallel, and a pile-up
        # made all of them time out. Lines that waited too long are skipped to catch up.
        async with self._one_at_a_time:
            if gone() or time.monotonic() - queued > config.PLAIN_MAX_WAIT_S:
                return None
            try:
                text, engine, secs = await self.llm.generate(system, seg.text)
            except Exception as e:
                self.emit({"type": "error", "text": f"{label}: {ui_text(e)}"})
                return None
        return None if gone() else {"text": text.strip(), **self._timing(engine, secs)}

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
        return [{"id": p["id"], "hotkey": p["hotkey"], "action": p["action"], "label": localized(p["label"]),
                 "query": localized(p["query"])} for p in config.PRESETS]

    async def run_preset(self, preset_id):
        """A preset button or hotkey: its action, "ask" (its question) or "summary" (fold the meeting memory)."""
        preset = next((p for p in self.presets() if p["id"] == preset_id), None)
        if preset is None:
            return
        await {"ask": self.ask, "summary": self.summary}[preset["action"]](preset["query"])

    def set_provider(self, provider_id):
        self.llm.set_primary(provider_id)
        self.emit(self.llm.describe())

    @staticmethod
    def _limit(suggestions):
        return suggestions[:1 if config.SINGLE_SUGGESTION else 2]

    def _card(self, event):
        if self.active:
            self.meeting.cards.append(event)
        self.emit(event)

    def _engine(self, engine, secs):
        self.engine = engine
        self.emit({"type": "engine", **self._timing(engine, secs)})

    @staticmethod
    def _timing(engine, secs, asked=None):
        """The engine and latency an event carries, and `ts`, the time a card was asked."""
        timing = {"engine": engine, "ms": round(secs * 1000)}
        return timing if asked is None else {**timing, "ts": asked}

    def _merge_facts(self, result):
        new = result.get("new_facts")
        if isinstance(new, list) and self.facts.merge(new):
            self.emit({"type": "facts", "text": self.facts.render()})
