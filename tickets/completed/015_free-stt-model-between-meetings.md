# Ticket: Free the STT model between meetings

## Meta
- ID:             015
- Typ:            feature
- Status:         completed
- Erstellt:       2026-11-25
- Erstellt von:   Adeel Nayyer Sheikh
- Verantwortlich: Adeel Nayyer Sheikh
- Referenzen:     copilot/stt.py (SttWorker, FasterWhisperEngine.close) · copilot/assistant.py (start_meeting, stop_meeting) · copilot/__main__.py · copilot/web/index.html (stt events)

## Kontext
The app starts idle, but `SttWorker.run()` builds and warms the STT engine at startup and keeps it until the app quits. On 2026-11-24 one WhisperKit server lived through five meetings and the idle time between them (see ticket 014). `FasterWhisperEngine.close()` is `pass`, so faster-whisper's STT model is never freed either.

## Ziel
While no meeting runs, no STT model is in RAM. Starting a meeting loads it again, and the window shows the loading state until it is ready.

## Anforderungen
- `stop_meeting()` closes the STT engine once the chunks of the ended meeting are transcribed; `start_meeting()` builds and warms it again (`stt` events: not ready, then ready with `engine`).
- Chunks that arrive before the engine is ready wait in the queue and are transcribed once it is; none are dropped.
- `FasterWhisperEngine.close()` drops the model so its RAM is freed.
- A run that switched to faster-whisper after a WhisperKit failure keeps faster-whisper for the next meetings of that run.
- Stop then start in quick succession (or start during a build) builds the engine once.

## Akzeptanzkriterien
- [x] After a meeting stops, the engine's `close()` has been called and the worker holds no engine.
- [x] Starting the next meeting builds a warm engine, and early chunks are transcribed, not lost.
- [x] A fast stop/start sequence builds exactly one engine.
- [x] Real run: with no meeting running, no `whisperkit-cli` process exists.

## Risiken
- Each meeting start now waits for the STT model to load (seconds; up to minutes on a first Core ML specialization).

## Offene Fragen

## Erste Umsetzungsidee
`SttWorker` gets `want_engine` (a `threading.Event`) set by `start_meeting` and cleared by `stop_meeting` via the loop; the worker builds when it is set and closes when it is cleared and the queue is empty. Remember the run's engine choice after a switch.

## Kommentare
- 2026-11-25: done in addeelnayyer/meeting-copilot#46 (PR #50). Real run: the app started idle (`--no-browser --no-hotkeys`) runs no `whisperkit-cli` process. Stopping a meeting closes the engine once no chunk has arrived for `CLOSE_GRACE_S` (1.5 s); a segment cut before the current meeting started is dropped, so a late chunk never lands in the next meeting. The window shows "STT off" while idle.
