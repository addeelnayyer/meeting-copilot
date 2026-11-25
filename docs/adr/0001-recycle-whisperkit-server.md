# Recycle the WhisperKit server on a RAM ceiling

`whisperkit-cli serve` 1.1.1 retains roughly 12–13 MB per transcription request and never gives it back: on 2026-11-24 one server grew to 47 GB over 3,669 requests in five meetings and froze a 48 GB M4 Max, while the copilot's Python process stayed flat at 3.2 GB (macOS Jetsam reports). We cannot fix the leak in Homebrew's binary, so the copilot treats the server as disposable: it is stopped when a meeting ends, and during a meeting it is replaced as soon as its RAM has grown `WHISPERKIT_MAX_GROWTH_GB` (default 1.5) past its size after warm-up. The replacement is started and warmed in the background while the old server keeps serving, and the switch happens between two requests, so transcription never pauses.

## Considered Options

- **Restart only between meetings:** a single two-hour meeting still reaches about 40 GB.
- **Restart every N requests:** the leak per request depends on the model and compute units, so N would need retuning for each one.
- **Absolute RAM ceiling:** a small model would leak for a long time below a ceiling sized for large-v3; growth since warm-up works the same for every model.
- **Plain stop-and-restart when the ceiling is hit:** server start takes 2–23 s, and transcription would stall for that long mid-meeting.

## Consequences

- RAM briefly holds two servers (about 2× the model) during a switch.
- `sessions/whisperkit.log` is opened for appending, so a switch does not wipe the log.
- Drop this mechanism once an upstream release no longer leaks, which the benchmark's RAM-growth measurement shows.
