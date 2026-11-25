# Ticket: WhisperKit server RAM ceiling

## Meta
- ID:             014
- Typ:            bug
- Status:         completed
- Erstellt:       2026-11-25
- Erstellt von:   Adeel Nayyer Sheikh
- Verantwortlich: Adeel Nayyer Sheikh
- Referenzen:     copilot/whisperkit.py · copilot/stt.py · copilot/config.py · .env.example · docs/adr/0001-recycle-whisperkit-server.md · /Library/Logs/DiagnosticReports/Retired/JetsamEvent-2026-11-24-*.ips

## Kontext
On 2026-11-24 the Mac froze. The macOS Jetsam reports name `whisperkit-cli serve` (1.1.1, large-v3, `cpuAndGPU`) as the largest process: 36.6 GB at 15:05, 47.3 GB at 15:35, on a 48 GB M4 Max. One server ran from 11:20 to 15:36 through five meetings and 3,669 requests without a restart or an error, which is roughly 12–13 MB kept per request (estimated, not measured per request). The copilot's Python process stayed at 3.2 GB, there were no orphaned servers, and crash recovery never ran. No upstream issue names this leak; argmax-oss-swift #517 (native memory kept on transcriber re-creation) is the closest.

## Ziel
The WhisperKit server never holds more than its size after warm-up plus a set growth, however long the app or a meeting runs, and replacing it causes no pause in transcription.

## Anforderungen
- `WHISPERKIT_MAX_GROWTH_GB` (default 1.5): the server's RAM is measured after warm-up, then after each request; once it has grown past the limit, the server is replaced.
- Replacement in the background: a new server starts on a new port and is warmed up while the old one keeps serving; the switch happens between two requests, then the old one is stopped. The chunk in flight is never lost or sent twice.
- A replacement that fails to start keeps the old server serving and is retried after the next request; it is not a crash and does not count toward the one crash restart per run.
- One console line per replacement (old and new RAM, request count). Nothing in the window.
- `sessions/whisperkit.log` is opened for appending, with one marker line per server start, so a replacement does not wipe the log.
- Tests with the existing fake HTTP servers and an injectable RAM probe; CI stays on all three OSes.

## Akzeptanzkriterien
- [x] A fake server whose RAM probe grows past the limit is replaced, the next chunk goes to the new server, and the old one is stopped.
- [x] No chunk is dropped or transcribed twice across a replacement.
- [x] A failed replacement leaves transcription running on the old server.
- [x] Real run: `stt_bench` looping a WAV for 2,000+ requests keeps the server's RAM under warm-up size + 1.5 GB (+ one model during a switch).
- [x] `.env.example`, README and CLAUDE.md describe the setting; ADR 0001 is linked.

## Risiken
- During a switch two servers are in RAM (about 2× the model).
- Server start takes 2–23 s; at 20 GB/h of growth, a 1.5 GB limit triggers a switch every few minutes with large-v3, so the cost of each switch (CPU, GPU, disk for Core ML loading) matters. A lighter model (ticket 017) lowers both.

## Offene Fragen
- Does the leak grow per request or with time? The benchmark in ticket 017 measures it.

## Erste Umsetzungsidee
Read the child's RSS with `ps -o rss= -p <pid>` (no new dependency) behind a `rss_probe(pid)` seam. `WhisperKitEngine` keeps `self.server` and an optional `self._next` being started on a worker thread; `transcribe()` checks after each reply and swaps when `_next` is healthy and warm.

## Kommentare
- 2026-11-25: done in addeelnayyer/meeting-copilot#45 (PR #50). Real run on the M4 Max, large-v3 `cpuAndGPU`, replacement on with the default 1.5 GB: 2,500 requests of a 29 s English WAV; server RSS 1.59 GB after warm-up, peak 1.91 GB (ceiling 3.09 GB). The leak did not reproduce with this repeated WAV (growth -0.03 GB per 1,000 requests), so no replacement triggered at 1.5 GB. With `WHISPERKIT_MAX_GROWTH_GB=0.1` (600 requests) and `0.01` (300 requests) the real server was replaced 6 and 73 times, every request was served, and no server was left running. Whether the growth follows requests, time or varied audio is open for ticket 017 (#49).
