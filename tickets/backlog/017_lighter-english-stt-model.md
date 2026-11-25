# Ticket: Lighter English STT model on Apple Silicon

## Meta
- ID:             017
- Typ:            feature
- Status:         proposed
- Erstellt:       2026-11-25
- Erstellt von:   Adeel Nayyer Sheikh
- Verantwortlich: Adeel Nayyer Sheikh
- Referenzen:     copilot/stt_bench.py · copilot/whisperkit.py · tickets/completed/013_whisperkit-stt-engine-on-apple-silicon.md (step 0) · docs/adr/0001-recycle-whisperkit-server.md

## Kontext
The user's meetings are English only now (Urdu dropped). The STT model is still the multilingual large-v3 (`openai_whisper-large-v3-v20240930`, 1.5 GB on disk) on WhisperKit, and the faster-whisper fallback is the multilingual `small`. Ticket 013 compared only these two and measured no RAM. English-only and distilled STT models are much smaller; how much they leak in `whisperkit-cli serve` (ticket 014) is unknown.

## Ziel
The user's `.env` uses the smallest English STT model whose word error rate stays within 2 points of today's 4.3 %, measured on accuracy, latency and RAM. The open-source defaults stay multilingual.

## Anforderungen
- Candidates on WhisperKit (`cpuAndGPU`): `distil-whisper_distil-large-v3` (plus its turbo variant if published), `openai_whisper-small.en`, `openai_whisper-base.en`; `large-v3-v20240930` as the reference. Fallback candidates on faster-whisper CPU int8: `small.en`, `base.en`.
- Per candidate: median and p95 per chunk, English WER on ticket 013's English set, server RAM after warm-up, and RAM growth per 1,000 requests. Also large-v3 on the Neural Engine, to see whether compute units change the leak.
- `stt_bench` reports RAM (warm-up and growth) next to latency and WER.
- Result table in this ticket's Kommentare; the chosen model goes to the user's `.env` only.
- A minimal reproduction of the leak (serve command, request loop, RAM curve) is drafted for argmaxinc; it is posted only after the user has read and approved it.

## Akzeptanzkriterien
- [ ] Table with all candidates and all five figures.
- [ ] The chosen model is within 2 points of 4.3 % English WER and is the smallest that is.
- [ ] Leak report drafted and shown to the user.

## Risiken
- The English set is synthetic speech; real meeting audio may rank the candidates differently.
- `tiny.en` is kept out of the defaults; it is only a choice for very weak machines.

## Offene Fragen
- Does the leak per request scale with model size?

## Erste Umsetzungsidee
Loop the WAV through `build_stt_engine()` for N requests and sample the server PID's RSS every 100 requests, reusing ticket 014's RAM probe.

## Kommentare
