# Ticket: WhisperKit STT engine on Apple Silicon

## Meta
- ID:             013
- Typ:            feature
- Status:         completed
- Erstellt:       2026-11-19
- Erstellt von:   Adeel Nayyer Sheikh
- Verantwortlich: Adeel Nayyer Sheikh
- Referenzen:     copilot/whisperkit.py · copilot/stt.py · copilot/segment.py · copilot/config.py · install.sh · .env.example · GLOSSARY.md · https://github.com/addeelnayyer/meeting-copilot/issues/1

## Kontext
On a Mac there is no CUDA, so faster-whisper ran `small` in int8 on the CPU: less accurate than the `large-v3-turbo` used on NVIDIA machines, and heavy on the CPU during a meeting. Apple Silicon has a Neural Engine and a GPU that faster-whisper cannot use. Many Mac users already have large-v3-turbo in Core ML form from MacWhisper (`openai_whisper-large-v3-v20240930`, about 1.5 GB), which the copilot ignored.

## Ziel
A second STT engine, WhisperKit, runs large-v3-turbo on the Apple GPU. `STT_ENGINE=auto` picks it on macOS arm64; Windows, Linux and Intel Macs are unchanged. Anything that goes wrong with WhisperKit falls back to faster-whisper on the CPU, with the reason shown.

## Anforderungen
- `STT_ENGINE` = `auto` (WhisperKit only on macOS arm64) | `faster-whisper` | `whisperkit`.
- STT model lookup: `WHISPERKIT_MODEL_PATH`, then MacWhisper's models folder (read-only, the configured `WHISPERKIT_MODEL`, never MacWhisper's own selection), then the Hugging Face cache, else only that model's subfolder of `argmaxinc/whisperkit-coreml` is downloaded once into the cache.
- Homebrew's `whisperkit-cli serve` as a child on a free localhost port, warmed up before the meeting, stopped on exit; `--model-path`, so it never downloads the model by itself.
- `WHISPERKIT_COMPUTE_UNITS` (default `cpuAndGPU`) for encoder and decoder; `WHISPER_DEVICE/MODEL/COMPUTE` stay faster-whisper only.
- Fallback at startup (no binary, no model, failed download or start), one restart on a crash during a run, then faster-whisper for the rest of the run with a note in the UI.
- `install.sh` runs `brew install whisperkit-cli` on macOS arm64, or prints a hint without Homebrew. The app never installs anything.
- `stt_bench --engine faster-whisper|whisperkit` for comparisons on the real code paths.
- Unit tests need neither a Mac nor WhisperKit (fake HTTP server), so CI keeps running on all three OSes.

## Akzeptanzkriterien
- [x] Step 0: WhisperKit beats faster-whisper `small` on Apple Silicon (see Kommentare).
- [x] `STT_ENGINE=auto` picks WhisperKit on macOS arm64 and faster-whisper elsewhere, with a reason when it falls back (issue 4).
- [x] MacWhisper's model is reused as is; without it the model is downloaded once (issues 3, 5).
- [x] A crashed server is restarted once, then the run continues on faster-whisper (issue 6).
- [x] Installer, `.env.example`, README and CLAUDE.md describe the engine, its settings and the fallback (issue 7).
- [ ] Tested in a real meeting, on other Apple chips, and with MacWhisper running at the same time.

## Risiken
- Step 0 used synthetic speech on one M4 Max; real meetings, slower chips and two sources talking at once are not measured.
- The server reports no language probabilities, so detection can't be limited to `LANGUAGES`; a misdetection costs a second decode of that chunk.
- The Core ML encoder always takes a 30 s window, so a 2 s chunk costs as much encoder time as 30 s of audio. Slower GPUs may not keep up.
- The first start specializes the Core ML model for the chip and can take minutes (start timeout 300 s).
- `whisperkit-cli` still fetches the tokenizer (a few MB) into `~/Documents/huggingface` on first use.

## Offene Fragen
- Is `cpuAndGPU` also the fastest on M1/M2 and the base chips, or should `auto` compute units depend on the chip?
- Does a GPU-heavy meeting app (screen sharing, video) slow WhisperKit down enough to prefer the Neural Engine?

## Erste Umsetzungsidee
Decisions taken:
- **Server, not bindings:** run Homebrew's `whisperkit-cli serve` and post each VAD chunk as a WAV to `/v1/audio/transcriptions`. Our own Swift helper or coremltools stay the fallback plan; the chunked VAD pipeline stays (no SSE streaming).
- **GPU by default:** `cpuAndGPU` took about 290 ms per chunk against about 720 ms on the Neural Engine, at the same word error rate.
- **No text prompt:** WhisperKit pre-fills the prompt through the decoder, which doubled latency, and with temperature fallback off it set off repetition loops in German. faster-whisper keeps its prompt.
- **Temperature fallback off** (`--temperature-fallback-count 0`, `temperature=0`), with segments above a compression ratio of 2.4 dropped as the safety net.
- **Language:** sticky per source and re-detected every `LANG_DETECT_EVERY` chunks; a language outside `LANGUAGES` is thrown away and the chunk is sent again in the sticky one (`LANGUAGES[0]` on a source's first chunk).
- **Licensing:** only the open (MIT) `argmaxinc/whisperkit-coreml` models. MacWhisper's Parakeet and WhisperKit Pro files, its settings and its `cli.sock` are never touched, and nothing is written under its folder.
- **Vocabulary:** STT engine, STT model and Whisper as in `GLOSSARY.md`.

## Kommentare
- 2026-11-19: Step 0, the go/no-go gate, passed.

### Step 0 results

**Setup:**
- Measured 2026-11-19 on an Apple M4 Max with macOS 26.6, `whisperkit-cli` 1.1.1 (Homebrew), and the model loaded directly from MacWhisper's folder (no download).
- Audio: 35 s of English and 41 s of German, synthetic speech from the macOS voices Samantha and Anna with a known script.
- Both engines got the same real VAD chunks (15 and 18 chunks of about 2.3 s), run one after another, with the planned sticky-language logic.
- The word error rate is computed against the script. It counts number style ("400" against "vierhundert") as errors for both engines.

| STT engine / settings | Median per chunk | p95 | Word error rate EN / DE |
|---|---|---|---|
| faster-whisper `small`, CPU int8 (current macOS default) | 840 ms | 1,550 ms | 8.7% / 17.5% |
| WhisperKit, Neural Engine (server default), prompt on | 1,040 / 1,340 ms | 2,400 ms | 7.0% / 254% (repetition loops) |
| WhisperKit, Neural Engine, no prompt | 720 ms | 750 ms | 4.3% / 11.7% |
| WhisperKit, `all` compute units, no prompt | 740 ms | 770 ms | 4.3% / 11.7% |
| **WhisperKit, GPU (`cpuAndGPU`) for encoder and decoder, no prompt** | **290 ms** | **310 ms** | **4.3% / 11.7%** |
| WhisperKit, GPU, prompt on | 590 / 830 ms | 1,820 ms | 6.1% / 175% (repetition loops) |

**Conclusions:**
- **Go,** with GPU compute units and no prompt. That beats the 500 ms target with margin and roughly halves errors against today's default.
- **The prompt caused the loops.** With it on, German produced "400.000.000…" and the same sentence repeated a dozen times. Turning fallback back on would cost latency, so the prompt is dropped and the compression-ratio filter is the safety net.
- **Server start** took 2 to 23 s depending on compute units.

**Not covered:** real meeting audio, other Apple chips, MacWhisper running at the same time, and both sources talking at once.
