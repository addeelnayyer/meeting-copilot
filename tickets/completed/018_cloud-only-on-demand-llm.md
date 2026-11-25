# Ticket: Cloud-only LLM, called only on demand

## Meta
- ID:             018
- Typ:            feature
- Status:         completed
- Erstellt:       2026-11-25
- Erstellt von:   Adeel Nayyer Sheikh
- Verantwortlich: Adeel Nayyer Sheikh
- Referenzen:     docs/adr/0002-cloud-only-llm.md · GLOSSARY.md (LLM, Meeting memory, Checkpoint) · supersedes the Ollama parts of ticket 016

## Problem Statement

On 2026-11-25, Ollama's `llama-server` held 26.5 GB of RAM on a 48 GB Mac while the user was in a meeting. Claude was the LLM and answered every request; the local LLM only existed as a fallback. Ollama opens every model with a context sized from the machine's RAM (262,144 tokens here), keeps a prompt cache of up to 8 GB and stays loaded for 30 minutes, so the fallback alone can push the machine into memory pressure.

Separately, the copilot calls the LLM without being asked: proactive suggestions fire after every participant turn, and the meeting memory is folded every five minutes. The user wants the LLM to answer only when they ask, and each request to send only what is new since the last one, not the whole call.

## Solution

The copilot no longer runs a language model on the user's machine. The LLM is always the cloud model the user picks in the window. Transcription stays local and unchanged (WhisperKit on Apple Silicon, faster-whisper elsewhere).

LLM calls happen only on demand: when the user clicks a preset button, presses its hotkey or types a question. Each such request sends the meeting memory (everything before the checkpoint) plus the segments since the checkpoint. After the answer is shown, the new segments are folded into the meeting memory in the background, and the checkpoint moves forward. The Summary button is that fold. A new Questions button suggests what the user could ask now. Proactive suggestions stay available behind their toggle, but every meeting starts with them off. Stopping a meeting folds the rest once, so the meeting record covers the whole meeting.

## User Stories

1. As a user, I want the copilot to never start a local language model, so that my machine keeps its RAM for the meeting app and the STT engine.
2. As a user, I want `llama-server` and Ollama to play no part in a meeting, so that I can uninstall Ollama without losing anything.
3. As a user, I want the LLM to always be the cloud provider I picked in the window, so that I know where my transcript goes.
4. As a user, I want to switch between the configured cloud providers in the window as today, so that I can move to another one when one is slow or out of quota.
5. As a user, I want the provider list in the window to show only cloud providers, so that I am not offered Ollama models that no longer work.
6. As a user, I want a failed LLM request to be explained in the window (timeout, quota, missing key, refusal), so that I know why no card appeared.
7. As a user, I want the next request after a failure to try the LLM again, so that one hiccup does not switch the copilot off for the rest of the meeting.
8. As a user, I want a rate-limited provider to pause for the time it asks for, so that the copilot does not hammer it and burn quota.
9. As a user without any API key, I want the copilot to start and transcribe anyway, so that I still get a transcript and a meeting record.
10. As a user without any API key, I want the window to tell me that no LLM is configured, so that I know why the buttons do nothing useful.
11. As a user with an OpenAI-compatible gateway (OmniRoute), I want the `custom` provider to keep working, so that my gateway setup survives.
12. As a user with Ollama settings left in my `.env`, I want them ignored with a console hint, so that the app starts and I know I can remove them.
13. As a user, I want no LLM call to happen while I just listen, so that my quota and my data are spent only when I ask.
14. As a user, I want an Answer button that tells me what to answer now, so that I can respond to the participant.
15. As a user, I want a Questions button that suggests which questions I should ask now given what was said, so that I can steer the conversation.
16. As a user, I want a Summary button, so that I can catch up on the meeting at any point.
17. As a user, I want the Weak points button and the free-text question box to stay, so that I can still ask anything specific.
18. As a user, I want each button to keep its own hotkey, and the Questions button to get one too, so that I can ask without leaving the meeting window.
19. As a user, I want the button labels and their questions in my UI language (English or German), so that they read naturally.
20. As a user, I want each request to send the meeting memory plus only the segments since the checkpoint, so that requests stay small and fast in a long call.
21. As a user, I want each request to include at least the last few turns even right after a previous request, so that a quick follow-up still has the immediate context.
22. As a user, I want the answer shown before the meeting memory is updated, so that I am not waiting for the summary to get my answer.
23. As a user, I want the meeting memory updated after each request and the checkpoint moved to the latest segment, so that the next request only sends what is new.
24. As a user, I want the window to show the checkpoint's time (for example "Summary up to 14:32"), so that I know how current the meeting memory is.
25. As a user, I want each card to show when I asked for it, so that I can tell older answers from new ones.
26. As a user, I want the Summary button to show the updated meeting memory as its card, so that the card and the meeting memory panel never disagree and only one call is made.
27. As a user, I want a failed fold to leave the checkpoint where it was, so that nothing said since then is lost from the meeting memory.
28. As a user, I want the next request after a failed fold to send the larger set of segments and fold them, so that the meeting memory catches up.
29. As a user, I want a fold skipped when almost nothing new was said, so that a second click seconds later costs only the answer.
30. As a user, I want no meeting memory update on a timer, so that the LLM is called only when I ask or when the meeting stops.
31. As a user, I want proactive suggestions to stay available behind their button and `ctrl+alt+p`, so that I can turn them on for a meeting where I want them.
32. As a user, I want every meeting to start with proactive suggestions off, so that no automatic LLM call happens unless I turn them on.
33. As a user with proactive suggestions on, I want them to use the meeting memory plus the last 15 turns, so that they behave as they do today.
34. As a user with proactive suggestions on, I want them never to move the checkpoint, so that only my own requests and the meeting stop fold the meeting memory.
35. As a user, I want stopping a meeting to fold the segments since the checkpoint once, so that the meeting record's meeting memory covers the whole meeting.
36. As a user, I want the meeting record written even when the final fold fails or no LLM is configured, so that I always keep the transcript.
37. As a user who turned translation on, I want translations to keep arriving per segment, so that the live translation still works.
38. As a user with transliteration configured, I want transliterations to keep arriving per segment, so that the Latin-script line still appears.
39. As a user, I want facts the LLM picks up in an answer to keep being merged into my facts, so that later answers know them.
40. As a maintainer, I want the hedge, the breaker, the Ollama warm-up, the Ollama model polling and the local LLM benchmark removed, so that the LLM code has one path.
41. As a maintainer, I want README, `.env.example`, the installers and CLAUDE.md to stop mentioning Ollama, so that new users are not told to install it.
42. As a maintainer, I want the glossary and ADR 0002 to describe the cloud-only, on-demand model, so that the reasons survive.

## Implementation Decisions

- **Providers:** the Ollama provider is removed. The provider interface (`complete(system, prompt, schema) -> str`), `retry_after()` and the Gemini, Claude and OpenAI-compatible providers (`openai`, `grok`, `custom`) are unchanged.
- **LLM module:** one path. A request goes to the selected provider; there is no hedge, no breaker, no fallback, no warm-up and no Ollama polling. `CLOUD_TIMEOUT_S` and the Ollama settings are removed. On an error the LLM sets its `note` (shown in the window) and the caller gets a failure it reports as an `error` event. A 429 pauses the provider for `retry_after()`, and a request during the pause fails at once with a note saying when it ends. `describe()` no longer carries `ollama_models` or `ollama_note`, and `set_primary` no longer accepts `ollama:<model>`. With no configured provider, `LLM` exists but every request fails with the "no LLM configured" note.
- **Startup:** no warm-up call. Leftover `OLLAMA_*` or `CLOUD_TIMEOUT` keys in `.env` print one console hint and are otherwise ignored.
- **Checkpoint:** the existing meeting memory cursor (the id of the last folded segment) is the checkpoint. The meeting memory additionally records the time of the checkpoint's segment, sent in `memory` events and shown in the window.
- **Request context (on demand):** `[FACTS]` + `[MEMORY]` + `[KNOWLEDGE]` + `[TRANSCRIPT]`, where the transcript is every segment after the checkpoint, extended backwards to at least the last few turns when fewer are new. Segments already folded are not sent beyond that minimum.
- **Fold after an answer:** after a preset, hotkey or typed question has produced its card, the assistant folds the segments after the checkpoint into the meeting memory with a second LLM call, in the background. Success moves the checkpoint to the last folded segment and emits a `memory` event. Failure emits an `error` event and leaves the checkpoint unchanged. The fold is skipped when the new segments have fewer than `MEMORY_MIN_WORDS` words. At most one fold runs at a time; a request while a fold runs does not start a second one, and its segments are folded by the next one.
- **Summary preset:** it runs the fold directly (not the suggestion prompt), and its card shows the resulting meeting memory. With nothing new to fold, the card shows the current meeting memory without an LLM call.
- **Questions preset:** a new entry in the presets with an English and a German label and query, asking which questions the user should ask now given what was said, and its own hotkey. It uses the normal suggestion prompt and `SUGGESTION_SCHEMA`.
- **Timer removed:** the periodic memory loop and `MEMORY_INTERVAL_S` are removed.
- **Proactive suggestions:** kept, with their toggle and hotkey, but `Assistant` starts each meeting with proactive off. A proactive request uses the meeting memory plus the last `MAX_TURNS` turns, as today, and never folds or moves the checkpoint.
- **Meeting stop:** the stop task replaces the "last minutes" summary with one fold of the segments after the checkpoint, within the existing 30 s limit, then writes the meeting record. A failed fold or a missing LLM still writes the record with the transcript.
- **Cards:** suggestion and answer events carry the time they were asked; the window shows it on the card.
- **Translation and transliteration:** unchanged; they run per segment only when switched on.
- **UI strings:** the Ollama strings go; new keys for the Questions preset, the "no LLM configured" note, the checkpoint time and the card time, in both `en` and `de` tables.
- **Removed outright:** the local LLM benchmark, its data file and its wizard script, the Ollama steps and hints in both installers, and the Ollama sections in README, `.env.example` and CLAUDE.md.

## Testing Decisions

- Tests check behaviour seen from outside the module: what reaches the (fake) LLM, which events the window gets, and what the meeting record holds. They do not check private fields or call order inside a module.
- **Seam 1, `Assistant` with a fake LLM** (prior art: the preset-hotkey, proactive and invalid-JSON tests built on `_meeting_assistant` with a `FakeLLM` that records prompts):
  - a preset, hotkey or typed question sends the meeting memory and the segments after the checkpoint, plus the minimum recent turns, and no other folded segments;
  - after the answer, a fold runs, the checkpoint moves and a `memory` event carries the checkpoint time;
  - a failed fold leaves the checkpoint, and the next request sends the larger set;
  - a fold is skipped below `MEMORY_MIN_WORDS`;
  - the Summary preset makes exactly one LLM call and its card is the meeting memory;
  - the Questions preset exists with `en` and `de` labels and a hotkey;
  - a meeting starts with proactive off; with proactive on, a proactive card sends the last 15 turns and does not move the checkpoint;
  - stopping a meeting folds once and the record holds the full meeting memory; with a failing or missing LLM the record still holds the transcript;
  - card events carry the time they were asked.
- **Seam 2, `LLM` with `FakeProvider`s** (prior art: the `_llm` helper and the breaker/hedge tests, rewritten without Ollama):
  - a failing provider reaches no other provider, sets the note and reports a failure;
  - a 429 pauses the provider for `retry_after()`;
  - with no configured provider, requests fail with the "no LLM configured" note;
  - `describe()` lists only cloud providers.
- The tests for the hedge, the breaker, the Ollama warm-up, the Ollama label and the missing-model note are deleted together with the code. `test_strings.py` keeps checking the parity of the `en` and `de` tables.
- CI keeps running on all three OSes; the import smoke check for the local LLM benchmark is removed.

## Out of Scope

- Changing the STT engines, the STT models or anything about transcription.
- Failover between cloud providers.
- Streaming answers or prompt caching (ticket 003).
- Quota and cost control (ticket 007).
- Making translation or transliteration on-demand.
- Removing Ollama from the user's machine; that is a manual step in README.

## Further Notes

- On 2026-11-25 the largest prompts in the Ollama log were about 20.7k tokens; a full hour of transcript after the checkpoint is about 12k tokens, which every supported cloud provider accepts, so no cap is set.
- `GLOSSARY.md` already drops Local LLM and Fallback and defines Checkpoint.
- Ticket 016's Ollama warm-up work is superseded by this ticket.

## Kommentare
- 2026-11-25: done on branch cloud-only-on-demand-llm. Unit tests pass (136); the window changes were not yet checked in a running app.
