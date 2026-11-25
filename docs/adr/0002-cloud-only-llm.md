# Cloud-only LLM, called only on demand

On 2026-11-25 Ollama's `llama-server` held 26.5 GB of RAM on a 48 GB Mac. Ollama 0.40 opens every model with a 262,144-token context sized from the machine's RAM, adds a host prompt cache of up to 8 GB, and keeps the model loaded for 30 minutes. That happened while Claude, the LLM in use, answered every request. We removed the local LLM completely, so the copilot runs no language model on the user's machine; transcription (WhisperKit or faster-whisper) stays local. At the same time the LLM is no longer called on a timer: the meeting memory is folded only when the user asks something and once when the meeting stops, and each request sends the meeting memory plus the segments since the checkpoint instead of the whole meeting.

## Considered Options

- **Cap the local LLM's context (`num_ctx`):** Granite would drop to about 5–6 GB, which is still several GB held for a fallback that is rarely used, and a user who picks a larger model is back to 20+ GB.
- **Make the fallback opt-in:** it keeps the hedge, the breaker, the warm-up and the model list in the code for a feature nobody turns on.
- **Fail over between cloud providers:** it needs a second API key and brings back a smaller version of the hedge.

## Consequences

- Without network access or an API key, the copilot only transcribes. The meeting record then holds the transcript without a meeting memory.
- A failed or rate-limited LLM call is dropped and explained in the window. Nothing answers in its place.
- The `custom` OpenAI-compatible provider still accepts any endpoint, including a local one, but the copilot neither ships nor documents a local setup.
