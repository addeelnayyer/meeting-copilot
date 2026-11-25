# Ticket: Warm up Ollama only when it is needed

## Meta
- ID:             016
- Typ:            feature
- Status:         completed
- Erstellt:       2026-11-25
- Erstellt von:   Adeel Nayyer Sheikh
- Verantwortlich: Adeel Nayyer Sheikh
- Referenzen:     copilot/__main__.py (llm.warmup) · copilot/llm.py · copilot/providers.py (keep_alive)

## Kontext
`llm.warmup()` runs at every startup, even when a cloud provider is the primary LLM, and each request keeps the model loaded for 30 minutes. On 2026-11-24 Ollama's model runner (`granite4:7b-a1b-h`) held 7–8 GB of RAM all day while Claude answered everything.

## Ziel
The local LLM is loaded only when it is the primary LLM or once the cloud has actually failed.

## Anforderungen
- Startup warm-up only when Ollama is the primary LLM.
- Switching the primary to Ollama from the UI warms it up.
- When the breaker trips or the hedge starts Ollama, the fallback is warmed if it is not already; the first fallback answer may be slower, which is accepted.
- `ollama_status` polling (`/api/tags`) stays, since it loads no model.

## Akzeptanzkriterien
- [x] With a cloud primary, no request reaches Ollama's chat API at startup.
- [x] With Ollama primary, the warm-up still runs at startup.
- [x] A cloud failure still gets an answer from Ollama.

## Risiken
- The first hedged answer after a cloud failure pays the model load time (seconds).

## Offene Fragen

## Erste Umsetzungsidee
Make `LLM.warmup()` a no-op unless `self.primary is self.fallback`, and call it from `set_primary` and from the first failover.

## Kommentare
- 2026-11-25: done in addeelnayyer/meeting-copilot#47 (PR #50). Real run: the app started idle with Claude as primary LLM; `ollama ps` showed no loaded model. A failed fallback warm-up is not cached; the fallback request still goes out.
