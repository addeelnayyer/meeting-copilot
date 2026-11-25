"""The LLM: one cloud provider, selectable live (Gemini, Claude, OpenAI, Grok, any
OpenAI-compatible URL). A request goes to the selected provider only. A failure sets `note`
(shown in the window) and is raised to the caller; a 429 pauses the provider for its
`retry_after()`, and a request during the pause fails at once.
"""

import json
import re
import time

from . import config
from .strings import UiError, t, ui_text
from .providers import build_providers


def _closed(schema):
    """Add additionalProperties: false to every object (required by Claude and OpenAI strict mode)."""
    if isinstance(schema, dict):
        schema = {k: _closed(v) for k, v in schema.items()}
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
    return schema


SUGGESTION_SCHEMA = _closed({
    "type": "object",
    "properties": {
        "trigger": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "text": {"type": "string"},
                    "tone": {"type": "string"},
                },
                "required": ["intent", "text", "tone"],
            },
        },
        "new_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
            },
        },
    },
    "required": ["trigger", "suggestions", "new_facts"],
})


def parse_json(text):
    """Strict parse first, then the outermost {...} block (models sometimes wrap JSON)."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class LLM:
    def __init__(self, providers=None):
        self.providers = {p.id: p for p in (providers or build_providers())}
        self.primary = None  # the selected provider; None while none is configured
        self.note = ""  # why the last request failed or the next one will, shown in the UI
        self._paused_until = 0.0  # after a 429
        self.set_primary(config.LLM_PROVIDER)

    def set_primary(self, provider_id):
        """Switch the provider at runtime. An unknown or unconfigured id keeps the current one
        and leaves a note."""
        p = self.providers.get(provider_id)
        if p is None or not p.configured:
            print(f"LLM provider '{provider_id}' is not configured (missing key/URL/model).")
            self.note = t("llm.no_key", provider=provider_id)
            return
        self.primary, self.note, self._paused_until = p, "", 0.0

    def describe(self):
        """Provider list for the UI selector."""
        return {"type": "providers", "active": self.primary.id if self.primary else "",
                "list": [{"id": p.id, "label": p.label, "model": p.model, "configured": p.configured}
                         for p in self.providers.values()]}

    async def generate(self, system, prompt, schema=None):
        """Returns (text, provider_id, seconds). schema set -> JSON mode. Raises on failure."""
        provider = self.primary
        if provider is None:
            self.note = t("llm.none")
            raise UiError("llm.none")
        wait = self._paused_until - time.monotonic()
        if wait > 0:
            self.note = t("llm.quota", provider=provider.label, minutes=_minutes(wait))
            raise UiError("llm.quota", provider=provider.label, minutes=_minutes(wait))
        start = time.monotonic()
        try:
            text = await provider.complete(system, prompt, schema)
        except Exception as e:
            if provider is self.primary:  # not switched while the request ran
                self._failed(provider, e)
            raise
        if provider is self.primary:
            self.note = ""
        return text, provider.id, time.monotonic() - start

    def _failed(self, provider, error):
        print(f"{provider.label} failed ({type(error).__name__}: {str(error)[:300]})")
        wait = provider.retry_after(error)
        if wait is not None:
            self._paused_until = time.monotonic() + wait
            self.note = t("llm.quota", provider=provider.label, minutes=_minutes(wait))
        else:
            self.note = t("llm.failed", provider=provider.label, error=ui_text(error)[:200])

    async def generate_json(self, system, prompt, schema=SUGGESTION_SCHEMA):
        text, engine, secs = await self.generate(system, prompt, schema)
        return parse_json(text), engine, secs


def _minutes(seconds):
    return max(1, round(seconds) // 60)
