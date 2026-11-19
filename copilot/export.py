"""Meeting record: one readable Markdown file per meeting in MEETINGS_DIR (default `meetings/`;
point it at an Obsidian vault folder to collect them there)."""

import time

from . import config
from .audio import USER
from .strings import t


def _turns(segments):
    """Consecutive segments of one speaker become one paragraph."""
    turns = []
    for s in segments:
        if turns and turns[-1][0] == s.source:
            turns[-1][1].append(s.text)
        else:
            turns.append((s.source, [s.text]))
    return [(source, " ".join(texts)) for source, texts in turns]


def _card(c):
    head = f"{t('card.question')} {c['query']}" if c["type"] == "answer" else c.get("trigger", "")
    lines = [f"- **{time.strftime('%H:%M', time.localtime(c['ts']))} {head}**"]
    lines += [f"  - {s.get('text', '')}" for s in c.get("suggestions") or []]
    return "\n".join(lines)


def render(segments, memory, facts, cards, started, ended):
    """The record; the headings and labels follow UI_LANGUAGE, the LLM-written memory stays as written."""
    start, end = time.localtime(started), time.localtime(ended)
    return "\n".join([
        "---",
        "type: meeting",
        f"date: {time.strftime('%Y-%m-%d', start)}",
        f'start: "{time.strftime("%H:%M", start)}"',
        f'end: "{time.strftime("%H:%M", end)}"',
        f"duration_min: {round((ended - started) / 60)}",
        "---",
        f"# Meeting {time.strftime('%Y-%m-%d %H:%M', start)}",
        "",
        f"## {t('record.summary')}",
        memory.text or t("record.no_summary"),
        "",
        f"## {t('facts.title')}",
        facts.render(empty=t("none_yet")),
        "",
        f"## {t('record.cards')}",
        "\n".join(_card(c) for c in cards) or t("record.none"),
        "",
        f"## {t('record.transcript')}",
        "\n\n".join(f"**{t('who.user' if source == USER else 'who.participant')}:** {text}"
                    for source, text in _turns(segments)),
        "",
    ])


def write(transcript, memory, facts, cards, started, ended=None):
    """Returns the file path, or None when nothing was said."""
    if not transcript.segments:
        return None
    ended = ended or time.time()
    config.MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MEETINGS_DIR / time.strftime("Meeting %Y-%m-%d %H-%M.md", time.localtime(started))
    path.write_text(render(transcript.segments, memory, facts, cards, started, ended), encoding="utf-8")
    return path
