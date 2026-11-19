# Changelog

## Unreleased

**The window is now English by default. To keep the German window, add `UI_LANGUAGE=de` to your `.env`.**

- New setting `UI_LANGUAGE` (`en` or `de`, default `en`): the language of the window's buttons, tooltips and hints, the status and error notes, the preset buttons and their questions, and the meeting record's headings. The hotkeys and the window title stay the same in every language. An unknown value falls back to English with a hint in the console.
- Suggestions follow `REPLY_LANGUAGE` as before, and the meeting memory and the transcript are unchanged by `UI_LANGUAGE`.
