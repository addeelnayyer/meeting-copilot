# Meeting Copilot

A real-time meeting assistant that listens to the user's mic and the meeting's system audio, transcribes both, and shows suggestions and answers while the meeting runs.

## Transcription

**STT engine**:
The software that turns an audio chunk into a transcript segment, for example faster-whisper or WhisperKit.
_Avoid_: Whisper (for the software), backend, transcriber

**STT model**:
The trained weights an STT engine loads, for example large-v3-turbo in CTranslate2 or CoreML form.
_Avoid_: Whisper (for the weights), engine

**Whisper**:
OpenAI's family of speech models. It names a model family only, never an STT engine or a specific set of files.

**Transliteration**:
Rewriting a segment in another script without changing its language.
_Avoid_: translation, romanization

**Roman Urdu**:
Urdu written in Latin letters.
_Avoid_: Urdu (for this form)

## Assistant

**LLM**:
The language model that writes suggestions, answers, translations and the meeting memory.
_Avoid_: model (unqualified), AI

**Local LLM**:
An LLM that runs on the user's machine rather than with a cloud provider.
_Avoid_: Ollama (for the model), offline model

**Meeting memory**:
The rolling summary of the meeting so far, written by the LLM and kept in the meeting record.
_Avoid_: memory (unqualified), summary (for the running one); say RAM for the machine's working memory

**Fallback**:
The local LLM's role when the primary LLM is slow, fails or is out of quota.
_Avoid_: backup, secondary

## People

**User**:
The person running the copilot, heard through the mic.
_Avoid_: me, host, operator

**Participant**:
Anyone heard through the meeting's system audio rather than the mic.
_Avoid_: attendee, remote, other side

**Speaker**:
One distinct voice detected within a single meeting, either unnamed ("Speaker A") or matched to a known speaker.
_Avoid_: speaker (for the loudspeaker hardware; say loudspeaker), voice, cluster

**Known speaker**:
A named person whose voice the copilot remembers across meetings.
_Avoid_: contact, profile, enrolled speaker

**Voiceprint**:
The stored fingerprint of a known speaker's voice, kept only on the user's machine.
_Avoid_: voice embedding, voice profile, sample
