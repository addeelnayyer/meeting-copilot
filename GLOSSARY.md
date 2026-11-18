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
