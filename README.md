# AI Clip Timeline Generator

Takes a YouTube URL, reads its transcript, uses an LLM to suggest interesting
moments, and returns a timeline of timestamped clip suggestions.
It does NOT cut, render or download video.

## Status
- Phase 1: Flask foundation (done)
- Phase 2: URL validation and transcript retrieval (done)
- Phase 3: LLM draft analysis (done)
- Phase 4: validation, scoring, ranking (next)

## Setup (Windows PowerShell)
    cd backend
    python -m venv venv
    .\venv\Scripts\python.exe -m pip install -r requirements.txt
    Copy-Item .env.example .env      # then put your own API key in .env
    .\venv\Scripts\python.exe -m pytest
    .\venv\Scripts\python.exe -m flask --app app run --debug

## Limitations
- Needs a video with accessible captions.
- Transcript source uses an unofficial YouTube endpoint and may break or be blocked.
- LLM output is a suggestion only; timestamps are validated against the transcript.
- Only use videos you are allowed to process.