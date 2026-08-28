# 🎤 Perfect-Timed AI Presentation Generator

A lightweight, zero-shot voice cloning presentation tool built using **Streamlit Cloud** and powered by the remote **F5-TTS Cloud API**. This application automatically parses custom script timestamps to compile a single, perfectly synced presentation audio track without manual editing.

---

## 🛠️ Required File Structure

To run this application successfully on the web without crashing, ensure these exact files exist in your repository root:

*   **`app.py`** — The frontend interface handles layout rendering, asset temporary storage, and session versions.
*   **`utils.py`** — The text-segment chunking engine manages API requests and aggregates timeline tracks.
*   **`requirements.txt`** — Holds Python tier dependencies (`streamlit`, `gradio_client`, `audioop-lts`, `pydub`).
*   **`packages.txt`** — Essential system-level Linux audio rendering layer package (`ffmpeg`).
*   **`README.md`** — Project guide and formatting map documentation.

---

## 📝 Document Asset Formatting

### 1. Presentation Timestamps (`.txt`)
Your script timeline markers are managed via a strict `[MM:SS]` timestamp notation framework. Ensure your uploaded script flows sequentially:

```text
[00:00] Welcome to our autonomous presentation overview. The engine verifies location metrics before launching.
[00:30] Notice that this slide profile is metered. The system will auto-terminate sessions at exactly ten minutes.
[01:15] Our analytics dashboard evaluates high-probability default exposures and risk updates instantly before your eyes.
```

### 2. Reference Audio Inputs (`.wav`)
*   **Audio Sample:** Upload a clean, noise-free **5 to 10 second** voice recording.
*   **Reference Text Input:** Type out the **exact verbatim transcription** of what is spoken *inside that 5-second sample file* so the model can correctly map your vocal signatures (accent, pitch, tone).

