# 🎙️ Enterprise AI Voice Cloning Presentation Studio
### Zero-Shot Acoustic Synthesis & Timestamp-Synchronized Narration Framework

This repository serves as a self-contained, enterprise-grade framework for automated presentation narration, zero-shot acoustic voice cloning, and hybrid cloud/local synthesis workflows built for **GitHub Web + Streamlit Web**.

The architecture is explicitly optimized to ingest raw reference voice recordings (`.wav`) alongside `[MM:SS]` timestamped presentation scripts (`.txt`), normalize acoustic profiles to mono 24kHz tensors, and synthesize gap-aligned master narration tracks via either **No-API F5-TTS** execution or **Groq-accelerated** Whisper auto-transcription and preset voice pipelines.

### About

**Enterprise Voice Presentation Studio** built with **Streamlit**, **F5-TTS zero-shot voice cloning**, and **Groq Whisper/PlayAI hybrid acceleration**. Features **session-isolated audio normalization**, **dynamic Gradio Space endpoint failover**, **regional English accent cadence profiles (India, US, UK, AU)**, and **automated GitHub Actions CI/CD**.

---

### ⭐ Show Your Support

If you like the UI design, resilient endpoint failover architecture, or zero-shot acoustic engineering behind **Voice Clone Presentation Studio**, please consider giving this repository a **Star ⭐** on GitHub! It helps showcase this work and motivates future builds.

---

## Architecture Overview

### System Data Flow Architecture
The diagram below illustrates how user audio samples and timestamped scripts flow through the session-isolated normalization, routing, and synthesis layers before compiling into a master presentation track:

```mermaid
flowchart TD
    subgraph Client_Intake_Layer [Client Intake & Security Layer]
        A[auth.py<br>Enterprise SSO & Consent Gate] e1@-->|Authenticated Session| B[app.py<br>Streamlit Web Gateway]
        B e2@-->|Raw .wav & .txt Payloads| C[parser.py<br>Timestamp & Sentence Chunker]
    end

    subgraph Acoustic_Processing_Layer [Acoustic Normalization Layer]
        B e3@-->|Raw Reference Bytes| D[audio_processor.py<br>Mono 24kHz & Peak Normalizer]
        D e4@-->|Isolated Temp Directory| E{Execution Mode Router<br>No-API vs With-API}
    end

    subgraph Synthesis_Engine_Layer [Hybrid TTS Synthesis Subsystem]
        E e5@-->|Zero-Shot Voice Cloning| F[tts_client.py<br>F5-TTS Failover & Local Engine]
        F e6@-->|Dynamic Schema Discovery| G[Hugging Face Spaces Cluster<br>E2-F5-TTS / Local PyTorch]
        E e7@-->|With-API Acceleration| H[groq_helper.py<br>Whisper Auto-Transcribe & PlayAI]
        H e8@-->|Low-Latency Inference| I[Groq Cloud Gateway<br>whisper-large-v3-turbo / playai-tts]
    end

    subgraph Master_Compilation_Layer [Timeline Stitching & Governance Layer]
        G e9@-->|Synthesized Segment .wav| J[audio_processor.py<br>Millisecond Timeline Stitcher]
        I e10@-->|Synthesized Segment .wav| J
        J e11@-->|Master Presentation .wav| K[st.audio & Download Player]
        J e12@-->|Emit Structured Event| L[(audit_logger.py<br>JSON Compliance Trail)]
    end

    %% Running Flow Animation Definitions
    e1@{ animate: true }
    e2@{ animate: true }
    e3@{ animate: true }
    e4@{ animate: true }
    e5@{ animate: true }
    e6@{ animate: true }
    e7@{ animate: true }
    e8@{ animate: true }
    e9@{ animate: true }
    e10@{ animate: true }
    e11@{ animate: true }
    e12@{ animate: true }

    %% Clean Light-Friendly Subgraph Styling
    style Client_Intake_Layer fill:#f4f6f9,stroke:#90a4ae,stroke-width:1.5px,color:#1a1a1a
    style Acoustic_Processing_Layer fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#1a1a1a
    style Synthesis_Engine_Layer fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a1a1a
    style Master_Compilation_Layer fill:#fff8e1,stroke:#f9a825,stroke-width:1.5px,color:#1a1a1a
```

---

### Hybrid Synthesis & Failover Logic Flow
The Synthesis Subsystem routes requests deterministically based on secret availability, user execution mode, and upstream endpoint health:

```mermaid
flowchart TD
    Start[User Triggers Presentation Synthesis] --> Mode_Eval{Evaluate Execution Mode}
    
    Mode_Eval -->|Without API Mode| No_API_Branch{Select No-API Backend}
    Mode_Eval -->|With API Mode| API_Branch[Load HF_TOKEN & GROQ_API_KEY from st.secrets]
    
    No_API_Branch -->|Cloud Free Tier| Public_Space[Unauthenticated Gradio Space Failover]
    No_API_Branch -->|Local Hardware| Local_Engine[Cached Local F5-TTS PyTorch Inference]
    
    API_Branch -->|Auto-Transcribe Requested| Groq_Whisper[Groq Whisper-Large-V3-Turbo Extraction]
    API_Branch -->|Zero-Shot Cloning| Auth_Space[Authenticated HF Space + Retry Backoff]
    API_Branch -->|Fast Preset Mode| Groq_TTS[Groq PlayAI Sub-Second Synthesis]
    
    Public_Space --> Stitch[Insert Exact Silence Padding & Stitch Master WAV]
    Local_Engine --> Stitch
    Auth_Space --> Stitch
    Groq_TTS --> Stitch
    
    Stitch --> Audit[Record JSON Audit Log & Clean Up Temp Workspace]

    %% Clean Light-Friendly Node Styling
    style Start fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#1a1a1a
    style Mode_Eval fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#1a1a1a
    style No_API_Branch fill:#fff8e1,stroke:#f57c00,stroke-width:1.5px,color:#1a1a1a
    style API_Branch fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px,color:#1a1a1a
    style Stitch fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#1a1a1a
    style Audit fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#1a1a1a
```

---

## 🛠️ Enterprise Repository Structure

```text
voice_clone_presentation/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                  # Automated GitHub Actions (ruff linting + pytest on every push)
│   └── dependabot.yml              # Automated weekly dependency security updates
├── .streamlit/
│   ├── config.toml                 # Corporate UI theme, maxUploadSize (200MB), & XSRF protection
│   └── secrets.toml.example        # Template for optional HF_TOKEN, GROQ_API_KEY, & SSO settings
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── auth.py                     # Enterprise SSO/OAuth gate & speaker-consent verification
│   ├── parser.py                   # [MM:SS] timestamp parser & natural sentence chunker
│   ├── audio_processor.py          # Mono 24kHz audio normalization & millisecond timeline stitching
│   ├── tts_client.py               # Dynamic F5-TTS client (Local No-API + Remote Space failover)
│   ├── groq_helper.py              # Optional Groq Whisper auto-transcription & PlayAI preset TTS
│   └── audit_logger.py             # Structured JSON compliance & observability logging
├── tests/
│   ├── __init__.py                 # Test package initializer
│   ├── test_parser.py              # Unit tests for timestamp parsing & sentence splitting
│   └── test_audio_processor.py     # Unit tests for audio normalization & silence padding math
├── app.py                          # Main Streamlit Web interface
├── packages.txt                    # System-level Linux dependency for Streamlit Cloud (ffmpeg)
└── requirements.txt                # Pinned Python dependencies
```

---

## 📝 Document & Audio Formatting Guide

### 1. Reference Audio Sample (`.wav`)
* **Duration:** Record or upload a clean **5 to 12 second** voice clip in a quiet environment.
* **Language & Accent Profile:** Select your matching regional cadence in the sidebar (e.g., **🇮🇳 India (English) — Clear Cadence**, **🇺🇸 US**, **🇬🇧 UK**, or **🇦🇺 AU**).
* **Reference Transcript:** Ensure the text box contains the **exact word-for-word transcription** spoken inside your reference `.wav` file (or click **Auto-Transcribe** when using Groq API mode).

### 2. Presentation Script (`.txt`)
Format your presentation script using sequential `[MM:SS]` time markers. Long paragraphs are automatically split into natural sentence chunks while anchoring the start of each block to your exact timestamp:

```text
Reference Text: Hello everyone, welcome to today's quarterly product review.
[00:00] Welcome to our autonomous presentation overview. The engine verifies location metrics before launching.
[00:30] Notice that this slide profile is metered. The system will auto-terminate sessions at ten minutes.
[01:15] Our executive dashboard evaluates high-probability risk exposures and updates instantly before your eyes.
```
## ⚠️ Disclaimer & Ethical AI Governance

This project is an **AI-powered zero-shot voice cloning and presentation narration platform** intended strictly for educational, research, portfolio demonstration, and authorized content production purposes. Users must only synthesize voices that they own or for which they hold explicit, verifiable consent. Any unauthorized impersonation, deceptive audio generation, or misuse of synthetic voice technology is strictly prohibited.

Users are responsible for securing their own optional `HF_TOKEN` and `GROQ_API_KEY` credentials locally or within Streamlit Cloud Secrets (`.streamlit/secrets.toml`), adhering to Hugging Face and Groq API terms of service and rate limits, and complying with applicable biometric and voice-likeness regulations.

---

## 🛡️ License, Copyright & Compliance Notices

**All Rights Reserved © 2026 T A Srinivas.**

> **Strict Intellectual Property Notice:** This repository is strictly for portfolio viewing and technical evaluation purposes. **DO NOT COPY, CLONE, FORK, OR REDISTRIBUTE** this source code, architecture, or documentation. Stolen copies or unauthorized forks will be reported immediately for a **GitHub DMCA copyright takedown**.

For corporate licensing, enterprise deployment, or commercial inquiries, please contact **`tasrinivass@gmail.com`**.

* **Lead Architect & Developer:** [Srinivasta](https://github.com/SRINIVASTA)

### 🌐 Let’s Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srinivas-t-a-557637119/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/srinivasta)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:tasrinivass@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/srinivasta)
[![Website](https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=website&logoColor=white)](https://srinivasta.github.io)

💡 Engineered with automated acoustic normalization and millisecond timeline stitching logic to transform raw reference recordings and timestamped scripts into studio-grade presentation audio.

