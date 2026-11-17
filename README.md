# 🎙️ Enterprise AI Voice Cloning Presentation Studio
# 🎙️ Own-Voice AI Cloning Presentation Studio
### Zero-Shot Acoustic Synthesis, Speaker Verification & Timestamp-Synchronized Narration Framework

This repository serves as a self-contained, enterprise-grade framework for automated presentation narration, zero-shot acoustic voice cloning (`F5-TTS`), and hybrid **Google Colab GPU / Streamlit Web** execution workflows.

The architecture ingests raw reference voice recordings (`.wav` or live microphone input) alongside `[MM:SS]` timestamped presentation scripts (`.txt`), auto-transcribes reference speech via **Groq Whisper**, normalizes acoustic profiles to mono 24kHz PCM, synthesizes speech **chunk-by-chunk with disk-streamed stitching** to prevent memory exhaustion, verifies speaker fidelity via an **Acoustic Voice Match (%)** meter, and exports verified master `.wav` files for **Google Drive** archiving.

### About

**Own-Voice AI Cloning Presentation Studio** built with **Streamlit**, **F5-TTS zero-shot voice cloning**, **Groq Whisper auto-transcription**, and **pure-Python acoustic similarity verification**. Features **chunk-by-chunk disk streaming (< 150 MB RAM footprint)**, **local Google Colab T4 GPU acceleration**, **automatic Hugging Face Space discovery with zero-key regional fallback**, and **automated GitHub Actions CI/CD**.

### ⭐ Show Your Support
If you explore, evaluate, or find inspiration in **Own-Voice AI Cloning Presentation Studio**, please consider giving this repository a **Star ⭐** on GitHub—it helps others discover the project and supports continued development!

---
<img src="assets/desktop.png" width="100%">

---
---

### ⭐ Show Your Support

If you like the UI design, memory-safe chunk-by-chunk audio pipeline, or zero-shot acoustic engineering behind **Own-Voice AI Cloning Presentation Studio**, please consider giving this repository a **Star ⭐** on GitHub! It helps showcase this work and motivates future builds.

---

## Architecture Overview

### System Data Flow Architecture
The diagram below illustrates how reference recordings and timestamped scripts flow through the intake, normalization, synthesis, and verification layers before exporting your master presentation track:

```mermaid
flowchart TD
    subgraph Client_Intake_Layer [Client Intake & Auto-Transcription Layer]
        A[auth.py<br>Enterprise SSO & Consent Gate] e1@-->|Authenticated Session| B[app.py<br>Streamlit Web Gateway]
        B e2@-->|Raw .wav / Mic Clip| C[groq_helper.py<br>Whisper-Large-V3-Turbo Auto-Transcribe]
        B e3@-->|Timestamped .txt Script| D[parser.py<br>MM:SS & Sentence Chunker]
    end

    subgraph Acoustic_Processing_Layer [Acoustic Normalization & Routing Layer]
        B e4@-->|Raw Reference Bytes| E[audio_processor.py<br>Mono 24kHz & Peak Normalizer]
        E e5@-->|Isolated Session Tempdir| F{Compute Backend Router<br>Colab GPU vs Cloud Spaces}
    end

    subgraph Synthesis_Engine_Layer [Chunk-by-Chunk Synthesis Subsystem]
        F e6@-->|Local T4 GPU / CPU| G[tts_client.py<br>Local F5-TTS Inference]
        F e7@-->|Remote Cloud Mode| H[tts_client.py<br>HF Space Discovery & Zero-Key Fallback]
    end

    subgraph Master_Compilation_Layer [Disk Streaming, Verification & Export]
        G e8@-->|Sequential Chunk .wav| I[audio_processor.py<br>Disk-Streamed WAV Stitcher]
        H e9@-->|Sequential Chunk .wav| I
        I e10@-->|Master WAV| J[compute_voice_similarity<br>Acoustic Match % Verifier]
        J e11@-->|Verified Audio + Score| K[Download Master WAV & Save to Google Drive]
        J e12@-->|Emit Compliance Event| L[(audit_logger.py<br>JSON Audit Trail)]
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

### Hybrid Compute & Failover Logic Flow
The Synthesis Subsystem automatically adapts to available hardware—running locally on a **Google Colab T4 GPU** when `f5-tts` is installed, or routing through **Hugging Face Spaces** with regional neural fallback on lightweight cloud containers:

```mermaid
flowchart TD
    Start[User Triggers Voice Cloning] --> Env_Check{Local f5-tts Installed?}
    
    Env_Check -->|Yes: Google Colab / GPU| Local_F5[Run Local F5-TTS GPU Inference]
    Env_Check -->|No: Streamlit Cloud 1GB RAM| Mode_Check{Check Execution Mode}
    
    Mode_Check -->|With API Vault Keys| Auth_HF[Authenticated HF Space Discovery]
    Mode_Check -->|Without API Zero-Key| Public_HF[Public HF Space Scan + Edge Regional Fallback]
    
    Local_F5 --> Disk_Stitch[Write Chunk to Disk & Free RAM via gc.collect]
    Auth_HF --> Disk_Stitch
    Public_HF --> Disk_Stitch
    
    Disk_Stitch --> Verify[Compute Acoustic Voice Match % vs Reference WAV]
    Verify --> Export[Preview Audio & Download Master WAV for Google Drive]

    %% Clean Light-Friendly Node Styling
    style Start fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#1a1a1a
    style Env_Check fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#1a1a1a
    style Local_F5 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1a1a1a
    style Mode_Check fill:#fff8e1,stroke:#f57c00,stroke-width:1.5px,color:#1a1a1a
    style Verify fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#1a1a1a
    style Export fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#1a1a1a
```

---


---

## 🛠️ Repository Structure

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
│   ├── auth.py                     # Enterprise SSO/OAuth gate & safe secrets lookup
│   ├── parser.py                   # [MM:SS] timestamp parser & natural sentence chunker
│   ├── audio_processor.py          # Mono 24kHz normalizer, disk WAV stitcher & voice match scorer
│   ├── tts_client.py               # Hybrid F5-TTS engine (Local Colab GPU + HF Spaces + fallback)
│   ├── groq_helper.py              # Groq Whisper-Large-V3-Turbo auto-transcription helper
│   └── audit_logger.py             # Structured JSON compliance & observability logging
├── tests/
│   ├── __init__.py                 # Test package initializer
│   ├── test_parser.py              # Unit tests for timestamp parsing & sentence splitting
│   └── test_audio_processor.py     # Unit tests for audio normalization & silence padding math
├── app.py                          # Main Streamlit Web interface
├── packages.txt                    # System-level Linux dependency for Streamlit Cloud (ffmpeg)
└── requirements.txt                # Cloud-safe Python dependencies (< 150 MB RAM footprint)
```

---

## ⚠️ Disclaimer & Ethical AI Governance

This project is an **AI-powered zero-shot voice cloning and presentation narration platform** intended strictly for educational, research, portfolio demonstration, and authorized content production purposes. Users must only synthesize voices that they own or for which they hold explicit, verifiable consent. Any unauthorized impersonation, deceptive audio generation, or misuse of synthetic voice technology is strictly prohibited.



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

💡 Engineered with chunk-by-chunk acoustic normalization, disk-streamed WAV stitching, and speaker similarity verification to transform raw voice recordings and timestamped scripts into studio-grade presentations.
