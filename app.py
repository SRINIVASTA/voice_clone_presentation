import gc
import os
import tempfile
import streamlit as st

from src.auth import enforce_enterprise_auth
from src.parser import parse_presentation_script
from src.audio_processor import (
    normalize_reference_audio,
    stitch_timeline_to_disk,
    compute_voice_similarity,
)
from src.tts_client import F5TTSClient
from src.groq_helper import is_groq_available, transcribe_reference_wav
from src.audit_logger import log_generation_event

st.set_page_config(page_title="Own-Voice AI Presentation Studio", page_icon="🎙️", layout="wide")

active_user = enforce_enterprise_auth()

# Securely read secrets on the backend without exposing keys in the browser
secret_hf_token = str(st.secrets.get("HF_TOKEN", "")).strip()
secret_groq_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
secret_custom_space = str(st.secrets.get("CUSTOM_HF_SPACE", "")).strip()

ACCENT_PROFILES = {
    "🇮🇳 India (English) — Natural Cadence": 0.90,
    "🇺🇸 United States (English) — Standard": 0.95,
    "🇬🇧 United Kingdom (English) — Crisp": 0.95,
    "🇦🇺 Australia (English) — Natural": 0.95,
    "🌐 Slower / Clear Presentation Pace": 0.85,
}

st.title("🎙️ Own-Voice AI Cloning Presentation Studio")
st.caption("Zero-shot acoustic voice cloning — trains on your own voice sample to narrate your entire presentation.")

with st.sidebar:
    st.header("⚙️ Voice Cloning Engine")
    cloning_mode = st.radio(
        "Compute Backend:",
        [
            "With API (Your HF Space + Auto-Transcribe)",
            "Without API (Free Public HF Spaces)",
        ],
    )

    use_api_secrets = cloning_mode.startswith("With API")
    hf_token = secret_hf_token if use_api_secrets else ""
    groq_api_key = secret_groq_key if use_api_secrets else ""
    custom_hf_space = secret_custom_space

    if use_api_secrets:
        st.divider()
        st.subheader("🔒 Vault Status")
        if secret_hf_token:
            st.success("HF_TOKEN: Active")
        else:
            hf_token = st.text_input("Enter HF_TOKEN (Optional)", type="password")

        if secret_groq_key:
            st.success("GROQ_API_KEY: Active (Auto-Transcribe)")

        custom_hf_space = st.text_input(
            "Your F5-TTS Space ID",
            value=secret_custom_space,
            placeholder="SRINIVASTA/f5-tts-cloner",
            help="Enter your own Hugging Face Space ID or leave blank to auto-scan live F5-TTS Spaces.",
        )

    st.divider()
    st.subheader("🌍 Speaking Cadence")
    selected_accent = st.selectbox(
        "Select Pacing Profile",
        list(ACCENT_PROFILES.keys()),
        index=0,
    )
    speech_speed = ACCENT_PROFILES[selected_accent]

groq_enabled = is_groq_available(groq_api_key)

col_left, col_right = st.columns([1.05, 0.95], gap="large")

with col_left:
    st.subheader("1. Your Reference Voice Sample (.wav)")
    input_method = st.radio("Capture Your Voice", ["Upload .wav File", "Record Microphone"], horizontal=True)

    if input_method == "Upload .wav File":
        uploaded_wav = st.file_uploader("Upload clean 5–12 second recording of your voice", type=["wav"])
        if uploaded_wav is not None:
            wav_data = uploaded_wav.getvalue()
            if wav_data and st.session_state.get("voice_bytes") != wav_data:
                st.session_state["voice_bytes"] = wav_data
                st.session_state["auto_ref_text"] = ""
    else:
        mic_audio = st.audio_input("Record 5–12 seconds of your voice clearly")
        if mic_audio is not None:
            mic_data = mic_audio.getvalue()
            if mic_data and st.session_state.get("voice_bytes") != mic_data:
                st.session_state["voice_bytes"] = mic_data
                st.session_state["auto_ref_text"] = ""

    voice_bytes = st.session_state.get("voice_bytes", None)
    if voice_bytes:
        st.audio(voice_bytes, format="audio/wav")

        if groq_enabled and not st.session_state.get("auto_ref_text"):
            with st.spinner("✨ Auto-transcribing your voice sample..."):
                try:
                    extracted = transcribe_reference_wav(voice_bytes, groq_api_key)
                    if extracted:
                        st.session_state["auto_ref_text"] = extracted
                        st.success("Reference speech transcribed automatically!")
                except Exception as exc:
                    st.warning(f"Auto-transcription skipped: {exc}")

    st.subheader("2. Presentation Script (.txt)")
    uploaded_script = st.file_uploader("Upload [MM:SS] timestamped script", type=["txt"])

    parsed_ref = st.session_state.get("auto_ref_text", "")
    timeline_items = []

    if uploaded_script is not None:
        raw_script_text = uploaded_script.getvalue().decode("utf-8")
        try:
            script_ref, timeline_items = parse_presentation_script(raw_script_text)
            if not parsed_ref:
                parsed_ref = script_ref
            st.success(f"Validated {len(timeline_items)} sequential speech chunks.")
        except ValueError as err:
            st.error(f"Script Format Error: {err}")

    ref_text = st.text_input(
        "Reference Transcript (Must match the exact words spoken in your voice clip)",
        value=parsed_ref,
    )

    consent_ok = st.checkbox("I confirm this is my own voice or I hold explicit consent to clone it.")
    run_btn = st.button("🚀 Clone My Voice & Generate Presentation", type="primary", use_container_width=True)
with col_right:
    st.subheader("3. Cloned Voice Output (Chunk-by-Chunk & Master)")
    st.info(f"**Target Engine:** Zero-Shot F5-TTS Voice Clone | **Cadence:** {selected_accent}")

    if run_btn:
        active_voice_bytes = st.session_state.get("voice_bytes", None)
        if not consent_ok:
            st.error("Please confirm voice authorization before cloning.")
        elif not timeline_items:
            st.warning("Please upload a valid .txt script with [MM:SS] markers.")
        elif not active_voice_bytes or not ref_text.strip():
            st.warning("Please provide both your reference voice clip (.wav) and its exact reference text.")
        else:
            progress = st.progress(0, text="Preparing isolated voice cloning workspace...")
            chunk_feed = st.container()
            tts_engine = None
            similarity_score = None

            try:
                with tempfile.TemporaryDirectory() as session_dir:
                    norm_ref_path = os.path.join(session_dir, "norm_ref.wav")
                    normalize_reference_audio(active_voice_bytes, norm_ref_path, max_duration_ms=12000)

                    tts_engine = F5TTSClient(hf_token=hf_token, custom_space=custom_hf_space)

                    stitched_inputs = []
                    total = len(timeline_items)

                    for idx, (ts_ms, text_line) in enumerate(timeline_items):
                        progress.progress(idx / total, text=f"Cloning your voice for chunk {idx + 1} of {total}...")
                        seg_path = os.path.join(session_dir, f"seg_{idx}.wav")

                        tts_engine.clone_segment(
                            norm_ref_path,
                            ref_text.strip(),
                            text_line,
                            seg_path,
                            speed=speech_speed,
                        )

                        with chunk_feed.expander(f"✅ Cloned Chunk {idx + 1}/{total}: {text_line[:55]}...", expanded=False):
                            with open(seg_path, "rb") as f_seg:
                                st.audio(f_seg.read(), format="audio/wav")

                        stitched_inputs.append((ts_ms, seg_path))
                        gc.collect()

                    progress.progress(0.92, text="Combining all cloned chunks into master presentation...")
                    master_wav_path = os.path.join(session_dir, "master_presentation.wav")
                    stitch_timeline_to_disk(stitched_inputs, master_wav_path)

                    progress.progress(0.97, text="Verifying acoustic similarity against your original voice...")
                    similarity_score = compute_voice_similarity(norm_ref_path, master_wav_path)

                    with open(master_wav_path, "rb") as f_master:
                        final_wav_bytes = f_master.read()

                    progress.progress(1.0, text="Voice Cloning Complete!")
                    gc.collect()

                active_space_used = getattr(tts_engine, "_active_space", "Hugging Face F5-TTS")
                log_generation_event(
                    active_user,
                    f"F5-TTS Clone ({active_space_used}) | {selected_accent}",
                    len(timeline_items),
                    "SUCCESS",
                )

                st.success(f"All {total} chunks cloned in your own voice and combined!")

                if similarity_score is not None:
                    m1, m2 = st.columns(2)
                    m1.metric(
                        label="🎯 Acoustic Voice Match",
                        value=f"{similarity_score}%",
                        delta="Cloned From Your Sample",
                    )
                    m2.metric(
                        label="🔊 Cloned Via Space",
                        value=str(active_space_used),
                    )

                st.audio(final_wav_bytes, format="audio/wav")
                st.download_button(
                    label="⬇️ Download My Cloned Presentation (.wav)",
                    data=final_wav_bytes,
                    file_name="my_cloned_presentation_master.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )
            except Exception as exc:
                gc.collect()
                log_generation_event(active_user, "F5-TTS Clone", len(timeline_items), "FAILED", str(exc))
                st.error(f"Voice cloning failed: {exc}")
