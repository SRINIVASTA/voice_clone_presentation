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
from src.groq_helper import is_groq_available, transcribe_reference_wav, synthesize_preset_groq
from src.audit_logger import log_generation_event

st.set_page_config(page_title="Enterprise Voice Presentation Studio", page_icon="🎙️", layout="wide")

active_user = enforce_enterprise_auth()
secret_hf_token = st.secrets.get("HF_TOKEN", "")
secret_groq_key = st.secrets.get("GROQ_API_KEY", "")

ACCENT_PROFILES = {
    "🇮🇳 India (English) — Clear Cadence": 0.90,
    "🇺🇸 United States (English) — Standard": 0.95,
    "🇬🇧 United Kingdom (English) — Crisp": 0.95,
    "🇦🇺 Australia (English) — Natural": 0.95,
    "🌐 Neutral / Slower Presentation Pace": 0.85,
}

st.title("🎙️ Enterprise AI Voice Cloning Presentation Studio")
st.caption("Memory-optimized chunk-by-chunk voice cloning and timeline stitching engine.")

with st.sidebar:
    st.header("⚙️ Execution Mode")
    operation_mode = st.radio(
        "Choose Architecture Mode:",
        [
            "Without API (Free Public Cloud + Regional Fallback)",
            "With API (Enterprise Keys & Groq)",
        ],
    )

    use_api_mode = operation_mode.startswith("With API")
    hf_token = None
    groq_api_key = None

    if use_api_mode:
        st.subheader("🔑 API Credentials")
        hf_token = st.text_input(
            "Hugging Face Token (Optional)",
            value=secret_hf_token,
            type="password",
            help="Authenticates private F5-TTS Spaces and avoids public rate limits.",
        )
        groq_api_key = st.text_input(
            "Groq API Key (Optional)",
            value=secret_groq_key,
            type="password",
            help="Unlocks Whisper auto-transcription and instant preset voices.",
        )

    st.divider()
    st.subheader("🌍 Language & Accent Profile")
    selected_accent = st.selectbox(
        "Select Output Dialect & Cadence",
        list(ACCENT_PROFILES.keys()),
        index=0,
    )
    speech_speed = ACCENT_PROFILES[selected_accent]

    groq_enabled = use_api_mode and is_groq_available(groq_api_key)

    engine_choice = "Zero-Shot Voice Cloning (F5-TTS)"
    preset_voice = "Fritz-PlayAI"
    if groq_enabled:
        st.divider()
        engine_choice = st.radio(
            "Synthesis Engine",
            ["Zero-Shot Voice Cloning (F5-TTS)", "Ultra-Fast Preset Voice (Groq)"],
        )
        if engine_choice == "Ultra-Fast Preset Voice (Groq)":
            preset_voice = st.selectbox(
                "Select Groq Preset Voice",
                ["Fritz-PlayAI", "Arista-PlayAI", "Atlas-PlayAI", "Celeste-PlayAI", "Mason-PlayAI"],
            )

col_left, col_right = st.columns([1.05, 0.95], gap="large")

with col_left:
    if engine_choice == "Zero-Shot Voice Cloning (F5-TTS)":
        st.subheader("1. Reference Voice Sample (.wav)")
        input_method = st.radio("Audio Source", ["Upload .wav File", "Record Microphone"], horizontal=True)

        if input_method == "Upload .wav File":
            uploaded_wav = st.file_uploader("Upload clean 5–12 second speaker clip", type=["wav"])
            if uploaded_wav is not None:
                wav_data = uploaded_wav.getvalue()
                if wav_data:
                    st.session_state["voice_bytes"] = wav_data
        else:
            mic_audio = st.audio_input("Record 5–12 seconds of clear speech")
            if mic_audio is not None:
                mic_data = mic_audio.getvalue()
                if mic_data:
                    st.session_state["voice_bytes"] = mic_data

        voice_bytes = st.session_state.get("voice_bytes", None)
        if voice_bytes:
            st.audio(voice_bytes, format="audio/wav")

        if groq_enabled and voice_bytes:
            if st.button("✨ Auto-Transcribe Reference Audio (Groq Whisper)"):
                with st.spinner("Transcribing reference voice..."):
                    try:
                        st.session_state["auto_ref_text"] = transcribe_reference_wav(voice_bytes, groq_api_key)
                        st.success("Transcript extracted!")
                    except Exception as exc:
                        st.error(f"Transcription failed: {exc}")
    else:
        voice_bytes = None

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

    ref_text = ""
    if engine_choice == "Zero-Shot Voice Cloning (F5-TTS)":
        ref_text = st.text_input(
            "Reference Transcript (Must match exact words spoken in the reference clip)",
            value=parsed_ref,
        )

    consent_ok = st.checkbox("I confirm I own or have explicit authorization to synthesize this voice.")
    run_btn = st.button("🚀 Generate Chunk-by-Chunk & Combine", type="primary", use_container_width=True)
with col_right:
    st.subheader("3. Live Chunk-by-Chunk & Combined Output")
    st.info(f"**Active Profile:** {selected_accent} (Speed: `{speech_speed}x`)")

    if run_btn:
        active_voice_bytes = st.session_state.get("voice_bytes", None)
        if not consent_ok:
            st.error("Please confirm voice authorization before generating.")
        elif not timeline_items:
            st.warning("Please upload a valid .txt script with [MM:SS] markers.")
        elif engine_choice == "Zero-Shot Voice Cloning (F5-TTS)" and (not active_voice_bytes or not ref_text.strip()):
            st.warning("Please provide both a reference .wav clip and its exact reference text.")
        else:
            progress = st.progress(0, text="Preparing low-memory disk workspace...")
            chunk_feed = st.container()
            tts_engine = None
            similarity_score = None

            try:
                with tempfile.TemporaryDirectory() as session_dir:
                    norm_ref_path = os.path.join(session_dir, "norm_ref.wav")
                    if engine_choice == "Zero-Shot Voice Cloning (F5-TTS)":
                        normalize_reference_audio(active_voice_bytes, norm_ref_path, max_duration_ms=12000)
                        tts_engine = F5TTSClient(
                            use_api=True,
                            hf_token=hf_token,
                            accent_label=selected_accent,
                            allow_fallback=True,
                        )

                    stitched_inputs = []
                    total = len(timeline_items)

                    # Process strictly ONE chunk at a time, flush to disk, and free RAM
                    for idx, (ts_ms, text_line) in enumerate(timeline_items):
                        progress.progress(idx / total, text=f"Generating chunk {idx + 1} of {total}...")
                        seg_path = os.path.join(session_dir, f"seg_{idx}.wav")

                        if engine_choice == "Zero-Shot Voice Cloning (F5-TTS)":
                            tts_engine.clone_segment(
                                norm_ref_path,
                                ref_text.strip(),
                                text_line,
                                seg_path,
                                speed=speech_speed,
                            )
                        else:
                            synthesize_preset_groq(text_line, preset_voice, groq_api_key, seg_path)

                        with chunk_feed.expander(f"✅ Chunk {idx + 1}/{total}: {text_line[:55]}...", expanded=False):
                            with open(seg_path, "rb") as f_seg:
                                st.audio(f_seg.read(), format="audio/wav")

                        stitched_inputs.append((ts_ms, seg_path))
                        gc.collect()

                    progress.progress(0.92, text="Combining all chunks on disk into master presentation...")
                    master_wav_path = os.path.join(session_dir, "master_presentation.wav")
                    stitch_timeline_to_disk(stitched_inputs, master_wav_path)

                    # Compute acoustic voice similarity before deleting temp workspace
                    if engine_choice == "Zero-Shot Voice Cloning (F5-TTS)" and os.path.exists(norm_ref_path):
                        progress.progress(0.97, text="Verifying acoustic speaker similarity...")
                        similarity_score = compute_voice_similarity(norm_ref_path, master_wav_path)

                    with open(master_wav_path, "rb") as f_master:
                        final_wav_bytes = f_master.read()

                    progress.progress(1.0, text="Complete!")
                    gc.collect()

                used_fallback = bool(tts_engine and getattr(tts_engine, "_fallback_triggered", False))
                engine_summary = (
                    f"{engine_choice} (Regional Neural Fallback)"
                    if used_fallback
                    else engine_choice
                )

                log_generation_event(
                    active_user,
                    f"{operation_mode} | {selected_accent} | {engine_summary}",
                    len(timeline_items),
                    "SUCCESS",
                )

                if used_fallback:
                    st.warning(
                        f"Public F5-TTS GPU Spaces were offline, so your presentation used **{selected_accent}** neural speech."
                    )
                else:
                    st.success(f"All {total} cloned chunks combined into your master presentation!")

                # Display Acoustic Voice Match Verification Metrics
                if similarity_score is not None:
                    m1, m2 = st.columns(2)
                    m1.metric(
                        label="🎯 Acoustic Voice Match",
                        value=f"{similarity_score}%",
                        delta="Zero-Shot Cloned" if not used_fallback else "Regional Neural Voice",
                    )
                    m2.metric(
                        label="🔊 Active Synthesis Engine",
                        value="F5-TTS Clone" if not used_fallback else "Edge Regional TTS",
                    )

                st.audio(final_wav_bytes, format="audio/wav")
                st.download_button(
                    label="⬇️ Download Combined Master Presentation (.wav)",
                    data=final_wav_bytes,
                    file_name="enterprise_presentation_master.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )
            except Exception as exc:
                gc.collect()
                log_generation_event(active_user, operation_mode, len(timeline_items), "FAILED", str(exc))
                st.error(f"Synthesis failed: {exc}")
