import os
import tempfile
import streamlit as st

from src.auth import enforce_enterprise_auth
from src.parser import parse_presentation_script
from src.audio_processor import normalize_reference_audio, stitch_timeline_segments
from src.tts_client import F5TTSClient
from src.groq_helper import is_groq_available, transcribe_reference_wav, synthesize_preset_groq
from src.audit_logger import log_generation_event

st.set_page_config(page_title="Enterprise Voice Presentation Studio", page_icon="🎙️", layout="wide")

active_user = enforce_enterprise_auth()
secret_hf_token = st.secrets.get("HF_TOKEN", "")
secret_groq_key = st.secrets.get("GROQ_API_KEY", "")

st.title("🎙️ Enterprise AI Voice Cloning Presentation Studio")
st.caption("Hybrid zero-shot voice cloning supporting both No-API and API-accelerated execution.")

with st.sidebar:
    st.header("⚙️ Execution Mode")
    operation_mode = st.radio(
        "Choose Architecture Mode:",
        [
            "Without API (Free Public / Local Mode)",
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
    else:
        backend_type = st.selectbox(
            "No-API Backend Engine",
            [
                "Public HF Cloud Spaces (No Key Needed)",
                "Local Python F5-TTS Engine (Offline CPU/GPU)",
            ],
        )

    groq_enabled = use_api_mode and is_groq_available(groq_api_key)

    engine_choice = "Zero-Shot Voice Cloning (F5-TTS)"
    preset_voice = "Fritz-PlayAI"
    if groq_enabled:
        engine_choice = st.radio(
            "Synthesis Engine",
            ["Zero-Shot Voice Cloning (F5-TTS)", "Ultra-Fast Preset Voice (Groq)"],
        )
        if engine_choice == "Ultra-Fast Preset Voice (Groq)":
            preset_voice = st.selectbox(
                "Select Groq Preset Voice",
                ["Fritz-PlayAI", "Arista-PlayAI", "Atlas-PlayAI", "Celeste-PlayAI", "Mason-PlayAI"],
            )

col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    voice_bytes = None
    if engine_choice == "Zero-Shot Voice Cloning (F5-TTS)":
        st.subheader("1. Reference Voice Sample (.wav)")
        input_method = st.radio("Audio Source", ["Upload .wav File", "Record Microphone"], horizontal=True)

        if input_method == "Upload .wav File":
            uploaded_wav = st.file_uploader("Upload clean 5–10 second speaker clip", type=["wav"])
            if uploaded_wav:
                voice_bytes = uploaded_wav.read()
                st.audio(voice_bytes, format="audio/wav")
        else:
            mic_audio = st.audio_input("Record 5–10 seconds of clear speech")
            if mic_audio:
                voice_bytes = mic_audio.read()

        if groq_enabled and voice_bytes:
            if st.button("✨ Auto-Transcribe Reference Audio (Groq Whisper)"):
                with st.spinner("Transcribing reference voice..."):
                    try:
                        st.session_state["auto_ref_text"] = transcribe_reference_wav(voice_bytes, groq_api_key)
                        st.success("Transcript extracted!")
                    except Exception as exc:
                        st.error(f"Transcription failed: {exc}")

    st.subheader("2. Presentation Script (.txt)")
    uploaded_script = st.file_uploader("Upload [MM:SS] timestamped script", type=["txt"])

    parsed_ref = st.session_state.get("auto_ref_text", "")
    timeline_items = []

    if uploaded_script:
        raw_script_text = uploaded_script.read().decode("utf-8")
        try:
            script_ref, timeline_items = parse_presentation_script(raw_script_text)
            if not parsed_ref:
                parsed_ref = script_ref
            st.success(f"Validated {len(timeline_items)} timestamped segments.")
        except ValueError as err:
            st.error(f"Script Format Error: {err}")

    ref_text = ""
    if engine_choice == "Zero-Shot Voice Cloning (F5-TTS)":
        ref_text = st.text_input(
            "Reference Transcript (Must match exact words spoken in the reference clip)",
            value=parsed_ref,
        )

    consent_ok = st.checkbox("I confirm I own or have explicit authorization to synthesize this voice.")
    run_btn = st.button("🚀 Generate Presentation Audio", type="primary", use_container_width=True)

with col_right:
    st.subheader("3. Master Timeline Output")
    if run_btn:
        if not consent_ok:
            st.error("Please confirm voice authorization before generating.")
        elif not timeline_items:
            st.warning("Please upload a valid .txt script with [MM:SS] markers.")
        elif engine_choice == "Zero-Shot Voice Cloning (F5-TTS)" and (not voice_bytes or not ref_text.strip()):
            st.warning("Please provide both a reference .wav clip and its exact reference text.")
        else:
            progress = st.progress(0, text="Initializing isolated session workspace...")
            try:
                with tempfile.TemporaryDirectory() as session_dir:
                    norm_ref_path = os.path.join(session_dir, "norm_ref.wav")
                    if engine_choice == "Zero-Shot Voice Cloning (F5-TTS)":
                        normalize_reference_audio(voice_bytes, norm_ref_path, max_duration_ms=15000)
                        use_remote_api = use_api_mode or (backend_type == "Public HF Cloud Spaces (No Key Needed)")
                        tts_engine = F5TTSClient(use_api=use_remote_api, hf_token=hf_token)

                    stitched_inputs = []
                    total = len(timeline_items)

                    for idx, (ts_ms, text_line) in enumerate(timeline_items):
                        progress.progress(idx / total, text=f"Synthesizing segment {idx + 1} of {total}...")
                        seg_path = os.path.join(session_dir, f"seg_{idx}.wav")

                        if engine_choice == "Zero-Shot Voice Cloning (F5-TTS)":
                            tts_engine.clone_segment(norm_ref_path, ref_text.strip(), text_line, seg_path)
                        else:
                            synthesize_preset_groq(text_line, preset_voice, groq_api_key, seg_path)

                        stitched_inputs.append((ts_ms, seg_path))

                    progress.progress(0.95, text="Stitching master presentation timeline...")
                    final_wav = stitch_timeline_segments(stitched_inputs)
                    progress.progress(1.0, text="Complete!")

                log_generation_event(active_user, f"{operation_mode} | {engine_choice}", len(timeline_items), "SUCCESS")
                st.success("Master presentation audio generated!")
                st.audio(final_wav, format="audio/wav")
                st.download_button(
                    label="⬇️ Download Master Presentation (.wav)",
                    data=final_wav,
                    file_name="enterprise_presentation_master.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )
            except Exception as exc:
                log_generation_event(active_user, operation_mode, len(timeline_items), "FAILED", str(exc))
                st.error(f"Synthesis failed: {exc}")
