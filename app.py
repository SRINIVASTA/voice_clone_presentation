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
from src.tts_client import F5TTSClient, synthesize_regional_neural
from src.groq_helper import is_groq_available, transcribe_reference_wav, synthesize_preset_groq
from src.audit_logger import log_generation_event

st.set_page_config(page_title="Enterprise Voice Presentation Studio", page_icon="🎙️", layout="wide")

active_user = enforce_enterprise_auth()
secret_hf_token = st.secrets.get("HF_TOKEN", "")
secret_groq_key = st.secrets.get("GROQ_API_KEY", "")
secret_custom_space = st.secrets.get("CUSTOM_HF_SPACE", "")

ACCENT_PROFILES = {
    "🇮🇳 India (English) — Clear Cadence": 0.90,
    "🇺🇸 United States (English) — Standard": 0.95,
    "🇬🇧 United Kingdom (English) — Crisp": 0.95,
    "🇦🇺 Australia (English) — Natural": 0.95,
    "🌐 Neutral / Slower Presentation Pace": 0.85,
}

st.title("🎙️ Enterprise AI Voice Cloning Presentation Studio")
st.caption("Select your dedicated synthesis engine in the sidebar: Zero-Shot Voice Cloning, Groq Preset TTS, or Free Regional Neural TTS.")

with st.sidebar:
    st.header("⚙️ Select Synthesis Option")
    selected_mode = st.radio(
        "Choose Exact Engine:",
        [
            "1. Zero-Shot Voice Cloning (F5-TTS)",
            "2. Ultra-Fast Preset Voices (Groq API)",
            "3. Free Regional Neural Voice (No API)",
        ],
    )

    hf_token = None
    custom_hf_space = ""
    groq_api_key = None
    preset_voice = "Fritz-PlayAI"

    if selected_mode.startswith("1."):
        st.subheader("🔑 F5-TTS Space Configuration")
        custom_hf_space = st.text_input(
            "Hugging Face Space ID",
            value=secret_custom_space,
            placeholder="e.g., SRINIVASTA/f5-tts-cloner",
            help="Enter your own Hugging Face Space ID for guaranteed cloning, or leave blank to scan public Spaces.",
        )
        hf_token = st.text_input(
            "Hugging Face Token (Optional)",
            value=secret_hf_token,
            type="password",
        )
        groq_api_key = st.text_input(
            "Groq Key for Auto-Transcribe (Optional)",
            value=secret_groq_key,
            type="password",
        )

    elif selected_mode.startswith("2."):
        st.subheader("🔑 Groq API Configuration")
        groq_api_key = st.text_input(
            "Groq API Key (Required)",
            value=secret_groq_key,
            type="password",
        )
        preset_voice = st.selectbox(
            "Select Groq Preset Voice",
            ["Fritz-PlayAI", "Arista-PlayAI", "Atlas-PlayAI", "Celeste-PlayAI", "Mason-PlayAI"],
        )

    st.divider()
    st.subheader("🌍 Language & Accent Profile")
    selected_accent = st.selectbox(
        "Select Output Dialect & Cadence",
        list(ACCENT_PROFILES.keys()),
        index=0,
    )
    speech_speed = ACCENT_PROFILES[selected_accent]

groq_enabled = is_groq_available(groq_api_key)
is_cloning_mode = selected_mode.startswith("1.")
is_groq_mode = selected_mode.startswith("2.")
is_edge_mode = selected_mode.startswith("3.")

col_left, col_right = st.columns([1.05, 0.95], gap="large")

with col_left:
    st.subheader("1. Reference Voice Sample (.wav)")
    if not is_cloning_mode:
        st.caption("Optional in Preset / Regional modes (required for Option 1 Zero-Shot Voice Cloning).")

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
        "Reference Transcript (Must match exact words spoken in the reference clip for Option 1)",
        value=parsed_ref,
    )

    consent_ok = st.checkbox("I confirm I own or have explicit authorization to generate this audio.")
    run_btn = st.button("🚀 Generate Chunk-by-Chunk & Combine", type="primary", use_container_width=True)
with col_right:
    st.subheader("3. Live Chunk-by-Chunk & Combined Output")
    st.info(f"**Mode:** `{selected_mode}` | **Dialect:** {selected_accent}")

    if run_btn:
        active_voice_bytes = st.session_state.get("voice_bytes", None)
        if not consent_ok:
            st.error("Please confirm authorization before generating.")
        elif not timeline_items:
            st.warning("Please upload a valid .txt script with [MM:SS] markers.")
        elif is_cloning_mode and (not active_voice_bytes or not ref_text.strip()):
            st.warning("Option 1 requires both a reference .wav clip and its exact reference text.")
        elif is_groq_mode and not groq_enabled:
            st.error("Option 2 requires a valid Groq API Key (`gsk_...`) in the sidebar.")
        else:
            progress = st.progress(0, text="Preparing low-memory disk workspace...")
            chunk_feed = st.container()
            tts_engine = None
            similarity_score = None

            try:
                with tempfile.TemporaryDirectory() as session_dir:
                    norm_ref_path = os.path.join(session_dir, "norm_ref.wav")
                    if active_voice_bytes:
                        normalize_reference_audio(active_voice_bytes, norm_ref_path, max_duration_ms=12000)

                    if is_cloning_mode:
                        tts_engine = F5TTSClient(hf_token=hf_token, custom_space=custom_hf_space)

                    stitched_inputs = []
                    total = len(timeline_items)

                    for idx, (ts_ms, text_line) in enumerate(timeline_items):
                        progress.progress(idx / total, text=f"Generating chunk {idx + 1} of {total}...")
                        seg_path = os.path.join(session_dir, f"seg_{idx}.wav")

                        if is_cloning_mode:
                            tts_engine.clone_segment(
                                norm_ref_path,
                                ref_text.strip(),
                                text_line,
                                seg_path,
                                speed=speech_speed,
                            )
                        elif is_groq_mode:
                            synthesize_preset_groq(text_line, preset_voice, groq_api_key, seg_path)
                        else:
                            synthesize_regional_neural(text_line, selected_accent, seg_path)

                        with chunk_feed.expander(f"✅ Chunk {idx + 1}/{total}: {text_line[:55]}...", expanded=False):
                            with open(seg_path, "rb") as f_seg:
                                st.audio(f_seg.read(), format="audio/wav")

                        stitched_inputs.append((ts_ms, seg_path))
                        gc.collect()

                    progress.progress(0.92, text="Combining all chunks on disk into master presentation...")
                    master_wav_path = os.path.join(session_dir, "master_presentation.wav")
                    stitch_timeline_to_disk(stitched_inputs, master_wav_path)

                    if active_voice_bytes and os.path.exists(norm_ref_path):
                        progress.progress(0.97, text="Verifying acoustic speaker similarity...")
                        similarity_score = compute_voice_similarity(norm_ref_path, master_wav_path)

                    with open(master_wav_path, "rb") as f_master:
                        final_wav_bytes = f_master.read()

                    progress.progress(1.0, text="Complete!")
                    gc.collect()

                active_space_used = getattr(tts_engine, "_active_space", "Hugging Face F5-TTS")
                log_generation_event(
                    active_user,
                    f"{selected_mode} | {selected_accent}",
                    len(timeline_items),
                    "SUCCESS",
                )

                st.success(f"All {total} chunks synthesized and combined!")

                if similarity_score is not None:
                    m1, m2 = st.columns(2)
                    m1.metric(
                        label="🎯 Acoustic Voice Match",
                        value=f"{similarity_score}%",
                        delta="Verified Zero-Shot Clone" if is_cloning_mode else "Comparison vs Uploaded WAV",
                    )
                    m2.metric(
                        label="🔊 Active Engine",
                        value=str(active_space_used) if is_cloning_mode else selected_mode,
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
                log_generation_event(active_user, selected_mode, len(timeline_items), "FAILED", str(exc))
                st.error(f"Synthesis failed: {exc}")
