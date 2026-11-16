import os
import shutil
import time
from typing import Optional
import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
from gradio_client import Client, handle_file

FALLBACK_SPACES = [
    "mrfakename/E2-F5-TTS",
    "SWivid/F5-TTS",
]


@st.cache_resource(show_spinner="Loading local F5-TTS model (No-API Mode)...")
def _load_local_f5_model():
    from f5_tts.api import F5TTS
    return F5TTS()


class F5TTSClient:
    """Supports both Local No-API F5-TTS and Remote API F5-TTS with failover."""

    def __init__(self, use_api: bool = True, hf_token: Optional[str] = None, custom_space: Optional[str] = None):
        self.use_api = use_api
        self.hf_token = hf_token if hf_token and hf_token.strip() else None
        self.spaces = ([custom_space] if custom_space else []) + FALLBACK_SPACES

    def clone_segment(self, ref_wav_path: str, ref_text: str, gen_text: str, dest_wav_path: str) -> str:
        if not self.use_api:
            return self._clone_locally(ref_wav_path, ref_text, gen_text, dest_wav_path)
        return self._clone_via_api(ref_wav_path, ref_text, gen_text, dest_wav_path)

    def _clone_locally(self, ref_wav_path: str, ref_text: str, gen_text: str, dest_wav_path: str) -> str:
        """Runs 100% locally without any external API keys or Space endpoints."""
        engine = _load_local_f5_model()
        wav, sr, _ = engine.infer(
            ref_file=ref_wav_path,
            ref_text=ref_text,
            gen_text=gen_text,
        )
        wavfile.write(dest_wav_path, sr, (wav * 32767).astype(np.int16))
        return dest_wav_path

    def _clone_via_api(self, ref_wav_path: str, ref_text: str, gen_text: str, dest_wav_path: str) -> str:
        """Runs via Hugging Face Gradio API (works with or without HF_TOKEN)."""
        last_error = None
        for space_id in self.spaces:
            for attempt in range(3):
                try:
                    client = Client(space_id, token=self.hf_token)
                    result = client.predict(
                        ref_audio_input=handle_file(ref_wav_path),
                        ref_text_input=ref_text,
                        gen_text_input=gen_text,
                        remove_silence=False,
                        cross_fade_duration_slider=0.15,
                        speed_slider=1.0,
                        api_name="/basic_tts",
                    )
                    generated_path = result[0] if isinstance(result, (list, tuple)) else result
                    if generated_path and os.path.exists(generated_path):
                        shutil.copyfile(generated_path, dest_wav_path)
                        return dest_wav_path
                except Exception as exc:
                    last_error = exc
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"All F5-TTS API endpoints failed after retries. Last error: {last_error}")
