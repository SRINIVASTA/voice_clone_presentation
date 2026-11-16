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
    "abidlabs/E2-F5-TTS",
    "emilalvaro/E2-F5-TTS",
]


@st.cache_resource(show_spinner="Loading local F5-TTS model (No-API Mode)...")
def _load_local_f5_model():
    from f5_tts.api import F5TTS
    return F5TTS()


def _extract_audio_path(result) -> Optional[str]:
    """Recursively extracts the first existing .wav/.audio file path from Gradio output."""
    if isinstance(result, str) and os.path.exists(result):
        return result
    if isinstance(result, dict):
        for key in ("path", "url", "name"):
            val = result.get(key)
            if isinstance(val, str) and os.path.exists(val):
                return val
    if isinstance(result, (list, tuple)):
        for item in result:
            found = _extract_audio_path(item)
            if found:
                return found
    return None


class F5TTSClient:
    """Supports both Local No-API F5-TTS and Remote API F5-TTS with schema-aware keyword mapping."""

    def __init__(self, use_api: bool = True, hf_token: Optional[str] = None, custom_space: Optional[str] = None):
        self.use_api = use_api
        self.hf_token = hf_token.strip() if (hf_token and hf_token.strip()) else None
        self.spaces = ([custom_space.strip()] if (custom_space and custom_space.strip()) else []) + FALLBACK_SPACES

    def clone_segment(
        self,
        ref_wav_path: str,
        ref_text: str,
        gen_text: str,
        dest_wav_path: str,
        speed: float = 0.92,
    ) -> str:
        if not self.use_api:
            return self._clone_locally(ref_wav_path, ref_text, gen_text, dest_wav_path, speed=speed)
        return self._clone_via_api(ref_wav_path, ref_text, gen_text, dest_wav_path, speed=speed)

    def _clone_locally(
        self,
        ref_wav_path: str,
        ref_text: str,
        gen_text: str,
        dest_wav_path: str,
        speed: float = 0.92,
    ) -> str:
        """Runs 100% locally without any external API keys or Space endpoints."""
        engine = _load_local_f5_model()
        try:
            wav, sr, _ = engine.infer(
                ref_file=ref_wav_path,
                ref_text=ref_text,
                gen_text=gen_text,
                speed=speed,
            )
        except TypeError:
            wav, sr, _ = engine.infer(
                ref_file=ref_wav_path,
                ref_text=ref_text,
                gen_text=gen_text,
            )
        wavfile.write(dest_wav_path, sr, (wav * 32767).astype(np.int16))
        return dest_wav_path

    def _predict_schema_aware(
        self,
        client: Client,
        ref_wav_path: str,
        ref_text: str,
        gen_text: str,
        speed: float = 0.92,
    ):
        """Maps inputs by parameter name from the Space's live API schema."""
        audio_input = handle_file(ref_wav_path)

        # 1. Primary verified signature for mrfakename/E2-F5-TTS (/predict)
        try:
            return client.predict(
                ref_audio=audio_input,
                ref_text=ref_text,
                gen_text=gen_text,
                remove_silence=False,
                api_name="/predict",
            )
        except Exception:
            pass

        # 2. Schema-aware parameter mapping for any F5-TTS fork
        api_info = client.view_api(return_format="dict")
        named_endpoints = api_info.get("named_endpoints", {})

        for ep_name, ep_meta in named_endpoints.items():
            params = ep_meta.get("parameters", [])
            param_names = [p.get("parameter_name", "") for p in params]
            if any("audio" in n for n in param_names) and any("gen" in n or "text" in n for n in param_names):
                kwargs = {}
                for p in params:
                    p_name = p.get("parameter_name", "")
                    if "audio" in p_name:
                        kwargs[p_name] = audio_input
                    elif "ref" in p_name and "text" in p_name:
                        kwargs[p_name] = ref_text
                    elif "gen" in p_name or p_name == "text":
                        kwargs[p_name] = gen_text
                    elif "silence" in p_name:
                        kwargs[p_name] = False
                    elif "speed" in p_name:
                        kwargs[p_name] = speed
                    elif "model" in p_name:
                        kwargs[p_name] = "F5-TTS"
                return client.predict(**kwargs, api_name=ep_name)

        # 3. Last-resort 3-argument call on /predict
        return client.predict(audio_input, ref_text, gen_text, api_name="/predict")

    def _clone_via_api(
        self,
        ref_wav_path: str,
        ref_text: str,
        gen_text: str,
        dest_wav_path: str,
        speed: float = 0.92,
    ) -> str:
        """Runs via Hugging Face Gradio API with schema-aware mapping and failover."""
        last_error = None
        for space_id in self.spaces:
            for attempt in range(3):
                try:
                    client = Client(space_id, token=self.hf_token) if self.hf_token else Client(space_id)
                    result = self._predict_schema_aware(client, ref_wav_path, ref_text, gen_text, speed=speed)
                    generated_path = _extract_audio_path(result)
                    if generated_path and os.path.exists(generated_path):
                        shutil.copyfile(generated_path, dest_wav_path)
                        return dest_wav_path
                except Exception as exc:
                    last_error = exc
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"All F5-TTS API endpoints failed after retries. Last error: {last_error}")
