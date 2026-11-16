import concurrent.futures
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

REQUEST_TIMEOUT_SECONDS = 35


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
    """Timeout-protected F5-TTS client that reuses active connections and fails over fast."""

    def __init__(self, use_api: bool = True, hf_token: Optional[str] = None, custom_space: Optional[str] = None):
        self.use_api = use_api
        self.hf_token = hf_token.strip() if (hf_token and hf_token.strip()) else None
        self.spaces = ([custom_space.strip()] if (custom_space and custom_space.strip()) else []) + FALLBACK_SPACES
        self._active_client: Optional[Client] = None
        self._active_space: Optional[str] = None

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
        engine = _load_local_f5_model()
        wav, sr, _ = engine.infer(ref_file=ref_wav_path, ref_text=ref_text, gen_text=gen_text)
        wavfile.write(dest_wav_path, sr, (wav * 32767).astype(np.int16))
        return dest_wav_path

    def _connect_with_timeout(self, space_id: str) -> Client:
        def _init_client():
            return Client(space_id, token=self.hf_token) if self.hf_token else Client(space_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_init_client)
            return future.result(timeout=20)

    def _predict_with_timeout(
        self,
        client: Client,
        ref_wav_path: str,
        ref_text: str,
        gen_text: str,
        speed: float = 0.92,
    ):
        def _run_predict():
            audio_input = handle_file(ref_wav_path)
            try:
                return client.predict(
                    ref_audio=audio_input,
                    ref_text=ref_text,
                    gen_text=gen_text,
                    remove_silence=False,
                    api_name="/predict",
                )
            except Exception:
                return client.predict(audio_input, ref_text, gen_text, api_name="/predict")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_predict)
            return future.result(timeout=REQUEST_TIMEOUT_SECONDS)

    def _clone_via_api(
        self,
        ref_wav_path: str,
        ref_text: str,
        gen_text: str,
        dest_wav_path: str,
        speed: float = 0.92,
    ) -> str:
        last_error = None
        # Try the already-connected warm Space first, then fail over through remaining Spaces
        ordered_spaces = (
            [self._active_space] + [s for s in self.spaces if s != self._active_space]
            if self._active_space
            else self.spaces
        )

        for space_id in ordered_spaces:
            for attempt in range(2):
                try:
                    if self._active_client is None or self._active_space != space_id:
                        self._active_client = self._connect_with_timeout(space_id)
                        self._active_space = space_id

                    result = self._predict_with_timeout(
                        self._active_client, ref_wav_path, ref_text, gen_text, speed=speed
                    )
                    generated_path = _extract_audio_path(result)
                    if generated_path and os.path.exists(generated_path):
                        shutil.copyfile(generated_path, dest_wav_path)
                        return dest_wav_path
                except concurrent.futures.TimeoutError:
                    last_error = f"Timed out after {REQUEST_TIMEOUT_SECONDS}s on {space_id}"
                    self._active_client = None
                    break  # Immediately switch to next Space instead of waiting
                except Exception as exc:
                    last_error = exc
                    self._active_client = None
                    time.sleep(1)

        raise RuntimeError(
            f"Public F5-TTS Spaces are currently busy or asleep ({last_error}). "
            "Tip: Switch to 'With API' mode and add your free HF_TOKEN or GROQ_API_KEY in the sidebar."
        )
