import asyncio
import concurrent.futures
import os
import re
import shutil
import time
from typing import List, Optional
import numpy as np
import scipy.io.wavfile as wavfile
import streamlit as st
from gradio_client import Client, handle_file
from huggingface_hub import HfApi

SEED_SPACES = [
    "SRINIVASTA/f5-tts-cloner",
    "mrfakename/E2-F5-TTS",
    "mrfakename/F5-TTS",
]

EDGE_VOICE_MAP = {
    "🇮🇳 India (English) — Natural Cadence": "en-IN-PrabhatNeural",
    "🇺🇸 United States (English) — Standard": "en-US-AndrewNeural",
    "🇬🇧 United Kingdom (English) — Crisp": "en-GB-RyanNeural",
    "🇦🇺 Australia (English) — Natural": "en-AU-WilliamNeural",
    "🌐 Slower / Clear Presentation Pace": "en-IN-NeerjaNeural",
}

REQUEST_TIMEOUT_SECONDS = 60


@st.cache_resource(show_spinner="Loading local F5-TTS GPU/CPU engine...")
def _get_local_f5_engine():
    """Loads local F5-TTS model if installed (e.g., in Google Colab); returns None on Streamlit Cloud."""
    try:
        from f5_tts.api import F5TTS
        return F5TTS()
    except ImportError:
        return None


@st.cache_data(ttl=180, show_spinner=False)
def _discover_running_f5_spaces(hf_token: Optional[str] = None) -> List[str]:
    discovered: List[str] = []
    try:
        api = HfApi(token=hf_token if hf_token else None)
        spaces = api.list_spaces(search="F5-TTS", sort="likes", direction=-1, limit=20)
        for sp in spaces:
            runtime = getattr(sp, "runtime", None)
            stage = getattr(runtime, "stage", "") if runtime else ""
            if stage == "RUNNING":
                discovered.append(sp.id)
    except Exception:
        pass

    combined: List[str] = []
    for s in SEED_SPACES + discovered:
        if s and s not in combined:
            combined.append(s)
    return combined[:8]


def _extract_audio_path(result) -> Optional[str]:
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


def _synthesize_edge_fallback(gen_text: str, accent_label: str, dest_wav_path: str) -> str:
    import edge_tts
    from pydub import AudioSegment

    clean_text = re.sub(r"\s+", " ", gen_text).strip() or "."
    primary_voice = EDGE_VOICE_MAP.get(accent_label, "en-IN-PrabhatNeural")
    fallback_voices = [primary_voice, "en-IN-NeerjaNeural", "en-US-JennyNeural"]
    temp_mp3 = f"{dest_wav_path}.mp3"

    last_err = None
    for attempt in range(3):
        voice_to_use = fallback_voices[attempt % len(fallback_voices)]
        try:
            time.sleep(0.4 + (attempt * 0.8))

            async def _run():
                communicate = edge_tts.Communicate(clean_text, voice_to_use)
                await communicate.save(temp_mp3)

            asyncio.run(_run())

            if os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 100:
                sound = AudioSegment.from_mp3(temp_mp3)
                sound.export(dest_wav_path, format="wav")
                os.remove(temp_mp3)
                return dest_wav_path
        except Exception as exc:
            last_err = exc
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)

    raise RuntimeError(f"Zero-key neural TTS failed after 3 retries: {last_err}")


class F5TTSClient:
    """Runs local F5-TTS automatically in Colab/GPU environments, or remote Spaces on Streamlit Cloud."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        custom_space: Optional[str] = None,
        accent_label: str = "🇮🇳 India (English) — Natural Cadence",
        allow_fallback: bool = True,
    ):
        self.hf_token = hf_token.strip() if (hf_token and hf_token.strip()) else None
        self.accent_label = accent_label
        self.allow_fallback = allow_fallback
        clean_custom = custom_space.strip() if (custom_space and custom_space.strip()) else None
        if clean_custom and "/" not in clean_custom:
            clean_custom = f"SRINIVASTA/{clean_custom}"

        self.spaces = [clean_custom] if clean_custom else _discover_running_f5_spaces(self.hf_token)
        self._active_client: Optional[Client] = None
        self._active_space: Optional[str] = None
        self._fallback_triggered: bool = False

    def _connect_with_timeout(self, space_id: str) -> Client:
        def _init_client():
            return Client(space_id, token=self.hf_token) if self.hf_token else Client(space_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_init_client).result(timeout=25)

    def _predict_with_timeout(self, client: Client, ref_wav_path: str, ref_text: str, gen_text: str):
        def _run():
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
            return pool.submit(_run).result(timeout=REQUEST_TIMEOUT_SECONDS)

    def clone_segment(
        self,
        ref_wav_path: str,
        ref_text: str,
        gen_text: str,
        dest_wav_path: str,
        speed: float = 0.92,
    ) -> str:
        # 1. If running in Google Colab (local f5-tts installed), clone directly on local T4 GPU
        local_engine = _get_local_f5_engine()
        if local_engine is not None:
            self._active_space = "Local Colab GPU (F5-TTS)"
            wav, sr, _ = local_engine.infer(
                ref_file=ref_wav_path,
                ref_text=ref_text,
                gen_text=gen_text,
            )
            wavfile.write(dest_wav_path, sr, (wav * 32767).astype(np.int16))
            return dest_wav_path

        # 2. Otherwise use remote Hugging Face Spaces + fallback (Streamlit Cloud)
        if self._fallback_triggered:
            return _synthesize_edge_fallback(gen_text, self.accent_label, dest_wav_path)

        ordered_spaces = (
            [self._active_space] + [s for s in self.spaces if s != self._active_space]
            if self._active_space
            else self.spaces
        )

        for space_id in ordered_spaces:
            try:
                if self._active_client is None or self._active_space != space_id:
                    self._active_client = self._connect_with_timeout(space_id)
                    self._active_space = space_id

                result = self._predict_with_timeout(self._active_client, ref_wav_path, ref_text, gen_text)
                generated_path = _extract_audio_path(result)
                if generated_path and os.path.exists(generated_path):
                    shutil.copyfile(generated_path, dest_wav_path)
                    return dest_wav_path
            except Exception:
                self._active_client = None
                time.sleep(0.5)

        if self.allow_fallback:
            self._fallback_triggered = True
            self._active_space = "Edge Regional Neural TTS (Zero-Key)"
            return _synthesize_edge_fallback(gen_text, self.accent_label, dest_wav_path)

        raise RuntimeError("All voice cloning endpoints failed.")
