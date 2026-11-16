import os
import shutil
import time
from typing import Optional
from gradio_client import Client, handle_file

FALLBACK_SPACES = [
    "mrfakename/E2-F5-TTS",
    "SWivid/F5-TTS",
]


class F5TTSClient:
    """Fault-tolerant F5-TTS voice cloning client with retry backoff and Space failover."""

    def __init__(self, hf_token: Optional[str] = None, custom_space: Optional[str] = None):
        self.hf_token = hf_token if hf_token and hf_token.strip() else None
        self.spaces = ([custom_space] if custom_space else []) + FALLBACK_SPACES

    def clone_segment(self, ref_wav_path: str, ref_text: str, gen_text: str, dest_wav_path: str) -> str:
        last_error = None
        for space_id in self.spaces:
            for attempt in range(3):
                try:
                    # Updated for gradio_client >= 1.0: uses 'token' instead of 'hf_token'
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
        raise RuntimeError(f"All F5-TTS endpoints failed after retries. Last error: {last_error}")
