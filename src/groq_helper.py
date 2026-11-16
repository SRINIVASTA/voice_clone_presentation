import io
from typing import Optional
from groq import Groq

GROQ_ORPHEUS_VOICES = [
    "troy",
    "hannah",
    "autumn",
    "diana",
    "daniel",
    "austin",
]


def is_groq_available(api_key: Optional[str]) -> bool:
    return bool(api_key and api_key.strip().startswith("gsk_"))


def transcribe_reference_wav(wav_bytes: bytes, api_key: str) -> str:
    """Uses Groq Whisper-Large-V3-Turbo to auto-transcribe the reference voice clip."""
    client = Groq(api_key=api_key)
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "reference.wav"

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3-turbo",
        response_format="text",
        temperature=0.0,
    )
    return str(transcription).strip()


def synthesize_preset_groq(text: str, voice_name: str, api_key: str, dest_wav_path: str) -> str:
    """Ultra-fast preset voice narration powered by Groq Canopy Labs Orpheus TTS."""
    client = Groq(api_key=api_key)
    selected_voice = voice_name if voice_name in GROQ_ORPHEUS_VOICES else "troy"

    response = client.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice=selected_voice,
        input=text,
        response_format="wav",
    )
    response.write_to_file(dest_wav_path)
    return dest_wav_path
