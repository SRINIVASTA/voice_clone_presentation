import gc
import io
import os
import wave
from typing import List, Tuple
from pydub import AudioSegment, effects


def normalize_reference_audio(raw_bytes: bytes, output_path: str, max_duration_ms: int = 12000) -> str:
    """Converts uploaded audio to mono 24kHz WAV, normalizes volume, and frees RAM immediately."""
    audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
    audio = audio.set_channels(1).set_frame_rate(24000)
    audio = effects.normalize(audio)
    if len(audio) > max_duration_ms:
        audio = audio[:max_duration_ms]
    audio.export(output_path, format="wav")
    del audio
    gc.collect()
    return output_path


def _standardize_chunk_wav(input_path: str, target_rate: int = 24000) -> None:
    """Ensures every synthesized chunk matches mono 24kHz 16-bit PCM before disk concatenation."""
    clip = AudioSegment.from_file(input_path)
    clip = clip.set_channels(1).set_frame_rate(target_rate).set_sample_width(2)
    clip.export(input_path, format="wav")
    del clip
    gc.collect()


def stitch_timeline_to_disk(timed_audio_paths: List[Tuple[int, str]], output_wav_path: str) -> str:
    """Streams chunks and silence frames directly to a disk WAV file using minimal RAM."""
    sample_rate = 24000
    sample_width = 2  # 16-bit PCM
    channels = 1
    bytes_per_ms = int((sample_rate * sample_width * channels) / 1000)

    current_cursor_ms = 0

    with wave.open(output_wav_path, "wb") as out_wav:
        out_wav.setnchannels(channels)
        out_wav.setsampwidth(sample_width)
        out_wav.setframerate(sample_rate)

        for start_ms, file_path in timed_audio_paths:
            if not os.path.exists(file_path):
                continue

            _standardize_chunk_wav(file_path, target_rate=sample_rate)

            # Write exact silence frames directly to disk if a timestamp gap is needed
            if start_ms > current_cursor_ms:
                silence_ms = start_ms - current_cursor_ms
                silence_bytes = b"\x00" * (silence_ms * bytes_per_ms)
                out_wav.writeframes(silence_bytes)
                del silence_bytes
                current_cursor_ms = start_ms

            # Stream chunk frames directly from temp file to master file
            with wave.open(file_path, "rb") as in_wav:
                frames = in_wav.readframes(in_wav.getnframes())
                duration_ms = int((in_wav.getnframes() / sample_rate) * 1000)
                out_wav.writeframes(frames)
                del frames
                current_cursor_ms += duration_ms

            # Remove processed chunk file immediately to save disk & RAM
            os.remove(file_path)
            gc.collect()

    return output_wav_path
