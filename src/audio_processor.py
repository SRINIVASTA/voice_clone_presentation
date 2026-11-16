import gc
import io
import math
import os
import struct
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


def _extract_acoustic_profile(wav_path: str) -> List[float]:
    """Extracts a lightweight 16-band energy & zero-crossing vocal signature using pure Python."""
    if not os.path.exists(wav_path):
        return [0.0] * 16

    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        total_samples = len(raw) // 2
        if total_samples < 500:
            return [0.0] * 16
        sample_count = min(total_samples, 48000)
        samples = struct.unpack(f"<{sample_count}h", raw[: sample_count * 2])

    voiced = [s for s in samples if abs(s) > 300]
    if len(voiced) < 256:
        return [0.0] * 16

    step = len(voiced) // 16
    profile: List[float] = []
    for i in range(16):
        chunk = voiced[i * step : (i + 1) * step]
        rms = math.sqrt(sum(x * x for x in chunk) / len(chunk))
        zcr = sum(1 for j in range(1, len(chunk)) if (chunk[j - 1] >= 0) != (chunk[j] >= 0)) / len(chunk)
        profile.append(math.log1p(rms) * (1.0 + zcr * 5.0))

    norm = math.sqrt(sum(v * v for v in profile))
    return [v / norm for v in profile] if norm > 0 else profile


def compute_voice_similarity(ref_wav_path: str, gen_wav_path: str) -> float:
    """Returns a 0-100% cosine similarity score comparing reference and generated vocal signatures."""
    p_ref = _extract_acoustic_profile(ref_wav_path)
    p_gen = _extract_acoustic_profile(gen_wav_path)

    dot = sum(a * b for a, b in zip(p_ref, p_gen))
    scaled = max(0.0, min(100.0, ((dot - 0.75) / 0.25) * 100.0))
    return round(scaled, 1)
