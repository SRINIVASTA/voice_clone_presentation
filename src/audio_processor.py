import io
import os
from typing import List, Tuple
from pydub import AudioSegment, effects


def normalize_reference_audio(raw_bytes: bytes, output_path: str, max_duration_ms: int = 10000) -> str:
    """Converts uploaded audio to mono 24kHz WAV, normalizes volume, and trims to 10s."""
    audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
    audio = audio.set_channels(1).set_frame_rate(24000)
    audio = effects.normalize(audio)
    if len(audio) > max_duration_ms:
        audio = audio[:max_duration_ms]
    audio.export(output_path, format="wav")
    return output_path


def stitch_timeline_segments(timed_audio_paths: List[Tuple[int, str]]) -> bytes:
    """Stitches generated audio segments onto an exact millisecond timeline."""
    master_track = AudioSegment.empty()
    current_cursor_ms = 0

    for start_ms, file_path in timed_audio_paths:
        if not os.path.exists(file_path):
            continue
        clip = AudioSegment.from_file(file_path)

        if start_ms > current_cursor_ms:
            silence_gap = start_ms - current_cursor_ms
            master_track += AudioSegment.silent(duration=silence_gap)
            current_cursor_ms = start_ms

        master_track += clip
        current_cursor_ms += len(clip)

    out_buffer = io.BytesIO()
    master_track.export(out_buffer, format="wav")
    return out_buffer.getvalue()
