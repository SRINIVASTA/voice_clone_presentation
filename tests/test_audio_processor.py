import io
import os
import tempfile
from pydub import AudioSegment
from src.audio_processor import normalize_reference_audio, stitch_timeline_segments


def _make_dummy_wav_bytes(duration_ms: int = 2000) -> bytes:
    tone = AudioSegment.silent(duration=duration_ms, frame_rate=44100).set_channels(2)
    buf = io.BytesIO()
    tone.export(buf, format="wav")
    return buf.getvalue()


def test_normalize_reference_audio_trims_and_converts():
    raw_bytes = _make_dummy_wav_bytes(duration_ms=15000)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "norm.wav")
        normalize_reference_audio(raw_bytes, out_path, max_duration_ms=10000)
        processed = AudioSegment.from_wav(out_path)
        assert processed.channels == 1
        assert processed.frame_rate == 24000
        assert len(processed) == 10000


def test_stitch_timeline_segments():
    clip_bytes = _make_dummy_wav_bytes(duration_ms=1000)
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = os.path.join(tmpdir, "c1.wav")
        p2 = os.path.join(tmpdir, "c2.wav")
        with open(p1, "wb") as f:
            f.write(clip_bytes)
        with open(p2, "wb") as f:
            f.write(clip_bytes)

        # Segment 1 at 0s (1s long), Segment 2 at 3s (1s long) -> Total 4000ms
        stitched_bytes = stitch_timeline_segments([(0, p1), (3000, p2)])
        master = AudioSegment.from_wav(io.BytesIO(stitched_bytes))
        assert len(master) == 4000
