import re
import numpy as np
from pydub import AudioSegment
from f5_tts.api import F5TTS

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    # Lazy-load model inside the function to optimize app startup memory
    f5tts = F5TTS()

    generated_segments = []
    max_required_duration = 0

    for start_time, speech_text in zip(timestamps, texts):
        if not speech_text: 
            continue

        wav, sr, _ = f5tts.infer(
            ref_file=voice_path,
            ref_text=ref_text,
            gen_text=speech_text
        )

        wav_bytes = (wav * 32767).astype(np.int16).tobytes()
        speech_segment = AudioSegment(data=wav_bytes, sample_width=2, frame_rate=sr, channels=1)
        
        generated_segments.append((start_time, speech_segment))
        
        end_position = start_time + len(speech_segment)
        if end_position > max_required_duration:
            max_required_duration = end_position

    # Construct the silent backing track timeline
    padding_ms = int(padding_seconds * 1000)
    final_presentation_audio = AudioSegment.silent(duration=max_required_duration + padding_ms)

    for start_time, speech_segment in generated_segments:
        final_presentation_audio = final_presentation_audio.overlay(speech_segment, position=start_time)

    final_presentation_audio.export(output_path, format="wav")
